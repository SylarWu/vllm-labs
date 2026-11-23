# Lab 05 · 分页 KV 与端到端 Engine

**状态：已完成。Lab 05 验收 `3 passed`；Lab 01～05 回归 `23 passed, 1 skipped`。**

## 本 lab 要打通的路径

```text
Request
→ Scheduler 在 tick 边界尝试准入
→ BlockTable 为请求预留物理页
→ PagedKV 作为该请求的 KVStore 视图
→ TinyQwen3.forward 通过 KVStore 写入并读取历史
→ 从最后位置 logits 贪心采样
→ Scheduler.complete 提交游标、输出计数和 TokenEvent
→ 结束或取消时释放该请求的页
```

Lab 03 解决“逻辑位置怎样映射到物理页”，Lab 04 解决“本轮哪些请求可以执行”。本 lab 新增的是**物理 KV 张量池**和**把各组件组合成同步 Engine 的执行循环**。当前仍逐个执行 `Step`；同一 tick 有多个 decode 请求，并不代表已经做了一次真正的 batched forward，后者属于 Lab 06。

## 区分索引、存储和请求视图

| 对象 | 职责 | 不应承担的职责 |
| --- | --- | --- |
| `BlockPool` | 管理物理块 ID 的分配与归还 | 不知道 K/V 张量元素 |
| `BlockTable` | 管理一个请求的逻辑保留范围及逻辑块到物理块的映射 | 不保存 K/V 数值 |
| `PagedStorage` | 持有全模型共享的 K、V 两个物理张量 | 不知道调度和请求生命周期 |
| `PagedKV` | 绑定一个 `BlockTable`，实现 `KVStore` 的按层 append/read | 不拥有或复制整个物理池 |

`storage.view(table)` 创建的是共享底层存储的请求视图，不是 KV 数据副本。每个活跃请求有自己的 `BlockTable` 和 `PagedKV`，但所有视图操作同一个 `PagedStorage`。

## 物理张量布局

K、V 都是：

```text
[L, num_blocks, block_size, Hkv, D]
```

- `L`：模型层数。每层有独立的 K/V，层之间不能混读。
- `num_blocks`：全池物理块数，不是当前请求块数。
- `block_size`：每块能容纳的 token 位置数。
- `Hkv`：KV head 数，不是 Q head 数。
- `D`：每个 head 的维度。

可以把中间两维看成扁平的物理 token 轴，其长度为 `num_blocks * block_size`。对于逻辑位置 `p`：

```text
logical_block, offset = divmod(p, block_size)
physical_block = table.block_ids[logical_block]
flat_slot = physical_block * block_size + offset
```

`BlockTable.slots(start, count)` 已按逻辑顺序返回扁平 slot。例如块大小为 2、映射为 `[3, 1]` 时，逻辑位置 `[0, 1, 2, 3]` 对应物理 slot `[6, 7, 2, 3]`。

在测试 toy 模型中，`L=2, Hkv=2, D=8`；`tests/test_lab05_engine.py` 的独立存储测试则使用 `D=3`。实现应从构造参数读取维度，不能写死 toy 配置。

## Scatter 与 Gather

模型每层产生的 `key`、`value` 形状均为 `[T, Hkv, D]`，其中 `T` 是本次输入长度。

**写入是 scatter**：先由 `table.slots(start_pos, T)` 得到 T 个物理位置，再把 K/V 写入对应层、对应 slot。物理位置可以不连续，但输入的第 `i` 个 token 必须写入第 `i` 个映射 slot。分页只改变存放地址，不改变 token 的逻辑顺序。

**读取是 gather**：读取 `[0, length)` 时，先取得这段逻辑范围的 slots，再按这个顺序取出 K/V，返回 `[length, Hkv, D]` 的普通连续张量。这样现有 GQA、RoPE 和 causal mask 无需因分页改变数学语义。

初版采用 gather 后普通 attention，是教学上的正确性优先方案。它会产生连续历史副本，并未获得 PagedAttention kernel 直接读页的计算和带宽收益；本 lab 主要学习内存管理与正确地址翻译。

## PagedKV 的按层写入长度

`BlockTable.reserved_tokens` 表示已经预留的逻辑容量，不代表 K/V 已经写入。应另外维护每层已写入长度，语义与 `ContiguousKV` 一致：

```text
write(layer, start_pos, K, V)
要求 start_pos == written_length[layer]
成功后将该层长度推进到 start_pos + T

read(layer, length)
要求 0 < length <= written_length[layer]
```

一次 forward 顺序经过所有层，所以不同层暂时具有不同已写长度是正常的。不能用某一层的写入长度代替其他层，也不能把预留长度直接当作历史长度，否则会读到未初始化或别的请求遗留的数据。

写入前至少检查：层号、K/V 形状一致、 `[T,Hkv,D]` 中各维合法、`start_pos` 与当前层长度一致、写入范围不超过已保留范围。沿用 Lab 02 的原则：先校验，再提交；失败不能推进长度，更不能留下只写了一半的状态。

`release()` 释放的是块所有权，不要求逐元素清零物理张量。只要分配、写入长度和读取边界正确，旧字节不会被当作新请求的有效历史。

## 为什么多个 layer 共用一个 BlockTable

一个请求的 `BlockTable` 描述的是**逻辑 token 位置到物理块位置的映射**，而不是某一层的专属张量。每个逻辑 token 在每一层都各有一份 K/V，因此完整地址由两部分组成：

```text
layer                 选择物理张量的层轴
BlockTable.slots(...) 选择该层内的物理 token slot

K/V 元素地址 = [layer, physical_block, offset, kv_head, head_dim]
```

`BlockTable` 不需要 layer 维，因为同一套逻辑位置映射会应用到所有层。假设请求有两层、已经预留 8 个 token 位置：

1. layer 0 从位置 5 写入 3 个 token，写入 `[5,8)`，只推进 layer 0 的写入长度。
2. layer 1 随后也从位置 5 写入 3 个 token，使用同一组 slots，但位于物理张量的另一个 layer 切片。
3. 两层都完成后，请求仍有 8 个逻辑 token 位置，不是 16 个；每层分别保存了这 8 个位置的 K/V。

“每层写 3 个，一共写 6 个”描述的是两层产生的 K/V 数据份数，不是新增了 6 个逻辑 token 位置。页池按整个模型共同计数：一个物理块 ID 对应所有层中该块的存储切片，容量计算中的 `block_size` 仍表示每块的逻辑 token 数。若给每层独立建表并分别分配，就会重复申请页，违背本 lab 的池容量语义。

当前扁平存储写法 `[L, num_blocks * block_size, Hkv, D]` 与文档的五维写法等价，只要先合并 block 和块内偏移两维，并始终使用 `physical_block * block_size + offset` 寻址即可。

## PagedStorage / PagedKV 第一次代码复盘

定向命令：

```bash
.venv/bin/python -m pytest -q tests/test_lab05_engine.py -k paged_store
```

结果：`1 failed, 2 deselected`。非连续页读取、层隔离和请求隔离断言已经通过；最后一个“超出预留范围的写入必须报错”未通过。

### 已确认正确的部分

- 构造参数和共享 K/V 张量已建立，扁平物理 slot 轴与五维布局语义一致。
- `view(table)` 返回共享 storage 的请求视图，没有复制物理数据。
- 每层写入长度分别记录；`read` 按该层实际写入长度限制读取。
- 写入按 `slots` 顺序 scatter，读取按相同顺序 gather，层索引和 slot 索引的组合正确。

### 需要亲自修正的语义问题

1. **`write` 不应自动扩容。** `_write` 中的 `table.reserve(start_pos + token_length)` 把预留管理混进了数据写入。Lab 05 的 Engine 会在准入时一次性预留足够容量，PagedKV 应只检查本次写入是否落在既有预留范围内。测试里的 table 只预留 3 个位置，最后从位置 3 再写一个 token 时，当前实现自动扩到 4，因而没有按约定报错。
2. **append 校验缺少按层写入长度。** 当前只检查 `start_pos <= table.reserved_tokens`，所以允许覆盖旧位置或从未写位置跳跃写入。应与 Lab 02 一样要求 `start_pos == written_length[layer]`，否则写入长度可能倒退，或历史出现空洞。
3. **变量命名混淆容量和写入进度。** `allocated_token_length` 实际记录的是每层已写入长度；`allocated/reserved` 已由 block table 表达。改成能体现 `written_length` 含义的名字，有助于避免把两种长度混用。
4. **空写入与错误原子性值得补齐。** K/V 的 `T=0` 当前没有被明确拒绝；同时应避免在全部边界检查之前改变 table、张量或写入长度。先沿用 `ContiguousKV` 的输入契约，再考虑优化。

逐 slot Python 循环在当前 CPU toy 阶段能清楚表达地址翻译，先保证语义正确。后续可用张量索引减少 Python 循环，但这只是性能优化，不应与当前正确性修正混在一起。

### 修正后复验

当前实现已移除写入时的自动扩容，并加入 `start_pos == writen_length[layer]` 的按层 append 约束；写入终点越界由 `slots()` 在复制数据前拒绝。重新执行分页存储定向测试，结果为 `1 passed, 2 deselected`。

职责边界可准确表述为：`PagedStorage` 一次性创建共享物理张量；`BlockPool` 管理物理块使用权；`BlockTable` 管理一个请求的逻辑块映射和预留长度；`PagedKV` 将 layer 与共享 slot 映射组合成最终地址。不是 BlockTable 完全不参与物理块分配，而是它不直接创建或持有物理张量。

仍有两个不阻塞当前验收的小问题：`written_length` 应拼写为 `written_length`；`T=0` 的空写入尚未像 `ContiguousKV` 一样明确拒绝。当前正向测试已通过，不代表这些输入契约已经完整。

## Engine 的准入与容量预留

对 prompt 长度为 `P`、最多生成 `N` 个 token 的请求，实际需要缓存的最多位置数为：

```text
reserved_tokens = P + N - 1
required_blocks = ceil(reserved_tokens / block_size)
```

减 1 是因为最后采样的 token 不会再次输入模型，因而不需要写入 KV cache。例如 `P=3,N=2` 时，prefill 写 prompt 的 3 个位置，之后只 decode 第 1 个生成 token，总计 4 个位置。

本 lab 采用**活跃请求准入时一次性预留最大所需页数**的策略：

1. `submit` 时若 `required_blocks > pool.capacity`，说明请求即使独占整个池也永远无法运行，应立即拒绝。
2. 若请求可以放入整个池，但当前空闲块不足，则不是永久非法请求，应留在等待队列。
3. Scheduler 每轮按 FIFO 调用 `try_admit`；成功时才创建/增长该请求的预留并让它进入活跃集合。
4. 已经活跃的请求不需要随 decode 再申请页，因此不会因彼此占走剩余页而在生成中途饿死。

需要区分“请求数准入”和“KV 容量准入”：`max_active` 限制同时活跃请求数量，块池限制这些请求合计的 KV 预留。两者任一不足都会让队首继续等待。严格 FIFO 也意味着不越过暂时放不下的队首去接纳队尾请求。

一次性预留的代价是保守占用：请求可能提前遇到 EOS，或尚未接近最大输出长度，预留尾页却已经不能给别的请求使用。这是用空间闲置换取活跃请求执行期间资源确定性，不是唯一可行的调度策略。

## 请求资源的生命周期

Engine 需要维护 `request_id → BlockTable/PagedKV` 的映射，只在准入成功后为请求建立分页资源。

- **等待请求取消**：尚未持有页，只从 Scheduler 移除，不应误释放其他资源。
- **活跃请求取消**：释放对应 `BlockTable`，删除资源映射。
- **EOS 或长度完成**：Scheduler 的 `_on_retire` 回调触发相同释放路径。
- **重复结束或重复取消**：不能重复归还同一批页。

可以让退休与取消统一经过一个幂等的有资源释放路径：只有确实存在该请求的资源记录时才执行 `release()`。`BlockTable.release()` 本身已幂等，但 Engine 的资源字典也应及时清理，避免元数据泄漏和把旧视图误交给同 ID 的新请求。

## 一个 Step 的端到端时序

Lab 05 中每个 `Step` 都会发出一个 token：

1. 调用 `scheduler.plan()`，完成本 tick 的准入并取得步骤列表。
2. 对每个步骤找到其 `PagedKV`。
3. 将 `step.input_ids` 构造成 `[1,T]`、dtype 为 `torch.long`、位于模型设备上的张量。
4. 调用 `model(input_ids, cache=paged_kv, start_pos=step.start_pos)`。
5. 取 `logits[:, -1, :]`，沿词表维 `argmax` 得到下一个 token ID。
6. 调用 `scheduler.complete(step, sampled_token)`，由 Scheduler 推进输入位置、输出计数并生成 `TokenEvent`。
7. 若请求结束，`_on_retire` 在 `complete` 内同步释放页；否则下一轮会把刚采样的 token 作为 decode 输入。
8. 收集本轮事件并返回；没有计划步骤时返回空列表。

注意时序：本次 forward 只把**本次输入 token**的 K/V 写入 cache；本次 logits 采样出的 token 要等下一次 forward 才会产生自己的 K/V。这与容量公式中的 `-1` 是同一个事实。

## Engine 初始化与回调事务

Engine 是这些组件的单 owner。初始化时从 `model.config` 读取层数、KV head 数和 head dim，从模型参数确定 device 和 dtype，再建立一组共享资源：一个 BlockPool、一个 PagedStorage，以及请求资源字典。真实模型下 KV storage 的 dtype 应与模型一致，否则写入时隐式转换会改变内存占用和数值路径。

`max_batch_tokens`、`max_prefill_chunk` 在 Lab 05 中尚未改变执行策略，但应原样传给 Scheduler，为 Lab 06 保留统一入口。

Scheduler 的 `try_admit` 在本实验中不仅回答“是否放得下”，还在返回 True 前完成该请求的页预留和视图建立。返回 False 时必须不改变任何状态。它每次被调用只处理队首请求；成功返回 True 后，Scheduler 才把请求从 waiting 移入 running。因此不要在 `plan()` 后再重复分配一次。

`submit` 的永久容量校验必须发生在请求进入 Scheduler 之前，避免抛出“太大”错误后队列里却残留请求。暂时容量不足不走这个拒绝分支，而由后续 tick 的 `try_admit` 处理。

`has_work` 必须同时覆盖 active 和 waiting。只要还有等待请求，外部驱动循环就不能提前退出。`free_blocks` 直接反映共享池的当前空闲块数。

执行 forward 和采样可使用 `torch.inference_mode()`，避免为推理建立 autograd 图。Lab 05 逐 Step 运行；即使本轮返回多个事件，也只是连续执行了多个单请求 forward。

## Engine 第一次代码复盘：临时视图丢失写入进度

定向结果：

```text
small_pool：1 passed, 2 deselected
serving：1 failed, 2 deselected
```

失败发生在 A 的第一次 decode：Scheduler 正确给出了 `start_pos=4`，但传入 forward 的 PagedKV 中 layer 0 的写入长度仍为 0，于是 append 校验拒绝写入。

根因不是 `written_length` 没有执行更新，而是 Engine 只持久化了 BlockTable，并在每次 forward、甚至打印调试信息时重新调用 `storage.view(table)`。该方法每次执行 `PagedKV(...)`，返回一个新对象。每个新对象都有自己的 `written_length=[0,0]`，即使它们共享同一个 storage 和 table，也不会共享这份 Python 列表状态。

```text
准入：创建 BlockTable，预留 A 的页
Prefill forward：创建 PagedKV #1 → 写入长度推进到 [4,4] → 对象随后无人持有
打印：创建 PagedKV #2 → 观察到 [0,0]
Decode forward：创建 PagedKV #3 → start_pos=4，但 layer 0 长度为 0 → 报错
```

BlockTable 只保存地址和预留范围，不保存各层写入进度。因此，**同一活跃请求的 PagedKV 也必须与 BlockTable 一样具有跨 tick 的稳定身份**。在准入时创建一次，后续 prefill/decode 都取得同一实例，退休时同时清理请求资源。

`small_pool` 没捕获这个问题，是因为 A 在 prefill 后立即取消，B 在首次 prefill 发出一个 token 后即达到长度上限，二者都没有进行第二次 forward。测试通过只能证明覆盖到的行为正确，不能证明未覆盖的 decode 连续性正确。

调试此类问题可打印视图对象的 `id` 以及每轮写入前后的长度。如果每轮对象 ID 改变，就应先检查对象生命周期，而不是怀疑列表的加法。

另一个独立问题是 `_prepare_kv_cache` 标注返回 `BlockTable`，实际没有 return。若调用方只依赖副作用，暂时不触发失败，但类型标注和实现应保持一致。

Engine 其余已检查部分符合当前约定：从模型配置建立 storage，提交时拒绝永久超限请求，准入时一次性预留，退休回调释放 table，`has_work` 同时覆盖等待和活跃请求，forward/采样处于 inference mode。

## Lab 05 完成总结

最终实现将 `request_id → PagedKV` 保存在 Engine 中。准入时创建一次视图并预留页，所有 Step 直接复用同一个实例；退休时弹出该实例并通过 `cache.table.release()` 归还页。这样三种状态的生命周期已经对齐：

- 共享 PagedStorage 由 Engine 长期持有。
- 请求的 BlockTable 由稳定的 PagedKV 持有。
- 请求的按层 written length 随 PagedKV 跨 tick 保留，退休时一起清理。

最终验收：

```text
tests/test_lab05_engine.py：3 passed
Lab 01～05 回归：23 passed, 1 skipped
```

覆盖了非连续页 scatter/gather、请求与层隔离、混合请求输出与串行完整重算一致、小页池等待、等待/活跃请求取消、正常完成后的页回收。跳过项仍是可选真实模型测试，不影响 CPU toy 验收。

本 lab 最核心的心得是：**共享底层数据不代表可以丢弃有状态视图。** BlockTable 保存地址，PagedStorage 保存数值，PagedKV 保存逐层写入进度；只有把三者的所有权和生命周期分清，append 校验才有意义。

当前逐 Step forward、gather 成连续历史以及准入时最大长度预留，都是 Lab 05 有意保留的简化。Lab 06 将在此基础上合并 decode forward，并引入 chunked prefill，不需要反过来推翻本 lab 的分页存储抽象。

## 建议实践顺序与验收观察

1. 先独立实现 `PagedStorage/PagedKV`，用手算的非连续页映射验证 scatter/gather、跨页读写、层隔离和请求隔离。
2. 再实现 Engine 的持有关系与准入回调，不接模型时也可用 `Scheduler` trace 检查页分配、等待、退休和取消。
3. 最后连接真实 toy forward，将输出与 `greedy_generate` 的串行完整重算结果对拍。
4. 每轮记录 `active_ids`、`waiting_ids`、`free_blocks` 和事件，解释为什么等待请求在某个完成/取消边界后才开始运行。

```bash
.venv/bin/python -m pytest -q tests/test_lab05_engine.py -k paged_store
.venv/bin/python -m pytest -q tests/test_lab05_engine.py
```

本阶段重点不是“测试刚好变绿”，而是能解释：分页地址为什么仍保持逻辑顺序、为什么容量是 `P+N-1`、为什么等待请求不能分配页、为什么退休回调能避免页泄漏。

[实现入口](../src/tinyserve/paging.py)、[Engine 入口](../src/tinyserve/engine.py)、[验收测试](../tests/test_lab05_engine.py)、[Lab 03 页映射](lab03_paging.md)、[Lab 04 调度](lab04_scheduler.md)
