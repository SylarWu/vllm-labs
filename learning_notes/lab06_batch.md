# Lab 06 · 真 Batch Decode 与 Chunked Prefill

**状态：学习中。模型纯 decode、混合 batch、mask 和非法输入边界均已验收；下一步接入 Engine。**

模型与调度侧验收：`tests/test_lab06_batch.py -k 'not engine'` 为 `12 passed, 2 deselected`。新增测试覆盖位置 0 安全占位的手推 mask、混合长度 1/3/2 的真实 logits 对拍、全输出有限性、逐层 cache 推进、后续 decode，以及非法元数据在写 cache 前被拒绝。

## Lab 01～06 的整体串联

```text
Lab 01：单请求/单序列的数学计算
  Token IDs → Embedding → L × Decoder → logits
  建立 RMSNorm、RoPE、causal GQA、SwiGLU 和残差结构

Lab 02：让多轮生成不必重算全部历史
  prefill 写整段 prompt KV → 每次 decode 只输入上一个 token
  建立 KVStore、start_pos、processed/generated 的时序语义

Lab 03：让 KV 的物理位置不连续
  BlockPool 管块所有权，BlockTable 管逻辑 token → 物理 slot

Lab 04：让多个请求按 iteration 交替前进
  Scheduler 管 waiting/running、Step、complete、EOS/长度和取消

Lab 05：把数学、分页和调度接成 Engine
  PagedStorage 持有物理 K/V，PagedKV 持有请求视图和逐层写入长度
  Engine 准入时预留页，执行 Step 后采样并提交，退休时回收资源

Lab 06：减少重复 kernel 调用，并限制长 prefill 对 decode 的阻塞
  多个 decode 行合并成一次模型 forward
  长 prompt 分为多个 chunk，中间 chunk 只写 KV，不发 token
```

本 lab 不是重写前五周，而是保留现有抽象：

- `KVStore` 协议不变，PagedKV 和 ContiguousKV 都能被 batched decode 使用。
- 分页地址仍由 PagedKV 处理，模型无需知道 block ID。
- PagedKV 实例仍需跨 tick 稳定持有。
- Scheduler 仍负责计划与状态提交，Engine 仍负责执行。

## 第一部分：什么是“真 batch decode”

Lab 05 同一 tick 有两个 decode Step 时，Engine 实际执行：

```text
forward(A 的一个 token)
forward(B 的一个 token)
```

这只是迭代级调度，不是张量级 batching。Lab 06 要变成：

```text
input_ids: [B,1]
caches:    B 个独立 KVStore
positions: [S0, S1, ..., SB-1]
→ decode_batch 一次
→ logits: [B,1,V]
```

每一行仍对应一个独立请求。batch 维让各请求共享同一组模型权重并共同执行 Linear、Norm、MLP 和 attention 计算，但不共享请求历史。

例如 A 已有 2 个 prompt 位置，B 已有 5 个：

```text
A: 当前输入 token 的绝对位置 = 2，写后历史长度 = 3
B: 当前输入 token 的绝对位置 = 5，写后历史长度 = 6
```

二者可以在同一个 `[B,1]` 张量里进入模型，但 RoPE 位置、cache 写入起点和可读历史长度必须逐行独立。

## decode_batch 的张量路径

`TinyQwen3.decode_batch` 的主干仍是 Lab 01 的计算：

```text
[B,1] token IDs
→ embedding [B,1,D_model]
→ 每层 Pre-Norm
→ Q/K/V projections
→ 每行独立 RoPE
→ 每行写自己的 KV cache
→ 读历史、pad、mask、batched GQA
→ O projection + 残差
→ MLP + 残差
→ final norm + LM head
→ [B,1,V]
```

Q/K/V 投影后仍转换成：

```text
Q: [B,Hq,1,D]
K: [B,Hkv,1,D]
V: [B,Hkv,1,D]
```

Norm、Linear、残差和 MLP 本来就能处理 batch 维，不应为每行重新调用完整模型。

## 每行独立 RoPE

现有 `apply_rope` 已支持：

```text
x:        [B,H,T,D]
positions: [B,T]
```

因此 decode 时传入 `[B,1]` 的绝对位置即可。不能把 A、B 的历史长度取最大值后共用一个位置，也不能把所有行都从位置 0 开始。

历史 K 在写入时已经旋转，本步只旋转当前新 token 的 Q/K；V 始终不做 RoPE。

实现位点：[model.py 的 apply_rope](../src/tinyserve/model.py) 已完成；`decode_batch` 需要构造每行 positions。

## 每层写各自的 KVStore

对第 `b` 行、第 `l` 层：

```text
cache[b].write(
  layer=l,
  start_pos=positions[b],
  K/V=[1,Hkv,D],
)
```

随后读取：

```text
length[b] = positions[b] + 1
cache[b].read(l, length[b]) → [length[b],Hkv,D]
```

读长度包含刚写入的当前 token。不能再把当前 K/V 额外拼一次。

允许用 Python 循环完成 B 个请求的 cache 写入与 gather；这不否定模型计算已经 batch 化。教学版的开销优化可以留到后面。

PagedKV 会继续完成 scatter/gather，所以 batched attention 收到的是普通连续张量。无需在 model 中读取 BlockTable。

## 为什么需要 padding

不同请求的历史长度不同，而矩形张量要求每行形状一致。设最大读长度为：

```text
Smax = max(lengths)
```

可将每行 K/V 转为 `[Hkv,length_b,D]`，再放入：

```text
padded K/V: [B,Hkv,Smax,D]
```

每行真实历史放在前缀，剩余尾部只是补齐轴长度。padding 建议置零，但**零填充本身不足以保证正确性**。

如果只用零填充而不 mask：

```text
attention = softmax(QK scores) @ V
```

padding 的 V 虽然是零，不产生分子向量，但 padding 的 Q/K 分数仍进入 softmax 分母，会分走真实历史的概率质量，导致输出改变。

## 变长历史 mask

对每行构造：

```text
valid[b,j] = j < length[b]
```

可以形成 `[B,1,1,Smax]` 的布尔 mask，广播到 attention scores：

```text
scores: [B,Hq,1,Smax]
```

在 softmax 前把无效位置填成 `-inf`。沿用 Lab 01 的教训：`masked_fill` 的 True 表示“要屏蔽”，所以传入的是允许集合的反面。

decode 每行只有一个 query，其真实 K/V 是截至当前位置的完整前缀，因此这里主要需要 padding mask。每行至少包含当前 token，不会出现整行全被 mask。

## 为什么不能原样复用 causal_gqa

当前 `causal_gqa` 假设：

- 所有行使用同一个 `query_start`；
- K/V 中所有位置都是真实有效位置；
- mask 只表达标量位置下的因果关系。

batched decode 增加了逐行长度和 padding 维度的语义。可以扩展一个共享 GQA 辅助函数以接受可选 key padding mask，也可以为单步 decode 建立专用路径，但必须保留原有 GQA head 展开、缩放和 FP32 中间计算，避免数值对拍出现无意义偏差。

实现位点：

- [model.py](../src/tinyserve/model.py) 的 `TinyQwen3.decode_batch` 是入口。
- 为减少重复，可在 `TinyQwen3Attention` / `TinyQwen3Decoder` 增加 batched decode 辅助路径。
- 现有单请求 `forward` 的行为应保持不变，Lab 01～05 回归是约束。

## decode_batch 的两层等价性验收

批量 decode 不仅要让本步输出正确，还要保证它写下的 KV 能支撑后续生成。因此需要连续检查两个不变量：

1. **本步 logits 等价**：不同历史长度的请求组成 batch 后，每行最后的 logits 等于该请求完整历史加当前 token 的一次性重算结果。这能发现 RoPE 位置、历史 gather、padding mask 和行顺序错误。
2. **下一步 KV 等价**：batch decode 后，再分别输入一个新 token 做普通 decode，其结果仍等于完整重算。这能进一步发现 cache 写入位置、写入内容和层索引错误。

只通过第一项并不充分：如果 attention 错用了临时拼接的当前 K/V，本步输出可能看似正确，但 cache 中并未留下正确历史，下一步才会暴露。

排错时应先固定单行、逐层检查，再扩展到不同长度的多行。若 batch 与逐条结果不一致，依次定位“投影前是否一致 → RoPE 后是否一致 → 写入 cache 后是否一致 → gather/pad 后是否一致 → mask 后 scores 是否一致”，不要一开始只观察最终 token；argmax 相同也可能掩盖 logits 的数值错误。

## decode 与 prefill chunk 的统一长度语义

`decode_batch` 只接收 decode 行，每行恰好输入一个 token，因此输入恒为 `[B,1]`。当 Scheduler 在一个 tick 同时返回 decode 和 prefill chunk 时，本 lab 的 Engine 将它们拆开执行：

- 所有 decode 行合并，调用一次 `decode_batch`。
- 每个 prefill chunk 仍分别调用普通 `forward`，使用自己的连续 KV 视图和 `start_pos`。
- 不为凑成 batch 把不同长度的 prompt chunk 填进 `decode_batch`。

这只是教学版的实现边界，并不表示 prefill 和 decode 在原理上不能进入同一个 batch。更一般的 `forward_batch` 可以让每行有自己的输入长度 `T_b`：decode 行满足 `T_b=1`，prefill chunk 行满足 `T_b=C_b`。但需要同时解决以下问题：

1. 输入按行变长，不能直接沿用 `[B,1]` 接口。
2. 每行有独立 RoPE positions，且只把真实 token 的 K/V 写入 cache。
3. attention 同时需要因果约束、历史 K/V padding mask 和输入 query padding mask。
4. 中间 chunk 不采样，最后 chunk 在自己最后一个真实位置采样，decode 在其唯一位置采样。

一种实现是把输入右填充为 `[B,Tmax]`。对第 b 行、chunk 内第 i 个 query 和历史第 j 个 key，允许可见的条件为：

```text
i < T_b                    # query 是真实输入
j < p_b + T_b              # key 是写入后的真实历史
j <= p_b + i               # 不能看未来
```

这里不能统一取 `logits[:,-1]`：右填充后，最后一个位置可能是无效 query；中间 prefill chunk 甚至没有输出事件。若把无效 query 的全部 scores 都屏蔽，softmax 还可能出现全 `-inf` 行，因此需要避免计算或妥善处理这些占位行，并最终丢弃其结果。

另一种更接近高性能实现的方式是把所有真实 token 展平成 `[sum(T_b),D]`，配合每个请求的起止偏移和变长 attention，不做矩形 padding。它能减少无效计算，但元数据和 kernel 接口明显更复杂。

本 lab 先采用“decode 合批、prefill 分开”的方案，是为了独立验证 KV 与 mask 正确性；掌握后可以再扩展为统一混合 batch。

更一般的 cache 长度公式不局限于 decode。设本次输入长度是 `T`，起始绝对位置是 `p`：

```text
本次写入位置：[p, p + T)
写入 K/V：[T,Hkv,D]
写入后长度：S = p + T
本层读取范围：[0, p + T)
```

decode 只是 `T=1` 的特例，所以 `S=p+1`。prefill chunk 的长度为 `C` 时，一次写入整个 `[C,Hkv,D]`，随后读取长度 `p+C`。例如 prompt 长 7，按 3、3、1 分块：

| Chunk | 输入位置 | 本次写入数 | 写后/读取长度 |
| --- | --- | --- | --- |
| 1 | `[0,3)` | 3 | 3 |
| 2 | `[3,6)` | 3 | 6 |
| 3 | `[6,7)` | 1 | 7 |

中间 chunk 虽然不采样，仍要正常经过每层并写入 K/V；后续层和后续 chunk 都依赖这些结果。

prefill chunk 的 attention 还需要因果约束。chunk 内第 `i` 个 query 的绝对位置是 `p+i`，只能读取满足 `j<=p+i` 的 key：它可以看旧 cache 和本 chunk 中不晚于自己的 token，不能看本 chunk 中更晚的 token。最后一个 chunk 与中间 chunk 的计算规则相同，区别只是前者会使用末尾 logits 采样，后者不采样。

## padding 对齐的是历史 K/V，不是 decode 输入

decode 的输入本来就统一为 `[B,1]`，不需要给 token IDs 补 padding。需要补齐的是从各请求 cache 读出的历史 K/V，因为它们的 `S_b=p_b+1` 不同，而 batched attention 需要统一的 `[B,Hkv,Smax,D]`。

因此 mask 表达的是“这个 key/value 槽位是否属于该行的真实历史”，不是“这个输入 token 是否有效”。无效槽位必须在 attention scores 进入 softmax 前被屏蔽；仅仅把输入或 V 填零并不能替代 mask。

## Engine 如何只调用一次 batched decode

`scheduler.plan()` 可能同时返回 decode 和 prefill。Engine 应先分类：

```text
decode_steps
prefill_steps
```

对于所有 decode：

1. 按 decode_steps 的顺序堆叠 input IDs 为 `[B,1]`。
2. 按完全相同的顺序收集 PagedKV 和 start_pos。
3. 调用一次 `model.decode_batch`。
4. 第 i 行 logits 对应第 i 个 Step，逐行 argmax。
5. 分别调用 `scheduler.complete(step_i, token_i)`。

顺序映射不能打乱，否则会把 A 的 token 提交给 B。prefill chunk 在本 lab 仍可逐个普通 forward。

实现位点：[engine.py](../src/tinyserve/engine.py) 的 `step()`。验收 spy 要求第二轮有两个 decode 行时，`decode_batch` 恰好调用一次，普通 `forward` 不调用。

## 第二部分：为什么需要 chunked prefill

Lab 05 一次 prefill 吃掉整个 prompt。一个长 prompt 的计算期间，已有 decode 请求必须等这一轮结束，表现为 ITL 变长。

chunked prefill 把长 prompt 分为多轮：

```text
prompt 长度 7，chunk 上限 3
第 1 轮：位置 [0,3)，只写 KV
第 2 轮：位置 [3,6)，只写 KV
第 3 轮：位置 [6,7)，写 KV，并采样第一个新 token
```

每一轮都可同时安排其他请求的 decode。这样用长请求自己的 TTFT 和更多调度轮次，换取其他生成请求更平滑的 ITL。

chunking 改变的是**一次 forward 消费多少 prompt**，不改变最终逻辑历史。所有 chunk 完成后，KV 必须与整段 prefill 等价。

## Step.emit_token 的完整语义

Lab 04/05 中每个 Step 都发 token；Lab 06 引入第三种执行结果：

| Step | 是否 forward/写 KV | 是否采样 | complete 的 sampled_token | 返回值 |
| --- | --- | --- | --- | --- |
| 中间 prefill chunk | 是 | 否 | `None` | `None` |
| 最后 prefill chunk | 是 | 是 | token ID | TokenEvent |
| decode | 是 | 是 | token ID | TokenEvent |

`emit_token=False` 不表示“这轮不执行模型”，而表示“不消费 logits、不增加输出计数、不设置 last_token、不产生对外事件”。

因此测试里取消半个 prefill 前，`engine.step()` 可以返回空列表，但 cache 已经写入了一个 chunk。空事件列表不等于空执行。

## Scheduler 的三个状态如何推进

`processed_tokens`：已消费并写入 KV 的输入位置数。每个 prefill chunk 和 decode 都按 `len(input_ids)` 增加。

`generated_tokens`：已经对外发出的新 token 数。仅 `emit_token=True` 时增加。

`last_token`：上一轮发出的、下一轮 decode 要消费的 token。中间 chunk 必须保持不变，最后 prefill chunk 和 decode 才更新。

prefill 是否结束应由：

```text
processed_tokens < len(prompt_ids)
```

判断，而不能继续只靠 `last_token is None`。中间 chunk 后 last_token 仍是 None，但下一轮 start_pos 必须等于 processed_tokens，input_ids 必须取 prompt 的对应切片。

`complete` 需要先校验 emit 与 sampled_token 的组合，再提交状态：

- `emit_token=False`：只接受 None，推进 processed，不判断 EOS/长度，返回 None。
- `emit_token=True`：要求 token，推进 processed/generated/last_token，再按 EOS 优先于长度判断退休。

实现位点：[scheduler.py](../src/tinyserve/scheduler.py) 的 `_generate_step`、`plan`、`complete`。

## token budget 与 decode 优先

每轮先做 tick 边界准入，再为活跃请求安排工作：

1. 所有 decode 行优先，每行消耗 1 个 token budget。
2. 剩余 budget 按活跃 prefill 顺序分配。
3. 每个 prefill 请求本轮至多得到一个 chunk。
4. chunk 同时受剩余 prompt、`max_prefill_chunk` 和剩余 budget 限制。
5. 只有吃到 prompt 末尾的 chunk 设置 `emit_token=True`。

`max_batch_tokens` 有值时必须至少等于 `max_active`。否则如果所有活跃请求都处于 decode，连每行一个 token 都容纳不下，无法满足 decode 优先并可能造成停滞。

需要区分这些“token 数”：

| 配置 | 限制什么 |
| --- | --- |
| `max_active` | 活跃请求行数 |
| `max_batch_tokens` | 一轮计划消费的输入 token 总量 |
| `max_prefill_chunk` | 单个请求一轮最多消费的 prompt token 数 |
| `block_size` | 一个物理块可容纳的 KV token 位置数 |
| `P+N-1` | 活跃请求需要预留的 KV 位置数 |

token budget 是每轮计算配额，不会替代 Lab 05 的 KV 页预留。中间 prefill 取消时，仍由 Lab 05 的退休回调释放该请求的全部预留页。

## 手推验收 trace

配置：

```text
max_active=2, max_batch_tokens=4, max_prefill_chunk=3
chat：prompt 长 2，已 prefill 并发出 token 20
long：prompt 长 7，max_new_tokens=1
```

| Tick | chat | long | budget |
| --- | --- | --- | --- |
| 1 | decode，1 token | prefill `[0,3)`，不发 token | 1+3=4 |
| 2 | decode，1 token | prefill `[3,6)`，不发 token | 1+3=4 |
| 3 | decode，1 token | prefill `[6,7)`，发 token 31 | 1+1=2 |

long 前两次 complete 都只推进 processed；最后一次发出 token 并因长度上限退休。chat 每轮都能继续 decode。

## 实践顺序与实现清单

1. **Scheduler 纯状态测试**：先实现 chunk 切片、decode 优先、emit/None 提交和构造参数校验，不接模型。
2. **模型 decode_batch**：完成逐行 RoPE、独立 cache 写入、gather/pad/mask 和 logits 对拍。
3. **Engine batch 接入**：将 decode 行合并一次调用，prefill 仍逐个执行，注意行顺序和事件过滤。
4. **端到端 chunk**：验证中间 chunk 无事件、取消释放页、最终输出与完整重算一致。
5. **回归 Lab 01～05**：尤其检查原单请求 forward 和 Lab 04 的默认调度语义未被破坏。

```bash
.venv/bin/python -m pytest -q tests/test_lab06_batch.py -k scheduler
.venv/bin/python -m pytest -q tests/test_lab06_batch.py -k real_batch
.venv/bin/python -m pytest -q tests/test_lab06_batch.py -k one_batched
.venv/bin/python -m pytest -q tests/test_lab06_batch.py -k chunked_engine
.venv/bin/python -m pytest -q tests/test_lab06_batch.py
```

[模型入口](../src/tinyserve/model.py)、[调度器入口](../src/tinyserve/scheduler.py)、[Engine 入口](../src/tinyserve/engine.py)、[验收测试](../tests/test_lab06_batch.py)

## decode_batch 第一次代码复盘

### 当前测试阻塞点：K/V 聚合张量形状错误

`forward_batch` 将 `final_k/final_v` 创建为 `[B,Smax,hidden_size]`，但 `KVStore.read` 返回 `[S_b,Hkv,D]`，而 `general_causal_gqa` 的 Q/K/V 布局约定是 `[B,H,S,D]`。这里同时缺少 head 轴，并误用了 `hidden_size`；K/V 的最后一维应是 `D`，head 轴长度应是 `Hkv`，不是 `Hq`，也不一定满足 `hidden_size=Hkv*D`。

实测 `test_real_batch_logits_match_different_length_cached_and_full_rows` 在首次给 `final_k` 赋值时失败：

```text
RuntimeError: expand(torch.FloatTensor{[3,2,8]}, size=[3,24]):
the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (3)
```

其中 `[3,2,8]` 正是 `[S_b,Hkv,D]`。修正方向是统一 gather 后的布局与 attention 的 `[B,Hkv,Smax,D]` 约定，并在从 cache 的 `[S_b,Hkv,D]` 布局转换时显式调整轴顺序。

### 混合 batch 不能只把 actual_length 传给 mask

当前 KV 写入使用 `k[b].permute(1,0,2)`，包含该行全部 `token_length=Tmax` 个位置，没有按 `actual_length[b]` 截断。对纯 decode，因为 `Tmax=T_b=1`，暂时不会暴露；对混合 decode/prefill，右侧 padding 会被当作真实 token 写入，导致写后游标和后续历史错误。只写入 `[0,T_b)`，读取长度仍应为 `p_b+T_b`。

公共 `TinyQwen3.decode_batch` 目前还把 `actual_length` 固定为全 1，所以它仍是纯 decode 接口。要支持混合 batch，需要从上层传入每行真实长度，并明确输入 padding、采样位置和中间 chunk 不发事件的契约。

### general_causal_gqa 的位置边界

`end_position=p+T` 是排他的写入后长度，最后一个真实 query 的绝对位置应为 `p+T-1`。当前 `prepare_positions <= end_positions` 多接受了一个位置，把越界位置 clamp 到 `end_positions` 也不是最后一个真实位置。对真实 query 尚未造成错误，但 padding query 的位置语义不正确。

更清晰的 mask 应分别表达：

```text
valid_query[b,i] = i < T_b
valid_key[b,j]   = j < p_b + T_b
causal[b,i,j]    = j <= p_b + i
```

真实 query 中，正确的 causal mask 实际上已经能间接屏蔽尾部 KV padding，因为最后一个真实 query 的位置是 `S_b-1`；显式 `valid_key` 仍能让契约更清晰、更稳健。padding query 若全部被 mask，softmax 可能出现全 `-inf` 行，需要安全处理并最终丢弃其输出，不能把它采样或写入 cache。

### 其他实现细节

- 聚合 K/V 的 `torch.zeros` 未指定 dtype，会默认使用 FP32；真实 BF16 模型中应显式保持与输入/cache 一致的 dtype。
- 增加混合 batch 测试时，应同时检查每行真实 logits、每层 `write_length=p_b+T_b` 和后续 decode 对拍，防止 padding 写入被最终输出掩盖。

## decode_batch 第二次代码复盘

### 已修正并验证

- gather 后 K/V 已改为 `[B,Hkv,Smax,D]`，cache 读出的 `[S_b,Hkv,D]` 通过显式轴转换写入对应前缀。
- 聚合张量已显式继承 K/V 的 dtype 和输入 device。
- KV 写入已按 `actual_lengths[b]` 截断，不再写入右侧 padding。
- attention mask 已增加 head 广播轴，`[B,Tq,Tkv] → [B,1,Tq,Tkv]` 与 scores 对齐。

验证结果：

```text
Lab 06：4 passed, 2 failed；失败均为 Engine 尚未接入 batch decode / emit_token=False
Lab 01～05：23 passed, 1 skipped
```

另用三个请求直接驱动底层混合长度路径，实际输入长度分别为 1、3、2。所有真实 query 的 logits 与逐条完整重算最大绝对误差为 `2.98e-7`，各层 cache 长度均正确推进到 `p_b+T_b`。这说明当前真实 token 的 RoPE、cache、gather、mask 和 GQA 数值路径一致。

### 仍需明确的边界与优化

1. 公共 `decode_batch` 应校验输入严格为 `[B,1]`。当前若误传 `[B,T>1]`，仍把每行 actual length 固定为 1，会只写一个 token 却继续计算整行，属于静默错误，最好尽早拒绝。
2. padding query 仍被赋予 `end_position=p+T`，因而可能看到一个额外的零填充 key。当前这些输出不会作为真实结果被使用，所以混合对拍仍正确；若保留矩形 padding 实现，可将占位 query 明确视为待丢弃结果，并使用最后真实位置等安全占位，避免其语义被误用。
3. 混合长度目前只由底层辅助函数支持，公共 `decode_batch` 仍固定 actual length 为 1。这是当前纯 decode 接口的边界，不代表已完成混合 batch 的输入、采样和事件接口。
4. 每层都会从 Python 序列创建 positions/mask 张量并分配 gather 缓冲区，且 cache gather 仍是逐行 Python 循环。教学版先保留以便观察正确性；性能优化应在端到端验收后通过 profiling 决定。
5. 普通 forward 与 adjustable forward 的投影、RoPE、残差流程存在重复。后续可抽取共享计算段，但应先保留独立实现对拍，避免为了去重同时改变两条路径而丢失参考基准。

## generate_mask 重构复盘：全掩码 query 的 NaN

将最后真实位置改为 `end_position=p+T-1`，并用 `prepare_position<=end_position` 判断真实 query，位置边界已经正确。padding query 被赋值为 -1 后，与非负 key 下标比较会得到全 False，因此其全部 key 都被屏蔽。

但这会让该 query 的整行 scores 都是 `-inf`：

```text
softmax([-inf,-inf,...,-inf]) → NaN
```

实测纯 decode 测试通过；混合长度 1、3、2 时，真实 logits 与完整重算的最大误差仍为 `2.98e-7`，cache 长度正确，但长度为 1 和 2 的两行在 padding logits 中出现 NaN。NaN 当前被隔离在占位位置，没有污染真实位置。

需要准确区分“真实结果错误”和“接口约束”：如果所有消费者都先按 `T_b` 选择真实位置，再进行采样、比较、归约或序列化，那么 padding NaN 不影响真实 token，保留它可以是一种合法设计。当前网络不会把 NaN 横向传播到真实位置：softmax 沿 key 轴独立处理每个 query，attention 中真实 query 不读取 padding query，残差、RMSNorm 和 MLP 也都按位置计算；padding K/V 又不会写入 cache。

因此，“必须修复 NaN 才能进入 Engine”是过强表述。真正必须保证的是消费契约，而不是 padding 数值必须有限。

可选修正方向：

1. 保留全 False mask，单独标记 `valid_query=allowed.any(key_dim)`；对无有效 key 的占位行使用安全 scores（例如全零）完成 softmax，再把对应 attention 输出置零。残差路径仍可继续，但最终丢弃占位输出。
2. 回到安全位置 clamp 的方案，attention 内不直接 mask 整行 query，只用 query validity 选择输出。

不建议在最终 logits 上笼统使用 `nan_to_num` 掩盖问题，因为它也会隐藏真实位置的数值错误。

### 两种可接受的设计契约

**允许 padding 未定义/为 NaN。** 明确 `forward_batch` 只有每行 `[0,T_b)` 是有效输出；Engine 必须先在 `T_b-1` 取最后真实 logits，再沿 vocab 维 argmax，且中间 chunk 不采样。任何 loss、指标、比较、序列化或全张量归约都要先过滤 padding。测试重点检查真实位置有限且对拍、cache 长度和下一步 decode，无需为占位位置增加安全 softmax 开销。

**要求整个返回张量有限。** 对无有效 key 的 query 做安全 softmax，并将占位 attention 输出清零。这对通用调用方和全张量测试更友好，但增加少量计算与代码，不是当前正确性的必要条件。

风险来自违反契约，而不是 NaN 会自动跨行扩散。例如在完整 `[B,Tmax,V]` 上直接 `argmax/max/sum/mean`、统一取 `[:, -1]`、做未过滤的 JSON 序列化或全量 `assert_close`，都会把无效占位值带入结果。

### 用位置 0 作为 padding query 的安全占位

在当前不变量下，也可把无效 `query_positions` 从 -1 改为 0：

```text
query_position=0 且 key_position>=0
→ 只有 key 0 满足 query_position>=key_position
→ softmax 在唯一可见的 key 0 上得到概率 1
→ 不再出现全 -inf 行和 NaN
```

每行 `T_b>=1` 时，写入后的历史前缀 `[0,p+T)` 至少包含位置 0，因此 key 0 确实是该请求的真实历史，而不是尾部 padding。真实 query 的位置和可见性完全不受这个占位值影响。

这只是用“attend 到第一个 token”构造一个有限但无业务意义的占位输出，仍不能采样或写回 padding 位置。它也不会减少矩形 attention 的计算量，只是省去专门的空行 softmax 处理。

代价是 `allowed_mask.any(key_dim)` 对 padding query 也会为 True，不能再据此推导 query validity；仍须以 `actual_lengths` 构造独立的有效位置判断。该方案依赖每行至少一个真实 token，因此公共入口的 `T_b>=1` 校验应作为明确前置条件。

### 当前 Lab 推荐的定稿方案

在进入 Engine 前，优先完成一小组边界清晰的优化，而不是继续改写 attention 数据结构：

1. padding query 使用位置 0 作为安全占位，避免全掩码 softmax；真实输出仍只认 `[0,T_b)`。
2. 在 `forward_batch` 校验二维输入、所有元数据的 batch 长度、`1<=T_b<=Tmax`、`max(T_b)=Tmax` 和 `p_b>=0`。`decode_batch` 继续只接受 `[B,1]`。
3. 在批次入口一次构建 RoPE positions、attention allowed mask 和 `Smax`，传给各层，消除逐层重复的 Python→device 张量创建和最大值计算。
4. 保留 `final_k/final_v` 的全零预分配和按行前缀复制。F.pad/stack 不是必然优化，varlen/paged kernel 留到性能实验阶段。
5. 输出契约写清 `[B,Tmax,V]`，占位位置即使有限也无业务意义；后续 Engine 必须按 `T_b-1` 选择最后真实 logits，中间 chunk 不采样。
6. 用长度 1、3、2 的混合 batch 验收：全输出有限、真实前缀逐条对拍、每层 cache 长度为 `p_b+T_b`、后续 decode 继续对拍；另为位置 0 占位的 allowed mask 保留手推小用例。

这套方案避免 NaN 和复杂空行分支，同时不混淆“有限占位值”和“有效 token 输出”。完成这些后即可进入 Engine，不必提前做共享 forward 去重或底层 kernel 优化。

mask 在一次 `forward_batch` 开头创建并跨层复用是合理的：所有层的 `B/Tq/Smax` 和逐行 `p/T` 相同，每层都写入同样的真实 token 数。cache 的逐层长度不变量由 KVStore 维护。

其他待完善点：

- `forward_batch` docstring 仍写 `[B,1]`，应描述混合接口的 `[B,Tmax]` 和真实长度语义。
- `max(actual_lengths)==Tmax` 只验证最大长度，还应保证每个 `1<=T_b<=Tmax` 且 `p_b>=0`，避免零长或负长在 cache 层以不直观的方式失败。
- RoPE positions 和 `max_length` 仍在每层重复构建；它们与 mask 一样只依赖批次元数据，可在正确性修复后一并上移。

## final K/V 的预分配与可选替代

当前先创建全零 `[B,Hkv,Smax,D]`，再把每行 cache 的真实前缀复制进去，属于“预分配 + 按行 scatter”，语义正确。真实区被完整覆盖，尾部保留零值，再由 attention mask 排除；dtype 和 device 也已显式指定。

也可用逐行 `F.pad` 或拼接零张量后再 `stack`，让代码更接近函数式写法，但不会产生根本性的性能优势：仍需逐行处理变长 cache，并增加临时张量或最后的堆叠复制。当前每个 cache 的 read/gather 本身已有明显开销，优化顺序应先 profiling，而不是仅替换预分配写法。

不能简单把 `zeros` 换成 `empty`：尾部并未被赋值。即使设计上会 mask padding scores，未初始化的 V 若含 NaN/Inf，也可能在后续矩阵乘中造成 `0 * NaN` 一类污染。保留零初始化更直观、安全。

真正减少这类 dense gather/pad 的方向是变长/分页 attention 直接消费请求历史，或展平真实 token；那属于 kernel 和元数据设计升级，不是本次正确性修正的前置条件。

## padding query 的完整语义

矩形混合 batch 中，第 b 行只有前 `T_b` 个 query 是真实输入，其余到 `Tmax` 的位置都是占位。它们也会经过 embedding、projection、RoPE、attention 残差和 MLP，但必须满足：

- 不写入 KV cache。
- 不采样、不产生事件。
- 不推进 processed/generated 等真实状态。
- 不参与最终有效 logits、指标或损失的汇总。

当前实现把 padding query 的位置设为 `end_position=p+T`。例如 `p=5,T=1,Tmax=3`，位置为 `[5,6,6]`，只有位置 5 真实。真实 query 5 只能看到 key `0..5`；假 query 6 却能看到 key `0..6`，其中 key 6 是尾部零 padding。

零 K 会产生一个额外 score，零 V 虽不贡献分子向量，但 softmax 分母会增加一项，因此假 query 的输出通常既不等于最后一个真实输出，也没有业务意义。这不会反向污染同一行位置 5：attention 的每个 query 独立，后续 Norm/MLP 也按位置计算，真实 query 不读取假 query 的 Q。

真正的风险发生在后续消费者误用占位结果，例如统一取 `logits[:,-1]`、再次写入整行 K/V，或对全部位置做无 mask 的指标/损失汇总。

一种稳妥的矩形实现是：

1. 用合法的安全位置计算占位 query，例如 clamp 到最后真实位置 `p+T-1`，避免它访问额外 key。
2. 单独保留 `valid_query[b,i]=i<T_b` 元数据。
3. 占位 query 可以继续走完网络，但输出侧只选择真实位置；采样位置是 `T_b-1`，且中间 prefill chunk 不采样。

不建议不加处理地把 `valid_query=False` 直接变成整行全 `-inf` 的 attention scores，这可能让 softmax 产生 NaN。query padding mask 常用于输出选择，key padding mask 和 causal mask 用于 attention 可见性，三者可以分开处理。

当前阶段优先补齐这些契约和混合测试即可；若后续要消除 padding 计算本身，再转向真实 token 展平和 varlen attention。

## Scheduler 第一次代码复盘

### list 解包不等于要求 tuple

Python 的序列解包作用于可迭代对象，不要求右侧必须是 tuple：

```python
step, = [step_a]                 # 列表中恰好一个元素
decode, chunk = [step_a, step_b] # 列表中恰好两个元素
```

因此 `plan() -> list[Step]` 与验收里的 `prefill, = plan()`、`decode, chunk = plan()` 完全兼容。元素数量不匹配才会报 too many/not enough values to unpack。

当前实现实际返回 `(decode_steps, prefill_steps)`，这是两个列表组成的嵌套 tuple，违反了扁平 `list[Step]` 契约。单个 prefill 时得到 `([], [step])`，长度 2 导致单元素解包失败；混合轮次即使解包成功，得到的也会是两个列表而不是两个 Step。应保持 decode 在前、prefill 在后，并返回一个扁平列表。

### 本轮测试结果

```text
Lab 06 scheduler：1 failed, 1 passed, 4 deselected
Lab 01～05 回归：6 failed, 17 passed, 1 skipped
```

构造时拒绝 `max_batch_tokens < max_active` 的测试已通过，其余失败暴露了以下问题。

### 需要逐项修正的实现语义

1. **无预算时反而没有 Step。** `filter_req_with_budget` 只有 `resource is not None and resource.take(...)` 才 append；默认 `max_batch_tokens=None` 时应表示不受 token budget 限制，而不是全部过滤。
2. **无 chunk 参数时无法生成 prefill。** `processed_tokens + self.max_prefill_chunk` 在 None 下直接 TypeError。Lab 04/05 的默认行为必须保留：整段 prompt 一次 prefill，且 emit token。
3. **decode 输入错误。** 当前 decode 返回整个 prompt，应该只返回 `(status.last_token,)`。否则既违背逐 token decode，也会让每行消耗的 budget 不再恰好是 1。
4. **prefill chunk 没有按剩余 budget 截短。** 当前先生成完整 Step，再判断整个 input 是否放得下；放不下就跳过。正确语义是 decode 优先后，以剩余 budget 作为本轮 chunk 上限之一，允许生成更短的非空 chunk。
5. **非发 token 的 complete 被误判为 EOS。** 中间 chunk 的 sampled_token=None，而默认 eos_token_id 也是 None；直接比较二者相等会错误退休。结束判断只能发生在实际发出 token 的路径。
6. **非发 token 的 complete 应返回 None。** 当前末尾仍构造 TokenEvent，且 token_id 可能是 None。中间 chunk 只推进 processed，不产生事件。
7. **last_token 不应被中间 chunk 覆盖。** 只有 emit_token=True 时才更新 last_token/generated，并按既有规则判断 EOS 和长度。
8. **emit 与 token 的反向组合也要校验。** emit=True 要求 token；emit=False 应要求 None，避免调用方把未计划输出的 token 错提交进状态。

### 默认参数与扩展参数的兼容矩阵

| 参数状态 | 应保持的行为 |
| --- | --- |
| 两者均 None | Lab 04/05：整段 prefill、逐 token decode、无 token budget 过滤 |
| 仅 max_prefill_chunk 有值 | decode 不受预算过滤，prefill 按 chunk 上限拆分 |
| 仅 max_batch_tokens 有值 | decode 优先，prefill 使用剩余预算 |
| 两者均有值 | chunk 长度取剩余 prompt、chunk 上限、剩余预算三者的最小值 |

`max_prefill_chunk` 有值时还应拒绝非正数，否则可能产生空 Step、状态不推进和无限 prefill。

本次回归说明：增加可选功能时，**None 通常代表关闭该限制并保留旧路径**，不能仅让新测试通过而忽略默认构造方式。

## Scheduler 第二次代码复盘

复验结果：

```text
Lab 06 scheduler：2 passed, 4 deselected
Lab 04 + Lab 05：8 passed
Lab 01～05 回归：23 passed, 1 skipped
```

扁平 `list[Step]`、无参数旧路径、decode 的 last_token 输入、默认 EOS 判断和公开 chunk trace 均已恢复/实现。

### 公开测试尚未覆盖的阻塞问题

当前 `_generate_step` 仍只根据 max_prefill_chunk 生成 Step，随后由 `resource.take(len(step.input_ids))` 做“全收或全拒”。这并不等价于按剩余 budget 截短。

最小复现：

```text
Scheduler(max_active=1, max_batch_tokens=2)，prompt 长度为 3
连续两次 plan() 都返回 []
```

仅配置 max_batch_tokens 时，max_prefill_chunk=None 使 prefill 试图一次吃整个 prompt；只要 prompt 超过总预算，请求就永远无法开始。另一个混合场景中，decode 消耗 1 后只剩 1 budget，prefill 生成长度 2 的 Step 而被整体跳过，可能一直等 decode 请求结束，违背 decode 优先但 prefill 使用剩余额度的设计。

修正方向是让 Step 生成本身获得本轮允许消费的上限，在切片前取：

```text
min(剩余 prompt，max_prefill_chunk 或无限，剩余 budget 或无限)
```

并保证只要 budget 有剩余，就能规划非空 chunk；不能先生成超限 Step 再整体拒绝。StepResource 可以只负责记录剩余额度，但需要和切片上限协同，而不只是事后过滤。

### 其他输入契约与可优化点

- `max_prefill_chunk <= 0` 尚未拒绝。零会产生空 Step 且不推进；负数还会让 `take` 增加预算，应在构造时校验正数。
- `complete` 尚未拒绝 `emit_token=False、sampled_token!=None` 的反向非法组合。当前会污染 last_token 后返回 None，应在任何状态修改前拒绝。
- `last_token = sampled_token` 可以只在实际 emit 时更新，使“中间 chunk 只改 processed”的不变量在代码结构上更清楚。
- `max_batch_tokens` 的错误文案应表达“at least / 大于等于”，因为当前允许等于 max_active。
- 等语义稳定后，再考虑用局部剩余预算整数简化 StepResource、用 deque 避免等待队列 `pop(0)` 的 O(n) 移动，以及省略 `in dict.keys()` 中的 `.keys()`。这些不是当前正确性的优先级。

这次复盘再次说明：公开 trace 通过代表样例路径正确，不代表所有可选参数组合都能取得进展。调度器除了输出对拍，还应专门检查“非空计划/状态推进”的活性条件。

## Scheduler 第三次代码复盘

再次复验公开结果：Lab 06 scheduler 为 `2 passed`，Lab 01～05 为 `23 passed, 1 skipped`。以下修正已经确认：

- max_prefill_chunk 非正数会在构造时拒绝。
- complete 同时校验 emit=True/token 与 emit=False/None 两种合法组合。
- last_token 只在 emit 时更新。
- max_batch_tokens 文案已改为大于等于。

剩余 budget 截短仍未真正生效，存在两个连接问题：

1. `_generate_step` 新增了 resource 参数，但 `plan()` 仍调用 `_generate_step(request, status)`，实际始终使用默认 None。
2. max_prefill_chunk=None 的提前分支直接取整段 prompt，即使以后传入 resource，也仍不会按 budget 截短。

因此两个最小复现保持原结果：budget-only 长 prompt 连续返回空计划；混合轮次 decode 后剩余 1 budget，prefill 仍被整体跳过。

下一步应统一 prefill 切片路径：无论 max_prefill_chunk 是否为 None，都先得到剩余 prompt，再按需取 resource 和 chunk 配置的最小上限；plan 调用时传入同一个 resource，让 decode 先扣减、prefill 后使用。resource 耗尽时应跳过 prefill，而不是规划 `input_ids=()` 的空 Step。

这次问题属于“辅助函数实现了能力，但调用方没有接线”。代码审查和自测不能只检查新函数本身，还要从公开入口跟踪参数是否到达目标分支。

## Scheduler 完成复验

budget 截短已正确接入：plan 将同一个 resource 传给 Step 生成函数，decode 优先扣减；prefill 统一按剩余 prompt、resource 和 chunk 配置计算 allowed_chunk；allowed_chunk=0 时跳过规划，不产生空 Step。生成后再按实际 input 长度扣减预算。

最小复现结果：

```text
仅 max_batch_tokens=2、prompt=3：
  可规划 prefill(1,2)，emit=False

decode 后剩余 1 budget、chunk 上限 2：
  decode + prefill(长度为1)，同一轮返回
```

第一次复现中未调用 complete 时，下一次 plan 返回相同 start_pos 是正确行为：计划本身不推进状态，只有 complete 才提交游标。

最终复验：

```text
Lab 06 scheduler：2 passed, 4 deselected
Lab 01～05 回归：23 passed, 1 skipped
```

Scheduler 至此形成以下不变量：返回扁平 Step 列表；decode 优先且每行占 1 budget；prefill 只要有剩余额度就能规划非空 chunk；无新参数时保持整段 prefill；中间 chunk 只通过 complete 推进 processed。
