# Lab 05 · 分页 KV 与端到端 Engine

**CPU 验收 `3 passed`，Lab 01–05 回归 `23 passed, 1 skipped`。** 本阶段调度可同时容纳多请求，模型按 Step 逐个 forward。实际 tensor batch 见 [Lab 06](06-batch-and-chunked-prefill.md)。

## 1. 一份物理池，多个持久请求视图

| 组件 | 闭环中的职责 |
|---|---|
| `BlockPool` | 管理物理块的使用权。 |
| `BlockTable` | 请求持有的块和逻辑范围，从池获取并归还 ID。 |
| `PagedStorage` | 由 Engine 长期持有，全模型共享的 K/V 张量各一份。 |
| `PagedKV` | `storage.view(table)` 创建的请求视图，绑定表、实现 `KVStore`、独立追踪各层写入长度。 |

K、V 各为 `[L,num_blocks,block_size,Hkv,D]`，也可扁平为 `[L,num_blocks*block_size,Hkv,D]`。完整地址含 `layer,physical_block,offset,kv_head,head_dim`。`B=2`、映射 `[3,1]` 时逻辑 `[0,1,2,3]` 翻译为 slot `[6,7,2,3]`。toy 模型 `L=2,Hkv=2,D=8`，独立存储测试 `D=3`，维度从配置取得。

- **Scatter** 以 `table.slots(start_pos,T)` 将本层 `[T,Hkv,D]` K/V 的第 i 项写入对应物理 slot。
- **Gather** 以同一逻辑顺序读 `[0,length)`，返回连续的 `[length,Hkv,D]` 给普通 GQA。该教学路径会复制历史，kernel 直接读页可省去这一步。
- 所有层共用一张地址表，分别选择 layer 轴。已保留 8 个逻辑位置时，两层都可从 `5` 写入同一组 `[5,8)` slot，分别推进本层游标。物理块 ID 表示所有层中该块的切片，一次准入给整个模型分配，容量仍按 8 个逻辑位置计。
- `reserved_tokens` 管容量，`written_length[layer]` 管已写。读要求 `0 < length <= written_length[layer]`。append 先校验层、K/V 三维及各轴、`start_pos == written_length[layer]`、终点在预留内，然后写两份张量并推进。层间暂时不同步是正常的。归还清除使用权，正确的所有权和游标校验能屏蔽物理池里前一请求的数据。

## 2. 准入、执行与回收

`reserved_tokens = P + N - 1`，`required_blocks = ceil(reserved_tokens / block_size)`。比如 prompt 3、输出上限 2，只处理 3 个 prompt 位置和第一个输出，需 4 个 KV 位置。

1. Engine 初始化从 `model.config` 取层数、KV heads 与 head dim，从参数取 device/dtype，建立一份池和 storage。模型与 cache dtype 对齐，避免隐式拷贝转换影响显存和数值。`max_batch_tokens`、`max_prefill_chunk` 原样交给 Scheduler。
2. `submit` 在入队前拒绝所需块超过全池的请求。能独占放下、只是当前空闲不足的请求仍入队。`max_active` 与总 KV 配额共同决定准入，严格 FIFO 保留队首次序。
3. `try_admit` 成功前一次性预留最坏页数并建立 `request_id → PagedKV`。返回 False 时无资源副作用。返回 True 后 Scheduler 才迁移到 active，避免 plan 后重复分配。一次预留避免活跃请求中途争页无法推进，代价是未用到的预留和提前 EOS 时的闲置。
4. `step()` 取得计划，按 ID 取持久视图，把输入整理为设备上的 `torch.long[1,T]`，在 `inference_mode` 下 forward。用末位置 logits 沿词表维贪心采样，再 `complete` 推进游标、输出数并收集 `TokenEvent`。无计划返回 `[]`。新采样的 token 下一轮才写 KV。
5. EOS/长度和 active 取消经 `on_retire` 弹出资源并 `cache.table.release()`。waiting 取消尚无资源，重复取消或退休也不会重复归还。`has_work` 包含 active 和 waiting，`free_blocks` 直接取共享池空闲数。

## 3. 源码复盘与验收

| 现象 | 根因与修复 |
|---|---|
| `-k paged_store` 首次 `1 failed, 2 deselected` | 非连续页、层隔离和请求隔离已过，末项发现 reserve 3 后在位置 3 写一项竟被接受。旧 `write` 自动 `table.reserve`，还只查起点不超保留长度，允许覆盖和跳写。分配移交准入，严格按层 append，`slots()` 在写前拒绝越界后变成 `1 passed, 2 deselected`。原记录指出 `writen_length` 拼写和空写入显式拒绝仍可补齐。 |
| A 第一次 decode 位置 4，视图游标却为 0 | 旧 Engine 只持有表，每次 forward 或打印都创建新视图，存储与表虽然共享，Python 列表 `written_length=[0,0]` 却独立。准入创建并保存单一 `PagedKV`，prefill 和 decode 跨 tick 复用，退休一并清理。可打印 `id(view)` 和各层游标定位。 |
| 小页池已经通过而 serving 失败 | 小页池用例里 A 在 prefill 后取消，B 首次 prefill 即结束，没有第二次 forward。混合 serving 对拍才覆盖跨 tick。初次分别为 `1 passed, 2 deselected` 与 `1 failed, 2 deselected`。 |
| 类型标注与数据所有权 | 早期 `_prepare_kv_cache` 标注回 `BlockTable` 却只有副作用。标注应与真实返回一致。按层写进度用 `written_length` 命名，容量由表表达。 |

```bash
.venv/bin/python -m pytest -q tests/test_lab05_engine.py -k paged_store
.venv/bin/python -m pytest -q tests/test_lab05_engine.py
```

先对拍非连续页和层/请求隔离，再通过 `active_ids`、`waiting_ids`、`free_blocks`、事件 trace 复核等待与释放，最后与 `greedy_generate` 完整重算比 ID。逐 slot Python 循环可先保证正确，后续可改张量索引。代码 `src/tinyserve/paging.py`、`engine.py`，原笔记 `learning_notes/lab05_engine.md`。
