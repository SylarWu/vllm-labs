# Lab 04 · 迭代级调度

**状态：已完成。Lab 04 验收 `5 passed`；Lab 01～04 回归 `20 passed, 1 skipped`。额外复验了同轮多请求准入、失败原子性、队首阻塞下的活跃请求推进、退休后的元数据清理与 ID 重用。**

## 调度为什么以轮为单位

Lab 02 的 `greedy_generate` 在一个循环内处理一个请求，直到 EOS 或预算用完。把这个过程拆成一轮一轮的 prefill/decode，就能在轮次边界重新决定谁参加下一轮。长请求 A 继续 decode，新来的 B 可以在空位上做 prefill。B 完成后，下一轮就能让 C 使用这个位置。Orca 把模型执行的调度单位从完整请求改为单个迭代，让等待请求和已完成请求都能更早得到处理。[Lab 02 的生成时序](lab02_cache.md#从-forward-到生成)、[Orca 论文](https://www.usenix.org/system/files/osdi22-yu.pdf)

本 Lab 用手工指定的 token 测调度行为，`plan()` 生成本轮任务列表。Lab 05 的 Engine 执行计划并提供真实 token，Lab 06 将同一轮的 decode 请求合并成一次批量 forward。调度器决定本轮谁运行，执行器决定这批工作怎么计算。

## 对外对象与内部状态

| 对象 | 含义 |
| --- | --- |
| `Request` | 请求 ID、prompt、生成预算和 EOS ID，提交时固定 |
| `Step` | 本轮要执行的一个请求，描述 `phase`、`start_pos`、`input_ids`、是否发 token |
| `TokenEvent` | 执行成功后给调用方的 token 事件，最后一个事件可带 `stop` 或 `length` |

可为活跃请求维护已处理的输入位置数、上一轮已采样的 token、已发出的 token 数。初始三个量分别为 `0`、空和 `0`。等待队列维持提交顺序，活跃集合维持准入顺序，两个集合里的活跃 ID 合起来唯一。

## 一轮的边界

1. `submit(request)` 校验 prompt 非空、输出预算为正、请求 ID 未出现在等待或活跃集合，之后把请求放到等待队尾。
2. `plan()` 先填可用的活跃位置。查看队首，`try_admit` 若返回 `False`，队首保持原位并停止本轮新增准入，已经活跃的请求仍要生成计划。`max_active` 限制活跃请求数。每个活跃请求在本轮至多拿到一个 `Step`。
3. 调用方执行各个 `Step`。Lab 04 的 prefill 传整个 prompt，decode 传上一步输出的单 token，两个阶段的 `emit_token` 都为 `True`。Lab 06 的中间 prefill chunk 会用到 `emit_token=False`。
4. `complete(step, sampled_token)` 提交成功执行的结果。输入位置增加 `len(step.input_ids)`，输出计数在发出 token 后增加 1，并记录该 token 供下一轮使用。当前轮的所有 `Step` 都完成后，调用方再调用下一次 `plan()`。计划阶段可以把请求移入活跃集合，计算游标和生成预算在提交时推进。

例如 A 的 prompt 长 3，prefill 成功后已处理位置数为 3，且已生成 1 个 token `11`。下一轮把 `11` 作为输入，`start_pos=3`。这轮完成后处理位置数为 4，生成数为 2。一个是模型消费过多少输入位置，另一个是返回给调用方多少输出，两者独立计数。

### 状态提交的原子性

`complete()` 可以看成一次小型事务：先校验本轮结果是否合法，再统一更新位置、输出计数和上一轮 token。若校验失败，运行状态应保持不变。否则可能出现位置已经推进、`last_token` 却被清空的不一致状态，下一次 `plan()` 便无法判断应该 prefill 还是 decode。

Lab 04 的每个 `Step` 都发 token，因此 `sampled_token=None` 是非法结果。Lab 06 的不发 token prefill chunk 则是另一种合法路径，届时应分别处理，不能提前混用两种语义。

### `Step` 与 `complete` 的对应关系

一次 `Step` 应当只被成功提交一次。若同一个 prefill Step 被重复 `complete`，请求位置和输出计数会被重复推进；若上一轮 Step 延迟到后续轮次才提交，也可能覆盖新一轮的推进顺序。

本 Lab 采用同步调用，并约定调用方在下次 `plan()` 前完成当前轮次的所有 `complete()`，因此可以暂时依赖这一接口契约。面向异步 Engine 加固时，可记录本轮已发出且尚未完成的 Step，对未知、重复或过期提交显式报错。进入 Lab 07 的断流取消后，还需要继续明确“取消后迟到的执行结果”如何处理。

### live 请求与元数据生命周期

“请求 ID 唯一”约束的是所有 live 请求，即 waiting 与 active 的并集，而不只是正在运行的请求。若只检查 active，重复提交等待中的 ID 会覆盖请求表里的原始 `Request`，并让等待队列出现两个相同 ID。

请求元数据也应随生命周期结束而清理：正常退休或取消时移除对应的 `Request` 和运行状态，再保证资源回调至多一次。长期运行的服务若永久保留已结束请求，会造成元数据无界增长。

同一轮准入应持续检查等待队首，填满所有可用活跃位置，而不是每轮最多接纳一个请求；遇到容量不足或 `try_admit=False` 才停止本轮新增准入。

## 手推三条请求

`max_active=2`。A 的 prompt 为 `(1, 2, 3)`，允许输出 4 个 token。A 第一轮完成后提交 B、C，它们各允许输出 1 个 token。

| Tick | A | B | C |
| --- | --- | --- | --- |
| 1 | prefill 从 0 吃 prompt，发 `11` | 还没提交 | 还没提交 |
| 2 | decode 从 3 吃 `11`，发 `12` | prefill，发 `13`，按长度结束 | 等待 |
| 3 | decode 从 4 吃 `12`，发 `14` | 已结束 | prefill，发 `15`，按长度结束 |
| 4 | decode 从 5 吃 `14`，发 `16`，按长度结束 | 已结束 | 已结束 |

第 2 轮的 B 消费整个 prompt 后就达到输出上限。B 在该轮 `complete()` 后释放活跃位置，C 在下一次 `plan()` 才获准进入。

## 结束、取消与资源回调

- `complete` 先记录并发出本轮 token，再决定是否终止。如果同一个 token 同时命中 EOS 和预算上限，事件的 `finish_reason` 为 `stop`。只达到预算时为 `length`。正常完成应移除活跃请求，并触发一次 `on_retire(request_id)`。
- `cancel(request_id)` 在轮次边界执行，等待或活跃请求都可移除，成功返回 `True`。找不到该 ID 返回 `False`，也就不会重复回调。取消没有本轮新 token，接口只返回布尔值。
- 严格 FIFO 允许产生队首阻塞。若队尾请求能获准但队首暂时不行，继续保持原顺序，下一轮重试。Lab 05 将通过 `try_admit` 判断是否有足够的 KV 块，通过 `on_retire` 释放请求资源。[Lab 03](lab03_paging.md)

## 动手与验收

先做内部状态、`submit`、`plan` 和 `complete`，把 A/B/C 的时序跑通。再加入 EOS 优先级、取消和准入回调。等待与活跃的 ID、活跃数上限、每轮发 token 的归属、完成时的游标变化都可以从 trace 中核对。

```bash
.venv/bin/python -m pytest -q tests/test_lab04_scheduler.py -k short_request
.venv/bin/python -m pytest -q tests/test_lab04_scheduler.py
```

[调度器入口](../src/tinyserve/scheduler.py)、[请求与事件定义](../src/tinyserve/requests.py)、[验收测试](../tests/test_lab04_scheduler.py)
