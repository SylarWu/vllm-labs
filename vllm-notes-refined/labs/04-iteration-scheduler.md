# Lab 04 · 迭代级调度

**CPU 验收 `5 passed`。** Lab 01–04 累计 `20 passed, 1 skipped`，复验了准入填满、失败原子性、队首阻塞、退休清理与 ID 重用。

## 1. 一轮为调度单位

单请求 `greedy_generate` 独占一个循环直到终止。Orca 将执行单位改为单次迭代，于是长请求 decode 时新请求可以开始，短请求完成后空位可在下一轮复用。`Scheduler.plan()` 决定谁运行，Lab 05 引擎决定怎么执行，Lab 06 才做 tensor batch。当前可手填 token 验证调度。[Orca 论文](https://www.usenix.org/system/files/osdi22-yu.pdf)

| 对象 | 内容 |
|---|---|
| `Request` | 固定的 ID、prompt、输出预算和 EOS ID。 |
| `Step` | 该请求本轮的 `phase`、`start_pos`、`input_ids`、`emit_token`。 |
| `TokenEvent` | 已发 token 与请求归属，最后一项可带 `stop` / `length`。 |
| 内部状态 | 已处理输入位置数、上一轮采样 token、输出数，初始 `0 / None / 0`。waiting 保持提交次序，active 保持准入次序。 |

## 2. Submit、plan、complete 与 retire

1. `submit` 校验 prompt 非空、预算为正、ID 在 waiting ∪ active 唯一。只查 active 会使同一个等待 ID 覆盖请求表、在队列重复出现。
2. `plan` 循环检查 FIFO 队首，填满 `max_active` 允许的位置。`try_admit=False` 时保持队首并停止新增准入，已有 active 仍推进。每个 active 每轮至多一项。Lab 04 prefill 输入完整 prompt，decode 输入上一轮 token，二者都发 token。
3. 每项执行后调用 `complete(step,sampled_token)`。先验证结果，再提交 `cursor += len(step.input_ids)`、输出计数和 last token。Lab 04 的 `None` 属非法结果。Lab 06 会为中间 prefill 定义合法的 `None`。计划可改变准入，执行游标与预算只在完成时推进。
4. 先记本轮 token 再终止。EOS 与上限同时命中时优先 `stop`，只命中上限为 `length`。正常完成与取消都清掉请求描述和运行状态并触发 `on_retire` 至多一次。`cancel(id)` 在轮次边界移除 waiting 或 active，成功 `True`，未知 ID `False`，无 token event。退休后 ID 可重用。

`complete` 是小事务，失败时游标、输出数和 last token 应一起保留。同步约定是本轮所有 Step 均恰好提交一次，再调下一次 `plan()`。若将来异步执行，可维护未完成 Step 集合，拒绝未知、重复、过期结果，并定义取消后迟到结果的处理。严格 FIFO 有队首阻塞，Lab 05 的暂时 KV 不足沿用该策略。

## 3. A/B/C 验收时序

`max_active=2`，A prompt 为 `(1,2,3)`，最多输出 4 项，首轮后 B/C 入队，各只输出 1 项。

| Tick | A | B | C |
|---|---|---|---|
| 1 | 位置 0 prefill，发 11 | 未提交 | 未提交 |
| 2 | 位置 3 吃 11，发 12 | prefill 发 13，完成 | 等待 |
| 3 | 位置 4 吃 12，发 14 | 已结束 | prefill 发 15，完成 |
| 4 | 位置 5 吃 14，发 16，完成 | 已结束 | 已结束 |

首轮 A 已处理输入数是 3，已发输出数是 1，下一轮完成后分别到 4 和 2。两者分别驱动 input/KV 游标和生成预算。B 在第二轮 complete 后释放位置，C 在第三轮 plan 时准入。

```bash
.venv/bin/python -m pytest -q tests/test_lab04_scheduler.py -k short_request
.venv/bin/python -m pytest -q tests/test_lab04_scheduler.py
```

检查同轮多准入、活跃数上限、每个事件归属、提交后游标、EOS 和双状态取消。代码 `src/tinyserve/scheduler.py`、`requests.py`，原笔记 `learning_notes/lab04_scheduler.md`。[下一篇：分页 Engine](05-engine.md)
