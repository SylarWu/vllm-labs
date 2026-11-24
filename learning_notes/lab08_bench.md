# Lab 08 · 请求回放与性能指标

**状态：本地实现与验收已完成，真实 GPU 压测待进行。** `summarize` 和 `run_trace` 均通过代码复盘；Lab 08 含新增边界回归共 `6 passed`，Lab 01～08 全量 CPU 回归为 `67 passed, 1 deselected`。下一阶段在 WSL2 RTX 4080 上完成可复现负载与官方 vLLM 对照。 [学习计划](../README.md#lab-08--指标与源码对照) · [验收代码](../tests/test_lab08_bench.py)

## 观测的是哪条时间线

`Arrival.at_s` 是相对回放起点的计划到达时间。`RequestTiming.submitted_s` 是实际调用 Engine 提交时记录的时间，`emitted_s` 是逐个输出 token 的观测时间，`completed_s` 是观察到终止事件的时间。当前 Engine 同步执行一整个 tick。tick 中间到达的请求，要等到这个 tick 返回后的调度边界才能提交。

Lab 07 记录的是怎么把 Engine 事件送进 SSE。Lab 08 在 Engine 层观测 `TokenEvent`，所以 TTFT 包含 submit 之后的 Engine 排队和 prefill。在线客户端还可以从发送 HTTP 请求起计时，分别观察首帧与首个非空文本。这些观测点适用于不同的性能问题。 [Lab 07 逐帧推演](lab07_api.md#从一次-post-逐帧推演)

## 第一步，手算后实现 summarize

### 先区分观测、采样和聚合

`RequestTiming` 保存一次请求的原始观测，`BenchStats` 保存跨请求的聚合结果。实现时可以分成三层推导，避免把不同总体混在一起：

1. **观测区间。** 时间统一先按秒计算，TTFT 和 ITL 最终再乘 `1000` 转成 ms。吞吐使用 `max(completed_s) - min(submitted_s)`，中间 Engine 空闲、等待后续请求到达的时间也属于整个 replay 的时间窗口，不能扣除。
2. **样本池。** 请求数是 `len(samples)`，输出 token 数是所有 `emitted_s` 长度之和。TTFT 每个有输出的请求最多贡献一个样本；ITL 只在同一请求的相邻 token 之间计算，再把各请求的间隔放进同一个池，绝不能连接两个不同请求的 token 时间线。
3. **聚合方式。** 最近秩分位数先排序，再取第 `ceil(p × n)` 个元素；换成 Python 的 0 基索引就是 `ceil(p × n) - 1`。`n = 0` 时没有分位数，应返回 `None`，不能调用索引或构造一个看似合理的 `0`。

这些指标回答的问题不同：TTFT 主要暴露排队与 prefill 对首 token 的影响，ITL 主要暴露持续 decode 的节拍，输出吞吐则衡量整个观测窗口交付了多少 token。提高 batching 可能提升总吞吐，却也可能让单请求 ITL 变差，因此不能只用吞吐判断服务体验。

本 lab 的 ITL 是**跨请求合并的 token 间隔总体**，不是“先求每请求 ITL 分位数，再对分位数求平均”。输出较长的请求会贡献更多 ITL 样本，这是当前指标定义的一部分；若将来要比较请求级体验，需要另建按请求加权的指标，不能悄悄改变本验收口径。

| 指标 | 采样与单位 |
| --- | --- |
| TTFT | 每个有输出的请求贡献一个样本。首个 token 时刻减实际提交时刻，转换为 ms。 |
| ITL | 每个请求的相邻 token 时刻相减，转换为 ms，把所有间隔汇总成一个样本池。只有 1 个 token 的请求没有 ITL。 |
| 输出吞吐 | 所有输出 token 个数除以最早实际提交到最晚完成的总秒数。 |
| p50 / p95 | 按最近秩法取值。排序后取第 `ceil(p * n)` 个，位置从 1 开始。没有样本时返回 `None`。 |

验收样例用相对时间来手算。A 在 0 ms 提交，20、50、80 ms 出 token。B 在 10 ms 提交，70 ms 出唯一一个 token。TTFT 样本是 `[20, 60]` ms，最近秩的 p50 为 20 ms、p95 为 60 ms。ITL 样本是 `[30, 30]` ms，两档均为 30 ms。总吞吐是 `4 / 0.08 = 50` token/s。空输入的请求数、输出 token 数和吞吐均为零，分位数无样本。 [指标验收](../tests/test_lab08_bench.py#L7)

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_lab08_bench.py -k summary
```

## 第二步，按实际提交时间回放

Fake clock 起点是 10.000 s。A 立刻提交，每个 tick 耗时 10 ms。B 的计划到达时刻为 10.015 s，第二个 tick 要到 10.020 s 才返回，所以 B 实际于 10.020 s 提交，在 10.030 s 观测到唯一输出并完成。A、B 在 10.030 s 完成后，Engine 空闲到 C 的计划到达时间 10.100 s，因此只 sleep 70 ms，再提交 C。

练习时先画出每个循环边界的三个事实：当前时刻、已到达但尚未提交的请求、Engine 是否仍有工作。用传入的 `clock` 与 `sleep` 掌控时间。TTFT 会使用实际提交时刻。 [Fake clock 验收](../tests/test_lab08_bench.py#L27)

### 回放器是时间驱动的小型状态机

回放起点只读取一次 `clock()`，记为 `replay_start`。`Arrival.at_s` 必须转换成绝对计划时刻 `replay_start + at_s` 后再与当前时间比较；`RequestTiming` 中仍保存 `clock()` 给出的绝对观测时刻，而不是相对秒数。

每个循环边界按以下顺序思考：

1. 读取当前时间，把所有计划时刻已经不晚于当前时间的请求逐个 `submit`，并记录**实际提交时刻**。一次可能提交多个请求，包括同一时刻到达的请求，以及上一个长 tick 期间积压的多个请求。
2. 若 Engine 有工作，执行且只执行一个 `step()`。step 返回前不能插入新请求，因为当前 Engine 是同步单 owner，新的到达只能在 tick 边界被受理。
3. step 返回后读取观测时间，处理本次全部 `TokenEvent`。同一 tick 中不同请求的事件可以拥有相同时间戳，不应为了处理事件额外 sleep 或推进 fake clock。
4. 若 Engine 没有工作且 trace 还有未来请求，sleep 到最近一个未来计划时刻。睡眠时间应是目标绝对时刻减当前时刻，不能写固定间隔，也不能对已经到期的请求再 sleep。
5. 只有当所有 trace 请求都已提交且 Engine 没有工作时结束。空 trace 直接返回空列表，不应额外 step 或 sleep。

这个顺序保证“先收下当前已经到达的请求，再推进下一次模型计算”。如果先 step 后 submit，会把本可参加当前 tick 的请求无意义地推迟一轮；如果 Engine 忙碌时 sleep 到下一次计划到达，又会暂停本可继续的 decode。

### TokenEvent 如何变成 RequestTiming

`RequestTiming` 是不可变对象，但回放期间 token 列表需要逐步增长。可以使用请求 ID 索引的内部可变记录，在完成或收尾时再生成 frozen dataclass；不要尝试原地修改 `emitted_s` tuple。

每个事件都表示一个真实输出 token，因此无论 `finish_reason` 是否为空，都要先把事件观测时刻加入该请求的 `emitted_s`。只有当 `finish_reason` 非空时，才把同一观测时刻记为 `completed_s` 并完成该请求的记录。最后一个 token 既计入输出数量，也决定完成时刻，不能因处理完成分支而漏记。

TTFT 使用实际提交时间，所以 B 的 TTFT 是 `10.030 - 10.020 = 10 ms`，不是相对计划到达的 15 ms。B 在 tick 内等待的 5 ms 已体现在 `10.015 → 10.020` 的实际提交推迟中；若再从计划到达计算，就会混入本指标定义之外的 replay 等待。

### run_trace 首次代码复盘

原有 fake clock 验收使用 `token_id = 2`，因此没有覆盖 Python 真值判断的一个典型陷阱：**token ID 是整数数据，不是“是否含有 token”的布尔标记**。`0` 是合法 token ID，`if event.token_id` 会把它误判为 False 并漏掉整个输出事件。本接口的每个 `TokenEvent` 都必然携带 token，因此记录输出不需要对 ID 做真值过滤；判断可选值是否存在时，也应根据字段契约使用 `is not None`，而不是依赖数值真值。回归用例见 `test_replay_records_zero_valued_token_ids`。

真实时钟还带来两个 fake clock 未覆盖的边界：

1. 到期请求的**筛选时刻**可以共用一个 `current`，但多个 `engine.submit` 本身也会消耗现实时间。若所有请求都保存筛选前的同一时间，后面的 `submitted_s` 就不是各自实际提交时刻。提交时间应在每个请求的实际提交边界记录。
2. 已用 `current` 判断下一请求仍在未来后，若在计算 sleep 时长时再次读取一个已经推进的时钟，目标时刻可能刚好被跨过，得到负数并触发 `time.sleep` 的 `ValueError`。同一次“判断是否需要等待—计算等待多久”的决策应使用一致的时间快照；若重新读钟，就需要重新处理已经到期的情况。

内部把未完成的 `completed_s` 先置为 `0.0` 也有数据质量风险：如果 Engine 异常地提前变为空闲，收尾会生成看似完整、实际未完成的时间记录。更稳妥的内部状态应能表达“尚未完成”，并在输出前检查所有请求都具有有效完成时刻，避免静默污染压测结果。

第二轮修改后，`None` 已能正确表达“尚未完成”，但收尾只保留完成记录仍不等于完成数据校验。过滤会把缺失请求从 `requests`、吞吐和延迟样本中一起隐藏，让一次异常回放看起来像一次较小但成功的实验。这里的检查应当**发现不一致并显式失败**，而不是删除证据。

负 sleep 修复也必须保持同一次判断的时间快照一致：先用 `current` 判断目标时刻在未来，却又用下一次 `clock()` 计算时长，仍然存在目标时刻在两次读取之间被跨过的窗口。`test_replay_never_sleeps_for_negative_duration` 用第四次读钟从 `0.000` 跨到 `0.020`、目标到达为 `0.010` 的场景稳定复现了 `-0.010` 的 sleep 参数。

最终收尾对任何 `completed_s is None` 显式抛错，并用 `test_replay_rejects_requests_without_completion_events` 固定该契约。这样 Engine 若异常地丢请求，实验会失败，而不会返回一个请求数偏少、看似成功的统计报告。

## 第三步，设计可复现的 GPU 性能实验

### 一次只改变一个变量

第一轮实验只改变 `max_active`：`1` 是串行基线，`8` 允许真实 decode batching。模型快照、dtype、页大小、页数、token 预算、chunk 大小、请求 trace、输入输出长度和随机 seed 都必须保持一致。若同时改变 batch 和输入长度，就无法把差异归因于 batching。

先运行短 smoke test，确认真权重、CUDA、页池和 JSON 输出正常，再执行正式负载。正式实验至少独立重复三轮，保留每轮原始 JSON，不用一次运行或只保留平均值代替波动观察。warmup 不计入正式指标；当前 `scripts.bench` 已在 warmup 后同步 CUDA 并重置 peak memory 统计。

推荐先在 WSL2 做 `mixed` 负载的两组三轮实验：

```bash
mkdir -p results/lab08

python -m scripts.bench --model-dir models/Qwen3-0.6B --device cuda \
  --max-active 1 --profile mixed > results/lab08/mixed-a1-r1.json
python -m scripts.bench --model-dir models/Qwen3-0.6B --device cuda \
  --max-active 8 --profile mixed > results/lab08/mixed-a8-r1.json
```

将文件名中的轮次替换为 `r2`、`r3`。每轮同时记录模型冻结 revision、GPU/驱动、torch/CUDA 和配额；当前 JSON 已包含 torch/CUDA、配置和显存指标，但 GPU/驱动与模型 revision 仍需额外记录。后续比较 `short/mixed` 或 `gap-ms` 时，也应一次只新增一个变量。

### 结果不只按“越快越好”解读

- `max_active=8` 若输出吞吐提高，说明 decode batching 可能摊薄了 kernel 启动和权重读取成本。
- TTFT 变差可能来自排队和混合 prefill；ITL 变差可能来自更大的 decode batch 或长 prompt 占用本轮计算。需要结合 `short/mixed` 与 `gap-ms` 实验继续分离原因。
- `peak_allocated_bytes` 是 PyTorch 实际张量分配的峰值，`peak_reserved_bytes` 是 caching allocator 向 CUDA 保留的峰值，后者包含缓存和碎片空间，不等于模型真实占用。两者都不是“本轮新增内存”，因为已加载的权重和持久 KV 仍计入高水位。
- `free_blocks_after_run` 用于检查请求退休后的页回收。若完整回放后仍少于初始页数，应先排查 KV 生命周期泄漏，再讨论性能。
- 当前 Engine 在采样后调用 `.item()` 取得 token ID，会引入 CPU/GPU 同步，因此事件时间包含相应 GPU 工作。若以后改为全异步执行，不能继续默认 `step()` 返回就代表 GPU 已完成，需要重新设计同步点或 CUDA events。

### 与官方 vLLM 对照时先统一观测层

本 lab 的 TTFT 从 Engine 实际 submit 起算，`vllm bench serve` 观测的是在线 HTTP 服务，两者还相差网络、HTTP 解析、tokenize/detokenize 和流式传输。不能把自己的 Engine 内 TTFT 与官方客户端 TTFT 直接相减后就宣称某部分 kernel 快了多少。

公平对照至少需要统一模型 revision、dtype、输入输出长度、temperature、并发/到达过程、warmup 和重复次数，并尽量让两边都使用客户端在线指标。官方基准若提前遇到 EOS 而停止，输出长度会与固定 token 预算不同，也会污染吞吐比较；应明确并控制这一差异。源码对照则沿着请求准入、KV block 分配、调度、模型执行和输出处理寻找与本项目对应的机制，解释设计取舍，而不只比较一个总分。

## 实现自检清单

- 没有用 `statistics.quantiles` 的默认插值替代最近秩定义。
- 单 token 请求仍计入请求数、输出 token 数和 TTFT，但不产生 ITL 样本。
- 吞吐分母使用所有请求共同的最早提交与最晚完成，不把各请求持续时间相加，也不使用计划到达时间。
- 空回放不会求 `min`、`max` 或执行除零；空 TTFT/ITL 样本池的分位数为 `None`。
- 所有时间差先保持秒为单位，仅在生成 TTFT/ITL 毫秒样本时换算，避免吞吐被意外放大一千倍。

## summarize 首次代码复盘

功能测试通过不等于实现已经适合真实压测。本次实现中，TTFT/ITL 样本池、最近秩索引和吞吐分子分母均符合验收定义，同时发现以下可复用问题：

1. **库函数不能随意写标准输出。** 调试 `print` 在 pytest 默认捕获下不明显，但 `scripts.bench` 最终向 stdout 输出 JSON。额外的调试行会让 stdout 不再是合法 JSON，破坏重定向保存和机器解析。临时调试输出应在提交前删除；长期诊断信息需要走明确的日志通道。
2. **观测顺序与统计排序不是一回事。** `emitted_s` 是按事件观测先后形成的序列，TTFT 应取第一个观测，ITL 应计算原始相邻观测。对每个请求的发送时间排序会掩盖上游乱序问题，甚至把负间隔改写成看似合理的正间隔。需要排序的是 TTFT/ITL 的**聚合样本池**，因为最近秩分位数与原始到达顺序无关。
3. **测试未覆盖的零时长窗口仍需显式处理。** 非空输入若最早提交与最晚完成相同，直接除法会抛 `ZeroDivisionError`。实现应明确零时长口径，或拒绝不合法观测，不能把“现有样例不会发生”当作健壮性保证。
4. **辅助函数的前置条件应清晰。** 最近秩函数依赖调用方传入非空、已排序序列。当前调用点满足条件，但函数名本身没有表达该契约；后续整理时可改成具有明确语义的私有辅助函数，并避免使用难读的 `l` 作为变量名。
