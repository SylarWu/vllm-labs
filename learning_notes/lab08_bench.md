# Lab 08 · 请求回放与性能指标

**状态：已开始。** Lab 01～07 的 CPU 回归为 `61 passed, 1 skipped`。Lab 08 当前 3 项验收在 `summarize` 和 `run_trace` 的占位实现处抛 `NotImplementedError`。先完成指标汇总，再用 fake clock 验证回放，最后做固定负载下的性能对比。 [学习计划](../README.md#lab-08--指标与源码对照) · [验收代码](../tests/test_lab08_bench.py)

## 观测的是哪条时间线

`Arrival.at_s` 是相对回放起点的计划到达时间。`RequestTiming.submitted_s` 是实际调用 Engine 提交时记录的时间，`emitted_s` 是逐个输出 token 的观测时间，`completed_s` 是观察到终止事件的时间。当前 Engine 同步执行一整个 tick。tick 中间到达的请求，要等到这个 tick 返回后的调度边界才能提交。

Lab 07 记录的是怎么把 Engine 事件送进 SSE。Lab 08 在 Engine 层观测 `TokenEvent`，所以 TTFT 包含 submit 之后的 Engine 排队和 prefill。在线客户端还可以从发送 HTTP 请求起计时，分别观察首帧与首个非空文本。这些观测点适用于不同的性能问题。 [Lab 07 逐帧推演](lab07_api.md#从一次-post-逐帧推演)

## 第一步，手算后实现 summarize

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
