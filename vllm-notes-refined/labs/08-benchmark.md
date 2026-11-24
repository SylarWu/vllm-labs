# Lab 08 · 请求回放与性能指标

**CPU 验收 `6 passed`，Lab 01–08 回归 `67 passed, 1 deselected`。** `summarize` 与 `run_trace` 已完成。真实 Qwen3 权重、WSL2 RTX 4080 压测和官方 vLLM 对照待验证。

## 1. 时间线与指标口径

`Arrival.at_s` 是相对回放起点的计划到达。`RequestTiming` 记录实际提交 `submitted_s`、引擎观测到的各输出 `emitted_s` 和终止 `completed_s`，以同一时钟的绝对读数保存。同步 Engine 在 tick 边界才接收新请求，tick 内计划到达者下轮提交。`BenchStats` 用这些记录跨请求聚合。

| 指标 | 定义与单位 |
|---|---|
| TTFT | 每个有输出的请求贡献一个 `(emitted[0] - submitted) × 1000` ms，包含实际 submit 后的排队和 prefill。 |
| ITL | 同一请求相邻观测时间之差 ×1000 ms，汇总进同一个 token 间隔样本池。单输出请求没有 ITL。长输出贡献更多样本，要看请求级体验可另建指标。 |
| 输出吞吐 | `sum(len(emitted)) / (max(completed) - min(submitted))`，单位 token/s。整窗含中间空闲，按共同窗宽计算。请求数为 `len(samples)`。 |
| p50/p95 | 对聚合样本排序，最近秩索引 `ceil(p*n)-1`，空池 `None`。保留每个请求的原始事件顺序，避免把乱序、负间隔改写成看似合理的结果。 |

A 在 0 ms 提交，20/50/80 ms 发三项。B 在 10 ms 提交，70 ms 发一项。TTFT 为 `[20,60]` ms，p50/p95 是 20/60。ITL 为 `[30,30]`，两档都 30。吞吐 `4/0.08=50` token/s。空输入请求数、输出数、吞吐均 0，分位数缺省。非空的零时长观测要显式定义返回口径或拒绝，以避免除零。最近秩辅助函数应明确非空、排序等前置条件，避免含糊命名。`statistics.quantiles` 的默认插值与最近秩有不同口径。所有时间差先用秒，仅延迟展示转 ms。

TTFT 观察首 token 的等待，ITL 观察持续 decode 节拍，吞吐观察总交付量。更大的 batch 可能提高 throughput 但降低单请求的节奏。服务端 HTTP 压测可另观察 200、首业务帧与首个非空文本，见 [Lab 07](07-streaming-api.md)。

## 2. run_trace 的状态机与异步到达

1. `replay_start=clock()` 读取一次，计划绝对时刻为 `replay_start+at_s`。
2. 每轮先提交所有已到期请求，包括同到达时刻的多条、上一长 tick 内积压的多条。判断到期可共用时钟快照，每次 `submit` 的实际时间要分别记录。
3. Engine 有工作就执行一个 `step()`，返回后取一次观测时刻处理其**全部事件**，同 tick 的请求可共用事件时间戳。先记本次 token，再在终止原因非空时用同一时刻完成请求。每个事件都算输出，`token_id=0` 合法。按 ID 累积内部可变记录，最后构造 frozen `RequestTiming` 和 tuple。
4. Engine 空闲且还有未来到达才 sleep 到下一计划时刻，使用同一快照判断与计算正等待，或重读后再检查已到期。忙时继续推进 decode。所有到达提交且无 work 后结束，空 trace 直接 `[]`。
5. 未完成以 `completed_s=None` 表示。若 Engine 无 work 却有请求缺终止事件，显式失败。用 `0.0` 占位或过滤未完成记录会隐藏请求丢失，扭曲全部指标。

Fake clock 起点是 10.000 s，tick 耗时 10 ms。B 计划 10.015 到达，实际 10.020 提交，10.030 完成。它的引擎 TTFT 是 10 ms，另外 5 ms 属回放推迟。A/B 完成后空闲至 C 的 10.100，sleep 70 ms。先 step 再 submit 会错过本可参加的 tick。

| 边界回归 | 旧错误与结论 |
|---|---|
| `test_replay_records_zero_valued_token_ids` | 旧 fixture 用 ID 2，`if event.token_id` 漏了合法的 0。按事件契约记录，可选值用 `is not None`。 |
| `test_replay_never_sleeps_for_negative_duration` | 判断未来后再次读钟，探针第 4 次读从 0 跨到 0.020 s，下一事件原为 0.010，旧代码 sleep `-0.010`。 |
| `test_replay_rejects_requests_without_completion_events` | Engine 提前空闲、缺终止时，应让回放失败且保留问题证据。 |
| stdout JSON | pytest 会捕获调试 print。库层 stdout 多一行就破坏 `scripts.bench > result.json`。长期诊断走日志通道。 |

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_lab08_bench.py -k summary
python -m pytest -q tests/test_lab08_bench.py
python -m pytest -q -m 'not real_model'
```

## 3. WSL2 真权重与可复现 GPU 实验

先做真实 Qwen 权重对拍和短 smoke，确认 CUDA、页池、JSON 输出，然后独立重复正式负载至少三轮。按原计划，先只改 `max_active=1/8`，固定模型快照、dtype、页大小与页数、budget、chunk、请求 trace、seed、输入输出长度。记录每轮 JSON、冻结 revision、GPU/驱动、Torch/CUDA 和资源配额，前两项当前脚本外补记。warmup 后脚本会同步 CUDA 并重置 peak memory。后续分开改变 `short/mixed` 或 `gap-ms`。

```bash
mkdir -p results/lab08
python -m scripts.bench --model-dir models/Qwen3-0.6B --device cuda \
  --max-active 1 --profile mixed > results/lab08/mixed-a1-r1.json
python -m scripts.bench --model-dir models/Qwen3-0.6B --device cuda \
  --max-active 8 --profile mixed > results/lab08/mixed-a8-r1.json
```

同样记录 r2/r3。吞吐提升可来自合批摊薄 kernel 启动与权重读取。TTFT 可能受服务排队和混合 prefill 影响，ITL 还受 decode batch 大小与长 prompt 占用影响。用 `short/mixed` 和间隔负载定位，而非只报一个吞吐数。

- `peak_allocated_bytes` 是 PyTorch 已分配张量高水位，`peak_reserved_bytes` 还包含 allocator 缓存与碎片。权重和持久 KV 都计入。
- `free_blocks_after_run` 应回到初始值。先排查页生命周期，再解释性能。
- 采样拿 `.item()` 会同步相关 CPU/GPU 工作。将来全异步时需要重定同步点或使用 CUDA events，重新定义 `step` 返回时的观测语义。

## 4. 官方 vLLM 对照

本地 Engine TTFT 从实际 submit 起算。`vllm bench serve` 经过网络、HTTP、tokenize/detokenize 和流式交付。对照时统一用同客户端观测层，并固定模型 revision、dtype、输入输出长度、temperature、并发/到达过程、warmup 和重复。核对 EOS 提前结束和固定输出预算的差异，以真实输出数算吞吐。读源码时追准入、KV block、调度、模型执行与输出处理这几段对应关系。

代码 `src/tinyserve/bench.py`、`scripts/bench.py`。原笔记 `learning_notes/lab08_bench.md`。真实权重测试入口见 [总索引](../README.md#下一步真实-qwen3)。
