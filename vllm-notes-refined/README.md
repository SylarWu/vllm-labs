# vLLM 学习笔记 · 提炼版

覆盖 `vllm-labs-main` 的 `learning_note.md` 和 `learning_notes/` 下全部 10 篇。正文保留计算公式、张量形状、接口边界、失败复现和最终验收，同一概念在主章节展开，相关章节通过链接跳转。

**当前进度**　Lab 01–08 的 CPU 实现和验收已完成，原记录的全量回归为 **`67 passed, 1 deselected`**。真 Qwen3 权重接入与对拍、WSL2 RTX 4080 压测、官方 vLLM 对照待进行。

## Lab 目录

| Lab | 可以查到什么 | CPU 验收记录 |
|---|---|---|
| [01 · Qwen3 Forward](labs/01-qwen3-forward.md) | dtype、RMSNorm、正弦位置编码和 RoPE、MHA/GQA/MQA、因果掩码、Pre-Norm 残差、SwiGLU、HF 权重映射、参数固定与后训练。 | 本关 `6 passed` |
| [02 · 单请求 Prefill / Decode](labs/02-single-request.md) | 右移预测、贪心与采样分布、KV 可复用的证明与数值实验、`start_pos`、cache 形状、`eval/inference_mode`、prefix cache、接口与生成边界。 | Lab 01+02 合计 12 通过。按命令筛除或跳过真模型用例。 |
| [03 · 分页地址与块池](labs/03-block-paging.md) | 连续 KV 与碎片、块归属、`reserve`、非连续物理 slot 翻译、失败原子性、末页和池满扩容。 | 本关 `3 passed` |
| [04 · 迭代级调度](labs/04-iteration-scheduler.md) | Orca 的迭代级思路、`Request/Step/TokenEvent`、FIFO、准入回调、EOS、取消、Step 提交与 ID 重用。 | 本关 `5 passed`，累计 `20 passed, 1 skipped` |
| [05 · 分页 KV 与 Engine](labs/05-engine.md) | 共享物理池、请求视图、scatter/gather、跨层地址、按层游标、页预算、等待与永久拒绝、跨 tick 持有和资源回收。 | 本关 `3 passed`，累计 `23 passed, 1 skipped` |
| [06 · 真 Batch Decode 与 Chunked Prefill](labs/06-batch-and-chunked-prefill.md) | 逐行 KV/RoPE/位置、变长 gather/pad、混合 batch、占位 NaN、调度预算、Step 活性、采样类型、下一轮 KV 对拍。 | 本关 `15 passed`，累计 `38 passed, 1 skipped` |
| [07 · FastAPI、SSE 与取消](labs/07-streaming-api.md) | 单 owner、事件路由、UTF-8 稳定前缀、队列积压、OpenAI 风格响应、HTTP 准入、ASGI 断流与例外清理。 | 本关 `23 passed`，累计 `61 passed, 1 skipped` |
| [08 · 请求回放与性能指标](labs/08-benchmark.md) | 实际提交时刻、TTFT/ITL、最近秩、可复现回放、时钟边界、GPU 显存高水位、官方对照的观测层。 | 本关 `6 passed`，全量 `67 passed, 1 deselected` |

## 通用附录

| 主题 | 内容 |
|---|---|
| [Python 环境与 uv](appendix/environment.md) | pip/Conda/uv、虚拟环境、editable 安装、extras、声明与锁文件。 |
| [Python 异步与 ASGI](appendix/python-async.md) | coroutine、Task、generator、协作式并发、Lock/Queue、取消、`aclose`、Uvicorn/Starlette、原始 ASGI 测试。 |

## 复习时常用的交叉索引

| 要确认的问题 | 对应章节 |
|---|---|
| Q/K 的绝对位置从哪里来，mask 划定什么？ | [Lab 01 的 RoPE 与 GQA](labs/01-qwen3-forward.md) · [Lab 02 的 `start_pos`](labs/02-single-request.md) · [Lab 06 混合行位置](labs/06-batch-and-chunked-prefill.md) |
| 有多少 KV 容量，哪些 token 已写，哪些 slot 当前可读？ | [Lab 02 的写入时序与 `P+N-1`](labs/02-single-request.md) · [Lab 03 的块与保留范围](labs/03-block-paging.md) · [Lab 05 的各层已写长度](labs/05-engine.md) |
| plan、forward、complete、退休与交付发生在什么时候？ | [Lab 04 的一轮事务](labs/04-iteration-scheduler.md) · [Lab 06 的 `emit_token`](labs/06-batch-and-chunked-prefill.md) · [Lab 07 的响应生命周期](labs/07-streaming-api.md) |
| 为什么最终结果对拍通过，系统仍可能有问题？ | [Lab 02 的重复拼接与实际 K/V 长度](labs/02-single-request.md) · [Lab 06 的 padding、下一步 decode 与事件类型](labs/06-batch-and-chunked-prefill.md) · [Lab 07 的 peer 终止、积压和首块断流探针](labs/07-streaming-api.md) |
| 如何解释吞吐、首 token 与尾延迟的变化？ | [Lab 06 的 budget 和切块](labs/06-batch-and-chunked-prefill.md) · [Lab 07 的 HTTP 与 Engine 边界](labs/07-streaming-api.md) · [Lab 08 的指标和负载设计](labs/08-benchmark.md) |

## 下一步真实 Qwen3

原定顺序是先用本地 Qwen3‑0.6B 快照跑 FP32 对拍，再在 WSL2 RTX 4080 上做 BF16 服务与压测。同一负载的基线和对照保留模型 revision、软件/驱动版本、运行参数和每轮原始结果。以下是项目中的真权重测试入口，负载及指标定义见 [Lab 08](labs/08-benchmark.md)。

```bash
QWEN_MODEL_DIR="$PWD/models/Qwen3-0.6B" QWEN_DEVICE=cuda \
  .venv/bin/python -m pytest -q tests/test_lab02_cache.py -m real_model
```

项目原实验计划在 `vllm-labs-main/README.md`。后续笔记可继续记录复用原理、形状推导、故障证据和验收结果，性能实验同时留环境、命令与结论。
