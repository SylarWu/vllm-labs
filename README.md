# 从零实现单卡 vLLM 风格推理引擎

**8 周 · 每周 8–10 小时 · Python / PyTorch · Mac CPU + WSL2 RTX 4080**

目标是让你亲手实现这一条路径：

`Qwen3 权重 → 自回归生成 → KV cache → 分页存储 → 迭代级调度 → 张量批处理 → 流式 API → 压测`

你已经写过 Transformer，做过 Needle 和 6.824 的实验。第 1 周只做 Qwen3 推理特有的结构和数值对拍。第 3–6 周用块池、请求状态和 token 预算检验系统设计。第 7–8 周把订单中心的异步治理、资源隔离和压测经验转到模型服务。

## 开始实验

```bash
cd vllm-labs
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[lab]'
python -m pytest --collect-only -q
python -m pytest -q tests/test_lab01_model.py
```

打开 `src/tinyserve/model.py`，先实现 `rms_norm`，可以用 `-k rmsnorm` 单独执行它的测试。每周修改 `src/tinyserve/` 下的实现，保留 `tests/` 里的验收断言。学生入口初始抛出 `NotImplementedError`。前 8 周的 37 条基础用例使用 CPU、确定性小模型或模拟器，安装依赖后可离线运行。

| 周 | 知识与动手任务 | 周末验收文件 |
|---|---|---|
| 1 | 纯张量实现 Qwen3 的算子、权重映射与整模型 forward | `tests/test_lab01_model.py` |
| 2 | Prefill/decode、连续 KV cache、贪心生成 | `tests/test_lab02_cache.py` |
| 3 | Block pool、block table、非连续页地址映射 | `tests/test_lab03_paging.py` |
| 4 | 迭代级 continuous batching 调度、请求生命周期 | `tests/test_lab04_scheduler.py` |
| 5 | 分页 KV 张量存储、真实模型联调、显存页准入 | `tests/test_lab05_engine.py` |
| 6 | 整批 decode、变长历史 mask、chunked prefill | `tests/test_lab06_batch.py` |
| 7 | FastAPI、SSE、断流取消、并发准入 | `tests/test_lab07_api.py` |
| 8 | 请求回放、TTFT/ITL、吞吐和显存，对照官方 vLLM | `tests/test_lab08_bench.py` |

周验收命令统一用 `python -m pytest -q tests/test_labNN_xxx.py`，把路径换成表中的文件。建议每周前半段留约 2 小时读资料和写状态图，余下时间用于实现、定位失败和记录一个实测观察。第 6 周优先通过多请求 decode 对拍，再加 chunked prefill。

## Lab 01 · Qwen3 Forward

**先读** [Tiny-LLM Week 1](https://skyzh.github.io/tiny-llm/) 和 [Qwen3 配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)。用 `hidden_size`、`head_dim` 和 Q/KV head 数画出四个 attention 投影，特别留意 `num_attention_heads × head_dim` 可以大于 `hidden_size`。

**实现** `model.py`。用基础 PyTorch 张量运算和 `nn.Linear` 或 `F.linear` 实现 RMSNorm、split-half RoPE、因果 GQA 和 SwiGLU。以 `QwenConfig` 驱动 embedding、各层、最终 RMSNorm 和 LM head。Qwen3 的 Q、K 在各 head 的最后一维上做 RMSNorm，再应用 RoPE。两条残差路径分别围绕 attention 和 MLP。`from_state_dict` 接收 HF 风格的权重名，示例键名在 `tests/toy.py`。对于 tied embeddings，LM head 复用 embedding 权重，权重文件可能省略独立的 `lm_head.weight`。

**验收** RMSNorm 遇到较大的 FP16/BF16 输入仍与 FP32 参考一致。RoPE 在位置 0 为恒等变换，位置 1 的旋转可手算。GQA 与 PyTorch SDPA 对拍，打乱未来的 V 不影响之前的位置。双层小 Qwen3 的全部 logits 与独立 PyTorch 参考生成的静态值对拍，FP32 的 `atol=rtol=2e-5`。

## Lab 02 · 单请求 Prefill / Decode

**先读** [Tiny-LLM Week 2 的 KV cache 章节](https://skyzh.github.io/tiny-llm/)。画出长度为 5 的 prompt 生成 3 个新 token 时，三次 forward 各自读写的输入位置和 logits。

**实现** `cache.py`、`generation.py` 和 `model.py` 的 cache 路径。每层 K/V 的逻辑格式为 `[T, Hkv, D]`。写入是按层 append，读到未写位置、覆盖、越界写入要报错，失败后原数据仍可读取。`forward` 本次输入为 `[1, T]`，RoPE 从 `start_pos` 开始，causal mask 用 query 的绝对位置。prefill 的最后一个位置直接给出第一个新 token，随后每步只输入上一步采样的 token。输出只包含新 ID，EOS 算作一个已发出的 token，0 个新 token 时直接返回空列表。

**验收** 全序列重算、逐 token decode、分段 prefill 的 logits 一致。固定采样下，带 cache 与不带 cache 的 token ID 完全相同。Mac 先过 toy tests，WSL 再跑真权重对拍。可选真实模型测试的命令在下面。

## Lab 03 · Block Pool 与 Block Table

**先读** [PagedAttention 论文](https://arxiv.org/abs/2309.06180) 的 KV 内存管理部分。用 4 个物理页画一遍逻辑页到物理页的地址翻译。

**实现** `paging.py` 中的 `BlockPool` 和 `BlockTable`。单进程单 owner。`allocate(n)` 要么一次获得 n 个页，要么一次都不取。`free` 对重复 ID、未分配 ID 或越界 ID 做整批校验。`reserve(n)` 只增长，旧映射保持稳定。`slots(start,count)` 只对已保留的逻辑范围返回扁平物理 slot，`release()` 幂等。

**验收** 强制造成中间有空洞的页池，跨页位置仍按逻辑顺序正确映射。用固定种子做 150 轮随机操作，每轮检查 `已占页数 + 空闲页数 = 总页数`，并检查活跃页归属唯一、失败时状态不变。

## Lab 04 · 迭代级调度

**先读** [Orca 论文](https://www.usenix.org/conference/osdi22/presentation/yu) 的 iteration-level scheduling。先用 fake logits 建立调度时序，然后连接模型。

**实现** `scheduler.py`。每个 tick 先按 FIFO 接纳，队首准入失败时保留次序。`plan()` 给每个活跃请求至多一个 `Step`，调用者在下一次 `plan()` 前完成当前迭代的全部 `complete()`。第一版的 prefill 吃完整 prompt 且发首 token，decode 消费上一步生成的 token。EOS 优先于长度上限。等待或活跃请求均可在 tick 边界取消，终止时触发一次 `_on_retire`。`try_admit` 返回 False 时请求继续等待。

**验收** A 长请求在 decode 时，B 能加入并结束，C 能接续占用空位。验证 EOS 和输出上限同 tick 的行为，等待态及运行态的取消，非法输入、重复活跃 ID 和队首阻塞后的再次尝试。阅读 nano-vLLM 的 `engine/scheduler.py` 可以用同一条 trace 对照。

## Lab 05 · 分页 KV 与端到端 Engine

**先读** [vLLM V1 的架构概览](https://docs.vllm.ai/en/stable/design/arch_overview/)。回看 nano-vLLM 的 `engine/block_manager.py`、`engine/model_runner.py` 和 `layers/attention.py`，追踪 block table 与 slot mapping 怎样进入 forward。

**实现** `PagedStorage`、`PagedKV` 和 `engine.py`。两块物理张量 K/V 采用 `[L, blocks, block_size, Hkv, D]`，通过 `BlockTable` 计算索引。先把逻辑历史 gather 成连续张量，再让普通 attention 消费它。初版 Engine 可以逐个运行 `Step`；下周把同一轮的 decode 行真正合并为一次 forward。

页池容量按整个模型、全部层共同计数，页归单个请求所有。活跃请求准入时预留 `prompt_len + max_new_tokens - 1` 个 KV token 的页，最后被采样的 token 不再进入 cache。超出总容量的请求直接拒绝，暂时放不下的队首等待。结束和取消都释放页。该策略能避免活跃请求各占一部分页后互相等待，可以量出它带来的闲置预留。

**验收** 非连续物理页能正确读回，两个请求和两层的 KV 相互独立。小页池满时其他请求等待，在已有请求结束或取消后继续。混合请求的输出等于逐条完整重算，结束后页池恢复初始容量。

## Lab 06 · 真 Batch Decode 与 Chunked Prefill

**先读** [vLLM 的 chunked prefill 调优说明](https://docs.vllm.ai/en/stable/configuration/optimization/) 和 [Tiny-LLM Week 3](https://skyzh.github.io/tiny-llm/)。

**实现** 两部分。先让 `TinyQwen3.decode_batch` 每层共同处理 `[B, 1, D_model]`，每行独立计算 RoPE 绝对位置、写入自己的 cache。把历史 gather/pad 到最大长度，用每行长度屏蔽 padding，再做一次 batched attention。`Engine.step` 把所有 decode 行交给同一次 `decode_batch`。prefill chunk 仍可逐请求执行。

随后将 chunked prefill 接入调度器。每次先排 decode，每行占 1 个 token。剩下的预算按活跃 prefill 顺序使用，每个请求至多取一个不超过 `max_prefill_chunk` 的片段。`max_batch_tokens` 有值时至少等于 `max_active`。只有最后一个 prompt chunk 可以采样，未完成的 prefill 只提交 KV 和游标。

**验收** 两条不同历史长度的序列，批量 logits 与逐条重算对拍，下一步读到的 KV 仍正确。spy 验证引擎只调用一次 batched decode。长 prompt 拆分时 decode 每轮仍出 token，提前取消的半个 prefill 能归还所有页。此时已有可运行的单卡简化引擎。

## Lab 07 · 流式服务与取消

**先读** [mini-vLLM 的服务架构](https://github.com/Mihawii/mini-vllm/blob/main/docs/architecture.md)。结合你做过的异步聚合框架，画出到达、入队、发出首 token、客户端断开、回收页的时序。

**实现** `api.py`。单一 engine owner 接收多路 chat 请求，并把输出按 ID 发送给各自 stream。启动 engine step 前主动让出一次事件循环，允许并发请求先完成入队。单进程实验可在协程中同步执行每个短 tick。前后端通过 `Tokenizer`、`ChatRequest` 和 `ChatEvent` 对接。

`POST /v1/chat/completions` 支持 `model`、`messages`、`max_tokens`、`stream`、`temperature=0`。普通响应使用 `chat.completion` 和 `choices[0].message`。流式响应以 `data: <JSON>\n\n` 发送 `chat.completion.chunk`，用 `data: [DONE]\n\n` 结束。响应包含同流一致的 `id`、`created`、`model`，以及 `index` 和终止原因。未知模型 404，格式或参数不合法 422，超过 `max_inflight` 时 429。关闭 SSE 时关闭 backend generator，并让后端在 tick 边界取消对应请求。tokenizer 可先对累计 token 解码，用稳定前缀发 delta，留住未完整的 UTF-8 尾部，等下一个 token 再解码。

**验收** 非流式与 SSE 的内容和终止原因一致。模拟过载后原请求能结束并恢复准入。用原始 ASGI 的 `http.disconnect` 验证断流关闭，避免被 HTTPX ASGI transport 的缓冲掩盖。两条并发 stream 的输出没有串到一起，取消后引擎 active 为空。

### WSL2 真权重试跑

在 Windows 侧安装支持 WSL 的 NVIDIA 驱动，在 WSL2 Ubuntu 中先跑 `nvidia-smi`。PyTorch wheel 从 [官方选择器](https://pytorch.org/get-started/locally/) 按驱动支持的 CUDA 版本选。若要编译 CUDA，参照 [NVIDIA WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/) 安装 toolkit-only。独立 venv 安装官方 vLLM 和 nano-vLLM，避免它们更改本实验的 Torch 依赖。

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
uv pip install -e '.[lab,hf]'
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

QWEN_MODEL_DIR="$PWD/models/Qwen3-0.6B" QWEN_DEVICE=cuda \
  python -m pytest -q tests/test_lab02_cache.py -m real_model

python -m scripts.serve --model-dir models/Qwen3-0.6B --device cuda
```

另一个 WSL 终端调用：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-lab","messages":[{"role":"user","content":"用一句话解释什么是 KV cache"}],"max_tokens":64,"stream":true}'
```

Qwen3-0.6B 的 BF16 权重约 1.1 GiB。普通 GQA KV 元素每 token 是 `2 × 28 层 × 8 KV heads × 128 head dim × 2 字节 = 112 KiB`。640 页、每页 16 token 的池约 1.09 GiB。运行时还需要临时张量、logits 和框架内存。调试先用 0.6B，FP32 做数值对拍，BF16 做真实负载。真模型命令使用本地快照，固定快照 revision 后可复现实验。

## Lab 08 · 指标与源码对照

**实现** `bench.py` 的 `run_trace` 和 `summarize`。`at_s` 相对 replay 开始计时，按到达顺序提交请求，记录实际提交时间和每个 token event 的观测时间。engine 没有工作时 sleep 到下一事件。同步单 owner 的 batch tick 期间到达的请求在下个 tick 被受理，延迟统计从实际提交算起。在线客户端的排队时间另用 HTTP 指标衡量。

- TTFT 是每请求第一个 token 时间减提交时间，包含 engine 排队。
- ITL 汇总所有请求内相邻 token 的时间差。单 token 请求只计入 TTFT。
- 吞吐用所有输出 token 数除以从第一个提交到最后一个完成的时间。
- p50/p95 用最近秩法。没有样本的分位数记为 `None`，空回放的吞吐记 `0`。

**验收** fake clock 验证乱入的长短请求、空闲等待、分位数、单 token 请求及 throughput 分母。在 4080 用同一份固定 seed 的负载，独立运行三个重复实验，记录每次结果。`max_active=1` 是串行基线，`max_active=8` 测真实 decode batching。两边共享精度、页大小、页池和输入输出长度。每轮记录冻结的模型 revision、torch/CUDA、GPU/驱动、配额、TTFT p50/p95、ITL p50/p95、输出 tok/s 和 PyTorch peak allocated/reserved。用 `--profile short` 对比全短请求，`--profile mixed` 保留长短混合。`--gap-ms 15` 模拟分批到达。性能无固定速度门槛，请解释瓶颈和取舍。

```bash
python -m pytest -q -m 'not real_model'
python -m scripts.bench --model-dir models/Qwen3-0.6B --max-active 1 --profile mixed
python -m scripts.bench --model-dir models/Qwen3-0.6B --max-active 8 --profile mixed
```

在独立环境启动官方 `vllm serve`，用 [vllm bench serve](https://docs.vllm.ai/en/stable/cli/bench/serve/) 验证线上服务。同图比较时，两个服务走同一个 HTTP 客户端、同一请求 trace，记录有效输出 token 数并注明 warm-up、prefix cache 与 CUDA graph 配置。实验包的 `scripts.bench` 是引擎内量测，适合做你自己实现的消融实验。

| 你的实验代码 | 官方 vLLM V1 中的阅读入口 |
|---|---|
| `engine.py` | [架构概览中的 EngineCore 和 Worker](https://docs.vllm.ai/en/stable/design/arch_overview/) |
| `scheduler.py` | [v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) |
| `paging.py` | [v1/core/kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) |
| 跨请求共享页的进阶方向 | [Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/) |

完成记录可以由一页请求时序、两张性能图和一条可复现命令组成。选择一个你实测过的现象，说明负载、指标变化、原因和代价。例如长 prompt 被切块后，正在生成的请求 ITL 怎样变化，长请求自己的 TTFT 又怎样变化。

## 接下来选一条扩展

- **MaaS 平台。** 在自己的引擎或官方 vLLM 前做租户级 token 配额和 admission control。用两个租户的混合长短请求做验收，记录一方过载时另一方的 p95 TTFT、限流计数和请求完成率。可用 Java 写网关并保留 Python 引擎。
- **GPU 推理。** 跟 [Stanford CS336 Assignment 2](https://github.com/stanford-cs336/assignment2-systems) 做 PyTorch/Triton FlashAttention 练习，再给当前引擎写一个直接读 block table 的 Triton decode kernel。以现有 gather+dense 为基准，测跨页、变长和 FP32/BF16 误差，再测 kernel 延迟和显存分配。
- **Agent 后端。** 用你的 OpenAI 风格服务驱动一个合成订单的只读查询助手。实现 tool schema、执行状态和超时，用固定的 mock model 回放正常调用、参数不合法、工具超时以及取消后恢复。另用真模型记任务完成率、错误工具调用率及执行延迟。Tiny-LLM 的 Week 4 有可对照的工具循环和评价样例。
