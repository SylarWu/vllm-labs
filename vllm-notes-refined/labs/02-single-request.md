# Lab 02 · 单请求 Prefill、Decode 与连续 KV Cache

**CPU 已完成。** Lab 01+02 合计 12 条通过，真权重用例待跑。原记录中筛掉该用例时为 `12 passed, 1 deselected`，未设置 `QWEN_MODEL_DIR` 时为 `12 passed, 1 skipped`。

## 1. Next-token logits 与生成时序

`forward([B,T]) → [B,T,V]`。输入 `[A,B,C]` 的三行分别对 `P(·|A)`、`P(·|A,B)`、`P(·|A,B,C)` 打分，训练目标右移一位，前两行分别预测 B、C。生成从 `logits[:, -1, :]` 取 `[B,V]`，沿词表轴 `argmax(dim=-1)` 得 `[B]`。比较 logits 可做贪心，需要概率时再 softmax。

| Forward | 输入 | `start_pos` | 新写 KV | Attention 读取 | 采样 |
|---|---|---:|---|---|---|
| 1，prefill | 5 个 prompt token | 0 | `[0,5)` | `[0,5)` | `y1`，位置 5 |
| 2，decode | `y1` | 5 | `[5,6)` | `[0,6)` | `y2`，位置 6 |
| 3，decode | `y2` | 6 | `[6,7)` | `[0,7)` | `y3`，位置 7 |

已知 prompt 可并行处理其 token，decode 每步输入上次输出。一个生成 token 被送入下一次 forward 后才产生自己的 K/V。`P` 个 prompt、`N>0` 个新 token 最终写入 `P+N-1` 项。已输出 `N` 项后，下一轮应把第 `N` 个输出送到位置 `P+N-1`。也可以在每轮 forward 后按本轮输入数推进游标，避免跳位。`N=0` 直接返回空列表，遇 EOS 先计入输出再停。

## 2. 采样配置与固定 logits 实验

| 配置 | 作用 |
|---|---|
| `temperature` | 正温度用 `softmax(logits/T)`。小于 1 更集中，大于 1 更平。排序不变，正温度再 argmax 的选择相同。vLLM 的 `temperature=0` 直接贪心。 |
| `top_k` | 保留分数最高的 k 项，再按权重抽样。 |
| `top_p` | 按概率从高到低累加，保留累计首次达到 p 的最短前缀，包括刚越过阈值的那项。重新归一化后抽样，`p=1` 关闭这一截断。 |
| `min_p` | 保留概率至少为 `min_p * max_prob` 的项。最高概率 0.40、`min_p=0.1` 时门槛 0.04。 |
| Penalty 和 seed | `presence_penalty` 看是否出现，`frequency_penalty` 看次数，`repetition_penalty` 也处理 prompt 与已生成中的 token。`seed` 固定随机数生成器起点。 |

同一前缀、同一模型的原始 logits 与采样配置无关。选出的 ID 进入下一轮前缀，首次分歧能改变后续生成。vLLM 的 PyTorch 参考实现组合 top-k/top-p 时先把 k 以外置 `-inf`，再基于过滤后的分布计算 top-p。参见 [SamplingParams](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/)、[组合采样代码](https://docs.vllm.ai/en/stable/api/vllm/v1/sample/ops/topk_topp_sampler/#vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p_pytorch) 和 [Nucleus Sampling](https://arxiv.org/abs/1904.09751)。

固定 A–E 的 logits `[3,2,1,0,-1]`，令 `top_p=0.9`。表中概率为截断前。

| T | A | B | C | D | E | 保留 |
|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 86.47% | 11.70% | 1.58% | 0.21% | 0.03% | A、B |
| 1.0 | 63.64% | 23.41% | 8.61% | 3.17% | 1.17% | A、B、C |
| 2.0 | 42.87% | 26.00% | 15.77% | 9.56% | 5.80% | A、B、C、D |

`T=1` 时 A+B ≈87.05%，加入 C ≈95.67%，归一化后约为 66.52%、24.47%、9.00%。复现实验用 `(logits/T).softmax(-1)`、`(probs.cumsum(-1) < .9).sum().item() + 1`，然后截取并除以保留概率之和。此处 logits 已降序。设 `top_k=2` 时三个温度都保留 A/B，但抽样权重随温度变化。

`argmin` 思考实验可同时取合法候选中分数最高和最低的项，任一项是 EOS 时按约定记一次并停止，否则输出分数最低项。EOS 可能一直位于中间，仍需 `max_new_tokens`。过滤用的 `-inf` 项要先排除，否则会被 argmin 选中。

## 3. 为什么历史 K/V 可复用

当前 Q 对可见 K 做点积，softmax 后加权同范围的 V。后续只产生新位置 Q，并读取历史 K/V。相同前缀在数学上也会重算出相同的旧 Q，但后续 query 无需消费它。缓存的是各层的激活。K 路径为 `w_k(pre_norm(h)) → 拆 head → k_rms_norm → 绝对位置 RoPE`，V 路径为 `w_v(pre_norm(h)) → 拆 head`，二者保持 `Hkv`。

Causal mask 是训练与逐步推理的共同条件概率约束。位置 A 若读到了目标 B，训练就泄漏答案。新增 token 后旧位置的 embedding、RoPE 位置和可见前缀仍相同，residual、RMSNorm、MLP 均按 token 作用。逐层归纳得到旧 K/V 可复用，可省掉旧位置的投影、attention 和 MLP，当前 attention 仍会读取历史。放开 mask 时，第 0 层旧 K/V 仍由相同 embedding 来，旧位置 attention 输出却可能变，随后的层会得到变化的旧 K/V。

| `[10]` 与 `[10,8]` 的旧位置 0 | 层 0 K / V 最大绝对误差 | 层 1 K / V 最大绝对误差 |
|---|---|---|
| 正常因果 | `1.19e-7 / 8.94e-8` | `4.77e-7 / 8.94e-8` |
| 全可见 | `1.19e-7 / 8.94e-8` | `3.21 / 0.95` |

对照组只在进程内包装 `causal_gqa`，令 mask 计算的 `query_start=k.shape[-2]-1` 以开放全部 key，RoPE 仍用真实位置。

## 4. 缓存接口、形状与位置

- `KVStore` 以 `Protocol` 声明 `read/write`。Python 使用结构化类型，签名匹配即可满足接口。`ContiguousKV` 为 K/V 各预分配 `[L,capacity,Hkv,D]`，每层独立写入游标。FP32 toy 每 token `2×2×2×8×4=256 B`，实际分配乘以 capacity。`torch.empty` 的容量可切片，读取时还要用本层已写长度确定有效前缀。
- 写入先校验层号、三维 K/V 完整形状、`start_pos == write_length[layer]` 和容量。`copy_` 会广播，不能只核对 token 数。切片为 `[S:S+T]`，禁止覆盖或跳写。K 与 V 都复制完再推进游标。当前契约拒绝空读写与零层/零容量，`read(layer,0)` 在写入前后都报 `ValueError`。
- `TinyQwen3.forward → Decoder → Attention` 传同一 `start_pos`、cache 和各自的 `layer_idx`。从 K 做 RoPE 后写本段 `[S,S+T)`，再读 `[0,S+T)`。写入用 `[1,Hkv,T,D] → [T,Hkv,D]`，只去 batch 轴后换 token/head。无参 `squeeze()` 会在 `T=1` 顺手删 token 轴。读回交换两轴并补 batch。用 `transpose/permute` 表示语义换轴。
- `S=3,T=3,Hq=4,Hkv=2,D=8,V=43` 时，Q `[1,4,3,8]`，本段 K/V 写 `[3,2,8]`，读 `[6,2,8] → [1,2,6,8]`，score `[1,4,3,6]`，logits `[1,3,43]`。query 3、4、5 分别见 key `0..3`、`0..4`、`0..5`。`start_pos` 传给 RoPE 和 mask 的 `query_start`，保证同一绝对坐标。
- 先写再读后，读出结果已包含本段。再拼本段会把 `Tkv` 从 `S+T` 变为 `S+2T`，额外项可能都被未来掩码遮住，导致 logits 照样通过。用 spy 核对 attention 入口 `Tk=Tv=S+T`，游标与 attention 实际长度要各查一次。

## 5. greedy、device 与 autograd

- 缓存路径的 `B=1`。无 cache 的整段 forward 实测支持 `B=2`。无 cache 的 query/key 都从 0 编号，显式非零 `start_pos` 会让 query 偏移并读到未来，适合在入口 `ValueError`。静默归零会掩盖调用端丢失的前缀。正生成预算要求非空 prompt，零预算保持直接返回。缺少入口校验时，带 cache 的 B=2 会在 `squeeze(0)` 后的 `permute` 报维数错误，空 prompt 则在 attention 的 reshape 报错。
- 单序列采样用 `logits[0,-1].argmax().item()` 得 Python int，生成器返回 `list[int]`，下一轮以 `[[next_id]]` 构造 `[1,1]`。Embedding 输出形状是 `input_ids.shape + [D]`，多做一次 unsqueeze 会使输入 `[1,1,1]`，输出成为 4 维。张量取模型参数 device 或沿用 `input_ids.device`，ID 用 `torch.long`。
- 无缓存每轮只追加最新选出的 token。错误地重新拼接全部 `generated_ids` 时，固定模型的输入长度从应有的 `[2,3,4,5]` 变为 `[2,3,5,8]`。重复选同一个 ID 的 toy 可能仍通过 token 对拍，需观测真实输入序列与长度。
- `eval()` 切换模块 `training`，影响 Dropout/BatchNorm。`inference_mode()` 在当前线程关闭推理范围内的 autograd，退出恢复外层状态。`no_grad()` 同样停止记录反向图，对后续需参与梯度的张量较宽松。只把预分配张量设为 `requires_grad=False` 时，`copy_` 仍可能随源 K/V 被 autograd 记录。
- 实测 `eval()` 后 `training=False`，默认梯度模式的 `logits.requires_grad` 仍为 True。`inference_mode()` 内为 False，退出后外部 `is_grad_enabled()` 为 True。`hf_io.load_model()` 和 `tests.toy.make_model()` 负责 eval，生成入口的 inference scope 覆盖整个循环和 cache 写入。[PyTorch Autograd](https://docs.pytorch.org/docs/2.15/notes/autograd.html#locally-disable-grad-doc) · [inference mode](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.grad_mode.inference_mode.html) · [copy_](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.copy_.html)

## 6. 请求内缓存、跨请求缓存与会话

| 机制 | 保存内容与效果 |
|---|---|
| 请求内 KV cache | 生成期间的每层历史状态，多请求 decode 每行 1 个输入组成 `[B,1]`。缓存失效可对完整前缀重做一次 prefill，恢复逐步 decode。 |
| Prefix / prompt caching | 跨请求保留并索引从位置 0 起完全匹配的 token 前缀，适合稳定 system prompt、工具描述、文档或共享会话前缀。复用要求模型、tokenizer、位置和 RoPE 一致，只减少重复 prefill。逻辑上下文和 decode 的 KV 读取依旧。 |
| 会话语义 | 无状态 API 由客户端重发完整消息列表。有状态产品可用会话 ID 和数据库保存消息或缓存。请求逻辑前缀决定语义，缓存淘汰只改变性能，模型权重始终独立。 |

## 7. 实测与排错

Mac arm64 CPU，Python 3.12.4，PyTorch 2.15.0。同步后在 `f5473d5` 复验。

| 探针 | 观测 |
|---|---|
| 分段 logits | `(4,1,1)` 的各层游标均按 `4→5→6`。`(1,2,3)` 按 `1→3→6`。沿 token 轴拼接与整段对拍，容差 `atol=rtol=2e-5`，后者实测最大绝对误差约 `2.38e-7`。 |
| GQA 长度 spy | `(1,2,3)` 的错误 `(start,Tq,Tkv)` 为 `(0,1,2),(1,2,5),(3,3,9)`。修复后两层 `(start,Tq,Tk,Tv)` 均为 `(0,1,1,1),(1,2,3,3),(3,3,6,6)`。 |
| 独立位置实验 | Q 取绝对位置 3、4 时，`query_start=3` 与完整 GQA 切片的最大误差 0，设为 0 时约 `2.47966`。无 cache 传 `start_pos=1`，在 `[10]` 后追加 `8` 使首位置 logits 变动约 `1.05`。传 0 约 `1.49e-7`。 |
| 生成 7 token | 4-token prompt 无 cache 输入长度 `4,5,6,7,8,9,10`。有 cache 一次 `[1,4]`、六次 `[1,1]`，位置 `0,4,5,6,7,8,9`，容量 10 与最终游标 `[10,10]`。两路 ID 一致且所有 forward 在 inference mode。 |
| 另一个无缓存样本 | `[10,8,1,6,42]` 重算三轮，logits `[1,5,43] → [1,6,43] → [1,7,43]`，每轮选到 42。相同 ID 在序列的不同位置分别处理。按行 RoPE 分项另有 `1 passed`。 |

```bash
.venv/bin/python -m pytest -q tests/test_lab02_cache.py -k contiguous_cache
.venv/bin/python -m pytest -q tests/test_lab02_cache.py -k cached_logits
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_lab01_model.py tests/test_lab02_cache.py
```

若第一段通过而后续偏差大，依次查各层游标、K/V 实际长度、RoPE 和 mask 偏移，然后再查生成循环的输入形状。下一步真实 Qwen 命令见 [Lab 08](08-benchmark.md)。本篇提炼自 `learning_notes/lab02_cache.md`。
