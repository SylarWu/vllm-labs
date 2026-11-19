# Lab 02 · 单请求 Prefill / Decode

**状态：进行中。已验证绝对位置 causal mask，下一步实现 `ContiguousKV`。**

Lab 01 已经能把 `[B, T]` 的输入 token ID 转成 `[B, T, vocab_size]` 的 logits。本 lab 用同一个模型生成多个 token，并以完整序列重算为基线，验证逐步复用历史 K/V 的结果。模型结构和算子见 [Lab 01](lab01_model.md)。

## 从 Forward 到生成

**Prefill** 指生成前处理已知的 prompt，用这些输入在每一层填充 KV cache。Lab 02 一次送入整个 prompt，各层可以并行计算多个 prompt 位置。**Decode** 指后续逐步生成，每轮把上一轮选出的 token 送进模型，追加该位置的 K/V，再预测下一个 token。Lab 06 会把长 prompt 的 prefill 分成多轮处理。

`TinyQwen3.forward` 的输出 `[B, T, vocab_size]` 是每个输入位置对下一个 token 的候选分数。以 `[A, B, C]` 为例，三行分别给出 `P(· | A)`、`P(· | A,B)`、`P(· | A,B,C)` 对应的 logits。训练时右移一位的目标让前两行分别预测 `B` 和 `C`。从这段 prompt 继续生成时，`logits[:, -1, :]` 取最后一行，得到 `[B, vocab_size]`。贪心生成对词表轴做 `argmax(dim=-1)`，得到形状为 `[B]` 的 token ID。需要概率时可以对词表做 `softmax`，贪心选择直接比较 logits 即可。

刚采样出来的输出要等到被送进下一次 forward，才会产生它自己的 K/V。最后一个 prompt 位置的 logits 已足够选出首个新 token。

设 prompt 有 5 个 token，位置从 `0` 开始，生成 `y1`、`y2`、`y3`，中途没有 EOS。带缓存的时序如下，区间采用左闭右开。

| Forward | 阶段 | 本次输入 | `start_pos` | 每层新写 KV | Attention 读取 KV | 采样结果 |
| --- | --- | --- | ---: | --- | --- | --- |
| 1 | Prefill | 整段 prompt | 0 | `[0, 5)` | `[0, 5)` | `y1`（位置 5） |
| 2 | Decode | `y1` | 5 | `[5, 6)` | `[0, 6)` | `y2`（位置 6） |
| 3 | Decode | `y2` | 6 | `[6, 7)` | `[0, 7)` | `y3`（位置 7） |

因此三次 forward 处理的输入长度依次为 `5, 1, 1`，停止时每层缓存长度是 `7`。一般地，prompt 长度为 `P`、实际生成 `N > 0` 个 token 时，最终需要缓存 `P + N - 1` 个位置。这个式子也会用于 Lab 05 的页容量计算。`N = 0` 时直接返回，连 prefill 都无需执行。采样到 EOS 时先记入输出，再结束生成。

## 从贪心到随机采样

模型对当前前缀输出一行 logits，生成策略据此选一个 token ID。**贪心**每步取分数最高的 ID。**随机采样**按候选的概率抽取一个 ID。每次选出的 ID 都会加入前缀，影响下一次 forward 得到的 logits，所以第一处选择差异可能让后续生成走向不同分支。相同前缀、相同模型的原始 logits 与采样配置无关。

| 参数 | 作用 |
| --- | --- |
| `temperature` | `T > 0` 时用 `softmax(logits / T)` 计算采样分布。`T < 1` 让分布更集中，`T = 1` 保持原分布，`T > 1` 让分布更平。正温度下分数排序相同，因此缩放后直接 `argmax` 仍做出同样的选择。vLLM 的 `temperature=0` 直接走贪心分支。 |
| `top_k` | 按分数保留排名最高的 `k` 个候选，在保留的候选中按权重抽样。`k` 控制候选数量上限。 |
| `top_p` | 从概率最高的候选开始累加，保留累积概率首次达到 `p` 的最短前缀，把刚跨过阈值的候选也留下。重新归一化后抽样。候选数随分布变化，`top_p=1` 关闭这项截断。这个方法也叫 nucleus sampling。 |
| `min_p` | 单个候选的概率至少要达到 `min_p × 最高候选概率`。例如最高概率为 `0.40`，`min_p=0.1`，则单项门槛是 `0.04`。 |

参考 [vLLM SamplingParams](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/) 和 [Nucleus Sampling 原论文](https://arxiv.org/abs/1904.09751)。在 vLLM 的 PyTorch 参考路径中，同时设 `top_k` 和 `top_p` 时，先按 `top_k` 将其余 logits 设为 `-inf`，然后基于过滤后的分布计算 `top_p`。复现组合采样时，过滤顺序也会影响结果。[vLLM 的 PyTorch 采样代码](https://docs.vllm.ai/en/stable/api/vllm/v1/sample/ops/topk_topp_sampler/#vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p_pytorch)

### 固定 logits，观察 temperature 如何改变 top-p 的候选数

用 PyTorch 对 `A, B, C, D, E` 的 logits `[3, 2, 1, 0, -1]` 计算分布。这里只启用 `temperature` 和 `top_p=0.90`。表中的概率为截断前的概率。

| `temperature` | A | B | C | D | E | `top_p=0.90` 保留 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5 | 86.47% | 11.70% | 1.58% | 0.21% | 0.03% | A、B |
| 1.0 | 63.64% | 23.41% | 8.61% | 3.17% | 1.17% | A、B、C |
| 2.0 | 42.87% | 26.00% | 15.77% | 9.56% | 5.80% | A、B、C、D |

`T=1` 时，前两项累计约 `87.05%`，加入 C 后约 `95.67%`，所以 C 也有机会被选中。三项分别除以累计概率 `0.956659`，实际抽样权重约为 `66.52%`、`24.47%`、`9.00%`。单独使用 `top_k=2` 时，这三个温度都保留 A 和 B，但它们的抽样权重仍随温度变化。

```python
import torch

logits = torch.tensor([3., 2., 1., 0., -1.], dtype=torch.float64)
for temperature in (0.5, 1.0, 2.0):
    probs = (logits / temperature).softmax(dim=-1)
    # 本例的 logits 已按降序排列。
    count = (probs.cumsum(dim=-1) < 0.9).sum().item() + 1
    retained = probs[:count]
    weights = retained / retained.sum()
    print(temperature, probs.tolist(), count, weights.tolist())
```

另外，`presence_penalty` 根据 token 是否已出现调整分数，`frequency_penalty` 根据出现次数调整分数，`repetition_penalty` 也会处理 prompt 和此前生成中出现过的 token。`seed` 控制随机数生成器的起点。Lab 02 用贪心建立带缓存与完整重算的对拍基线，Lab 07 的实验 API 对应 `temperature=0`。[vLLM SamplingParams](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/)、[项目计划](../README.md#lab-07--流式服务与取消)

### 思考题 · 用 `argmin` 生成时如何停下来

每步在合法候选中取 `argmax` 和 `argmin`，任一选到 EOS 就结束，否则输出 `argmin`，是一个确定性的解码实验。EOS 如果一直排在中间，这个规则就不会触发，所以仍需用 `max_new_tokens` 保证有输出上限。按本 lab 的约定，可把触发停止的 EOS 计入输出一次。若 logits 已用 `-inf` 标记非法候选，求最小值前要把这些候选排除，否则 `argmin` 会优先选到它们。

## 历史 K/V 为什么能复用

对当前 query 位置 `i`，每层会用该位置的 `q_i` 与位置 `0..i` 的 K 做点积，再用 softmax 得到的权重对同一范围的 V 加权。K 决定从当前位置看过去，每个可见位置获得多少权重，V 提供按这些权重汇总的向量。后续 query 还能读取同一组历史 K/V。旧 Q 已经用于计算旧位置的 attention 输出，后续步骤只需要新输入位置的 Q。随着位置推进，参与计算的 Q 来自不同 token 位置。固定的历史位置在相同前缀下重新计算时，Q 也在数学上保持相同。

Lab 02 将缓存每一层 `TinyQwen3Attention.forward` 传给 `causal_gqa` 的 K/V。它们是本次输入计算出的激活。具体顺序如下。

```text
x = decoder.pre_attn_layer_norm(hidden_states)
K: w_k(x) → 拆成 KV heads → k_rms_norm → 按绝对位置做 RoPE → 写入
V: w_v(x) → 拆成 KV heads → 写入
```

保存 `Hkv` 个 KV head，GQA 按需展开到 `Hq` 个 head。连续缓存可以分别为 K、V 预分配 `[L, capacity, Hkv, D]`，每次按一层读写 `[T, Hkv, D]`。本实验 toy 模型有 2 层、2 个 KV head、每个 head 8 个元素，FP32 下每个位置的逻辑 KV 数据为 `2 × 2 × 2 × 8 × 4 = 256` 字节。实际预分配字节数取决于 `capacity`。

### 因果掩码先保证预测成立，再让历史可复用

以训练输入 `[A, B, C]` 为例，位置 `A` 的目标是预测 `B`。如果 `A` 能通过 attention 读取 `B`，模型就在预测前拿到了答案，而且推理时只给 `[A]` 就无法提供同样的上下文。因果掩码把每个位置的依赖限制为自己和更早的位置，让已知序列上的并行训练与逐步生成遵循同一条条件概率规则。

它也使历史表示在追加 token 后保持稳定。第一个 Decoder 的旧位置输入来自同样的 embedding，因果掩码让旧 query 仍然读取同样的前缀，其中 Q/K 的 RoPE 位置也相同。残差、RMSNorm 和 MLP 在各 token 位置独立处理，所以第一层的旧位置输出保持相同。将这一点逐层推下去，后续各层旧位置的 K/V 也保持相同。这样可以省掉历史位置的投影、attention 和 MLP，每一轮仍要读取历史 K/V 来计算新位置的 attention。

临时放开掩码时，追加 `B` 不会直接改变第 0 层输入端 `A` 的 K/V 投影，因为该层仍从同一个 `A` 的 embedding 计算。但 `A` 的第 0 层 attention 能读取 `B`，层输出就可能变化。第 1 层接到变化后的 `A` 表示，再计算出的旧 K/V 也就可能变化。缓存失效是跨层依赖变化的结果。

## `start_pos`、RoPE 和 mask 怎么对齐

`start_pos` 是本次第一个输入 token 的绝对位置。设本次输入长度为 `T`，先用 `[start_pos, start_pos + T)` 作为 Q/K 的 RoPE 位置，把新 K/V 追加到对应层，再读取 `[0, start_pos + T)` 作为本层 attention 的 K/V。当前 Q 保持长度 `T`，K/V 使用整个已处理前缀。接口的形状转换如下。

```text
当前 Q   [1, Hq, T, D]
当前 K/V [1, Hkv, T, D] → [T, Hkv, D]                 写入
全部 K/V [start_pos + T, Hkv, D] → [1, Hkv, start_pos + T, D]  读回
```

例如已有位置 `0, 1, 2`，当前 chunk 的两个 query 在位置 `3, 4`。按 `key_position <= query_position` 得到下面的可见矩阵。`1` 代表可见。

| Query \ Key | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 3 | 1 | 1 | 1 | 1 | 0 |
| 4 | 1 | 1 | 1 | 1 | 1 |

把当前 chunk 的 Q 当成位置 `0, 1` 会让它们漏读大部分历史。Lab 01 的 `causal_gqa` 已经实现了 `query_start` 偏移，接入时让它与 RoPE 使用同一个 `start_pos`。参见 [Lab 01 的 Causal GQA](lab01_model.md#causal-gqa)。

## 动手顺序

1. **`ContiguousKV`。** 各层独立记录已写长度。同一轮里第 0 层写完新位置后，第 1 层可能还没写，同一个全局长度会混淆这两个状态。`write` 先检查层号、形状、`start_pos == length[layer]` 和容量，再修改 K/V 和长度。`read` 只允许读取该层已写过的前缀。先用已有测试验证失败的写入之后还能继续 append。
2. **模型接入。** 从 `TinyQwen3.forward` 将层号、cache 和 `start_pos` 传到 attention。用已有的 `(4, 1, 1)` 和 `(1, 2, 3)` 两种分段方式拼回 logits，与整段一次 forward 比较。逐位置 logits 对拍能发现被 `argmax` 掩盖的数值差异。
3. **`greedy_generate`。** 先用整段重算建立基线，再按上面的时序走带缓存的分支，最后检查两条路径生成的 ID 一致，并覆盖 EOS 和零预算。

第一步的独立验收命令如下。

```bash
.venv/bin/python -m pytest -q tests/test_lab02_cache.py -k contiguous_cache
```

## 当前实验记录

环境为 Mac arm64 CPU、Python 3.12.4、PyTorch 2.15.0。Lab 01 与 Lab 02 合跑为 `7 passed, 5 failed, 1 skipped`。其中 Lab 01 为 `6 passed`，Lab 02 的绝对位置掩码测试已通过，剩余 5 个 toy 用例触发缓存或生成函数的 `NotImplementedError`。真权重用例因未设置 `QWEN_MODEL_DIR` 跳过。Lab 06 的按行 RoPE 测试额外跑过，`1 passed`。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_lab01_model.py tests/test_lab02_cache.py
```

用 `tests.toy.make_model()` 对 prompt `[10, 8, 1, 6, 42]` 做三轮完整前缀重算，可观察到 logits 形状依次为 `[1, 5, 43]`、`[1, 6, 43]`、`[1, 7, 43]`，每次贪心采样都选出 ID `42`。相同的 ID 在不同位置仍是序列里的不同 token，分别按自己的绝对位置处理。下面的脚本可复现这条无缓存基线。

```python
import torch

from tests.toy import ids, make_model

model = make_model()
prefix = [10, 8, 1, 6, 42]
with torch.inference_mode():
    for _ in range(3):
        logits = model(ids(prefix))
        next_id = logits[0, -1].argmax().item()
        print(prefix, tuple(logits.shape), next_id)
        prefix.append(next_id)
```

用 `test_gqa_uses_absolute_query_positions_for_chunk` 的固定输入做对照，取绝对位置 `3, 4` 的 Q。传 `query_start=3` 时与完整 GQA 截取结果的最大绝对误差是 `0`，传 `0` 时是约 `2.47966`。若接入缓存后分段对拍失败，可以先核对 `start_pos`、两条时间轴及读出的 KV 长度。

另外用同一 toy 模型比较输入 `[10]` 与 `[10, 8]` 时旧位置 `0` 的 K/V。实验在进程内包装 `causal_gqa`，记录传入的 K/V。在放开掩码的对照组里，只把传给掩码计算的 `query_start` 设为 `k.shape[-2] - 1`，使所有 key 可见，Q/K 的 RoPE 仍使用实际位置。下表记录最大绝对误差。

| Attention 可见范围 | 第 0 层 K | 第 0 层 V | 第 1 层 K | 第 1 层 V |
| --- | ---: | ---: | ---: | ---: |
| 正常因果掩码 | `1.19e-7` | `8.94e-8` | `4.77e-7` | `8.94e-8` |
| 全部位置可见 | `1.19e-7` | `8.94e-8` | `3.21` | `0.95` |

因果组的差异处于 FP32 舍入量级。全可见组中，第 0 层输入端的旧 K/V 仍一致，从第 1 层开始能观察到旧 K/V 的变化。
