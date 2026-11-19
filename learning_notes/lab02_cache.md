# Lab 02 · 单请求 Prefill / Decode

**状态：进行中。连续缓存与绝对位置 causal mask 已通过，开始接入模型 forward。**

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

## 将 KV cache 接入 TinyQwen3 Forward

本步的目标是让同一段输入按不同方式切分后，各位置的 logits 与一次性完整 forward 对齐。输入仍然是一段 token ID，输出仍然是本段每个位置的 logits。带缓存路径中的 `B=1` 表示一份缓存对应一个请求。无缓存整段路径继续作为数值基线。

### 从入口把状态传到 Attention

- `TinyQwen3.forward` 已经接收 `cache` 和 `start_pos`。遍历 decoder 时需要知道当前 `layer_idx`。一轮 forward 的所有层处理相同的 token 位置，共享 `start_pos`，各层通过自己的 `layer_idx` 读写缓存。
- `TinyQwen3Decoder.forward` 把这些信息传给 attention。这里继续按原有顺序执行 Pre-Norm、attention 残差、MLP 残差。
- `TinyQwen3Attention.forward` 在 K 的 RMSNorm 与 RoPE 之后、`causal_gqa` 之前接入缓存。先把当前段的 K/V 写到当前层，再读取包括当前段在内的有效前缀。读出的前缀已经含有当前段，可直接用于 GQA。

### 手推最后一段 `[3, 6)`

测试把 6 个输入切成 `(1, 2, 3)` 时，第三段开始前每层已经缓存位置 `0, 1, 2`。位置 `3, 4, 5` 是本次输入。设 `S = start_pos = 3`，`T = 3`，`end_pos = S + T = 6`。Toy 模型有 `Hq=4`、`Hkv=2`、`D=8`、词表大小 `43`。

| 操作 | 序列位置 | 张量形状 |
| --- | --- | --- |
| 当前 Q，使用绝对位置做 RoPE | `[3, 6)` | `[1, 4, 3, 8]` |
| 当前 K/V，转换后写入当前层 | `[3, 6)` | `[1, 2, 3, 8] → [3, 2, 8]` |
| 读取当前层的完整 K/V 前缀 | `[0, 6)` | `[6, 2, 8] → [1, 2, 6, 8]` |
| GQA 的分数矩阵 | 每个当前 query 对全部 6 个 key | `[1, 4, 3, 6]` |
| 本次 forward 的 logits | `[3, 6)` | `[1, 3, 43]` |

缓存写入时，明确去掉 batch 轴，再交换 head 与 token 轴。注意 `T=1` 时，无参数的 `squeeze()` 还会把 token 轴去掉。换轴应使用 `transpose` 或 `permute`，`reshape` 只改变形状，无法表达这两个轴的交换。读回后交换 token 与 head 轴，再补回 batch 轴，交给原有 `causal_gqa` 做 head 展开。缓存里的历史 K 已经按各自的绝对位置旋转过。

这次三个 query 在绝对位置 `3, 4, 5`，分别能读取 key `0..3`、`0..4`、`0..5`。尽管读到了整段前缀，causal mask 会屏蔽当前段内在 query 之后的位置。先写再读让 prefill 和单 token decode 用同一路径，也保证当前 query 能看见自己。`start_pos` 同时传给 RoPE 和 `causal_gqa` 的 `query_start`。每层只为当前 `T` 个位置做投影和 MLP，历史 K/V 仍会被读取来计算新 query 的 attention。

### 验收与定位偏差

```bash
.venv/bin/python -m pytest -q tests/test_lab02_cache.py -k cached_logits
.venv/bin/python -m pytest -q tests/test_lab01_model.py
```

- `(4, 1, 1)` 覆盖单 token decode 形状以及 `T=1` 时的轴变换。每轮结束时，两层的有效长度应分别达到 `[4, 4]`、`[5, 5]`、`[6, 6]`。
- `(1, 2, 3)` 覆盖非零起点的多 token chunk，验证当前 chunk 内的未来位置仍被遮住。三轮结束长度应分别达到 `[1, 1]`、`[3, 3]`、`[6, 6]`。
- 测试沿 token 轴拼接本次的 logits，再与无缓存整段结果比较，FP32 的 `atol=rtol=2e-5`。如果第一块过、后续块偏差大，先查每层的游标、读回的 `Tkv`、RoPE 绝对位置和 mask 的 `query_start`。

接线前实测，两种切分方式的第一段最大绝对误差都约为 `2e-7`，后续各段约为 `1.02–1.16`，每一轮后缓存长度都还是 `[0, 0]`。当前 `TinyQwen3.forward` 收到了 cache 参数，但 attention 只在当前段的 K/V 上计算。两组对拍转绿后，再实现选 token 和停止条件。

## 连续缓存代码复盘

第一版采用 `[L, capacity, Hkv, D]` 预分配，按层切片，再用 `copy_` 写入。这些设计可以沿用。出错的是有效数据边界和切片终点。

| 第一版的问题 | 原因 | 本轮实现的修正 |
| --- | --- | --- |
| `read(1, 2)` 读到尚未写过的位置 | `torch.empty` 分配了每层的全部容量。第 1 层即使只写了 1 项，`[:2]` 也能切出 2 项，读取方必须检查有效长度。 | `write_length[layer]` 独立记录每层的有效前缀，读取时校验 `length <= write_length[layer]`。 |
| 从非零位置追加时切错范围 | `[start_pos:token_length]` 把新写入数量当成了绝对终点。起点为 `2`、新增 1 项时会得到 `[2:1]`。 | 用 `[start_pos:start_pos + token_length]`，此例写入 `[2:3]`。检查 `start_pos == write_length[layer]` 同时挡住覆盖和跳写。 |
| 原校验只比了 K/V 的 token 数 | K/V 每个轴都需要吻合存储约定。`copy_` 支持广播，形状不符时有时会静默复制。 | 先核对三维形状、head 数和 head dim，再校验容量。完成 K 和 V 的复制后才推进游标，报错时已提交的前缀仍有效。 |

[PyTorch `Tensor.copy_`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.copy_.html)

### 接口与空区间的复查

- **`Protocol` 的职责。** `KVStore` 现在声明 `read`、`write`，`ContiguousKV` 持有实际张量和游标。可以用 Java 的 `interface` 建立直觉，再记住 Python 此处使用结构化类型：方法签名匹配即可满足 `Protocol`，即使类没有显式继承它。Lab 05 的 `PagedKV` 可以沿用读写接口和自己的分页布局。[Python typing 规范](https://typing.python.org/en/latest/spec/protocol.html#protocol-members)
- **空读与空写是两条约定。** 当前构造器明确拒绝 `num_layers=0` 和 `capacity=0`，读写也都要求非空。用 toy cache 实测，`read(layer, 0)` 在写入前后都抛出 `ValueError`，正常的非空读仍返回期望的数据。接入模型时采用先写当前 chunk、再读完整前缀的顺序，因此首轮 prefill 的读长度也是正数。

### `eval()` 与 `inference_mode()` 放在哪里

`model.eval()` 切换模型及其子模块的 `training` 标志，影响 Dropout、BatchNorm 等依赖该标志的层。`torch.inference_mode()` 控制当前线程在代码块或被装饰的函数中如何使用 autograd，退出后会恢复外层状态。`torch.no_grad()` 也会在作用域内停止记录反向图；如果需要把这段代码新建的张量交给后续有梯度的计算，`no_grad` 更宽松。[PyTorch Autograd](https://docs.pytorch.org/docs/2.15/notes/autograd.html#locally-disable-grad-doc)、[PyTorch inference mode](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.grad_mode.inference_mode.html)

用已有 toy 模型实测，调用 `.eval()` 后在默认梯度模式下，`model.training` 为 `False`，但 `logits.requires_grad` 仍为 `True`。把同一次 forward 放进 `inference_mode()`，`logits.requires_grad` 变为 `False`，退出后 `torch.is_grad_enabled()` 恢复为 `True`。同理，`torch.empty(..., requires_grad=False)` 只描述缓存张量的初始状态，默认梯度模式下，`copy_` 仍可能随来源 K/V 被记录。

本项目的 `hf_io.load_model()` 和 `tests.toy.make_model()` 都在交出模型前调用 `.eval()`。实现 `greedy_generate` 时，可以让模型加载方负责设置 eval，生成入口用 `@torch.inference_mode()` 或对应的 `with` 作用域覆盖整个 prefill/decode 循环，包括 forward 里的 K/V 写入。

## 当前实验记录

环境为 Mac arm64 CPU、Python 3.12.4、PyTorch 2.15.0。复查空读和构造器的零值校验后，连续缓存与绝对位置掩码用例通过，两组分段 logits 对拍失败，合计 `2 passed, 2 failed, 3 deselected`。此前 Lab 01 的 6 项用例均通过，贪心生成尚待实现，真权重用例因未设置 `QWEN_MODEL_DIR` 跳过。Lab 06 的按行 RoPE 测试此前单独验证过，`1 passed`。

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
