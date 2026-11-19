# Lab 01 · Qwen3 Forward

**状态：已完成。验收：`6 passed`。**

## 本周总结

本 lab 从基础张量算子出发，实现了一个能加载 HF 风格权重、完成 Qwen3 forward 并输出 logits 的双层 toy 模型：

```text
token IDs
→ Embedding
→ L × [RMSNorm → QKV → QK Norm → RoPE → Causal GQA → O Proj → 残差
       RMSNorm → SwiGLU MLP → 残差]
→ 最终 RMSNorm
→ LM Head
→ logits [B,T,V]
```

### 已掌握的核心机制

- **RMSNorm**：在最后一维按均方根缩放；半精度输入使用 FP32 中间计算。
- **RoPE**：按绝对位置旋转 Q/K，Q/K 点积自然获得相对位置信息；不处理 V。
- **Causal GQA**：多个 Q head 共享 K/V head，以缩放 Q/K 点积和 causal mask 计算注意力。
- **Qwen3 Decoder**：两条 Pre-Norm 残差路径，QK Norm 位于 RoPE 前，MLP 使用 SwiGLU。
- **权重映射**：HF Linear 权重无需转置；所有本实验投影无 bias；tied LM head 共享 embedding 参数。
- **数值对拍**：不仅检查测试通过，还要能解释张量形状、dtype、device 和掩码语义。

### 本周最值得复盘的错误

| 问题 | 根因 | 形成的检查习惯 |
| --- | --- | --- |
| RMSNorm 数值过大 | 对 `rsqrt` 的结果又做了除法 | 先写出中间变量的数学含义，再选择乘除。 |
| RoPE 角度错误 | 混淆 token 轴与频率轴，遗漏 position，误用原地 `cos_` | 写出每个张量的 shape 和逐元素公式。 |
| GQA 出现 NaN | 把 `allowed` 直接传给 `masked_fill` | 区分“允许集合”和“需要屏蔽的集合”。 |
| 整模型无法匹配 | Norm 类型、Linear bias、投影宽度和 LM head 等结构问题 | 加载权重前先核对模块结构及参数表。 |

### 与后续 lab 的连接

- Lab 02 将复用当前 forward，加入连续 KV cache、非零 `start_pos` 和贪心生成。
- 当前已经提前支持 RoPE 的 `[B,T]` positions，Lab 06 会用它处理不同历史长度的 batched decode。
- 当前 GQA 为教学实现，会展开 K/V heads；后续真实引擎重点优化的是存储、调度和批处理，而不是改变这里的数学语义。

## 验收与复习

```bash
python -m pytest -q tests/test_lab01_model.py
python -m pytest -q tests/test_lab01_model.py tests/test_lab06_batch.py -k rope
```

复习时应能脱离代码回答：

1. 为什么 RMSNorm 要在半精度输入下使用 FP32 累加？
2. 为什么 RoPE 对 Q/K 的绝对旋转会在点积中变成相对位置？
3. `positions` 与 `inv_freq` 分别对应哪个轴？
4. GQA 如何减少 KV cache，为什么不缓存 Q？
5. `query_start` 如何同时影响 RoPE 和 causal mask？
6. 为什么 `Hq × head_dim` 不一定等于 `hidden_size`？
7. 为什么 tied embedding 的权重可以同时服务输入 embedding 和输出 LM head？

---

## 浮点数格式与 PyTorch dtype

FP16、BF16、FP32 和 FP64 都是浮点数表示格式。浮点数的位数主要分配给符号、指数和尾数：

- 指数位数主要影响可表示数值的范围。
- 尾数位数主要影响精度。

| PyTorch 类型 | 常见名称 | 总位数 | 指数位 | 存储尾数位 | 主要特点 |
| --- | --- | ---: | ---: | ---: | --- |
| `torch.float16` | FP16 / Half | 16 | 5 | 10 | 精度相对较好，但范围较小。 |
| `torch.bfloat16` | BF16 | 16 | 8 | 7 | 范围接近 FP32，但精度较粗。 |
| `torch.float32` | FP32 / Float | 32 | 8 | 23 | 精度、范围和开销较均衡。 |
| `torch.float64` | FP64 / Double | 64 | 11 | 52 | 精度高，常用于科学计算和数值验证。 |

表中的存储尾数位不包括隐含的最高位。

### FP16 与 BF16

- FP16 最大有限值约为 `65504`，超过后可能变成 `inf`。
- BF16 使用与 FP32 相同的 8 位指数，因此范围大、不易溢出，但有效精度低于 FP16。
- 两者都是 16 位，都能减少模型权重、激活和 KV cache 的显存占用与带宽压力。
- 训练和推理常使用 BF16 的重要原因之一，是它对较大的动态范围更宽容。

### 常见 dtype 转换

```python
x.float()       # 转为 torch.float32，不是 float64
x.half()        # 转为 torch.float16
x.bfloat16()    # 转为 torch.bfloat16
x.double()      # 转为 torch.float64

x.to(torch.float32)
x.to(dtype=torch.float32)  # 与 x.float() 的 dtype 转换等价
```

低精度模型计算中常使用 FP32 做累加或敏感算子的中间计算，最后再将结果转回原始 dtype。低精度存储节省显存，FP32 中间计算降低溢出和累计误差风险。

## PyTorch 基础张量运算回顾

### 逐元素运算与矩阵运算

```python
x * y           # 逐元素乘法，支持广播
torch.mul(x, y) # 逐元素乘法
x / y           # 逐元素除法
torch.div(x, y) # 逐元素除法
x @ y           # 矩阵乘法
torch.matmul(x, y)
```

`*` 和 `@` 不能混淆：前者要求形状可广播，后者遵循矩阵或批量矩阵乘法规则。

### 平方、平方根与倒数平方根

```python
x.square()      # 每个元素的平方：x^2
x.pow(2)        # 同样是 x^2
torch.sqrt(x)   # 平方根：sqrt(x)
torch.rsqrt(x)  # 倒数平方根：1 / sqrt(x)
```

`square` 和 `rsqrt` 不是一对互相替代的操作：

- `x.square()` 把数值平方。
- `x.sqrt()` 对数值开平方。
- `x.rsqrt()` 先开平方再取倒数。

以下两行等价：

```python
x * torch.rsqrt(mean_square + eps)
x / torch.sqrt(mean_square + eps)
```

但如果已经计算了 `inv_rms = torch.rsqrt(...)`，接下来应写 `x * inv_rms`，不能再写 `x / inv_rms`。

### 均值和维度

```python
x.mean(dim=-1, keepdim=True)
```

- `dim=-1`：沿最后一维归约。对形状 `[B, T, D]` 的张量，结果保留为 `[B, T, 1]`。
- `keepdim=True`：保留被归约的维度，便于和原张量广播。
- 若省略 `keepdim=True`，结果会变为 `[B, T]`，与 `[B, T, D]` 做除法时可能无法按预期广播。

### 广播

形状 `[B, T, D]` 的张量乘以形状 `[D]` 的权重时，权重会自动对齐最后一维并应用到所有批次和位置。RMSNorm 的可学习缩放正是利用这一机制。

## RMSNorm

### 数学定义

对最后一维计算：

\[
\operatorname{RMSNorm}(x)=
\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}\times weight
\]

与使用均值和方差的 LayerNorm 不同，RMSNorm 不做均值居中，只按均方根缩放。

### 当前实现的关键点

```python
original_dtype = x.dtype
x_fp32 = x.float()

rms = x_fp32.pow(2).mean(dim=-1, keepdim=True)
normalized = x_fp32 * torch.rsqrt(rms + eps)

return (normalized * weight).to(dtype=original_dtype)
```

- 对最后一维求平方均值。
- FP16、BF16 输入使用 FP32 中间计算。
- `eps` 防止零输入时分母为零，并提高数值稳定性。
- 乘上可学习的 `weight` 后恢复输入 dtype。

### 本次易错点

`torch.rsqrt` 已经得到归一化所需的倒数平方根，因此必须乘回 `x`，而不是除以它。变量命名使用 `rms`、`normalized` 等语义化名称，可以避免把中间量误认为其他含义。

验收命令：

```bash
python -m pytest -q tests/test_lab01_model.py -k rmsnorm
```

当前结果：`3 passed, 3 deselected`，分别覆盖 FP32、FP16 和 BF16。

## 从 Sinusoidal Encoding 到 RoPE

### 为什么需要位置编码

不含位置信息的 self-attention 对输入顺序具有置换等变性：它知道有哪些 token，却不知道 token 的排列顺序。位置编码的任务是把顺序信息注入模型。

### 正弦位置编码为什么能表示位置平移

把一个频率对应的位置编码写成：

\[
PE_i(p)=[\sin(\omega_i p),\cos(\omega_i p)]
\]

希望位置从 \(p\) 移到 \(p+k\) 时，可以写成：

\[
PE_i(p+k)=M_k PE_i(p)
\]

其中 \(M_k\) 只依赖偏移量 \(k\)，不依赖绝对位置 \(p\)。利用正弦、余弦和角公式：

\[
\sin(\omega(p+k))=\sin(\omega p)\cos(\omega k)+\cos(\omega p)\sin(\omega k)
\]

\[
\cos(\omega(p+k))=\cos(\omega p)\cos(\omega k)-\sin(\omega p)\sin(\omega k)
\]

得到：

\[
M_k=
\begin{bmatrix}
\cos(\omega k)&\sin(\omega k)\\
-\sin(\omega k)&\cos(\omega k)
\end{bmatrix}
\]

这就是一个旋转矩阵。原始 Transformer 运行时不一定真的计算这个矩阵；这里的推导是在证明正弦位置编码具有线性、相对的位置平移结构。

### RoPE 的关键变化

- Sinusoidal Encoding：生成独立位置向量，加到 token embedding 上。
- RoPE：不额外添加位置向量，而是按 token 位置旋转投影后的 Q/K。
- 对每个二维分量，\(q'_m=R(m\omega)q_m\)，\(k'_n=R(n\omega)k_n\)。
- V 不需要旋转；位置信息通过 attention logits 起作用。

旋转矩阵满足 \(R(a)^T=R(-a)\) 和 \(R(a)R(b)=R(a+b)\)，所以：

\[
(R(m\omega)q)^T(R(n\omega)k)
=q^TR((n-m)\omega)k
\]

attention 点积因此只依赖内容和相对位置 \(n-m\)，不再分别依赖绝对位置 \(m,n\)。RoPE 是用绝对旋转得到相对位置点积。

旋转还满足 \(R^TR=I\)，所以它保持向量长度，不会像直接加上一个很大的位置标量那样改变激活尺度。

### 正弦、余弦和“相近位置”的正确直觉

\[
[\cos(a),\sin(a)]\cdot[\cos(b),\sin(b)]
=\cos(a-b)
\]

当 \(\Delta=a-b\) 很小时，\(\cos\Delta\approx1-\Delta^2/2\)，相位接近时余弦接近 1。但余弦会周期振荡，并不随距离单调下降。因此 RoPE 并不保证相近 token 一定获得更高注意力，也不存在“越远余弦必然越小”的规则。

正弦和余弦共同表示二维相位。只用其中一个无法完整表达旋转，也无法形成上述平移和点积恒等式。多组频率提供不同的时间尺度：高频对局部位置变化敏感，低频可以表达较长距离。

### 阅读推导时需要保留的边界

- 可以计算任意位置不等于能在比训练更长的序列上稳定泛化。
- 周期性意味着不能把单个正弦值当作无限范围内严格唯一的位置编号。
- “加法会污染语义”是帮助理解 RoPE 动机的说法，不表示加法位置编码无效。
- 文章是直觉式重构；RoFormer 原论文直接以“Q/K 内积只依赖内容和相对位置”为约束进行推导。

## RoPE 的实现形状与角度映射

### 数学矩阵与实际实现

对于单个 token，RoPE 在数学上可以表示为 \(D\times D\) 的分块对角旋转矩阵，每个 \(2\times2\) 小块旋转一对维度。但矩阵高度稀疏，实际实现不构造它，而是直接做逐元素乘法和加减，将每 token 的计算量从 \(O(D^2)\) 降为 \(O(D)\)。

RoPE 没有可学习参数。对于 `[B,T,D]` 的 Q/K，只需为每个位置计算 `[T,D/2]` 的角度、正弦和余弦，并广播到 batch 维。进入多层 Decoder 后，每层都有自己投影得到的 Q/K，因此每层都要应用 RoPE；cos/sin 表可以复用。使用 KV cache 时，历史 K 已经旋转，decode 只处理新 token，V 始终不旋转。

### split-half 配对

若 `D=6`，split-half 将：

```text
[x0, x1, x2, x3, x4, x5]
```

分为 `[x0,x1,x2]` 和 `[x3,x4,x5]`，配对为 `(x0,x3)、(x1,x4)、(x2,x5)`。它与博客常见的相邻配对 `(x0,x1)、(x2,x3)` 数学等价，只是维度布局不同。

### 位置和频率是两个不同轴

设 `D` 为 head dim、`half=D/2`：

- 频率对下标 `i=0,...,half-1`，沿最后一维分布。
- token 位置 `p` 来自 `positions`，沿时间维 `T` 分布。

\[
inv\_freq_i=\theta^{-2i/D}
\]

\[
angle_{p,i}=positions_p\times inv\_freq_i
\]

`positions[:, None] * inv_freq[None, :]` 是外积式的广播，得到 `[T,half]`：每一行是一个 token 位置，每一列是一种旋转频率。不能用序列长度 `T` 代替 `D/2` 来构造频率。

### 本次实现中的易错点

- `torch.arange(0,t)` 构造的是序列下标，不是频率对下标；应使用 `torch.arange(half)`。
- `//` 是整除，频率指数需要浮点除法 `/`。
- `inv_freq` 只是频率，还必须乘 `positions` 才是旋转角度。
- `cos/sin` 需要形状 `[T,half]`，才能同时对应时间维和 split-half 维。
- `torch.cos_` 末尾下划线表示原地修改；若随后对同一 tensor 调用 `sin`，会变成 `sin(cos(angle))`。这里应使用非原地的 `torch.cos` 和 `torch.sin`。

Lab 01 和 Lab 06 的 RoPE 测试均已通过，已支持共享 `[T]` 位置和逐行 `[B,T]` 位置。CPU toy test 通过不等于 GPU 路径也已验证：后续在 CUDA 上运行时，频率下标、`positions` 和角度表应与 `x` 位于同一 device，避免 CPU/CUDA 张量混合运算。

### `unsqueeze`、`expand` 与自动广播

- `x.unsqueeze(dim)`：在指定位置增加长度为 1 的维度，不增加元素。`x[:,None]` 是常用简写。
- `x.expand(...)`：将长度为 1 的维度扩展到目标大小，返回共享底层数据的 view，通常不复制数据；`-1` 表示保持该维原大小。
- `x.expand_as(y)`：扩展为 `y.shape`。
- `x.repeat(...)`：按次数复制数据，会产生新的内存开销。
- `x.reshape(...)` / `x.view(...)`：重排已有形状，不把少量元素扩展成更多元素。

`expand` 不能把原本非 1 的维度直接改成其他大小。例如 `[2,3]` 不能直接 expand 为 `[4,3]`，需要先在最前面增加维度得到 `[1,2,3]`，再 expand 为 `[4,2,3]`。扩展产生的 view 不适合进行会互相覆盖的原地写入。

Lab 06 的 RoPE 可以统一两种 positions：

```python
# 共享位置：[T] -> [1,T] -> [B,T]
if positions_fp32.ndim == 1:
    positions_fp32 = positions_fp32.unsqueeze(0).expand(x.shape[0], -1)

# [B,T] × [D/2] -> [B,T,D/2] -> [B,1,T,D/2]
angles = positions_fp32.unsqueeze(-1) * inv_freq
cos = angles.unsqueeze(1).cos()
```

这样 cos/sin 可广播到 `[B,H,T,D/2]`，无需真的复制 H 份数据。

## Causal GQA

### 基本形状与计算

```text
Q:       [B,Hq,Tq,D]
K/V:     [B,Hkv,Tk,D]
scores:  [B,Hq,Tq,Tk]
output:  [B,Hq,Tq,D]
```

\[
scores=\frac{QK^T}{\sqrt D},\qquad
output=softmax(scores+mask)V
\]

除以 \(\sqrt D\) 防止 D 较大时点积数值过大，导致 softmax 过度饱和。`-2,-1` 转置只交换最后两维，不动 batch/head 维。

### GQA 的 head 映射

- MHA：`Hq=Hkv`，每个 Q head 有自己的 K/V head。
- GQA：多个 Q head 共享一个 K/V head。
- MQA：`Hkv=1`，所有 Q head 共享同一组 K/V。

设 `groups=Hq/Hkv`，则：

```python
k = k.repeat_interleave(groups, dim=1)
v = v.repeat_interleave(groups, dim=1)
```

例如 `[k0,k1]` 在 groups=2 时变为 `[k0,k0,k1,k1]`。这与重复整个序列的 `[k0,k1,k0,k1]` 不同。GQA 主要减少 K/V 投影和 KV cache 容量，Q 不共享也不写入 KV cache。

### 带历史缓存的 causal mask

query 的绝对位置为 `query_start + arange(Tq)`，key 的位置为 `arange(Tk)`，仅允许：

```text
key_position <= query_position
```

mask 形状为 `[Tq,Tk]`，可广播到 `[B,Hq,Tq,Tk]`。在 softmax 前把不允许的位置填为 `-inf`，不是等到乘 V 后才屏蔽。

- 普通 prefill：`query_start=0, Tq=Tk`，形成下三角 mask。
- 单 token decode：新 query 是最后一个位置，可以看见全部历史及自身。
- 分段 prefill：当前 chunk 能看见全部历史，但 chunk 内仍保持因果顺序。

RoPE 与 causal mask 的职责不同：RoPE 编码相对位置，causal mask 决定可见范围。

### `masked_fill` 的 True/False 语义

`scores.masked_fill(condition, -inf)` 会把 `condition == True` 的位置填成 `-inf`。若布尔矩阵叫 `allowed`，表示允许访问，则必须写：

```python
scores = scores.masked_fill(~allowed, float("-inf"))
```

误传 `allowed` 会反过来屏蔽历史、保留未来；当某一行全部被填成 `-inf` 时，softmax 会产生 `NaN`。此外，`query_positions` 和 `kv_positions` 应创建在 `q.device` 上，以支持 CUDA。

## TinyQwen3 的权重映射与 Forward

### 投影形状

设模型宽度为 `D_model`、Q head 数为 `Hq`、KV head 数为 `Hkv`、head dim 为 `D`：

```text
embedding:       [V,D_model]
q_proj.weight:   [Hq*D,D_model]
k_proj.weight:   [Hkv*D,D_model]
v_proj.weight:   [Hkv*D,D_model]
o_proj.weight:   [D_model,Hq*D]
gate/up.weight:  [I,D_model]
down.weight:     [D_model,I]
q/k_norm.weight: [D]
```

Q 内部宽度 `Hq*D` 可以大于 `D_model`，不能假设二者相等。PyTorch 与 HF 的 Linear 权重都是 `[out_features,in_features]`，`F.linear(x,w)` 内部使用 `w.T`，加载时不要再手动转置。

### 每层的 Pre-Norm 与残差

```text
h
├────────────────────────────── residual ─────────────────────────────┐
└→ input_layernorm → QKV → QK Norm → RoPE → GQA → o_proj → attention_out ┤
                                                                       ↓
                                                              h = h + attention_out

h
├──────────────────── residual ────────────────────┐
└→ post_attention_layernorm → SwiGLU MLP → mlp_out  ┤
                                                    ↓
                                           h = h + mlp_out
```

残差跨越整个 attention 或 MLP 子层，RMSNorm 只作用于子层输入。Qwen3 的 q_norm/k_norm 在拆分 heads 后作用于最后的 head-dim 维，并在 RoPE 之前执行；V 不做 QK Norm 或 RoPE。

### Q/K/V 与 head 布局转换

```text
线性投影: [B,T,H*D]
reshape:  [B,T,H,D]
transpose:[B,H,T,D]       ← apply_rope / causal_gqa 使用

attention 输出:
[B,H,T,D] → transpose → [B,T,H,D] → reshape → [B,T,H*D]
```

transpose 后内存可能不连续，合并 heads 时使用 `reshape` 比依赖 `view` 更稳妥。

### SwiGLU

\[
MLP(x)=W_{down}(SiLU(W_{gate}x)\odot W_{up}x)
\]

gate 和 up 接收同一个归一化输入，逐元素相乘后由 down 投影回模型宽度。

### Embedding、最终 Norm 与 Logits

所有 Decoder Layer 完成后，还要执行 `model.norm`，再投影到 vocab 维：

```text
input_ids [B,T] → hidden [B,T,D_model] → logits [B,T,V]
```

logits 是未经过 softmax 的原始分数。若 `tie_word_embeddings=True`，LM head 复用 embedding 权重，权重文件可能不含 `lm_head.weight`。使用 `nn.ModuleList` 注册各层，保证 `.parameters()` 和 `.to(device,dtype)` 能遍历它们。

### 权重加载前的结构检查

- Qwen3 使用的是 RMSNorm，不能用带 mean 居中和 bias 的 `nn.LayerNorm` 替代。
- Qwen3 本实验覆盖的 attention/MLP 投影均无 bias，创建 `nn.Linear` 时应显式 `bias=False`，否则默认 bias 会留下未加载的随机参数。
- `o_proj` 的输入宽度是 `Hq*D`，不是 `Hq`。
- `apply_rope` 的第二个参数是位置 ID，不是 QK Norm 权重。位置由 `arange(start_pos,start_pos+T)` 构造。
- 所有 Decoder Layer 后还需要最终 RMSNorm 和 LM head，forward 返回 `[B,T,V]` logits，不是 hidden states。

`from_state_dict` 推荐先创建模块，再在 `torch.no_grad()` 下用 `Parameter.copy_` 加载。复制前检查 key 是否存在及形状是否一致，避免缺失权重被随机初始化掩盖。源权重到目标参数的复制可以完成 dtype 转换；后续 `model.to(device,dtype)` 依赖参数已正确注册。

tied LM head 与 embedding 应共享同一个 Parameter，只加载 `model.embed_tokens.weight`，不能因为没有独立 `lm_head.weight` 而报错；非 tied 时才要求并加载该 key。

当前 Lab 01 全部 `6 passed`。权重映射、无 bias 投影、Pre-Norm 残差、最终 Norm 和 tied LM head 均已接通。进入 Lab 02 前需注意：`query_start` 必须继续传入 `causal_gqa`；若 RoPE 使用了非零起点而 attention mask 仍固定从 0 开始，分段 prefill/decode 的可见范围会不一致。
