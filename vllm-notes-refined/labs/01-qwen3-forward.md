# Lab 01 · Qwen3 Forward

**CPU 验收 `6 passed`。** 双层 toy 模型已完成 HF 风格权重映射和完整 forward。真实快照验证见 [Lab 02](02-single-request.md)。

## 1. dtype、广播与 RMSNorm

| PyTorch dtype | 位数 | 指数 / 存储尾数位 | 常用转换 |
|---|---:|---:|---|
| `float16` / FP16 | 16 | 5 / 10 | `x.half()` |
| `bfloat16` / BF16 | 16 | 8 / 7 | `x.bfloat16()` |
| `float32` / FP32 | 32 | 8 / 23 | `x.float()` 或 `x.to(dtype=torch.float32)` |
| `float64` / FP64 | 64 | 11 / 52 | `x.double()` |

指数决定动态范围，存储尾数决定精度，表中尾数不计隐含最高位。FP16 最大有限值 65504，超出可能成 `inf`。BF16 动态范围接近 FP32，尾数比 FP16 粗。两种半精度都能减少权重、激活、KV cache 的存储量与带宽。训练和推理常将低精度存储与 FP32 累加或敏感算子组合，结束后转回原 dtype。

- `*` / `torch.mul`、`/` / `torch.div` 是可广播的逐元素运算，`@` / `torch.matmul` 遵循矩阵乘法语义。`[B,T,D] * [D]` 会把最后一维的权重广播到所有 batch 和 token。
- `square()` / `pow(2)` 为平方，`sqrt()` 为平方根，`rsqrt()` 为倒数平方根。以 `mean(..., dim=-1, keepdim=True)` 沿最后一维归约，`[B,T,D] → [B,T,1]`，可直接与输入广播。
- **RMSNorm** 在 FP32 中求 `mean(x²)`，令 `y = (x * rsqrt(mean(x²) + eps)) * weight`，`weight` 为可学习缩放权重，最后恢复输入 dtype。`eps` 使零输入的分母稳定，乘的是倒数因子。对 `rsqrt` 再做除法会放大数值。RMSNorm 使用均方根缩放，LayerNorm 则按均值与方差归一化。

## 2. 位置编码与 RoPE

全可见且无位置信息的 self-attention 对顺序置换等变。正弦位置编码把频率对 `PE_i(p) = [sin(ω_i p), cos(ω_i p)]` 加到 embedding。由和角公式，`PE_i(p+k) = M(k) PE_i(p)`，其中 `M(k)` 两行分别为 `[cos(ωk), sin(ωk)]` 和 `[-sin(ωk), cos(ωk)]`，只依赖位移 `k`。这个矩阵用于解释平移结构，运行时可按三角函数直接得到位置向量。

RoPE 用绝对位置旋转投影后、归一化后的 Q/K。`R(a)ᵀ = R(-a)`、`R(a)R(b) = R(a+b)`，所以 `(R(mω)q)ᵀ(R(nω)k) = qᵀ R((n-m)ω) k`，点积引入相对位移。旋转保持向量范数，V 保留原值。RoFormer 直接从点积仅依赖内容与相对位置的约束推导。加法位置编码仍能提供位置信息。

`[cos a, sin a] · [cos b, sin b] = cos(a-b)`。小角度下 `cos Δ ≈ 1 - Δ²/2`，相位相近时接近 1。周期振荡使该关系在远距离不单调，注意力最终还取决于内容。正余弦共同表示二维相位，高频感受局部变化，低频提供更长的位置尺度。能计算任意位置与长序列泛化是两回事，单个三角分量也会周期性重复。

| 实现要点 | 形状与约束 |
|---|---|
| 旋转布局 | split-half 将 `D=6` 配成 `(0,3)、(1,4)、(2,5)`。与相邻配对经维度重排等价。两半为 a、b 时，输出两半是 `a*cos-b*sin` 与 `b*cos+a*sin`。分块对角矩阵按元素计算，单 token 从显式稠密乘法的 `O(D²)` 降为 `O(D)`。 |
| 位置轴与频率轴 | `i = arange(D/2)`，`inv_freq_i = θ^(-2i/D)`，`angle[p,i] = positions[p] * inv_freq_i`。共享位置采用 `positions[:,None] * inv_freq[None,:] → [T,D/2]`。频率指数用浮点除法 `/`，位置乘上频率后才是角度。 |
| 多行 positions | 从共享 `[T]` 增轴并 `expand(B,-1)` 得 `[B,T]`，或直接传独立位置。角度 `[B,T,D/2]`，在 head 轴增维至 `[B,1,T,D/2]` 后广播到 `[B,H,T,D/2]`。Lab 06 用它处理不等长前缀。 |
| 运行时 | RoPE 无可学习参数，每层都旋转自己的 Q/K，cos/sin 表可复用。历史 K 已旋转，decode 只旋转新位置。位置、频率下标、角度表与输入同 device。分别用非原地 `cos` / `sin`，先 `cos_()` 会覆盖原角度。 |

`unsqueeze(dim)` 或 `x[:,None]` 增长度为 1 的轴。`expand` 把单例轴扩为共享视图，`-1` 保持大小，`expand_as(y)` 使用 `y.shape`。例如 `[2,3] → [1,2,3] → [4,2,3]`，引用可能共享内存，避免重叠原地写。`repeat` 会复制数据。`reshape` / `view` 重排已有元素，换轴用 `transpose` / `permute`。

## 3. Causal GQA

`Q=[B,Hq,Tq,D]`，`K/V=[B,Hkv,Tk,D]`，score 为 `[B,Hq,Tq,Tk]`，输出为 `[B,Hq,Tq,D]`。用 `Q @ K.transpose(-2,-1) / sqrt(D)` 缩放点积，抑制大 head dim 下 softmax 饱和。在 softmax 前掩码，再乘 V。

| 类型 | 对齐关系 |
|---|---|
| MHA | `Hq=Hkv`，每个 Q head 对应自己的 K/V。 |
| GQA | `groups=Hq/Hkv`，每组 Q 共享一个 K/V head。参考实现用 `repeat_interleave(groups,dim=1)`，`[k0,k1] → [k0,k0,k1,k1]`。存储保留 `Hkv`，减少 K/V 投影与 cache，后续 query 不读取旧 Q。 |
| MQA | `Hkv=1`，所有 Q heads 共用一组 K/V。 |

令 `qpos = query_start + arange(Tq)`，`kpos = arange(Tk)`。`allowed = kpos <= qpos` 得 `[Tq,Tk]`，以 `scores.masked_fill(~allowed, -inf)` 屏蔽未来后再 softmax。误用 `allowed` 会屏蔽历史。整行 `-inf` 会产生 `NaN`。位置张量建在 `q.device`。普通 prefill 是下三角。单 token decode 看见整个历史及自己。分段 prefill 看见历史并保持块内因果。RoPE 用绝对位置编码相对关系，mask 决定可见性。

## 4. Qwen3 结构与权重表

| 权重 | HF / PyTorch 存储形状 |
|---|---|
| `embed_tokens.weight`、独立的 `lm_head.weight` | `[V,D_model]` |
| `q_proj` | `[Hq*D,D_model]` |
| `k_proj` / `v_proj` | `[Hkv*D,D_model]` |
| `o_proj` | `[D_model,Hq*D]` |
| `gate_proj` / `up_proj` | `[I,D_model]` |
| `down_proj` | `[D_model,I]` |
| `q_norm` / `k_norm` | `[D]` |

- token IDs `[B,T]` 经 embedding 得 `[B,T,D_model]`。每层先对残差输入做 RMSNorm，投影得到 `[B,T,H*D]`，reshape 为 `[B,T,H,D]`，transpose 至 `[B,H,T,D]`。Q/K 的 head-dim RMSNorm 在 RoPE 之前，V 直通。Attention 输出换轴与 reshape 后经 `o_proj` 回到模型宽度，再加原输入。MLP 同样 pre-norm 后加残差。
- SwiGLU 为 `W_down(SiLU(W_gate x) ⊙ W_up x)`，gate/up 使用同一个归一化输入。所有层结束再做 `model.norm` 和 LM head，得到 `[B,T,V]` 的原始 logits。
- `D_model` 与 `Hq*D` 可以不同，`o_proj` 输入取后者。投影显式 `bias=False`，避免保留随机 bias。用 `reshape` 合并 transpose 后可能非连续的 heads。模型层由 `nn.ModuleList` 注册，使 `.parameters()` 和 `.to(device,dtype)` 正常遍历。
- HF Linear 存 `[out,in]`，`F.linear(x,w)` 已处理转置。先核对 key/shape，在 `torch.no_grad()` 中用 `Parameter.copy_` 加载，可复制到目标 dtype。tied 时两个模块共享同一个 embedding Parameter，只加载 `model.embed_tokens.weight`。untied 才要求独立 `lm_head.weight`。

## 5. 推理与后训练的参数边界

部署固定模型版本后，请求只运行 forward。SFT、偏好优化和强化学习后训练得到新参数后再冻结部署。冻结底座而训练 LoRA / adapter 时，有效权重仍是 `W + ΔW`。相同有效参数、配置、tokenizer 和输入 token 给相同的 logits 函数，行为差异可来自采样、system prompt、解码参数、检索、工具或路由等服务侧输入。

## 6. 回归与定位顺序

```bash
python -m pytest -q tests/test_lab01_model.py
python -m pytest -q tests/test_lab01_model.py -k rmsnorm
python -m pytest -q tests/test_lab01_model.py tests/test_lab06_batch.py -k rope
```

RMSNorm 分项 `3 passed, 3 deselected`，覆盖 FP32/FP16/BF16。追偏差时依次对公式和 dtype、位置与频率两轴、mask 布尔方向、模块结构和参数表。带 cache 后同时检查 RoPE 和 GQA 的 `start_pos`，见 [Lab 02](02-single-request.md)。本篇提炼自 `learning_notes/lab01_model.md`。
