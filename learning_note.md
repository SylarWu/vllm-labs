# vLLM 学习笔记

这份笔记沉淀学习过程中具有复用价值的概念、命令和易错点，不作为逐句聊天记录。

## 1. Python 环境与 uv

### pip、Anaconda 和 uv 的区别

| 工具 | 核心定位 |
| --- | --- |
| pip | Python 包安装工具，主要负责从 PyPI 安装包和解析依赖，不负责创建隔离环境。 |
| Anaconda / Conda | Python 发行版和环境管理工具，可以管理 Python 版本、环境及部分非 Python 系统依赖。 |
| uv | 高速 Python 包、环境和项目管理工具，可替代许多 pip、venv 等命令的用法。 |

同一个环境尽量固定一种主要安装方式，避免依赖管理混乱。本实验使用 uv 和 `.venv`，不需要 Anaconda。

### 创建和激活虚拟环境

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

- `uv venv`：创建虚拟环境。
- `--python 3.12`：指定 Python 3.12。
- `.venv`：将环境放在当前项目的 `.venv/` 目录中。
- 创建不等于激活；激活后，包与系统 Python 和其他项目隔离。

### 安装当前实验项目

```bash
uv pip install -e '.[lab]'
```

- `uv pip install`：以 pip 风格安装依赖，但由 uv 执行。
- `-e`：editable（可编辑）安装，源码修改后通常无需重新安装。
- `.`：安装当前目录中由 `pyproject.toml` 描述的项目。
- `[lab]`：同时安装项目的 `lab` 可选依赖组。
- 引号用于避免 shell 对方括号进行路径匹配。

### 在指定环境中补装依赖

激活环境后：

```bash
uv pip install numpy
```

或者不激活环境，显式指定 Python：

```bash
uv pip install --python .venv/bin/python numpy
```

### `uv pip` 与 `uv add` / `uv sync`

- `uv pip install ...`：只安装，不修改项目依赖声明，也不生成锁文件。
- `uv add numpy`：把 NumPy 写入 `pyproject.toml`，更新 `uv.lock` 并安装。
- `uv add --optional lab numpy`：把 NumPy 写入 `lab` 可选依赖组。
- `uv sync --extra lab`：按照项目声明和锁文件同步环境，同时启用 `lab` 依赖组。
- `uv run pytest ...`：直接使用项目环境执行命令，通常不必手动激活环境。

如果源码直接使用 NumPy，应声明为核心依赖；如果只用于测试或实验，可放入 `lab`：

```toml
[project]
dependencies = [
    "torch>=2.7,<3",
    "numpy>=2.0",
]

[project.optional-dependencies]
lab = [
    "pytest>=8,<10",
    "fastapi>=0.115,<1",
    "httpx>=0.28,<1",
    "uvicorn>=0.30,<1",
]
```

本项目采用 uv 管理依赖、setuptools 构建和 src 布局：

```text
vllm-labs/
├── pyproject.toml       # 项目元数据、依赖、构建和工具配置
├── uv.lock              # 锁定完整依赖版本，建议纳入版本管理
├── .venv/               # 本地虚拟环境，不纳入版本管理
├── src/tinyserve/       # 实际 Python 包
├── tests/
└── scripts/
```

## 2. 浮点数格式与 PyTorch dtype

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

## 3. PyTorch 基础张量运算回顾

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

## 4. Lab 01：RMSNorm

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

## 5. 从 Sinusoidal Encoding 到 RoPE

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

## 6. RoPE 的实现形状与角度映射

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
