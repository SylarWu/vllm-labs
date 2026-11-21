# Lab 06 · 真 Batch Decode 与 Chunked Prefill

**状态：学习中。Scheduler 公开测试、默认兼容和剩余预算边界均已复验通过；下一步实现 `decode_batch`。**

## Lab 01～06 的整体串联

```text
Lab 01：单请求/单序列的数学计算
  Token IDs → Embedding → L × Decoder → logits
  建立 RMSNorm、RoPE、causal GQA、SwiGLU 和残差结构

Lab 02：让多轮生成不必重算全部历史
  prefill 写整段 prompt KV → 每次 decode 只输入上一个 token
  建立 KVStore、start_pos、processed/generated 的时序语义

Lab 03：让 KV 的物理位置不连续
  BlockPool 管块所有权，BlockTable 管逻辑 token → 物理 slot

Lab 04：让多个请求按 iteration 交替前进
  Scheduler 管 waiting/running、Step、complete、EOS/长度和取消

Lab 05：把数学、分页和调度接成 Engine
  PagedStorage 持有物理 K/V，PagedKV 持有请求视图和逐层写入长度
  Engine 准入时预留页，执行 Step 后采样并提交，退休时回收资源

Lab 06：减少重复 kernel 调用，并限制长 prefill 对 decode 的阻塞
  多个 decode 行合并成一次模型 forward
  长 prompt 分为多个 chunk，中间 chunk 只写 KV，不发 token
```

本 lab 不是重写前五周，而是保留现有抽象：

- `KVStore` 协议不变，PagedKV 和 ContiguousKV 都能被 batched decode 使用。
- 分页地址仍由 PagedKV 处理，模型无需知道 block ID。
- PagedKV 实例仍需跨 tick 稳定持有。
- Scheduler 仍负责计划与状态提交，Engine 仍负责执行。

## 第一部分：什么是“真 batch decode”

Lab 05 同一 tick 有两个 decode Step 时，Engine 实际执行：

```text
forward(A 的一个 token)
forward(B 的一个 token)
```

这只是迭代级调度，不是张量级 batching。Lab 06 要变成：

```text
input_ids: [B,1]
caches:    B 个独立 KVStore
positions: [S0, S1, ..., SB-1]
→ decode_batch 一次
→ logits: [B,1,V]
```

每一行仍对应一个独立请求。batch 维让各请求共享同一组模型权重并共同执行 Linear、Norm、MLP 和 attention 计算，但不共享请求历史。

例如 A 已有 2 个 prompt 位置，B 已有 5 个：

```text
A: 当前输入 token 的绝对位置 = 2，写后历史长度 = 3
B: 当前输入 token 的绝对位置 = 5，写后历史长度 = 6
```

二者可以在同一个 `[B,1]` 张量里进入模型，但 RoPE 位置、cache 写入起点和可读历史长度必须逐行独立。

## decode_batch 的张量路径

`TinyQwen3.decode_batch` 的主干仍是 Lab 01 的计算：

```text
[B,1] token IDs
→ embedding [B,1,D_model]
→ 每层 Pre-Norm
→ Q/K/V projections
→ 每行独立 RoPE
→ 每行写自己的 KV cache
→ 读历史、pad、mask、batched GQA
→ O projection + 残差
→ MLP + 残差
→ final norm + LM head
→ [B,1,V]
```

Q/K/V 投影后仍转换成：

```text
Q: [B,Hq,1,D]
K: [B,Hkv,1,D]
V: [B,Hkv,1,D]
```

Norm、Linear、残差和 MLP 本来就能处理 batch 维，不应为每行重新调用完整模型。

## 每行独立 RoPE

现有 `apply_rope` 已支持：

```text
x:        [B,H,T,D]
positions: [B,T]
```

因此 decode 时传入 `[B,1]` 的绝对位置即可。不能把 A、B 的历史长度取最大值后共用一个位置，也不能把所有行都从位置 0 开始。

历史 K 在写入时已经旋转，本步只旋转当前新 token 的 Q/K；V 始终不做 RoPE。

实现位点：[model.py 的 apply_rope](../src/tinyserve/model.py) 已完成；`decode_batch` 需要构造每行 positions。

## 每层写各自的 KVStore

对第 `b` 行、第 `l` 层：

```text
cache[b].write(
  layer=l,
  start_pos=positions[b],
  K/V=[1,Hkv,D],
)
```

随后读取：

```text
length[b] = positions[b] + 1
cache[b].read(l, length[b]) → [length[b],Hkv,D]
```

读长度包含刚写入的当前 token。不能再把当前 K/V 额外拼一次。

允许用 Python 循环完成 B 个请求的 cache 写入与 gather；这不否定模型计算已经 batch 化。教学版的开销优化可以留到后面。

PagedKV 会继续完成 scatter/gather，所以 batched attention 收到的是普通连续张量。无需在 model 中读取 BlockTable。

## 为什么需要 padding

不同请求的历史长度不同，而矩形张量要求每行形状一致。设最大读长度为：

```text
Smax = max(lengths)
```

可将每行 K/V 转为 `[Hkv,length_b,D]`，再放入：

```text
padded K/V: [B,Hkv,Smax,D]
```

每行真实历史放在前缀，剩余尾部只是补齐轴长度。padding 建议置零，但**零填充本身不足以保证正确性**。

如果只用零填充而不 mask：

```text
attention = softmax(QK scores) @ V
```

padding 的 V 虽然是零，不产生分子向量，但 padding 的 Q/K 分数仍进入 softmax 分母，会分走真实历史的概率质量，导致输出改变。

## 变长历史 mask

对每行构造：

```text
valid[b,j] = j < length[b]
```

可以形成 `[B,1,1,Smax]` 的布尔 mask，广播到 attention scores：

```text
scores: [B,Hq,1,Smax]
```

在 softmax 前把无效位置填成 `-inf`。沿用 Lab 01 的教训：`masked_fill` 的 True 表示“要屏蔽”，所以传入的是允许集合的反面。

decode 每行只有一个 query，其真实 K/V 是截至当前位置的完整前缀，因此这里主要需要 padding mask。每行至少包含当前 token，不会出现整行全被 mask。

## 为什么不能原样复用 causal_gqa

当前 `causal_gqa` 假设：

- 所有行使用同一个 `query_start`；
- K/V 中所有位置都是真实有效位置；
- mask 只表达标量位置下的因果关系。

batched decode 增加了逐行长度和 padding 维度的语义。可以扩展一个共享 GQA 辅助函数以接受可选 key padding mask，也可以为单步 decode 建立专用路径，但必须保留原有 GQA head 展开、缩放和 FP32 中间计算，避免数值对拍出现无意义偏差。

实现位点：

- [model.py](../src/tinyserve/model.py) 的 `TinyQwen3.decode_batch` 是入口。
- 为减少重复，可在 `TinyQwen3Attention` / `TinyQwen3Decoder` 增加 batched decode 辅助路径。
- 现有单请求 `forward` 的行为应保持不变，Lab 01～05 回归是约束。

## Engine 如何只调用一次 batched decode

`scheduler.plan()` 可能同时返回 decode 和 prefill。Engine 应先分类：

```text
decode_steps
prefill_steps
```

对于所有 decode：

1. 按 decode_steps 的顺序堆叠 input IDs 为 `[B,1]`。
2. 按完全相同的顺序收集 PagedKV 和 start_pos。
3. 调用一次 `model.decode_batch`。
4. 第 i 行 logits 对应第 i 个 Step，逐行 argmax。
5. 分别调用 `scheduler.complete(step_i, token_i)`。

顺序映射不能打乱，否则会把 A 的 token 提交给 B。prefill chunk 在本 lab 仍可逐个普通 forward。

实现位点：[engine.py](../src/tinyserve/engine.py) 的 `step()`。验收 spy 要求第二轮有两个 decode 行时，`decode_batch` 恰好调用一次，普通 `forward` 不调用。

## 第二部分：为什么需要 chunked prefill

Lab 05 一次 prefill 吃掉整个 prompt。一个长 prompt 的计算期间，已有 decode 请求必须等这一轮结束，表现为 ITL 变长。

chunked prefill 把长 prompt 分为多轮：

```text
prompt 长度 7，chunk 上限 3
第 1 轮：位置 [0,3)，只写 KV
第 2 轮：位置 [3,6)，只写 KV
第 3 轮：位置 [6,7)，写 KV，并采样第一个新 token
```

每一轮都可同时安排其他请求的 decode。这样用长请求自己的 TTFT 和更多调度轮次，换取其他生成请求更平滑的 ITL。

chunking 改变的是**一次 forward 消费多少 prompt**，不改变最终逻辑历史。所有 chunk 完成后，KV 必须与整段 prefill 等价。

## Step.emit_token 的完整语义

Lab 04/05 中每个 Step 都发 token；Lab 06 引入第三种执行结果：

| Step | 是否 forward/写 KV | 是否采样 | complete 的 sampled_token | 返回值 |
| --- | --- | --- | --- | --- |
| 中间 prefill chunk | 是 | 否 | `None` | `None` |
| 最后 prefill chunk | 是 | 是 | token ID | TokenEvent |
| decode | 是 | 是 | token ID | TokenEvent |

`emit_token=False` 不表示“这轮不执行模型”，而表示“不消费 logits、不增加输出计数、不设置 last_token、不产生对外事件”。

因此测试里取消半个 prefill 前，`engine.step()` 可以返回空列表，但 cache 已经写入了一个 chunk。空事件列表不等于空执行。

## Scheduler 的三个状态如何推进

`processed_tokens`：已消费并写入 KV 的输入位置数。每个 prefill chunk 和 decode 都按 `len(input_ids)` 增加。

`generated_tokens`：已经对外发出的新 token 数。仅 `emit_token=True` 时增加。

`last_token`：上一轮发出的、下一轮 decode 要消费的 token。中间 chunk 必须保持不变，最后 prefill chunk 和 decode 才更新。

prefill 是否结束应由：

```text
processed_tokens < len(prompt_ids)
```

判断，而不能继续只靠 `last_token is None`。中间 chunk 后 last_token 仍是 None，但下一轮 start_pos 必须等于 processed_tokens，input_ids 必须取 prompt 的对应切片。

`complete` 需要先校验 emit 与 sampled_token 的组合，再提交状态：

- `emit_token=False`：只接受 None，推进 processed，不判断 EOS/长度，返回 None。
- `emit_token=True`：要求 token，推进 processed/generated/last_token，再按 EOS 优先于长度判断退休。

实现位点：[scheduler.py](../src/tinyserve/scheduler.py) 的 `_generate_step`、`plan`、`complete`。

## token budget 与 decode 优先

每轮先做 tick 边界准入，再为活跃请求安排工作：

1. 所有 decode 行优先，每行消耗 1 个 token budget。
2. 剩余 budget 按活跃 prefill 顺序分配。
3. 每个 prefill 请求本轮至多得到一个 chunk。
4. chunk 同时受剩余 prompt、`max_prefill_chunk` 和剩余 budget 限制。
5. 只有吃到 prompt 末尾的 chunk 设置 `emit_token=True`。

`max_batch_tokens` 有值时必须至少等于 `max_active`。否则如果所有活跃请求都处于 decode，连每行一个 token 都容纳不下，无法满足 decode 优先并可能造成停滞。

需要区分这些“token 数”：

| 配置 | 限制什么 |
| --- | --- |
| `max_active` | 活跃请求行数 |
| `max_batch_tokens` | 一轮计划消费的输入 token 总量 |
| `max_prefill_chunk` | 单个请求一轮最多消费的 prompt token 数 |
| `block_size` | 一个物理块可容纳的 KV token 位置数 |
| `P+N-1` | 活跃请求需要预留的 KV 位置数 |

token budget 是每轮计算配额，不会替代 Lab 05 的 KV 页预留。中间 prefill 取消时，仍由 Lab 05 的退休回调释放该请求的全部预留页。

## 手推验收 trace

配置：

```text
max_active=2, max_batch_tokens=4, max_prefill_chunk=3
chat：prompt 长 2，已 prefill 并发出 token 20
long：prompt 长 7，max_new_tokens=1
```

| Tick | chat | long | budget |
| --- | --- | --- | --- |
| 1 | decode，1 token | prefill `[0,3)`，不发 token | 1+3=4 |
| 2 | decode，1 token | prefill `[3,6)`，不发 token | 1+3=4 |
| 3 | decode，1 token | prefill `[6,7)`，发 token 31 | 1+1=2 |

long 前两次 complete 都只推进 processed；最后一次发出 token 并因长度上限退休。chat 每轮都能继续 decode。

## 实践顺序与实现清单

1. **Scheduler 纯状态测试**：先实现 chunk 切片、decode 优先、emit/None 提交和构造参数校验，不接模型。
2. **模型 decode_batch**：完成逐行 RoPE、独立 cache 写入、gather/pad/mask 和 logits 对拍。
3. **Engine batch 接入**：将 decode 行合并一次调用，prefill 仍逐个执行，注意行顺序和事件过滤。
4. **端到端 chunk**：验证中间 chunk 无事件、取消释放页、最终输出与完整重算一致。
5. **回归 Lab 01～05**：尤其检查原单请求 forward 和 Lab 04 的默认调度语义未被破坏。

```bash
.venv/bin/python -m pytest -q tests/test_lab06_batch.py -k scheduler
.venv/bin/python -m pytest -q tests/test_lab06_batch.py -k real_batch
.venv/bin/python -m pytest -q tests/test_lab06_batch.py -k one_batched
.venv/bin/python -m pytest -q tests/test_lab06_batch.py -k chunked_engine
.venv/bin/python -m pytest -q tests/test_lab06_batch.py
```

[模型入口](../src/tinyserve/model.py)、[调度器入口](../src/tinyserve/scheduler.py)、[Engine 入口](../src/tinyserve/engine.py)、[验收测试](../tests/test_lab06_batch.py)

## Scheduler 第一次代码复盘

### list 解包不等于要求 tuple

Python 的序列解包作用于可迭代对象，不要求右侧必须是 tuple：

```python
step, = [step_a]                 # 列表中恰好一个元素
decode, chunk = [step_a, step_b] # 列表中恰好两个元素
```

因此 `plan() -> list[Step]` 与验收里的 `prefill, = plan()`、`decode, chunk = plan()` 完全兼容。元素数量不匹配才会报 too many/not enough values to unpack。

当前实现实际返回 `(decode_steps, prefill_steps)`，这是两个列表组成的嵌套 tuple，违反了扁平 `list[Step]` 契约。单个 prefill 时得到 `([], [step])`，长度 2 导致单元素解包失败；混合轮次即使解包成功，得到的也会是两个列表而不是两个 Step。应保持 decode 在前、prefill 在后，并返回一个扁平列表。

### 本轮测试结果

```text
Lab 06 scheduler：1 failed, 1 passed, 4 deselected
Lab 01～05 回归：6 failed, 17 passed, 1 skipped
```

构造时拒绝 `max_batch_tokens < max_active` 的测试已通过，其余失败暴露了以下问题。

### 需要逐项修正的实现语义

1. **无预算时反而没有 Step。** `filter_req_with_budget` 只有 `resource is not None and resource.take(...)` 才 append；默认 `max_batch_tokens=None` 时应表示不受 token budget 限制，而不是全部过滤。
2. **无 chunk 参数时无法生成 prefill。** `processed_tokens + self.max_prefill_chunk` 在 None 下直接 TypeError。Lab 04/05 的默认行为必须保留：整段 prompt 一次 prefill，且 emit token。
3. **decode 输入错误。** 当前 decode 返回整个 prompt，应该只返回 `(status.last_token,)`。否则既违背逐 token decode，也会让每行消耗的 budget 不再恰好是 1。
4. **prefill chunk 没有按剩余 budget 截短。** 当前先生成完整 Step，再判断整个 input 是否放得下；放不下就跳过。正确语义是 decode 优先后，以剩余 budget 作为本轮 chunk 上限之一，允许生成更短的非空 chunk。
5. **非发 token 的 complete 被误判为 EOS。** 中间 chunk 的 sampled_token=None，而默认 eos_token_id 也是 None；直接比较二者相等会错误退休。结束判断只能发生在实际发出 token 的路径。
6. **非发 token 的 complete 应返回 None。** 当前末尾仍构造 TokenEvent，且 token_id 可能是 None。中间 chunk 只推进 processed，不产生事件。
7. **last_token 不应被中间 chunk 覆盖。** 只有 emit_token=True 时才更新 last_token/generated，并按既有规则判断 EOS 和长度。
8. **emit 与 token 的反向组合也要校验。** emit=True 要求 token；emit=False 应要求 None，避免调用方把未计划输出的 token 错提交进状态。

### 默认参数与扩展参数的兼容矩阵

| 参数状态 | 应保持的行为 |
| --- | --- |
| 两者均 None | Lab 04/05：整段 prefill、逐 token decode、无 token budget 过滤 |
| 仅 max_prefill_chunk 有值 | decode 不受预算过滤，prefill 按 chunk 上限拆分 |
| 仅 max_batch_tokens 有值 | decode 优先，prefill 使用剩余预算 |
| 两者均有值 | chunk 长度取剩余 prompt、chunk 上限、剩余预算三者的最小值 |

`max_prefill_chunk` 有值时还应拒绝非正数，否则可能产生空 Step、状态不推进和无限 prefill。

本次回归说明：增加可选功能时，**None 通常代表关闭该限制并保留旧路径**，不能仅让新测试通过而忽略默认构造方式。

## Scheduler 第二次代码复盘

复验结果：

```text
Lab 06 scheduler：2 passed, 4 deselected
Lab 04 + Lab 05：8 passed
Lab 01～05 回归：23 passed, 1 skipped
```

扁平 `list[Step]`、无参数旧路径、decode 的 last_token 输入、默认 EOS 判断和公开 chunk trace 均已恢复/实现。

### 公开测试尚未覆盖的阻塞问题

当前 `_generate_step` 仍只根据 max_prefill_chunk 生成 Step，随后由 `resource.take(len(step.input_ids))` 做“全收或全拒”。这并不等价于按剩余 budget 截短。

最小复现：

```text
Scheduler(max_active=1, max_batch_tokens=2)，prompt 长度为 3
连续两次 plan() 都返回 []
```

仅配置 max_batch_tokens 时，max_prefill_chunk=None 使 prefill 试图一次吃整个 prompt；只要 prompt 超过总预算，请求就永远无法开始。另一个混合场景中，decode 消耗 1 后只剩 1 budget，prefill 生成长度 2 的 Step 而被整体跳过，可能一直等 decode 请求结束，违背 decode 优先但 prefill 使用剩余额度的设计。

修正方向是让 Step 生成本身获得本轮允许消费的上限，在切片前取：

```text
min(剩余 prompt，max_prefill_chunk 或无限，剩余 budget 或无限)
```

并保证只要 budget 有剩余，就能规划非空 chunk；不能先生成超限 Step 再整体拒绝。StepResource 可以只负责记录剩余额度，但需要和切片上限协同，而不只是事后过滤。

### 其他输入契约与可优化点

- `max_prefill_chunk <= 0` 尚未拒绝。零会产生空 Step 且不推进；负数还会让 `take` 增加预算，应在构造时校验正数。
- `complete` 尚未拒绝 `emit_token=False、sampled_token!=None` 的反向非法组合。当前会污染 last_token 后返回 None，应在任何状态修改前拒绝。
- `last_token = sampled_token` 可以只在实际 emit 时更新，使“中间 chunk 只改 processed”的不变量在代码结构上更清楚。
- `max_batch_tokens` 的错误文案应表达“at least / 大于等于”，因为当前允许等于 max_active。
- 等语义稳定后，再考虑用局部剩余预算整数简化 StepResource、用 deque 避免等待队列 `pop(0)` 的 O(n) 移动，以及省略 `in dict.keys()` 中的 `.keys()`。这些不是当前正确性的优先级。

这次复盘再次说明：公开 trace 通过代表样例路径正确，不代表所有可选参数组合都能取得进展。调度器除了输出对拍，还应专门检查“非空计划/状态推进”的活性条件。

## Scheduler 第三次代码复盘

再次复验公开结果：Lab 06 scheduler 为 `2 passed`，Lab 01～05 为 `23 passed, 1 skipped`。以下修正已经确认：

- max_prefill_chunk 非正数会在构造时拒绝。
- complete 同时校验 emit=True/token 与 emit=False/None 两种合法组合。
- last_token 只在 emit 时更新。
- max_batch_tokens 文案已改为大于等于。

剩余 budget 截短仍未真正生效，存在两个连接问题：

1. `_generate_step` 新增了 resource 参数，但 `plan()` 仍调用 `_generate_step(request, status)`，实际始终使用默认 None。
2. max_prefill_chunk=None 的提前分支直接取整段 prompt，即使以后传入 resource，也仍不会按 budget 截短。

因此两个最小复现保持原结果：budget-only 长 prompt 连续返回空计划；混合轮次 decode 后剩余 1 budget，prefill 仍被整体跳过。

下一步应统一 prefill 切片路径：无论 max_prefill_chunk 是否为 None，都先得到剩余 prompt，再按需取 resource 和 chunk 配置的最小上限；plan 调用时传入同一个 resource，让 decode 先扣减、prefill 后使用。resource 耗尽时应跳过 prefill，而不是规划 `input_ids=()` 的空 Step。

这次问题属于“辅助函数实现了能力，但调用方没有接线”。代码审查和自测不能只检查新函数本身，还要从公开入口跟踪参数是否到达目标分支。

## Scheduler 完成复验

budget 截短已正确接入：plan 将同一个 resource 传给 Step 生成函数，decode 优先扣减；prefill 统一按剩余 prompt、resource 和 chunk 配置计算 allowed_chunk；allowed_chunk=0 时跳过规划，不产生空 Step。生成后再按实际 input 长度扣减预算。

最小复现结果：

```text
仅 max_batch_tokens=2、prompt=3：
  可规划 prefill(1,2)，emit=False

decode 后剩余 1 budget、chunk 上限 2：
  decode + prefill(长度为1)，同一轮返回
```

第一次复现中未调用 complete 时，下一次 plan 返回相同 start_pos 是正确行为：计划本身不推进状态，只有 complete 才提交游标。

最终复验：

```text
Lab 06 scheduler：2 passed, 4 deselected
Lab 01～05 回归：23 passed, 1 skipped
```

Scheduler 至此形成以下不变量：返回扁平 Step 列表；decode 优先且每行占 1 budget；prefill 只要有剩余额度就能规划非空 chunk；无新参数时保持整段 prefill；中间 chunk 只通过 complete 推进 processed。
