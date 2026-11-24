# Lab 06 · 真 Batch Decode 与 Chunked Prefill

**CPU 验收 `15 passed`，Lab 01–06 回归 `38 passed, 1 skipped`。** 这关把多请求合进同一次模型 forward，并用输入 token 预算限制长 prefill 对已有 decode 的阻塞。模型沿用 `KVStore`，持久 `PagedKV` 负责页地址与层游标，Scheduler 负责计划和提交，Engine 负责组 batch 和执行。

## 1. 从迭代级调度到张量级 batch

Lab 05 的同一 tick 可以有 A、B 两条 Step，分别调用 `forward`。纯 decode 批次将它们组成 `input_ids=[B,1]`、B 个独立 cache 与 B 个绝对位置，调用一次 `decode_batch`，得到 `[B,1,V]`。共享模型权重并合批执行 embedding、Norm、Linear、attention、MLP、final norm 与 LM head，各行历史仍独立。例 A/B 原有 2/5 个位置，此轮位置是 2/5，写后读长 3/6。

| 环节 | 数据形状和不变量 |
|---|---|
| 投影与 RoPE | hidden `[B,1,D_model]`，Q `[B,Hq,1,D]`，K/V `[B,Hkv,1,D]`。positions 为 `[B,1]`，每行独立，使用 [Lab 01 的 RoPE](01-qwen3-forward.md#2-位置编码与-rope)。历史 K 已旋转，V 保留原值。 |
| 按层读写 | 第 b 行输入真长 `T_b`、起点 `p_b`，只向自己的 cache 写 `[p_b,p_b+T_b)` 的 `[T_b,Hkv,D]`，读 `[0,S_b)`，`S_b=p_b+T_b`。decode 的 `T_b=1`。读回含刚写项，不再拼一次。允许逐行 Python 读写，模型主体仍合批。 |
| Gather、pad | 从 cache 读 `[S_b,Hkv,D]`，显式交换轴并复制到 `[B,Hkv,Smax,D]` 的对应前缀，`Smax=max(S_b)`，dtype/device 与 K/V 一致。纯 decode 的输入已是 `[B,1]`，padding 的对象是变长**历史**。 |
| Decode mask | `valid[b,j] = j < S_b`，形状 `[B,1,1,Smax]`，广播到 `[B,Hq,1,Smax]` 的 score。softmax 前用 `masked_fill(~valid,-inf)`。每行至少看到自己。只补零时，padding K 的 score 仍进 softmax 分母，零 V 也会分走概率。 |

原 `causal_gqa` 的标量 `query_start` 及全真 K/V 的假设无法表达按行 padding。可加通用 padding mask 或专用路径，复用 GQA head 展开、缩放与 FP32 数值路径。保留原单序列 `forward` 作对照，再考虑提取投影、RoPE 与残差等重复逻辑。

## 2. 混合 batch 的位置和 padding 契约

更一般的 `forward_batch` 用右填充矩形 `[B,Tmax]`，每行有独立 `p_b,T_b`，输出 `[B,Tmax,V]`。只对真实区投递 KV。真 query `i<T_b` 的绝对位置为 `p_b+i`，允许 key `j < p_b+T_b` 且 `j <= p_b+i`。`p_b+T_b` 是排他写入终点，最后真 query 是 `p_b+T_b-1`。正确的 causal 条件已会挡掉真实 query 后的尾部，显式 `valid_key` 让契约易审查。mask 从 `[B,Tq,Tkv]` 增 head 轴为 `[B,1,Tq,Tkv]`。

`forward_batch` 入口校验二维、非空 batch、caches/positions/actual_lengths 与 B 一致、`1<=T_b<=Tmax`、`max(T_b)=Tmax` 与 `p_b>=0`，错误元数据在写 cache 前拒绝。`decode_batch` 严格要求 `[B,1]` 并以全 1 的真长调用底层，拒绝把多 token 行只写一项却继续算完。RoPE positions、attention mask 与 `Smax` 只依赖批次元数据，可入口建一次后跨层复用。docstring 要区分两个入口的形状和有效区域。

### Padding query 为什么会出现 NaN

- 第一种位置错误把 `p+T` 当最后真位置。例 `p=5,T=1,Tmax=3` 时得到 `[5,6,6]`，真 query 只读 `0..5`，占位 query 可能读到 tail 零 slot 6，softmax 分母多一项。这不会倒流至位置 5，但占位结果没有业务语义。
- 改成 `-1` 后占位 query 对所有非负 key 都被遮住，`softmax([-inf,…])` 得 `NaN`。实际长度 1/3/2 的混合批里，真 logits 仍与完整重算差约 `2.98e-7`，两条短行的占位才出现 NaN。按 query 独立做 attention，后续 residual、Norm 和 MLP 也按位置计算，且 padding KV 未写入，故不会横向污染真项。
- 一种合法契约只定义 `[0,T_b)`，消费方先过滤再采样、比对、做 loss/指标、归约或 JSON。另一种契约要求全输出有限，可对全遮蔽行用安全 scores 做 softmax 后将占位 attention 结果清零，或赋占位合法位置。笼统对最终 logits 做 `nan_to_num` 会隐藏真实错误。
- **本 Lab 的安全占位取位置 0。** 每行至少有一个真 token，key 0 在其真实历史内，pad query 只见 key 0，softmax 概率 1，输出有限。clamp 到最后真位置也是可考虑的安全值。`allowed.any(key_dim)` 此时对占位也为 True，仍要靠 `i<T_b` 标记真实输出。占位参与矩形网络计算，但不写 cache、不采样、不发事件、不推进状态、不纳入有效统计。`logits[b,T_b-1]` 才是该行末尾有效位置。

### K/V 缓冲区与优化方向

旧 `final_k/final_v=[B,Smax,hidden_size]` 在写 `[3,2,8]` 到 `[3,24]` 时触发 expand 维数错误。聚合需独立 `Hkv` 轴与 head dim `D`，正确形状是 `[B,Hkv,Smax,D]`。先 `zeros` 预分配并复制各行前缀，保持 dtype/device。用 `F.pad/stack` 仍逐行处理，可能增加临时张量与堆叠复制。若换 `empty`，尾部未赋值的 V 即使 score 被遮住，也可能经 `0 * NaN` 污染结果。用 profiling 判断逐行 gather、缓冲区和元数据分配开销。进一步可将真 token 展为 `[sum(T_b),D]` 并传每请求起止偏移给 varlen attention，或让分页 kernel 直接读物理页，代价是更复杂的元数据与 kernel。

## 3. Chunked prefill、预算和活性

长 prompt 一次 prefill 会拖长同 tick 的 decode ITL。切块后每轮先服务 decode，用长请求自己的 TTFT 和更多轮次换更平滑的 ITL，完成后逻辑 KV 与整段 prefill 一致。

| 量 | 约束 |
|---|---|
| `max_active` | 活跃请求行数。 |
| `max_batch_tokens` | tick 消费的总输入 token，有值时 `>= max_active`，以容纳所有 decode 行每行 1 个。 |
| `max_prefill_chunk` | 单请求单轮 prompt token 上限，有值须为正。 |
| `block_size`、`P+N-1` | 每物理块可容纳的 KV 位置与单请求的预留容量，分别见 [Lab 03](03-block-paging.md)、[Lab 05](05-engine.md)。 |

每轮先准入，再规划 decode 并扣掉 1/行，再按活跃 prefill 顺序分剩余预算，每行至多一块。`allowed_chunk=min(剩余 prompt, chunk 上限或 ∞, 剩余预算或 ∞)`。为 0 就跳过，正数用来切片然后按实际长度扣预算。

| chunk / budget 参数 | 预期行为 |
|---|---|
| 都 `None` | 整段 prefill、逐 token decode，无 token budget 筛选。 |
| 仅 chunk | 按 chunk 拆 prompt，decode 无预算筛选。 |
| 仅 budget | decode 优先，prefill 随剩余预算截短。 |
| 都给出 | 同时应用三项最小值。 |

判阶段用 `processed_tokens < len(prompt_ids)`。中间块始终可能保持 `last_token=None`，不能用它区分首次与继续 prefill。`processed` 每轮按输入更新，`generated` 与 `last_token` 只在 emit 时更新。`complete` 先校验 emit 与采样值，再按下表提交，EOS 优先于长度。取消半段 prefill 仍归还整份页预留。

| Step | 模型与 KV | 传 `complete` | 对外结果 |
|---|---|---|---|
| decode | 执行 | Python `int` | token 事件 |
| 最后 prefill chunk | 执行 | 最后真实位置采样的 `int` | 首 token 事件 |
| 中间 prefill chunk | 执行 | `None` | `None`，只推进 processed |

最小 trace 为 `max_active=2,budget=4,chunk=3`。chat 已消费长度 2 的 prompt 且发 20，新 long prompt 长 7、上限 1。三轮分别给 chat 1 项和 long `[0,3)`、`[3,6)`、`[6,7)`，总预算用量 4、4、2。long 的 KV 读写长度依次 3、6、7，最后才发 31 并按长度退休。

### Scheduler 复盘

| 故障 | 定稿语义与防回归 |
|---|---|
| `plan()` 返回 `(decode_steps,prefill_steps)` | 返回 decode 在前的扁平 `list[Step]`。`step, = [one]` 和 `decode, chunk = [a,b]` 都是合法的可迭代对象解包，关键在元素个数和类型。 |
| 无 budget 时所有 Step 被过滤，或把 `None` 加到游标 | `None` 表示对应限制关闭。默认整段 prefill，decode 只吃 `(last_token,)`。 |
| 中间 chunk 被误判 EOS | `sampled_token=None == eos_token_id=None` 会误退休。无 emit 时只改 processed，返回 None。emit/token 两种不匹配组合都在更新前报错。 |
| 先生成整块再按额度全收或拒绝，导致饿死 | budget-only `2`、prompt 长 3 时仍要生成前两项。decode 后余 1、chunk 上限 2 时仍安排一项 prefill。`plan` 要把同一个 resource 传入生成函数，`chunk=None` 也走统一切片。没有 complete 时重 plan 复现相同游标，说明计划本身不提交。 |
| 非正 chunk 长与文案 | 0 产生无进展空 Step，负数可能让扣减反向增加额度。入口校验正数，budget 文案用“至少”。 |

首次调度定向 `1 failed,1 passed,4 deselected`，旧 Lab 回归 `6 failed,17 passed,1 skipped`。扁平返回与默认路径修复后定向 `2 passed,4 deselected`，Lab 04+05 `8 passed`，Lab 01–05 `23 passed,1 skipped`。剩余预算的两个最小复现在公开用例已绿时仍失败，最后补齐入口接线与 `None` 分支才恢复实际进展。简化 `StepResource`、用 `deque` 避免 `pop(0)` 搬移和直接查 `key in dict` 属后续整理。

## 4. Engine 接线、采样与事件

- 基线做法将 Step 分成两组，decode 一次 `decode_batch`，prefill 逐条 `forward`。input/cache/position/Step 的顺序必须一致，按同一行取 logits 和提交 token。
- 本实现进一步合并混合 tick。空 plan 直接 `[]`，全 decode 仍调用一次 `decode_batch` 以满足公开 spy，混合或 prefill tick 用一次 `forward_batch`。按 plan 右填充 IDs（可用 0），同时传每 Step 的真长、起点和持久 cache。整批算完再逐行 complete。各请求的 KV 独立，已算完后某行退休不影响整批计算。识别纯 decode 可用 `all(...)`，避免依赖最后一项的相位和隐式顺序。
- 例如 chat 的 `start=2,input=(20,),emit=True`，long 的 `start=0,input=(3,4,5),emit=False`。打包 `[[20,0,0],[3,4,5]]`，真实长度 `[1,3]`、起点 `[2,0]`。chat 只从第 0 列采样，long 只提交 None，processed 分别加 1/3，输出只有 chat 事件。空事件也可能代表真的执行了某段 prefill。
- 过滤的是 `complete() is None`，带 `stop/length` 的事件保留。批量 argmax 可保留 Tensor，单行入 complete 前用 `.item()`。`.int()` 仍是 Tensor，0 维 Tensor 与 int 比较会掩盖类型错误，tokenizer 和 JSON/SSE 才可能失败。增加 `type(event.token_id) is int` 覆盖纯 decode 与混合分支，修正后通过。重复 batch 包装可整理，旧 `_prepare_input_ids` 和注释已清理。

## 5. 对拍顺序与验收记录

先比本步 logits 对逐条完整重算，再用这次写回的 KV 继续各自下一轮 decode 与重算对拍。第一步对了也可能只用了临时 K/V，后续才暴露写错。看单行逐层的投影、RoPE、cache 读写、gather/pad、mask 与 scores，避免只比 argmax。混合长度 1/3/2 还验全张量有限、逐层 `p_b+T_b`、失效占位未写入和非法元数据拒绝。

```bash
python -m pytest -q tests/test_lab06_batch.py -k scheduler
python -m pytest -q tests/test_lab06_batch.py -k real_batch
python -m pytest -q tests/test_lab06_batch.py -k one_batched
python -m pytest -q tests/test_lab06_batch.py -k chunked_engine
python -m pytest -q tests/test_lab06_batch.py
```

纯 decode spy 验 `decode_batch` 恰一次且普通 forward 未调用。混合 spy 验 `forward_batch` 恰一次和 shape/positions/actual lengths。端到端验中间块无事件、半段取消全回收、最终 token 与重算一致。中途模型已通但 Engine 未接入时为 `4 passed,2 failed`。Engine 初次完成与新增类型契约前为 `14 passed`。模型/调度定向 `-k 'not engine'` 曾为 `12 passed,2 deselected`。最终全关 15 通过。代码 `src/tinyserve/model.py`、`scheduler.py`、`engine.py`，原笔记 `learning_notes/lab06_batch.md`。
