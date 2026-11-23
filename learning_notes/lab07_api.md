# Lab 07 · 流式服务与取消

**状态：进行中，EngineChatBackend 已完成当前验收。** Backend 的并发路由、提前关闭、任务取消、UTF-8 稳定前缀、peer 终止事件交付和积压 delta 测试合计 `5 passed`；`create_app` 的普通响应、参数校验、SSE 和断流测试尚未实现。Lab 01～06 回归基线为 `38 passed, 1 skipped`。

异步语法与 `asyncio`、ASGI 工具包的基础解释见 [Python 异步语法与工具包](python_async.md)。

## 本 lab 在整条推理链路中的位置

Lab 01～06 已得到同步 Engine：调用一次 `submit` 提交请求，调用一次 `step` 推进一个 iteration，并返回本 tick 的 `TokenEvent`；调用 `cancel` 可在 tick 边界退休请求并回收分页 KV。

Lab 07 在其外增加异步服务层：

```text
HTTP 请求
  → 校验模型、messages、max_tokens、temperature
  → 获取 inflight 准入名额
  → ChatRequest
  → EngineChatBackend.stream
  → Request(request_id, prompt_ids, max_new_tokens, eos_token_id)
  → Engine.submit
  → Engine.step（多个请求共享 tick）
  → TokenEvent(request_id, token_id, finish_reason)
  → 按 request_id 路由、累计 token 并解码
  → ChatEvent(delta, finish_reason)
  → 普通 JSON 或 SSE
  → 正常结束 / 客户端断流
  → 关闭 async generator、取消未完成请求、释放准入名额
```

这里有两层不同的“流”：

- `EngineChatBackend.stream` 是 Python 异步生成器，输出与协议无关的 `ChatEvent`。
- FastAPI 的 `StreamingResponse` 是 HTTP/ASGI 响应体，把 `ChatEvent` 编码为 SSE 字节。

保持这两层分离，才能用同一后端同时支持普通响应和 SSE，也才能独立测试请求路由与 HTTP 序列化。

## 单一 Engine owner 与协作式调度

当前 Engine、Scheduler 和 BlockPool 都按单进程、单 owner 设计，没有为并发修改内部字典和页池加锁。因此不能让多个线程随意同时调用 `submit/step/cancel`。

本 lab 使用单事件循环中的协作式并发：

1. 每路 HTTP 请求是独立 coroutine。
2. Engine 调用仍是同步的；一个同步 tick 内没有 `await`，不会被另一 coroutine 插入。
3. tick 之间通过 `await asyncio.sleep(0)` 主动让出事件循环，让其他请求完成提交，让断流等 I/O 事件得到处理。
4. 所有输出都必须根据 `TokenEvent.request_id` 送回所属请求，不能按 `engine.step` 的返回值位置猜请求，也不能使用一个全局文本缓冲区。

在第一次 `step` 前让出事件循环尤其重要。若 A 提交后立刻同步执行完整个生成过程，B 根本没有机会提交，continuous batching 就退化为串行请求。

```text
错误时序：submit(A) → 跑完 A → submit(B) → 跑完 B
目标时序：submit(A) → yield → submit(B) → tick(A+B) → tick(A+B) → ...
```

`sleep(0)` 不是sleep 一段时间，也不是线程锁；它只表示“把当前 coroutine 挂起一次，让事件循环运行其他已就绪任务”。

## 请求状态与输出路由

每路请求至少需要独立维护：

- 唯一的 `request_id`；
- 累计的原始 `token_ids`；
- 已向客户端发送的稳定文本长度或稳定前缀；
- 是否收到终止事件；
- 可选的私有事件队列，用于解耦 Engine tick 与单路 stream 的消费速度。

一个 tick 可以同时返回 A、B 的事件。路由时必须按 ID 分发：

```text
TokenEvent(A, ...) → A 的 token_ids / queue
TokenEvent(B, ...) → B 的 token_ids / queue
```

不能假设返回列表只含当前 coroutine 的事件，也不能把别的请求的事件丢弃。若采用共享 driver 与私有队列，还要明确 driver 的启动、空闲、异常传播和退出条件；若采用各 stream 协作驱动 tick，则要避免重复消费事件和错误地结束仍在运行的请求。

请求 ID 同时用于 Engine 路由、取消和响应标识。必须在提交前生成，不能用列表下标或用户文本代替。

## 为什么不能简单地对每个 token 单独 decode

BPE/SentencePiece token 不等于一个完整字符，一个字节序列也可能跨 token。逐 token 解码容易得到空串或 `U+FFFD` 替换字符。

本 lab 使用“累计 token 解码 + 稳定前缀发 delta”：

```text
累计 IDs: [t1]       → decode 可能尚不稳定 → 暂不发送不完整尾部
累计 IDs: [t1,t2]    → decode 得到完整文本  → 只发送新增稳定部分
```

朴素地执行 `new_text[len(previous_text):]` 有一个陷阱：如果 `previous_text` 末尾含替换字符，而下一轮真实字符把它替换掉，按长度切片会留下错误的替换字符。应区分“已经确认并发送的前缀”和“本轮完整解码结果”，只发布仍然稳定的增量；至少不能急于发送末尾的 `U+FFFD`。

原始 token IDs 仍需保留，不能用解码后的文本反推历史。finish reason 来自 Engine 的终止事件，不由文本是否为空推断。

## async generator 的关闭与取消传播

客户端提前断开时，服务端通常不是收到一个普通业务返回值，而是响应体生成器被关闭或任务被取消。因此清理必须放在 `finally`，不能只放在正常循环结束之后。

完整清理链是：

```text
ASGI 检测 http.disconnect / 响应任务取消
  → 关闭 SSE body generator
  → 关闭 backend.stream async generator
  → backend.stream 的 finally 调用 engine.cancel(request_id)
  → Scheduler 退休请求，Engine.on_retire 释放 BlockTable
  → create_app 的 finally 释放 inflight 名额
```

只 `break` 消费循环不等于关闭生成器。应确保实际调用其 `aclose()`，或使用能保证关闭的异步上下文管理。

`finally` 中也要区分“已经正常退休”和“仍需取消”：重复取消应依赖 Engine 的幂等语义，而不能让清理异常覆盖正常响应。不要只捕获 `Exception` 来完成取消；任务取消和生成器关闭应依靠 `finally` 处理，并且不应吞掉取消信号。

本 lab 的取消点是 tick 边界。同步 `engine.step()` 运行期间不能立刻处理 `http.disconnect`；它返回后，事件循环才有机会执行取消和页回收。

## HTTP 与 SSE 契约

普通响应：

- object 为 `chat.completion`；
- `choices[0].message` 包含 assistant role 和完整拼接文本；
- `choices[0].finish_reason` 与后端最终事件一致。

流式响应：

- Content-Type 为 `text/event-stream`；
- 每帧格式为 `data: <JSON>\n\n`；
- object 为 `chat.completion.chunk`；
- 每帧的 `choices[0].delta.content` 是新增文本，不是累计文本；
- 最后一帧业务 chunk 带 finish reason，之后只发送一次 `data: [DONE]\n\n`。

同一请求的 `id`、`created`、`model` 必须在所有 chunk 中一致。`created` 应在请求开始时生成一次，不能每帧读取当前时间。

校验顺序应先于后端提交：

- 未知模型：404；
- 空 messages、非法 role、`max_tokens` 越界、非零 temperature 等：422；
- 合法但超过 `max_inflight`：429。

`max_output_tokens` 是服务端策略上限，不是请求默认生成量。非法请求不能触碰 backend，否则即使返回 422，也可能已经分配 KV 页。

## inflight 准入与资源释放

`max_inflight` 限制的是正在消费 backend stream 的 HTTP 请求，不等于 Scheduler 的 `max_active`：

- HTTP inflight：接入层限制客户端并发，拒绝返回 429；
- Scheduler active：模型执行层限制每 tick 活跃请求数；
- BlockPool：KV 页资源限制，可能让请求等待准入。

名额应在创建流式响应之前获取，否则响应已经返回 200，再发现超载就无法正确改成 429。名额释放必须覆盖正常结束、验证后异常、客户端断流和任务取消。

HTTPX 的 ASGI transport 会缓冲响应，不能充分证明真实断流清理正确。因此验收使用原始 ASGI `receive/send`，在收到首个 body 后注入 `http.disconnect`。

## 推荐实践顺序

1. **先实现 `EngineChatBackend`**：只运行 `-k engine_backend`，验证 A/B 同 tick、文本不串流和提前 `aclose` 后取消。
2. **再实现普通响应与输入校验**：确保普通 JSON 聚合正确，非法输入不启动 backend。
3. **实现 SSE 和 inflight**：统一响应元数据、delta、finish reason、`[DONE]` 和 429 恢复。
4. **最后验证原始 ASGI 断流**：沿着 body generator → backend generator → Engine.cancel → inflight release 检查完整清理链。
5. 每完成一个阶段，都运行整个 `tests/test_lab07_api.py`，不要只保留当前阶段的测试。

实现时优先画出每路请求的状态及 `try/finally` 边界，再写代码。公开测试通过后，还应检查：空 delta 是否被误当成终止、完成后是否仍调用 cancel、最后一个事件是否既保留文本又保留 finish reason，以及断流后 `engine.active` 是否为空。

## EngineChatBackend 第一次实现复盘

首次定向复验：

```text
tests/test_lab07_api.py -k engine_backend
结果：1 failed
直接原因：stream 第一次产出了 None，测试读取 finish_reason 时触发 AttributeError
```

### 1. 循环条件与初始状态矛盾

`latest_event` 初始为 None，而循环条件要求它非 None，因此循环一次也不会执行，Engine 提交后不会运行任何 tick，最终进入 `yield None`。

这里应先写出三种状态的活性表：

| 当前状态 | 含义 | 是否继续 |
| --- | --- | --- |
| 尚未收到任何事件 | 还没开始输出 | 继续 |
| 收到 finish_reason=None 的事件 | 生成中 | 继续 |
| 收到 finish_reason 非 None 的事件 | 已终止 | 发出最后事件后结束 |

类型注解也应与可空初始值一致：None 是合法的运行时状态，就不能只标注为 `TokenEvent`。

### 2. coroutine 对象不能代替 await 的结果

以下调用都返回 coroutine，而不是同步结果：

```python
lock.acquire()
queue.put(item)
queue.get()
```

把 `lock.acquire()` 放进 `if`，判断的是 coroutine 对象自身的真值；coroutine 对象为真，不代表锁已获取，而且该 coroutine 没有被等待。

Queue 同理：

- `await queue.put(item)` 才会真正执行异步入队；
- `await queue.get()` 才会返回队列元素；
- 直接写 `queue.get().token_id` 是在访问 coroutine 对象的属性，而不是 TokenEvent；
- 若明确使用非等待路径，应使用 `put_nowait/get_nowait`，并处理 `QueueEmpty/QueueFull`，不能把二者混用。

已确认当前环境中 `Lock.acquire`、`Queue.put/get` 都是 coroutine function，而 `put_nowait/get_nowait` 是同步方法。

### 3. `locked()` 不表示“锁属于当前任务”

`lock.locked()` 只说明锁当前被某个任务持有，不说明持有者是谁。若当前任务并未成功获取锁，却在 finally 中根据 `locked()` 调用 `release()`，可能释放其他任务持有的锁。

使用锁时，应让获取、临界区和释放形成明确的所有权关系；`async with lock` 能表达这一关系。若 Engine.step 到所有事件路由之间完全是同步代码，单事件循环本身不会在中途切换 coroutine，此时可以重新评估是否需要异步锁，避免无意义地增加状态。

### 4. 不能丢弃同 tick 其他请求的事件

当前只保留与本地 request_id 相等的事件，然后 break。A 驱动一次 Engine.step 时，如果返回 A、B 两个事件，B 的事件会被直接丢弃，且 Scheduler/Engine 已经推进了 B，无法靠下一次 tick 补回。

正确约束是：一次 tick 的每个事件都必须路由到对应请求的私有状态；当前 stream 只消费自己的状态，但必须完整处理本次 events 列表。

### 5. 流式输出不能在终止后只 yield 一次

当前唯一的 yield 位于整个生成循环之后，即使循环条件修正，也会等最终 finish 后才一次性输出，不符合逐 token 流式语义。提前 `anext` 的测试也要求首 token 在整段生成结束前可见。

应在每个可输出的 TokenEvent 到达时形成 ChatEvent。最后事件可以同时携带文本 delta 和 finish_reason，不需要把二者拆成互斥分支。

此外，若某次 tick 没有当前请求的新事件，不能重复入队上一次保存的 `latest_event`，否则真实 Engine 的混合 prefill/decode 路径可能产生重复 token。

### 6. 当前 decode 不是累计解码

`_get_ready_delta` 只解码本次从队列取出的 token IDs，下一次调用又从空列表开始。简单 ToyTokenizer 下拼接结果可能看似正确，但跨 token UTF-8/BPE 无法依赖这种方式。

每请求必须持久保存累计原始 token IDs，再对累计 IDs 解码，并只发布稳定新增前缀。队列是事件传输工具，不能兼任完整的文本历史。

### 7. 缺少覆盖 yield 挂起点的取消清理

当前 try/finally 只包围 Engine tick，没有包围最终 yield。消费者在 yield 处调用 aclose 时，这段 finally 不是整个 stream 生命周期的清理保障。

应让提交后的主循环和所有 yield 都位于统一的 try/finally 生命周期内，并记录是否正常收到终止事件：

- 正常终止：Engine 已退休请求，不应再次把取消计为一次主动取消；
- 提前关闭或异常：对未完成请求调用 engine.cancel；
- 两条路径都清理 backend 中该请求的队列和累计状态，避免内存泄漏。

测试 ToyEngine 会记录每次 cancel 调用，因此“反正真实 cancel 幂等，所有请求都再 cancel 一次”也不符合当前契约。

### 本轮需要建立的核心认识

异步生成器同时有三条控制流：正常逐事件 yield、正常终止、在任意 yield/await 处被提前关闭。实现不能只证明正常路径最终能返回，还要证明每次 tick 不丢事件、每个事件只处理一次、首个事件及时可见，以及提前关闭一定完成资源回收。

## EngineChatBackend 第二次实现复盘

第二次定向测试已通过 A/B 并发收集和 `peak_active == 2`，说明同 tick 事件路由已接通。当前失败变为：

```text
第一次 anext(C) 得到 CCCC，预期 C
```

这证明输出仍等到 finish 后才统一 yield。逐事件输出修正后，还需要继续验证 aclose 时的取消；当前仍没有覆盖整个生成器生命周期的取消清理。

### running_state 是否需要锁

当前 `_init_and_get_running_state` 的锁体内只有 dict 查询和赋值，没有 await。单事件循环不会在这段同步代码中切换任务，因此这里的锁不是必需的。

而且函数返回 state 后，对 `events/n_generated/status` 的修改都发生在锁外，所以这把锁并没有保护“整个请求状态”的一致性，只保护了初始化查表。

在当前单线程、单事件循环约束下，更简单的纪律是：

- 初始化、路由、Queue.put_nowait、状态修改放在一个无 await 的同步区间；
- 不让多个线程访问该 dict；
- 需要等待的地方才使用 async/await。

如果未来改成多线程访问，不能据此认为 dict 操作仍然足够，也不能把 asyncio.Lock 当跨线程锁；应重新设计 owner 和线程间消息传递。

### Engine 是否需要锁

当前 Engine.step 本身是同步函数，单事件循环不会让两个 coroutine 同时进入它的函数体。若 step 和完整事件路由之间没有 await，step-return 的事件也能作为一次同步临界区完成路由，因此本 lab 不强制需要 engine_lock。

当前锁有两个设计问题：

1. 锁内await `_accumulate_token_event`，并进一步获取 running_state_lock，把本可同步的路由变成嵌套异步临界区，增加调度与锁顺序复杂度。
2. Engine.submit 在锁外，未来 cancel 若也在锁外，那么 engine_lock 并没有统一表达“所有 Engine 操作”的访问纪律。

可以选择统一加锁，但必须明确保护哪些操作、所有访问是否遵守同一把锁，以及为什么临界区需要 await。对本 lab，更简单的方案是保持 Engine 单 owner，submit/step/cancel 都是同步调用，step 后用同步 nowait 路由；async 只用于请求等待和事件循环让出。

### async 污染与状态简化

当前 `_prepare_request/_accumulate_token_event/_get_ready_delta` 主要是为了获取状态锁而变成 async，调用方因此增加大量 await。这是从不必要同步原语传播出来的“async 染色”。纯 dict、list、token 解码等同步计算不必全部改成 async。

`n_generated` 与 init/running 状态目前没有独立业务作用，finish 布尔状态已经能控制退出。建议用有明确类型的每请求状态表达：事件队列、累计 token_ids、已发送稳定文本、finished/finish_reason。这样比任意 string key 的嵌套 dict 更容易检查不变量。

路由应假设请求状态已在 submit 前注册；收到未知 request_id 时静默创建新状态可能掩盖生命周期错误并造成泄漏。

## 如何判断哪些 token 已能稳定解码为字符

### Token 不等于字符

UTF-8 示例：

```text
A  = 41          （1 字节）
你 = E4 BD A0    （3 字节）
```

tokenizer 的 token 是词表条目，可能表示词、子词、空格、特殊标记或字节片段。不能根据 token 数量判断字符数量，也不能假设一个汉字一定对应一个 token。应由 tokenizer 的累计解码结果判断，而不是自己猜词表边界。

### 使用累计 IDs 与稳定前缀

每请求保留：

- 全部输出 token_ids；
- 已经发送的 stable_text（不是上一次包含替换符的完整 decode）。

每次新增 token 后：

1. 对全部累计 IDs 调 tokenizer.decode。
2. 末尾若存在 U+FFFD（`�`），说明尾部可能仍在等待后续 UTF-8 字节，先保留这段尾部。
3. 其余文本作为当前稳定文本；只有它以 stable_text 为前缀时，才能按 stable_text 的长度取得新增 delta。
4. 发出 delta 后，把 stable_text 更新为已确认前缀。

示例：

```text
累计字节       decode       可确认文本    本轮 delta
E4             �            ""           ""
E4 BD          ��           ""           ""
E4 BD A0       你            你           你
E4 BD A0 41    你A           你A          A
```

关键是上次没有发送 `�`，所以之后不需要撤回或覆盖已经发给客户端的字符。若新的稳定文本不再以已发送文本开头，说明前缀假设被破坏，不能继续盲目按长度切片，应将其视为解码状态错误并检查 tokenizer 行为。

空 delta 是合法事件：可能仍在等待 UTF-8 尾部，也可能 token 是被跳过的特殊标记。不能用空字符串判断请求结束；结束只能由 finish_reason 决定。终止时仍需明确最终尾部处理策略，并且无论是否还有文本可发，都要交付 finish_reason。

新增验收测试 `test_engine_backend_holds_incomplete_utf8_until_the_stable_prefix_grows` 将“你”的 E4/BD/A0 三个字节拆成三个 token，期望 delta 依次为 `""、""、"你"、"A"`。

## EngineChatBackend 第三次实现复盘

去掉 engine_lock 和 running_state_lock 后，单事件循环内的同步状态区间更清楚。UTF-8 拆分测试已通过，首个 token 也能及时 yield。

当前公开定向结果为：原并发/取消测试失败，UTF-8 测试通过。取消失败的直接原因是正常完成的 A、B 也进入了无条件 engine.cancel，加上提前关闭的 C，共记录 3 次 cancel，而预期只取消 C。

### 正常退休与主动取消必须区分

`finish_reason != None` 表示 Engine 已经退休请求并释放其内部资源，此时 backend 只需交付最后 ChatEvent 并清理自身 running_state。只有尚未收到终止事件时，提前关闭或异常路径才需要 engine.cancel。

真实 Engine.cancel 对不存在请求返回 False，可能掩盖重复取消；ToyEngine 记录每次调用，使语义错误可见。不能因为操作幂等，就把所有退出都称为主动取消。

清理顺序还应考虑 cancel 自身异常：若希望 running_state 无论 cancel 是否成功都被删除，需要让状态删除拥有独立的 finally 保障。

### 收到终止事件不等于已经交付终止事件

已通过公开 stream 接口复现：

```text
B 消费首 token，暂停在 yield
A 驱动下一 tick，本次事件包含 A 普通事件和 B 的终止事件
backend 正确把 B 终止事件写入 B.state
B 被再次 anext
B 从上次 yield 恢复，在循环顶部看到 finish_reason 后直接退出
B 的最后 token 和 finish_reason 没有交付
```

因此 while 顶部的 finish_reason 不能单独决定“是否还有事件需要输出”。循环需要区分：

- Engine/Scheduler 已终止：不再为该请求提交新计算；
- 当前请求还有待消费事件：仍须继续交付；
- 最后事件已交付：才可以结束 async generator。

这也是私有 events Queue 可以发挥作用的场景：它表达“已路由但尚未交付”的事件，而不仅是堆积 TokenEvent。

新增 `test_engine_backend_delivers_terminal_event_routed_by_a_peer_stream` 固定上述调度顺序，避免普通 gather 测试因任务交错不同而偶然通过。

### 当前 Queue 尚未参与消费

`_accumulate_token_event` 会 put_nowait，但 `_get_ready_delta` 不再 get，因此每 tick 都会 yield 一个 ChatEvent，即使本次 tick 实际只有其他请求的事件。这样会产生没有来源的空 delta 帧，也无法区分“本请求收到 token 但 UTF-8 尚未稳定”与“本请求根本没有新事件”。

后续设计应让 Queue 明确表达待交付事件，或者删除 Queue 并使用等价的明确计数/状态。不能保留一个只写不读的队列来暗示已经实现事件驱动。

## EngineChatBackend 第四次实现复盘

原有三项 backend 定向测试均通过：

```text
3 passed
```

当前改进包括：逐事件 yield、消费私有 Queue、交付 peer 路由的终止事件、正常 return 不再无条件 cancel，以及 UTF-8 稳定前缀输出。

### 不能用 except BaseException 吞掉取消

当前结构：

```text
except BaseException:
    engine.cancel
finally:
    destroy state
```

没有 re-raise，因此 CancelledError 被当作正常生成器结束处理。aclose 测试只观察 engine.cancel 和 active 清理，无法发现任务取消语义已被改变。类似地，KeyboardInterrupt/SystemExit 以及普通内部异常也会被这一宽泛分支捕获并吞掉。

取消语义要求：

- 清理必须执行；
- 取消信号必须继续传播；
- 调用方 await 被取消任务时应观察到 CancelledError，而不是 StopAsyncIteration 或正常结果。

这里不需要捕获异常来决定是否清理。finally 本身就能覆盖正常 return、GeneratorExit、CancelledError 和其他异常；根据 state 是否已收到 finish_reason 决定是否 cancel，然后让原控制流自动继续传播。

新增 `test_engine_backend_task_cancellation_cleans_up_and_propagates`：IdleEngine 永不发事件，取消正在等待 anext 的 Task，要求取消传播、Engine.active 清空且 backend.running_state 清空。

### 后续可优化但非当前阻塞点

- Queue 非空时已用同步 empty 判断，可以直接表达 get_nowait 的非等待语义；当前 await get 在非空队列上能工作，但不是必要 await。
- 路由未知 request_id 时仍会 lazy init 状态；更严格的生命周期可以要求状态在 submit 前注册，避免取消后的异常事件重新创建状态。
- 若多个事件在消费前累积，stable_text 是全部累计文本，首个待交付事件可能携带后续 token 的合并 delta。总文本仍正确，但若要求事件与逐 token delta 严格对应，需要在事件中保存文本快照或在消费时逐个推进 decode 游标。

## 终止 yield、消费者关闭与事件积压

当前四项 backend 测试为 `4 passed`。任务取消通过 finally 条件取消并继续传播，running_state 在退出时删除。

### yield 后的 return 不会立即执行

异步生成器在：

```text
yield final_event
if final_event.finish_reason:
    return
```

的 yield 处暂停。消费者拿到 final_event 时，生产者尚未执行后面的 if 和 return。

若消费者使用普通 async for 且没有 break，循环体会处理完 final_event，然后自动再请求一次 anext；生产者此时恢复，执行 return/finally，消费者收到 StopAsyncIteration 并结束循环。

若消费者：

- 看到 finish_reason 后 break；
- 只调用一次 anext 后不再请求；
- 停止消费但没有 aclose；

则生成器仍暂停在 yield，finally 不保证已经执行。Engine 因终止事件已经退休，但 backend 私有 running_state 可能残留。

消费者应通过 aclose 或 contextlib.aclosing 明确关闭不再消费的生成器。后续 create_app 的 SSE body 在发送 DONE 或提前退出时也必须关闭 backend.stream，不能只依赖 final yield 后面的 return。

若希望正常终止不依赖消费者再次恢复，也可以在最终 yield 之前先构造完整 ChatEvent 并完成正常路径的状态清理；提前关闭和异常仍由 finally 兜底。两种策略都依赖清晰的消费者/生产者关闭契约，不能假定 yield 后代码必然已经运行。

### 提前取消后的 ID 校验

`_accumulate_token_event` 在 lazy init 前先检查 request_id 是否存在，能够避免请求状态删除后又因迟到事件重新创建 running_state。这消除了 zombie state 风险。

当前同步 Engine 在 tick 边界取消，且 step+路由无 await，正常情况下不应出现取消后的迟到事件；该校验是防御性措施。对未来真正异步的 Engine，忽略已取消请求的迟到事件可以是合理策略，但对“本应存在的活跃请求 ID 不存在”则应通过日志或测试暴露，而不是让所有路由错误都静默通过。

### 多事件积压为什么会让 delta 错位

当前状态有两个不同游标：

- 路由时：append token_ids，并把 stable_text 推进到最新累计解码结果；
- 消费时：_get_ready_delta 读取“当前最新 stable_text”，再更新 emitted_text。

Queue 中的 TokenEvent 没有保存它路由时的文本快照。因此若 A 暂停消费，其他 stream 连续驱动两 tick：

```text
路由 e1(token 你)：stable_text = 你，emitted_text = ""
路由 e2(token 好)：stable_text = 你好，emitted_text = ""
Queue = [e1, e2]
```

随后消费：

```text
取 e1：读取最新 stable_text=你好 → delta=你好，emitted_text=你好
取 e2：stable_text 仍是你好     → delta=""
```

总拼接文本仍是“你好”，没有丢失或重复；但 e2 的文本被提前挂到了 e1。若 e2 是终止事件，就会出现“前一个普通事件携带最后文本，最终事件只有 finish_reason”。这在 OpenAI 风格 SSE 中可以是合法帧形，但事件与 token 的对应关系已不再确定。

UTF-8 场景更明显：

```text
e1 路由时只有半个汉字，stable_text=""
e2 路由后得到“你”，stable_text="你"
e3 路由后得到“你A”，stable_text="你A"
若三个事件都积压，消费 e1 时会直接得到“你A”
```

也就是说，输出由事件到达时的稳定前缀，变成了消费时的最新稳定前缀；同一 token 序列可能因调度时机不同而得到不同的 delta 分配。

一种整理方向是在路由时就把 TokenEvent 转为待交付 ChatEvent：使用“已入队稳定文本”游标计算本事件自己的 delta，再把 ChatEvent 放入 Queue。消费者只负责 yield，不再读取更晚事件的 stable_text。另一种方向是在消费时按事件对应的 token 前缀逐个 decode。若决定允许合并帧，则应明确合并规则并相应调整逐事件测试，而不是让调度偶然决定合并。

此外，Queue 无界时，慢消费者的待交付事件可能持续增长。本 lab 的 max_tokens 会限制单请求规模，因此不是当前阻塞问题；生产系统还需背压或集中 driver 策略。累计 IDs 每次从头 decode 的总工作量也可能随输出长度呈平方级增长，这是后续增量 detokenizer 的优化点。

## 积压问题的职责拆分：不需要 Queue.peek

第五次重构将 Queue 独立为 event_queue dict，但 Queue 中仍是 TokenEvent，消费时仍读取最新 stable_text，因此尚未改变 delta 归属时机。原有四项测试继续通过。

这里应拆成三个独立问题：

1. Unicode 完整性：累计 decode 后暂时保留末尾 U+FFFD；
2. 事件 delta 归属：在事件入队时记录它自己的稳定文本增量；
3. 待交付传输：Queue 保存已经构造完整的 ChatEvent，或 TokenEvent 与文本快照。

Queue.peek 只能观察队首，无法恢复每个历史事件入队时的 stable_text。问题不在 Queue 缺少查询能力，而在入队时丢掉了时间快照。

### 入队时计算 delta

每请求除 emitted_text 外，可概念上维护 queued_text：表示已经分配给 Queue 中事件的稳定文本。每个 TokenEvent 到达时：

```text
更新累计 token IDs
计算最新 stable_text
校验 stable_text 以 queued_text 为前缀
本事件 delta = stable_text 去掉 queued_text 前缀
queued_text = stable_text
Queue.put_nowait(ChatEvent(delta, event.finish_reason))
```

消费者只需 get_nowait 后原样 yield。即使事件积压，后续消费也不会重新解释旧事件。emitted_text 不再需要兼任 delta 计算游标。

UTF-8 仍由原累计 decode 规则处理：

```text
e1 stable=""    → 入队 delta=""
e2 stable="你"  → 入队 delta="你"
e3 stable="你A" → 入队 delta="A"
```

无论消费者何时恢复，三个事件的内容都固定。不需要 tokenizer 额外提供“是否完整 Unicode”查询，也不需要 Queue.peek。

新增 `test_engine_backend_backlogged_events_keep_their_own_deltas` 固定让 B 的第二、第三事件在 B 暂停时依次入队，再恢复 B，要求二者各自携带自己的 B，而不是第二个事件携带 BB、最终事件为空。

### 本次重构的其他观察

- event_queue 与 running_state 两个 dict 必须始终同步创建和删除；将 Queue 放回每请求状态或使用 typed RequestState 更容易维护该不变量。
- event_queue 声明为 Queue[TokenEvent]，但 `_init_running_state` 返回注解写成 Queue[ChatEvent]，二者目前不一致；`_destroy_running_state` 的 Queue[dict] 也不是实际元素类型。
- `state["finished"] = event.finish_reason is not None` 会让迟到普通事件把 True 重置为 False。正常序列不会这样，但状态语义更适合只在收到终止事件时从 False 转为 True。

## EngineChatBackend 当前验收结论

最新复验为 `5 passed, 4 deselected`。Queue[ChatEvent] 与 queued_text 已将 delta 固定在事件入队时，新增积压测试通过；任务取消仍传播 CancelledError，提前关闭和正常结束的资源路径均保持通过。

Backend 可以进入 HTTP 集成阶段。finished 已改为只在终止时单向置 True，冗余 stable_text 已删除，返回值和 token_ids 类型注解已规范化，finally 已处理 state=None。整理后复验仍为 `5 passed`。

最终 yield 后是否恢复仍由消费者决定，create_app 必须保证关闭 backend stream。
