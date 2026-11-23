# Lab 07 · 流式服务与取消

**状态：已完成。** Lab 07 为 `23 passed`，Lab 01～07 回归为 `61 passed, 1 skipped`。普通请求重复被拒绝、SSE 断流后关合失败的计数回归均已通过。见[父类退出后关闭 body 与最新复验](#父类退出后关闭-body-与最新复验)。

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

## create_app 知识导入：FastAPI、SSE 与准入

### FastAPI app factory 与路由

本项目使用应用工厂：

```python
def create_app(backend, *, max_inflight=8, max_output_tokens=128):
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(...):
        ...

    return app
```

`FastAPI()` 创建 ASGI app；`@app.post` 注册 HTTP method + path 与 endpoint 的映射；endpoint 返回 dict 时 FastAPI 自动编码为 JSON。每次 create_app 都产生独立 app 和独立 inflight 状态，避免不同测试共享计数。

backend、max_inflight、max_output_tokens 通过闭包提供给 endpoint，不需要全局变量。若 endpoint 修改外层 inflight 整数，需要使用 nonlocal；只读取则不需要。

当前 api.py 将 FastAPI 放在 TYPE_CHECKING 下，仅用于静态类型。create_app 真正运行时必须能取得 FastAPI、Body/HTTPException 和 StreamingResponse 等运行态类，因此需要增加正常导入或函数内导入。

### 请求体解析与校验

FastAPI 常见两种方式：

1. Pydantic BaseModel：声明 schema，自动解析和生成 422，适合稳定复杂模型；
2. dict/Request：自行检查字段，适合本 lab 学习精确状态码和动态 max_output_tokens。

无论采用哪种方式，都要在调用 backend.stream 或提交 Engine 前完成：

- model 必须等于 backend.model，否则 404；
- messages 必须是非空列表，每项 role 为 system/user/assistant，content 为字符串；
- max_tokens 为整数且在 1..max_output_tokens；
- temperature 只能为 0；
- stream 若出现应为布尔值，缺省为 False；
- 其他格式错误为 422。

Python 中 bool 是 int 子类，`isinstance(True, int)` 为 True。若字段要求真正的整数，应额外排除 bool。非法请求即使最终返回 422，也不能已经创建 backend stream 或分配 Engine 资源。

HTTPException(status_code=..., detail=...) 可显式产生 404/422/429。FastAPI 对 JSON 解析失败和声明 schema 校验失败通常自动生成 422。

内部 ChatRequest/Message 与 HTTP JSON 是不同层：HTTP 层解析后构造不可变内部对象，backend 不需要知道 Pydantic、HTTPException 或 ASGI。

### 普通 chat.completion 响应

普通响应消费同一 backend.stream，累计所有 event.delta，并保留最终 finish_reason：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "toy",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "你好"},
    "finish_reason": "stop"
  }]
}
```

id 和 created 在请求开始时生成一次。HTTP 响应 id 与 Engine 内部 request_id 可以不同，因为它们属于不同层；同一 HTTP 请求的所有响应帧必须复用同一个 HTTP id/created/model。

普通和流式响应必须消费同一个 backend 契约，不能分别实现两套生成逻辑。使用 contextlib.aclosing 包住 backend.stream，保证聚合异常、任务取消或提前退出时关闭生成器。

### StreamingResponse 与 SSE

StreamingResponse 接收 body iterator/async generator，把每次 yield 的字符串或字节编码为后续 http.response.body：

```text
endpoint return StreamingResponse(body(), media_type="text/event-stream")
→ 先发送 200 和响应头
→ 每次 body yield 发送一段响应体
```

SSE 业务帧：

```text
data: <JSON>\n\n
```

结束标记：

```text
data: [DONE]\n\n
```

JSON chunk 的 object 为 chat.completion.chunk，choices[0] 包含 index=0、delta={"content": event.delta} 和本事件 finish_reason。最后业务事件应同时保留 delta 与 finish_reason；[DONE] 只发送一次。空 delta 仍可有合法 chunk，它不代表结束。

json.dumps 可将中文转义为 Unicode escape，消费者 json.loads 后文本相同；ensure_ascii=False 则直接保留 UTF-8 中文，二者都符合 JSON。

状态码与校验必须在返回 StreamingResponse 前决定。响应头一旦以 200 发出，body 内部才发现过载或参数错误，就不能再可靠地改成 429/422。

### inflight 计数与所有权

本 lab 的 429 是立即拒绝，不是排队等待。单事件循环中，同步的“检查 inflight + 增加计数”没有 await，可作为原子临界区，不需要 semaphore/lock。

Semaphore.acquire 通常用于等待名额，而测试要求超载请求立刻 429，因此直接使用计数更直观。

普通响应：

```text
校验 → 占用名额 → try 聚合完整响应 → finally 释放 → return
```

流式响应不能在 endpoint return 的 finally 中释放：

```text
错误：endpoint try → return StreamingResponse → finally 立即释放
此时 body 尚未消费，实际并发请求数被低估
```

让 body generator 持有名额，能覆盖它已经开始迭代且能够退出的路径：

```text
endpoint 校验并占用名额
→ 构造 body
→ body try/finally 负责消费期间持有并在结束时释放
→ endpoint return StreamingResponse(body)
```

完整的名额作用域还要覆盖发送响应头到首次迭代之间，以及外层发送期间的取消。参见[准入、断流与取队列改动复查](#准入断流与取队列改动复查)。

### 关闭传播与原始 ASGI 测试

SSE body 中使用 aclosing 包住 backend.stream。取消落在 body 正等待下一条 backend 事件时，这条清理链可以在当前请求中执行：

```text
StreamingResponse 检测 disconnect/取消 body task
→ body 的 async with/finally 退出
→ backend.stream.aclose
→ backend 对未完成请求 cancel 并删除状态
→ body finally 释放 inflight
```

当前正常路径先发送末个 JSON 帧和 DONE，随后 Starlette 再拉一次 body，恢复到 `return` 后退出 `aclosing`，显式关闭仍挂在终止 `yield` 的 backend。提前停止消费时，要由消费方在自己的退出路径显式关合外层生成器。

HTTPX ASGITransport 的 client.post 会收集完整响应，适合检查最终 SSE 文本，不足以证明真实首帧后的断流。原始 ASGI 测试直接调用 app(scope, receive, send)：receive 首次给 http.request，首帧 hello 发出后给 http.disconnect；send 观察 response.body。验收要求 backend closed 且之后名额恢复。

### 推荐实现与验收顺序

1. 路由、请求解析和校验：先通过 test_http_validation_does_not_start_the_backend；
2. 普通响应与 SSE：通过 test_chat_completion_and_sse_share_the_same_text_and_finish_reason；
3. inflight 计数：通过 test_inflight_limit_recovers_when_request_finishes；
4. 关闭传播：通过 test_asgi_disconnect_closes_stream_and_restores_admission；
5. 最后运行整个 Lab 07 和 Lab 01～06 回归。

## create_app 第一次实现复盘

首次全文件复验：`6 passed, 3 failed`。Backend 和已有 HTTP 校验测试通过，普通/SSE 响应、inflight 和断流尚未通过。

### 普通 JSON 的编码与 schema

`JSONResponse(json.dumps(data))` 先把 dict 转成 JSON 字符串，再由 JSONResponse 把这个字符串编码成 JSON string，形成双重编码。客户端 response.json() 得到 str 而不是 dict，因此后续 `data["object"]` 报 string indices must be integers。

应让 JSON 序列化只发生一次：传 dict 给 JSONResponse，或直接从 endpoint 返回 dict。

普通 choices[0].message 还需是 `{"role":"assistant","content":...}`，不能只是累计字符串。created 应使用 Unix 秒；time.time_ns()//1000 是微秒，不是 OpenAI 风格秒时间戳。

### StreamingResponse 的调用对象和 SSE 帧

body 是 async generator function，body() 才是 async iterable。StreamingResponse(body) 会让 Starlette 把 function 当同步 iterable，出现 TypeError: function object is not iterable。

还需显式 media_type="text/event-stream"，每个业务帧以 `data: ` 开头，DONE 为 `data: [DONE]\n\n`。chunk 应与普通响应一样携带 model。

### inflight 的边界和释放位置

`if inflight > max_inflight` 在 inflight==max_inflight 时仍放行，正确拒绝边界应从等于上限开始。当前 max_inflight=1 的第二请求因此进入 SlowBackend 并等待，最终测试超时。

外层 endpoint finally 在 return StreamingResponse 时立即释放名额，不能覆盖 body 的实际消费期。流式名额应由 body generator 的 finally 持有和释放；普通聚合响应则可由 endpoint finally 释放。

### 现有校验通过不代表非法输入覆盖完整

直接索引 payload["model"]、payload["messages"]、payload["max_tokens"]，缺少键会 KeyError；messages=None、字符串或元素 None/{} 会在 len、迭代或 message["role"] 处产生 TypeError/KeyError，成为 500。

model 非字符串属于格式错误，应与未知模型字符串 404 区分。stream=None 和 temperature=None 当前被跳过校验；False==0 使 temperature=False 被当作合法零值。这些可选字段若出现，也应验证其声明类型。

新增 `test_http_validation_rejects_missing_and_wrong_shaped_fields` 覆盖缺少必需键、model/messages 错误类型、消息元素结构、bool max_tokens，以及 None/bool 可选参数。测试关闭 ASGI app exception 透传，以便观察服务端实际状态码并断言统一为 422。

## create_app 第二次实现复盘

最新全文件结果：`9 passed, 1 failed`。普通 JSON、SSE、inflight、ASGI 断流均已通过。

唯一当前失败是缺少 model 字段返回 404。需要区分：

| 输入 | 语义 | 状态码 |
| --- | --- | --- |
| 缺少 model | 请求格式错误 | 422 |
| model=None | 字段类型错误 | 422 |
| model=非字符串 | 字段类型错误 | 422 |
| model="missing" | 格式合法，但模型资源不存在 | 404 |

原 `test_http_validation_does_not_start_the_backend` 中 `"missing" -> 404` 仍通过。新增校验测试只把缺失/None/错误类型归入 422，没有改变未知字符串的语义。实现应先做存在性和类型校验，再做模型查找，不能用一个组合条件把所有失败都映射为 404。

后续仍需注意：

- temperature 当前仅检查非 None 且 ==0；False==0，因此 temperature=False 仍会被接受。新增用例在 model 分支修正后会继续检查到这一项。
- req_created=time.time() 是浮点秒，而 normal/stream helper 注解为 int；OpenAI 风格 created 通常使用整数 Unix 秒，可显式取整。

## create_app 第三次代码复盘

当前仍为 `9 passed, 1 failed`，失败已推进到 temperature=False：bool 是 int 子类，因此仅判断 isinstance(value, int) 或 isinstance(value, float) 不能排除布尔值。 temperature 应像 max_tokens 一样显式排除 bool。

Lab 01～06 回归保持 `38 passed, 1 skipped`。

响应测试已补充元数据契约：普通响应 created 为 int；同一次 SSE 的所有 chunk 只有一个 id、一个 created，model 全为 toy，首帧 created 为 int。当前 time.time() 浮点值会被该测试识别。

### 普通响应的非阻塞优化

当前 normal_chat_completion 先把全部 ChatEvent 收集为 list，再第二次遍历累计文本，并使用 events[-1] 取终止原因。backend 契约保证至少有终止事件时功能正确，但可以简化为单次 aclosing + async for：

- 逐事件把 delta append 到文本片段列表；
- 遇到 finish_reason 时保存；
- 最后 join 片段；
- 避免额外事件列表、二次遍历和 events[-1] 空列表风险；
- aclosing 保证聚合异常或任务取消时关闭 backend。

content 反复使用不可变字符串 += 可能产生重复拷贝；片段列表 + join 更适合累积较长文本。以上属于清晰度和资源使用优化，不应优先于 temperature/created 契约修正。

## 上一版复验与代码审查

Python 3.12.7、Starlette 1.10.0、Uvicorn 0.54.0，使用本地虚拟环境。两条命令分别得到 `10 passed` 和 `48 passed, 1 skipped`，跳过的是可选真权重测试。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_lab07_api.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q --ignore=tests/test_lab08_bench.py
```

当前实现已经对齐 404、422、429 契约，普通与流式响应共享后端，同流元数据稳定，入队时固定每条 delta，终止事件携带最后一段内容。HTTP 准入计数与引擎的 `max_active`、KV 页准入各自约束不同资源。待修的优先级如下。

| 优先级与位置 | 现象及原因 | 练习方向 |
| --- | --- | --- |
| 高 · [api.py:217](../src/tinyserve/api.py#L217)、[api.py:253](../src/tinyserve/api.py#L253) | endpoint 已取得名额，Starlette 先发送响应头。原始 ASGI 2.3 探针在发送头时阻塞并断开，body 从未进入 `try`，后续请求持续 429。此时没有 Engine 工作或页泄漏，泄漏的是 HTTP 名额。 | 让覆盖整个响应调用的外层承担释放，从发送响应头前持有至调用返回或抛错。使释放职责唯一，移除与该职责重叠的 body 释放。给尚未启动的生成器调用 `aclose()` 也不会执行其函数体内的 `finally`。 |
| 高 · [api.py:253](../src/tinyserve/api.py#L253)～[api.py:279](../src/tinyserve/api.py#L279) | body 已 `yield` 首块时，框架在 body 外执行 `await send`。原始 ASGI 2.3 探针在此处取消发送，2.4 探针令 `send` 抛 `BrokenPipeError`；两者在 ASGI 调用退出时都仍有 Engine 请求和保留的页，紧接着请求 429。此次运行在事件循环继续调度生成器收尾后恢复 200。 | 在消费 body 的外层 `finally` 显式等待 `body.aclose()`，再保证名额归还。顺序为结束响应体、关闭后端、释放名额，关闭抛错时仍要释放名额。AnyIO 已取消作用域中需要异步等待的短清理可用有边界的 shield。 |
| 中 · [api.py:130](../src/tinyserve/api.py#L130)、[api.py:135](../src/tinyserve/api.py#L135)、[engine.py:95](../src/tinyserve/engine.py#L95) | `max_tokens` 合法时，渲染后的 prompt 加输出预算仍可能永远放不进页池。6 页 × 2 token 的探针发送 13 token prompt，流已经发出 200，再由 `Engine.submit` 抛出 `ValueError`，没有 SSE 业务帧，下一条合法请求仍获 200。 | 在发送响应头前做可预测的、无副作用的输入和容量校验，并复用 render 后的 IDs。把无法容纳的请求和暂时无空闲页的请求区分开，前者可映射为 422，后者继续遵循 Engine 等待准入策略。 |
| 中 · [api.py:138](../src/tinyserve/api.py#L138)～[api.py:150](../src/tinyserve/api.py#L150) | 检查队列为空后经过 `sleep(0)`，其他流可能已经把本流事件投递进来。两个流各拉一次首 token 的探针中，A 的 tick 给 A/B 投递首 token，B 恢复时又先跑了一个 tick，再消费已经就绪的首 token。顺序正确，但 B 的首帧多等了一轮同步计算。 | `sleep(0)` 返回后重查本流队列，有现成事件就回到取事件路径。 |
| 优化 · [api.py:96](../src/tinyserve/api.py#L96)～[api.py:113](../src/tinyserve/api.py#L113) | 每步重新解码所有累计 ID，长输出累计解码工作量可能接近平方级。去掉末尾 U+FFFD 是启发式，合法的字面 `�` 或最后仍残缺的字节都需要明确策略。按旧长度切片依赖解码结果延长已承诺的前缀。 | 保留玩具实现时，记录前缀假设并决定终止时的尾部策略。若 tokenizer 暴露合适的字节级接口，再考虑真正的增量解码。 |

原始 ASGI 探针使用小模型、真实 `Engine` 和每次新建的单名额 app。6 页中的 3 页在首 token 后仍被请求占用。启动前断流保持 6 页空闲，后续名额不恢复。发送中断流时即刻只有 3 页空闲，探针让事件循环继续运行后恢复 6 页。既有断流用例在首块送出后让后端挂在等待下一条事件的位置，覆盖的是内层正在执行的取消路径。

## 从一次 POST 逐帧推演

以探针的 mock tokenizer 和真实 toy Engine 为例。prompt 有 3 个 token，`max_prefill_chunk=2`，输出限额 4，忽略 EOS，四个输出 ID 都被探针显示为 `B`。

1. FastAPI 读取 ASGI `http.request`，调用 endpoint 校验并占用 HTTP 名额，生成对外的 `chatcmpl-…` 和 `created`。同步 helper 调用 `body()` 得到异步生成器并构造响应。Starlette 先发 `http.response.start`（200），再调用 body 的 `__anext__`。
2. body 创建 backend 生成器并进入 `aclosing`。内层 `async for` 第一次拉取时，backend 才 render prompt、生成内部 ID、初始化状态、`Engine.submit`。此时请求进入等待队列。
3. 队列空时 `await asyncio.sleep(0)` 主动让出循环，其他 HTTP 请求有机会完成入队。恢复后同步 `Engine.step()`，同 tick 的所有输出按内部请求 ID 路由。tick 同步段里 tokenizer、forward、采样、入队都由当前任务执行。

| Tick 或消费动作 | 引擎与后端 | ASGI / SSE |
| --- | --- | --- |
| #1 · prefill 第一段 | 处理 prompt 的前 2 个 token，只更新 KV 和 processed 游标。无输出事件。 | 无业务帧，body 仍在等待首个 `ChatEvent`。 |
| #2 · prefill 最后一段 | 处理 prompt 最后 1 个 token，以最后位置 logits 采样首个输出 token。计算并入队 `ChatEvent("B", None)`。 | body 被唤回，序列化并 `yield` 首个 `data: <JSON>\n\n`，Starlette 随后 `await send`。 |
| #3、#4 · decode | 各消费上一轮生成的 token，分别采样一条新 token。 | 按顺序发送第二、第三条 JSON 帧。 |
| #5 · decode 触及长度上限 | scheduler 先退休请求并回收页，再返回带 `length` 的最后一个 `TokenEvent`。backend 标记 `finished` 并投递末个 delta。 | 发末个 JSON 帧，仍携带 `content="B"` 与 `finish_reason="length"`。遇 EOS 则优先标记 `stop`。 |
| 框架再拉 body | body 从末个 JSON `yield` 之后恢复，看到终止原因。 | `yield "data: [DONE]\\n\\n"`，Starlette 再发送一次。 |
| 框架再拉 body | body 从 DONE 之后恢复并 `return`，`aclosing` 显式关闭还挂在最后一次 `yield event` 的 backend。后端删状态，body `finally` 归还 HTTP 名额。 | `http.response.body` 空字节、`more_body=False`，响应传输结束。 |

真实 tokenizer 可能在输出字节尚未拼成完整字符时产生空 delta。本实现仍送 JSON 帧。累计字节 `e4 bd a0` 的三个输出 token 可以对应 `""`、`""`、`"你"`。SSE 客户端按空行组装消息，传输层可能拆分或合并这些字节。JSON 里的 `id` 属于业务数据，协议的 `id:` 行承担 SSE 的事件游标。当前协议用 `[DONE]` 标记业务结束，ASGI 用 `more_body=False` 标记响应体结束。 [SSE 规范](https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream) · [ASGI HTTP 规范](https://asgi.readthedocs.io/en/latest/specs/www.html)

断流时关注当时的栈位置。若取消落在 backend 的 `await asyncio.sleep(0)` 或其它内部等待，取消向外展开，backend 删除状态并 `Engine.cancel`，最后 body 归还名额。若取消落在框架的 `await send`，body 保持挂在 `yield`，需由框架层的消费方显式关合。若同步 `step()` 正在运行，事件循环要等这个 tick 返回后才能处理取消。

## 准入、断流与取队列改动复查

### 名额从通过准入开始持有

新版把流式 `inflight += 1` 放在第一条 `ChatEvent` 到达之后，把普通请求的占用放在 `await normal_chat_completion` 前。普通路径的检查和增加之间没有挂起点。流式路径的检查和增加之间可能已响应 200、提交 Engine、等待准入、执行多轮 prefill。通过准入的请求在这段期间也占服务容量。

| 最新探针 | 实测 |
| --- | --- |
| ASGI 2.3，发送响应头时断开 | body 未进入，计数保持 0，6 个 KV 页全空闲；上一版的这条泄漏路径已消除。 |
| body 等待第一条事件时断开 | `finally` 仍减 1，计数变成 -1。进入 `async for` 后、首事件前发生异常或空迭代也走未加先减的路径。 |
| `max_inflight=1`，两条请求都卡在 backend 首事件前 | 两条都进入 backend，计数仍为 0。释放等待后两条都收到 200。 |
| ASGI 2.3，首块 `send` 挂起时收到断流 | ASGI 返回时计数为 1、6 页中只有 3 页空闲、下一条请求 429。该次探针强制 GC 并让事件循环执行收尾后恢复 6 页和 200。 |
| ASGI 2.4，首块 `send` 抛 `BrokenPipeError` | Starlette 抛 `ClientDisconnect`，当时计数为 1，随后异步生成器收尾才回收资源。 |
| backend 的 `aclose()` 抛 `RuntimeError` | `await event_stream.aclose()` 后面的递减被跳过，计数仍为 1。 |

让检查和占用成为同一段同步操作，发生在 200 发出和后端工作开始之前。用覆盖整个 `StreamingResponse.__call__` 的作用域持有名额。这一层在发送响应头之前就已经进入，也能等 Starlette 的迭代任务退出后关合它的 body。清理需要两个独立的保证，内层尝试 `await body.aclose()`，外层归还这次成功取得的名额。需要屏蔽取消时，只保护有限的清理范围。

### 先找取消落在谁的调用栈上

Starlette 拉取 body 时，body 正在拉取 backend。若此时取消，异常沿这条调用链展开，body 内的 `finally` 有机会运行。Starlette 已获得一个 chunk 并开始 `await send` 时，body 与 backend 都停在各自的 `yield`。异常发生在迭代器消费者里，需由持有 body 的外层显式关合，依次触发 body 关闭 backend，backend 清理未完成的引擎请求。

原实现已经使用 `async for`。这次改的是把 `aclosing` 的退出行为展开成 `try/finally` 和 `await event_stream.aclose()`，两者负责的仍是同一个内层生成器。SSE 可能在首次进入 body 前断开，这种情况下关闭该异步生成器也不会执行它尚未进入的 `finally`。 [Python aclosing](https://docs.python.org/3.12/library/contextlib.html#contextlib.aclosing)

### API 和 Engine 通过后端契约衔接

`create_app` 通过 `ChatBackend` 使用业务接口。服务若要对可预测的拒绝返回 HTTP 422，可在响应头前调用一个后端 `prepare` 接口。具体 backend 渲染 prompt，并复用 Engine 的请求容量检查，让准备结果携带已渲染的 IDs，供实际 stream 提交。后端把可预期拒绝表达为稳定的业务错误，HTTP 层将其映射为状态码。页大小、总页数和当前分配由 Engine 持有。暂时缺少可用页仍由 Engine 调度等待。

### 只在本流队列确实为空时推进 tick

新版第二次 fetch 消除了一个首事件延迟。两个并发请求各拉一次时，探针只执行了一次 tick，两条首事件都能返回。剩下的问题在 `yield` 后面的续跑位置。B 在第二次 fetch 的 `yield` 处暂停，A 独自推进 tick 2 和 3，给 B 积压了两条。B 再拉一条时会从 `yield` 后恢复，先执行 tick 4，再从下一轮取已经排队的事件。统一两个 `yield` 位置，让有事件的路径在交付后回到循环顶部。只有第一次 fetch 缺失才 `sleep(0)`，第二次仍缺失才执行 `step()` 并按 ID 路由。

## 新增回归用例与最新复查

最新实现把流式计数移到 body 入口，加入第二次队列检查并统一到一个 `yield`。在 backend 等待首事件期间会持有名额，取消后能恢复准入。有积压时，由另一条流推进并入队的事件也能全部交付，消费方无需额外推进 tick。已有的 backlog 用例增加了 tick 数不变的断言。

| 覆盖场景 | 结果与待修方向 |
| --- | --- |
| chunked prefill 首 tick 无输出 · [测试](../tests/test_lab07_api.py#L568) | 真实 toy Engine 的第一轮仅写 KV。当前分支调用 `step()` 后直接 `queue.get_nowait()`，抛 `QueueEmpty`。 |
| 当前 tick 只给其它请求出 token · [测试](../tests/test_lab07_api.py#L581) | `max_active=1` 时 B 驱动 A 的终止 tick，B 的队列仍空，也抛 `QueueEmpty`。完成路由后回到循环顶部，让队列状态决定下一步。 |
| 在响应头或首事件处等待 · [测试](../tests/test_lab07_api.py#L445) | 等首事件时的拒绝、断流后的恢复已通过。发送响应头期间 backend 还未启动，另一个请求仍能通过，因为检查和占用分别处在 endpoint 与 body。让同一段同步逻辑在响应开始前检查并取得名额。 |
| 发送首个 chunk 中途断流 · [测试](../tests/test_lab07_api.py#L493) | ASGI 2.3 的 `http.disconnect` 和 2.4 的 `BrokenPipeError` 都能复现响应返回时 Engine 仍有工作。测试在响应返回后、下一次挂起或 GC 收尾之前检查请求和页是否释放。 |
| stream 创建或关闭失败 · [测试](../tests/test_lab07_api.py#L531) | 创建调用发生在 `try` 之前，关闭异常则会跳过后面的名额归还。两种情况下下一条合法请求都得到 429。清理作用域需覆盖创建，且关闭失败时仍归还成功取得的名额。 |
| 请求永远超出 KV 总容量 · [测试](../tests/test_lab07_api.py#L599) | 新验收将 prompt 与合法输出预算合计超出总容量视为无效输入，沿用 Lab 的 422 契约，同时验证下一条能恰好放下的请求通过。当前普通请求得到 500，SSE 已发出 200。建议通过后端准备接口传递类型明确的拒绝，由 HTTP 层映射状态码。 |

可先修两条 `QueueEmpty`，再一起处理准入和响应退出，最后处理永久容量校验。清理时可顺手删除已经没有调用者的 `_fetch_event`。测试使用本地 toy 权重和原始 ASGI 调用，依靠事件同步把取消落到指定位置。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_lab07_api.py --tb=short
```

## stream 空 tick 复验与下一步

内层 `while queue.empty()` 让同一条流可以推进多轮，直到它自己的队列有数据，解决了两条 `QueueEmpty` 回归。当前没有挂起点把退出内层循环和 `get_nowait()` 分开，所以单事件循环下这里的取队列操作安全。

本版漏掉了 `sleep(0)` 之后的第二次检查。A、B 同时等待首事件时，A 恢复后运行 tick 1，把两条首事件都入队。B 恢复时，本地队列已经有数据，但代码直接运行 tick 2，随后才读取 tick 1 已经生成的首事件。新增[并发首事件回归](../tests/test_lab07_api.py#L619)实际观察到 2 个 tick，预期 1 个。保留内层循环并在让步之后重查本流队列，只在仍为空时推进 tick。

下一项练习是 HTTP 流式响应的资源生命周期。请求通过 HTTP 准入时获取名额，覆盖发送响应头、迭代响应体、发送数据和退出清理。当前在 endpoint 检查、在 body 首次运行时占用，响应头发送处的挂起能让第二条请求越过上限。可保持 FastAPI 路由和 Starlette 发送逻辑，把检查及占用合成同步操作，由响应对象的 `__call__` 持有名额。外层应先等待 Starlette 的发送任务退出，再显式 `aclose()` 它的 body。body 自己负责关闭 backend。无论 body 尚未启动、处在 `yield`、创建 backend 抛错或关闭抛错，响应层都要归还成功取得的名额一次。两类资源用各自的清理保证托底。

先验证名额、断流及 backend 异常的六个测试参数，再处理普通与流式请求对永久容量拒绝的两条 422 验收。

## 容量校验复验与扩展流式响应

Backend 让出事件循环后再次确认队列状态，并发两条流只需一个 tick 就能拿到各自首事件。HTTP 入口在发送响应头前调用 `ChatBackend.is_applicable()`，然后把永久超容量映射为 422。Engine 以总页数判断请求能否容纳，调度准入依据当前空闲页数。`Engine.submit()` 复用容量校验。普通与 SSE 的超容量及恰好放下两组边界均通过。

`Protocol` 声明结构契约，实际调用需要每个后端自己提供该成员。测试中正常受理的简易 backend 复用 `AcceptingBackend.is_applicable()` 返回 `(True, "")`。专门验无效 HTTP 输入的替身让 `is_applicable()` 和 `stream()` 都抛断言，保证无效 payload 在 HTTP 层就被拦住。用真实 Engine 的容量用例继续验证实际计算。当前成功请求会在 `is_applicable` 和 `stream` 各调用一次 `_build_request`，重复渲染 prompt 及生成内部 ID。可在响应清理完成后再考虑复用一次准备的结果。

FastAPI 的路由返回一个 `Response` 实例，随后通过 `await response(scope, receive, send)` 执行它。继承 `StreamingResponse` 并覆盖异步 `__call__`，就能在项目里扩展此调用。调用 `await super().__call__(scope, receive, send)` 复用 Starlette 的响应头、迭代、发送和断流逻辑。也可以让一个继承 `Response` 的包装类持有原版 `StreamingResponse`，在自己的 `__call__` 内调用它。普通可调用对象直接从路由返回时可能会被 FastAPI 当作数据序列化。 [FastAPI 直接返回 Response](https://fastapi.tiangolo.com/advanced/response-directly/#return-a-response)

建议先用子类。在调用父类前原子检查并占用名额，超限时直接发送 429。保存明确可关闭的 body 生成器，让 `aclosing(body)` 包住父类调用。外一层 `finally` 归还已取得的 HTTP 名额。body 本身负责关闭 backend。这样响应头发送中断、body 挂在 `yield` 时断流、backend 创建抛错和关闭抛错，都由覆盖相应边界的清理路径处理。响应关合后再归还名额，名额的归还仍会在关合抛错时执行。

本轮五项失败分别是发送响应头时的超额准入、发送首块时的两种 ASGI 断流，以及 backend 打开和关闭异常。用下列命令可聚焦响应生命周期的六个参数化用例，其中等首事件的准入已通过。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_lab07_api.py \
  -k 'holds_admission or during_send or failure_restores'
```

## 父类退出后关闭 body 与最新复验

原验收 Lab 01～07 共 `59 passed, 1 skipped`，跳过的是需要本地 Qwen3 快照的可选测试。当前子类在调用 Starlette 之前检查并占用名额，保存传给父类迭代的同一个 `body()` 生成器，并在父类调用退出后显式关闭它。上一轮的响应头准入、首块断流和 backend 打开或关闭故障测试都通过。

### 首块发送时断流的模拟

用真实 toy Engine、总计 6 个 KV 页、每页 2 token，输入 2 个 prompt token，`max_tokens=4`。在原始 ASGI 调用中，于首块进入 `send` 时制造断流。

| 时刻 | 观察与解释 |
| --- | --- |
| 子类开始调用父类 | 已取得名额，Starlette 发送响应头。首轮迭代进入 body，body 拉取 backend，backend 提交请求。 |
| 首个 tick | 预留 5 个 KV token 的空间，占 3 页，剩余 3 页。prefill 产生首个 token，`finish_reason=None`。 |
| 首块进入 `send` | Backend 暂停在 `yield event`，body 暂停在 `yield SSE 字符串`，Starlette 正在消费者的 `await send`。 |
| 父类调用退出 | 本地 Starlette 在 ASGI 2.3 分支收到 `http.disconnect` 并取消发送任务，再等任务组退出。在 2.4 分支，探针让 `send` 抛 `BrokenPipeError`，父类转换为 `ClientDisconnect`。日志均先观察到父类退出。 |
| 子类显式等待 `body.aclose()` | Python 在 body 挂起的 `yield` 处注入 `GeneratorExit`，body 的 `finally` 关闭 backend。Backend 在自己的挂起点清理，因为请求尚未完成而执行 `Engine.cancel`。日志此时记录页数回到 6。 |
| 整个 ASGI 调用退出 | Engine 无工作、backend 状态表为空。下一条请求返回 200。 |

`async for` 取到值后会先执行循环体，再决定是否拉下一项。异常若出现在循环体内的 `send`，生成器处于挂起状态，消费者退出并不会沿着那两个 `yield` 执行生成器清理。等待父类调用退出后，父类已停止拉取迭代器，此时关闭由子类持有的生成器。若取消发生在正等待下一条事件时，取消会沿当前的等待链进入 body 和 backend，它们通常已经开始清理。若发响应头时就断开，body 还未启动，`aclose()` 不会运行 body 的函数体。正常迭代耗尽后再关合也是空操作。 [Python 异步生成器](https://docs.python.org/3.12/reference/expressions.html#asynchronous-generator-iterator-methods) · [ASGI 断流规范](https://asgi.readthedocs.io/en/latest/specs/www.html#disconnected-client-send-exception)

### 两个新增的计数回归

- **普通请求连续遭拒。** [api.py](../src/tinyserve/api.py#L317) 中 `try` 覆盖了获取名额，第二个请求的 `429` 会经过 `finally`，误减第一个请求的名额。保持第一个请求挂起时，第三条请求实测返回 200，预期仍为 429。让成功获取先于负责归还的 `try`。[普通请求回归](../tests/test_lab07_api.py#L132)
- **首块断流且显式关闭抛错。** [api.py](../src/tinyserve/api.py#L308) 中 `await body.aclose()` 与归还名额顺序放在同一个 `finally`。当 backend 的清理在这次显式关闭中抛错，名额归还被跳过，下一条请求实测返回 429。让尝试关闭与归还名额分别有自己的清理保证，最外层 `finally` 归还名额。先前的关闭失败用例走的是正常迭代末尾，抛错时 body 已经退出，所以子类随后显式关闭时没有再抛。[断流与关闭异常回归](../tests/test_lab07_api.py#L647)

最终复验中，普通请求先取得名额，再进入负责归还的 `try/finally`。SSE 响应用 `aclosing(body)` 包住 Starlette 的调用，并从外层 `finally` 归还名额。前者让 429 短路退出时保持计数，后者让关闭异常传播时仍能执行归还。两条新增回归已通过。 [普通响应分支](../src/tinyserve/api.py#L314) · [SSE 响应分支](../src/tinyserve/api.py#L303)

## 从 Lab 07 到 Lab 08

先实现 [bench.py](../src/tinyserve/bench.py) 的 `summarize`，用 [fake-clock 验收](../tests/test_lab08_bench.py) 读清分位数，再实现 `run_trace`。引擎内 TTFT 从实际 `submit` 到第一个 `TokenEvent` 被观测，ITL 只取同一请求相邻两个 token 的时间差，吞吐用总输出 token 数除以最早提交到最晚完成的时间。每个 tick 执行期间抵达的请求在 tick 之后提交。没有样本时分位数为 `None`，最近秩用排序后的位置 `ceil(p * n)`（从 1 开始）。在线接口另外观察首个 SSE 帧或首个非空文本，报告时明确起点和终点。200 响应头、空 delta、首个非空文本可能出现在不同时间。
