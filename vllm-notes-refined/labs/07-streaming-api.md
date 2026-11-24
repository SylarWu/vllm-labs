# Lab 07 · FastAPI、SSE 与取消

**CPU 验收 `23 passed`，Lab 01–07 回归 `61 passed, 1 skipped`。** 服务共享一个后端生成契约，普通响应聚合 `ChatEvent`，流式响应逐条发送 SSE。语法与工具包见 [Python 异步附录](../appendix/python-async.md)。

## 1. Engine、Backend 和 HTTP 的三道边界

| 边界 | 所有者与契约 |
|---|---|
| 计算 | [Lab 06](06-batch-and-chunked-prefill.md) 的 `Step` 决定消耗的 token、位置与是否采样，Engine 同步 `submit / step / cancel`。 |
| 引擎状态 | Scheduler `complete/cancel` 推进或退休，页由 Engine 的 `on_retire` 回收。 |
| 传输与交付 | Backend 按内部 ID 路由 `TokenEvent`，产出 `ChatEvent(delta,finish_reason)`。HTTP 层生成对外 ID 和元数据，控制名额、发送与关闭。引擎已退休时，其末事件仍可能在私有队列里等待发出。 |

Engine、Scheduler、BlockPool 按单进程、单 owner 设计。每路请求是独立 coroutine，`submit/step/cancel` 和 step 后遍历全部事件的路由保持同步且无 `await`，单事件循环不能插入此段。首次 tick 前和需要继续执行的轮次之间通过 `await asyncio.sleep(0)` 让其他请求提交和 I/O 就绪。只在一个流里同步跑完整段生成会使第二个请求来不及入队。同步 tick 正在运行时，断流只能等它返回后处理。

A 首轮处理 2-token prompt 发 `a1`。B 的 3-token prompt 随后到达，`chunk=2`、budget 至少 3、A/B 各最多发 2 项。本轮一次混合 batch 执行 A 的 decode 与 B 的前 2 个 prompt，A 发 `a2 / length` 并回收页，B 只推进 KV。下轮 B 才发 `b1`。若其首块正在 `send` 时断开，响应层关 body、body 关 backend、backend 取消 B，使页和 HTTP 名额分别回收。A 的末块与 `[DONE]` 能在其引擎退休后交付。

## 2. Backend 状态、事件路由与进展

- 提交前生成唯一内部 `request_id`，建立按请求隔离的累计 token IDs、已入队稳定前缀 `queued_text`、`Queue[ChatEvent]` 与 `finished`。状态结构可用类型明确的 RequestState。拆成两个 dict 时需同步建删并对齐注解。只在终止事件时将 finished 单向置 True，避免迟到普通事件重置。冗余 `n_generated`、init/running 字符串和额外 stable_text 可去掉。
- 一个 tick 的 A/B 事件都要按 `TokenEvent.request_id` 路由。只处理本 stream 会丢掉 peer 已在引擎中推进的输出。对已取消或未知的 ID 应避免 lazy init 造成 zombie state。异常活跃 ID 可由日志/测试暴露。共享 driver 与私有队列的替代方案需要定义启动、空闲、异常传播和退出。
- 有待交付事件时直接 `get_nowait()`、原样 `yield`，之后回循环顶部。只有队列空才 `await sleep(0)`，醒来再检查自己的队列，仍空才 step 并完整路由，接着重新判断队列。真实 chunked prefill 可以返回零事件，或当前轮只给其他请求产事件，两种情况都要继续推进，直接 get 会 `QueueEmpty`。
- 第二次队列检查避免 A 已投递 B 首事件、B 恢复后又多算一轮才读。统一 yield 位置可避免积压时恢复后先多跑 tick。探针两流首事件只需同一个 tick，backlog 的消费新增断言 tick 数不增。
- 尚未收到事件、或收到普通事件，仍继续。Engine 已终止且队列还有项时继续交付。末事件交付后才可退出。若循环顶部只看 finished，peer 先路由的最终 token 及 finish reason 会被跳过。空 delta 可以来自特殊 token 或不完整字符，应独立于“本 tick 本请求无事件”。
- `Queue.put/get`、`Lock.acquire` 调用得到 coroutine，需要 await。`put_nowait/get_nowait` 同步并可能抛 `QueueFull/QueueEmpty`。`locked()` 只表示某任务持有锁，按它贸然 release 会释放别人的锁。当前无 await 的单线程临界段可以同步建状态、路由与消费。若改用锁，`async with lock` 表达所有权，所有 Engine 操作遵守一致纪律。若变多线程，重建 owner/线程间消息策略。把纯 dict/list/decode 包成 async 再嵌套锁会增加挂起点和锁顺序复杂度。

## 3. Detokenize、Unicode 与积压

Token 可能代表词、子词、标记或字节片段。UTF-8 的 `A=41`，`你=E4 BD A0`，可拆到多个 token。每到一个事件，把 ID 追加到本请求序列，累计 decode。将尾部 `U+FFFD` 暂缓，确认新稳定文本以旧的 `queued_text` 为前缀，再取新增部分。在原示例，E4 和 E4 BD 两次 delta 为空，E4 BD A0 发“你”，加 41 发“A”。如新稳定文本不再延长已承诺前缀，定位 tokenizer 行为而非盲切。终止时必须交付 finish reason，也要对合法字面 `�` 和最后仍残缺的尾部给出策略。原始 token IDs 继续保留，累计 decode 全程可能近平方开销，可在合适 tokenizer 接口下换增量 detokenizer。

**在路由入队时固定 delta。** 若 e1 解出“你”、e2 解出“你好”，但队列只存 TokenEvent，消费 e1 时读最新 stable_text 就会提前发“你好”，e2 变空。总文本虽对，delta 的归属与终止帧携带的文本会受调度影响。处理每个到达事件时令 `delta = stable_text[len(queued_text):]`、更新 `queued_text`、入队完整 `ChatEvent`，消费方原样发。也可在消费时按该事件对应的前缀解码。`Queue.peek` 只能看队首，无法恢复入队时遗漏的文本快照。若业务允许合帧，也需明确定义合并规则。无界队列在本 Lab 受单请求 `max_tokens` 限制，长流生产场景需要背压或集中 driver。

## 4. 生成器关闭与取消传播

提交后到全部 yield 都纳入同一个 `try/finally`。尚未收到终止事件而提前关闭或异常时调用 `Engine.cancel(id)`。若已退休，Backend 清掉私有状态即可。任何退出都删状态，取消自身报错时也应有独立清理保障，初始化失败可能还没有 state。ToyEngine 会计数，能识别对正常 A/B 也调用 cancel 的问题。用 `except BaseException` 吞掉 `CancelledError` 会改变调用方看到的结果，也会吞掉内部错误、`KeyboardInterrupt` 和 `SystemExit`。清理后让原异常自然传播。

`yield final_event` 会挂起生产者，后面的 `return/finally` 要在下一次 `anext` 或显式 `aclose` 后执行。正常 `async for` 会继续拉一次，把 `StopAsyncIteration` 当结束。消费者 break、只取一项或不再拉取时应显式关合，或使用 `contextlib.aclosing`。也可先构造终止事件并在末次 yield 前完成正常清理。HTTP 消费方须拥有明确的 body 与 backend 关闭契约。[Python async generator](https://docs.python.org/3.12/reference/expressions.html#asynchronous-generator-iterator-methods) · [aclosing](https://docs.python.org/3.12/library/contextlib.html#contextlib.aclosing)

## 5. HTTP 工厂、入参和 SSE 输出

`create_app(backend, max_inflight=8, max_output_tokens=128)` 创建独立 FastAPI 实例，通过闭包向 `@app.post("/v1/chat/completions")` 提供 backend 与计数。修改外层整数用 `nonlocal`。`TYPE_CHECKING` 导入只供静态类型使用，FastAPI、Body/HTTPException、StreamingResponse 要运行态可用。可用 Pydantic BaseModel 声明校验，或 dict/Request 手工校验动态上限与状态码。解析后构造内部不可变 `ChatRequest/Message`，Backend 不依赖 HTTP 类。

| 入参与资源边界 | 状态与语义 |
|---|---|
| 缺字段、`model` 非字符串、消息列表为空/错形状、非法 role 或非字符串内容 | `422`。合法 role 是 system/user/assistant。先检查存在性与类型，再判断模型名。直接索引缺项会变 500。 |
| `max_tokens`、`temperature`、`stream` | token 数要求整数且 `1..max_output_tokens`，排除 bool。temperature 为数值 0，排除 `False` 与显式 `None`。stream 缺省 False，显式出现须 bool。`bool` 是 `int` 子类且 `False == 0`。`max_output_tokens` 是服务端上限。 |
| 格式合法、模型字符串未知 | `404`。畸形输入在 HTTP 层拦截，Backend 的容量校验和 stream 均不启动。 |
| 永久超 KV 总容量 | 在 200 响应头前把 `ChatBackend.is_applicable()` 的稳定拒绝映射为 `422`。判断基于 `P+N-1` 和全池总页数，Engine.submit 复用校验。只因当前空闲不足的请求继续按 FIFO 等待。 |
| 通过校验但达到 `max_inflight` | `429` 即时拒绝，`inflight >= max_inflight`。HTTP 请求数、Scheduler `max_active`、KV 空闲页是三个边界。 |

`Protocol` 的成员由每个后端提供。测试的 `AcceptingBackend.is_applicable()` 返回 `(True,"")`，拒绝无效 payload 的替身让它与 `stream()` 若被调用便断言失败。真实 Engine 探针为 6 页 × 2 token，13-token prompt 永久放不下，旧普通响应为 500、SSE 已发 200 后抛 `ValueError`。新验收覆盖两种响应均 422、恰好放下者通过。当前 `is_applicable` 与 stream 各调用一次 `_build_request`，会重复 render 和生成内部 ID。可把一次准备好的 IDs 传给执行阶段。

| 响应 | 契约 |
|---|---|
| 普通 JSON | `object="chat.completion"`，`choices[0]` 含 `index=0`、`message={"role":"assistant","content":...}` 与终止原因。用 `aclosing` 逐条累计 delta 到 list，最后 join，保存最终原因。避免额外事件全列表、`events[-1]` 空列表风险和反复 `+=` 的字符串复制。返回 dict 或 `JSONResponse(dict)`。先 `json.dumps` 再 JSONResponse 会双重编码。 |
| SSE | 传 `body()` 生成器对象给 `StreamingResponse`，media type `text/event-stream`。每帧 `data: <JSON>\n\n`，`object="chat.completion.chunk"`，`choices[0]` 含 `index=0`、`delta.content`、本事件的 finish reason，末业务帧后仅一次 `data: [DONE]\n\n`。空 delta 仍可发送。`ensure_ascii=False` 可保留中文，转义后由客户端解析也等价。 |
| 共享元数据 | 一次生成 `chatcmpl-…`、整数 Unix 秒 `int(time.time())`、`model`，同一流保持一致。HTTP 对外 ID 与 Engine 路由 ID 各有作用域。`time.time_ns()//1000` 是微秒，`time.time()` 是浮点秒。 |
| 帧与传输 | SSE 客户端按空行组帧，底层传输可拆、可合。JSON 的 `id` 属业务数据，SSE 的 `id:` 行是事件游标。`[DONE]` 是业务结束，ASGI 的 `more_body=False` 是响应体结束。 |

参见 [SSE 解析规范](https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream) · [ASGI HTTP](https://asgi.readthedocs.io/en/latest/specs/www.html)。

## 6. 名额与断流以 Response 生命周期为界

同步的“检查 + 增量”无 await，可立即拒绝，Semaphore 的等待语义与此处不同。普通响应在成功取得名额后才进入负责释放的 `try/finally`，聚合期间由 `aclosing` 负责关闭 backend。

流式响应可继承 `StreamingResponse` 并覆盖异步 `__call__(scope,receive,send)`。FastAPI 识别 Response 实例，再 await 它的调用。用子类调用 `super().__call__` 复用 Starlette 的响应头、迭代与发送逻辑。也可继承 Response，包装原 StreamingResponse。直接返回普通可调用对象有被数据序列化的风险。流式校验、永久容量拒绝和名额判定发生在 200 之前。最终结构是**先成功取名额，`aclosing(body)` 包住整个父类调用，最外层 finally 只归还该次取得的名额**。body 自己用 `aclosing` 或等价的 `finally → aclose` 关闭 backend，保存并关合同一个 `body()` 对象。需要异步清理时，只对必要的短清理做 AnyIO shield。

| 取消落点 | 为什么由这一层处理 |
|---|---|
| 发送响应头期间 | body 尚未进入。对未启动的生成器 `aclose()` 不会运行其函数体，HTTP 名额由 Response 外层归还。这时通常没有 Engine 请求或页。 |
| 正在 await 下一条 backend 事件 | 取消沿 body → backend 调用链展开，内层 finally 通常已开始关合并取消未完成请求。 |
| 发送首个 body chunk 期间 | 框架在两个生成器都暂停于 `yield` 后执行消费者的 `await send`。Starlette 退出迭代并不自动驱动挂起的 finally。等待父类调用完全退出，再显式 `body.aclose()` 注入 `GeneratorExit`，body 关闭 backend，未完成请求取消回收页。 |
| 同步 Engine tick 中 | 事件循环要先等 tick 返回，再处理 disconnect。 |
| 正常耗尽或关闭报错 | 正常耗尽后关合是空操作。关闭抛错时外层仍归还名额并让错误传播。Backend 创建抛错也要落在名额作用域内。 |

[FastAPI 返回 Response](https://fastapi.tiangolo.com/advanced/response-directly/#return-a-response) · [ASGI 断流规范](https://asgi.readthedocs.io/en/latest/specs/www.html#disconnected-client-send-exception)

### 一次 POST 的正常逐帧顺序

3-token prompt、`max_prefill_chunk=2`、输出上限 4、忽略 EOS，toy tokenizer 把选中的 ID 显示为 B。响应先发头，body 第一次被拉取时进入 backend。backend render、注册 ID 并入队，空队列时让出事件循环。Tick 1 只消费前两个 prompt token 和写 KV，没有帧。Tick 2 消费最后一个 prompt token 并发首 B。Tick 3/4 各 decode 一项。Tick 5 发 B 与 `length`，Scheduler 已先退休并释放页，EOS 命中则优先 `stop`。框架再拉一次获得 `[DONE]`，再拉一次退出 body，显式关闭可能还停在末 yield 的 backend，发送 `more_body=False`。HTTP 名额由最终 Response 退出时归还。

## 7. 回归索引

| 复盘发现 | 用例或证据 |
|---|---|
| 首轮循环判断 `latest_event is not None`，未运行 tick 就 yield None。修正后曾等整段结束才 yield，提前 `anext(C)` 得到 CCCC | 后端并发/取消测试要求 A/B 共 tick、`peak_active==2`、首个 C 提前可见、只取消提前关闭的 C。 |
| UTF-8、peer 终止、异常传播、积压归属 | `holds_incomplete_utf8` 拆 E4/BD/A0，`delivers_terminal_event_routed_by_a_peer_stream` 保留 peer 终止帧，`task_cancellation_cleans_up_and_propagates` 用永不出事件的 IdleEngine，`backlogged_events_keep_their_own_deltas` 暂停 B、检查两帧各自的 delta 和无需多一个 tick。 |
| 空 tick、只给 peer 出事件、两流首事件 | 真实 toy 第一段 prefill 不产事件。`max_active=1` 时 B 可驱动 A 结束。两流第一次拉取只需一个 tick。依次补齐循环续跑及让出后重查。 |
| 解析、序列化和限流 | 缺字段、错类型与 None/bool 参数统一 422，未知模型 404，app exception 透传关闭以观察真实状态码。普通/SSE 文本、末事件与元数据对拍，429 后恢复。 |
| 获得名额的时间 | endpoint 返回 StreamingResponse 就释放会低估并发。移到 body 入口漏掉发送头。首事件后才增加会放行两条请求、首事件前退出甚至减成 `-1`。最终从 Response 调用前覆盖至结束。 |
| 普通请求连续遭拒 | 第二条 429 若从包住 acquire 的 finally 递减，会错误归还第一条仍持有的名额，第三条变 200。成功取名额先于释放 try 后连续遭拒都返回 429。 |
| ASGI 首块断流 | 用原始 `app(scope,receive,send)`，先 `http.request`，再在指定时点触发断流。ASGI 2.3 的本地 Starlette 收 `http.disconnect` 后取消发送任务，2.4 的探针让 `send` 抛 `BrokenPipeError`，观察到转成 `ClientDisconnect`。HTTPX ASGITransport 缓冲整条响应，适合断言内容，首块中途需原始 ASGI。 |
| 页回收与双重失败 | toy 容量 6 页、每页 2 token，prompt 2、输出限 4，首帧时预留 `5` 位即 3 页。旧版首块断流返回时只剩 3 空闲页，下一条 429，要等 GC/事件循环收尾才恢复。新版父类退出后显式关 body，当场回到 6 页、Engine 无工作、状态为空、下一条 200。backend 打开或关闭抛错、特别是首块断流与显式关闭同时抛错，最外层仍归还名额。 |

定向 backend 从首次 1 fail，经 UTF-8、peer 终止、取消传播和 backlog 回归到 `5 passed,4 deselected`。HTTP 阶段依次 `6 passed,3 failed`、`9 passed,1 failed`、`10 passed`，当时前七关 `48 passed,1 skipped`。原始 ASGI 补齐后为 `59 passed,1 skipped`，最后再补连续拒绝和断流关闭抛错两个计数用例，最终 `61 passed,1 skipped`。原笔记的上一版环境为 Python 3.12.7、Starlette 1.10.0、Uvicorn 0.54.0。跳过的是未配置的真权重用例。

```bash
python -m pytest -q tests/test_lab07_api.py -k engine_backend
python -m pytest -q tests/test_lab07_api.py -k 'holds_admission or during_send or failure_restores'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_lab07_api.py --tb=short
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q --ignore=tests/test_lab08_bench.py
```

代码 `src/tinyserve/api.py`。原笔记 `learning_notes/lab07_api.md`。首响应头、首业务 SSE 帧、首个非空文本的时间可能不同，指标边界见 [Lab 08](08-benchmark.md)。
