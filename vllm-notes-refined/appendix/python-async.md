# 附录 · Python 异步与 ASGI 速查

配合 [Lab 07](../labs/07-streaming-api.md) 阅读。这里建立单线程、单事件循环的心智模型。

## 1. Coroutine、Task 与协作切换

| 写法 | 执行语义 |
|---|---|
| `result = fetch()` | 同步调用返回后继续。在事件循环的 Task 内执行耗时同步函数时，其他 coroutine 也要等待。 |
| `async def work()`（无 yield）与 `work()` | 定义 coroutine function，调用创建 coroutine 对象，函数体尚未运行。用 `await` 执行并取结果，或用 `create_task` 交给事件循环。 |
| `await awaitable` | 等待结果，真正挂起时把运行权交回 loop。已完成 Future 可能连续执行。通常在 async 函数内使用，依赖协作式 I/O 并发。 |
| `asyncio.run(main())` | 同步入口创建、运行、关闭事件循环，通常要求当前线程没有活动 loop。Lab 用它写同步 pytest，不依赖异步 pytest 插件。 |
| `create_task(work())` | 独立调度，下一句看到 task 对象时其函数体未必已执行。 |
| `await asyncio.gather(ask("A"),ask("B"))` | 并发等待多个 awaitable，结果顺序与传参一致，实际完成顺序可不同。 |
| `await asyncio.sleep(0)` | 确保当前 Task 挂起一次，其他已就绪任务有机会运行。给 submit、断流 I/O 让出执行权，具体谁先就绪仍由调度决定。 |

`engine.step()`、`queue.put_nowait`、`json.dumps` 在当前 Task 中同步运行。同步工作跨越较长时段时，HTTP 的取消响应也要等真正的挂起点。跨多个 await 的读改写需要重新验证或协调，I/O 并发与多核计算属于不同能力。

## 2. Async generator、消费与关闭

普通 `def + yield` 创建同步生成器，用 `for` 消费。`async def + yield` 创建 async generator function，调用返回 async iterator 对象，第一次 `await anext(gen)` 才开始执行。`async for` 逐次等待 `__anext__`，`StopAsyncIteration` 结束循环。`ChatBackend.stream(...) -> AsyncGenerator[ChatEvent,None]` 是“直接返回可逐条消费的异步生成器”的结构化契约，实现可用 `async def` 加 yield。调用结果不支持整体 await 或同步 for。`[event async for event in stream]` 收集全部事件。

一次 `yield` 只把值交给消费者并挂起，交接可全在同一个 Task 中。消费者不再拉取时，应显式等待 `aclose()`，在已挂起的 yield 处进入关闭，使 `finally` 运行。若生成器从未启动，它的函数体与内部 finally 均不会执行。外部事先取得的 HTTP 名额放在外部管理。`async with` 通过 `__aenter__/__aexit__` 管理异步资源，如 HTTPX 连接，`contextlib.aclosing` 则在提前退出或异常时显式等待生成器关闭。

```python
from contextlib import aclosing

async def first_event(backend, request):
    async with aclosing(backend.stream(request)) as stream:
        return await anext(stream)
```

## 3. 协调工具与取消语义

| 工具 | 契约与 Lab 用法 |
|---|---|
| `asyncio.Event` | 共享发生信号。`await started.wait()` 等首请求进入 backend，`started.set()` 同步唤醒等待者。另一个 release Event 控制它何时结束。业务数据用 Queue 传。 |
| `asyncio.Queue` | `await put/get` 可挂起，空队列等待时不会忙等。`put_nowait/get_nowait` 同步，无法完成时抛 `QueueFull/QueueEmpty`。每个请求的私有队列解耦生产和消费。同线程 loop 内使用。跨线程选相应队列和消息机制。 |
| `asyncio.Lock` | `async with lock` 使获取/释放相配。`locked()` 只表示有持有者，任务应只释放自己成功获取的锁。check+modify 全同步时 loop 无法插入。check→await→modify 需锁、重查或调整设计。所有相关路径遵守同一锁才有互斥。 |
| `asyncio.wait_for(coro, timeout=2)` | 超时取消被等待任务并抛超时异常，测试用来防无限挂起。 |
| `CancelledError` | Task 通常在可取消的挂起点收到，finally 收尾。如果显式捕获以补充处理，清完后继续传播。现代 Python 的取消继承 `BaseException`，靠 `except Exception` 的业务处理覆盖不到。 |
| AnyIO | `sleep_forever()` 模拟不会自然结束的 backend。取消作用域内的异步清理也可能立刻被再次打断。有边界的 `with anyio.CancelScope(shield=True)` 可保护短 `await` 清理，原测试用它确保清理标记设置。 |

常见错误是把 `lock.acquire()` coroutine 放在 `if` 中，或读 `queue.get().attribute`。要 await 后才有结果。`sleep(0)` 提供一个确定的让出点，无法保护跨多个 await 的状态。不必要的锁会把同步 dict helper 变成 async，扩大交错点。引入 `to_thread`、独立 worker 后要重新设计 owner 和线程间通信。`asyncio.Lock` 只协调 loop 中的 coroutine。避免在 finally 以 return 覆盖原异常或取消。

## 4. ASGI、FastAPI、Starlette、Uvicorn 与 HTTPX

`ASGI` 全称 Asynchronous Server Gateway Interface。一次 HTTP 请求对应一次 `await app(scope,receive,send)`，长连接上不同请求各有 scope。

| 层 | 职责 |
|---|---|
| `scope` | 请求静态信息，如 method/path/headers。`scope["asgi"]["version"]` 是接口版本，`spec_version` 是 HTTP 子规范版本。 |
| `receive` / `send` | 接收 `http.request/http.disconnect`。发送 `http.response.start` 与一或多次 `http.response.body`。`await send` 遇到背压可挂起，也可能立即完成。 |
| FastAPI / Starlette | 前者路由与响应，后者 `StreamingResponse` 从同步或异步迭代器读取并发送 body。SSE 的 `data: ...\n\n` 是 body 的字节内容，Backend 的 ChatEvent 由 body 负责编码。 |
| Uvicorn / HTTPX | Uvicorn 在真实 HTTP 字节与 ASGI 消息之间转换。HTTPX 的 ASGITransport 直接调用 app 而不走 socket，会缓冲流，适合内容验证。断流测试直接控制 `receive` 和 `send`。 |

记录环境里的 Starlette 1.10.0 小于 ASGI HTTP 2.4 时并发监听 receive 和发送，收到 disconnect 取消发送任务。2.4 起分支在断开的 send 上观察 `OSError`，转为 `ClientDisconnect`。原始 ASGI 可把断流落在发头、等事件或发首块。发块时生成器挂在 yield，要由外部消费者在退出时等待 aclose，回收与本次响应完成才绑定。[ASGI 总规范](https://asgi.readthedocs.io/en/latest/specs/main.html) · [HTTP 消息与版本](https://asgi.readthedocs.io/en/latest/specs/www.html#spec-versions)

## 5. 测试代码对照

先 `create_task(client.post(...))` 让第一条并发运行，`wait_for(backend.started.wait(),2)` 等它确实进入 backend，再起第二条验 429。完整聚合用异步推导式，提前关闭用 `await anext(stream)` 后 `await stream.aclose()`。事件数据按请求 ID 隔离，SSE body 与 backend stream 各关合自己的下游。原笔记 `learning_notes/python_async.md`。[Python 生成器](https://docs.python.org/3.12/reference/expressions.html#asynchronous-generator-functions) · [asyncio 任务](https://docs.python.org/3.12/library/asyncio-task.html)
