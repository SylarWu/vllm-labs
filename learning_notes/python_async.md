# Python 异步语法与工具包速查

本文服务于 Lab 07，重点是读懂和编写单事件循环中的异步代码，不覆盖多线程、多进程或完整 AnyIO 体系。

## 1. 同步、并发与并行

同步调用必须等当前函数返回后才执行下一句：

```python
result = fetch()
print(result)
```

异步函数允许在等待期间主动挂起，让事件循环执行其他任务：

```python
result = await fetch()
print(result)
```

这里的核心收益是**单线程中的 I/O 并发**，不是让 Python 计算自动并行。若 coroutine 内执行一个耗时的纯同步函数，事件循环仍会被它占住，其他 coroutine 无法运行。

Lab 07 的 `engine.step()` 就是同步计算。它在一个 tick 内独占事件循环；只有 tick 之间出现 `await`，其他请求和断流处理才有机会运行。

## 2. Event loop、coroutine 与 Task

### Event loop

事件循环负责选择当前可以运行的任务。一个任务遇到真正的异步等待时挂起，事件循环切换到其他就绪任务。

```python
async def main():
    ...

asyncio.run(main())
```

`asyncio.run` 创建事件循环、运行传入的 coroutine，并在结束时关闭循环。通常只能在“当前没有运行事件循环”的同步入口调用。项目测试用它是因为没有依赖 pytest 的异步插件。

### Coroutine

不含 `yield` 的 `async def` 定义 coroutine function。调用它只会创建 coroutine 对象，函数体尚未执行：

```python
async def work():
    print("running")

coro = work()       # 尚未打印
await coro          # 执行到完成
```

`await` 只能用于 awaitable 对象，且通常必须写在 async 函数中。它表示“等待结果；等待期间可以切换任务”，不表示“另开线程”。

### Task

Task 是由事件循环独立调度的 coroutine 执行单元：

```python
task = asyncio.create_task(work())
result = await task
```

`create_task` 让任务尽快开始调度，而不是保证调用下一行前它一定已经执行。需要主动 `await` 或让出事件循环，其他任务才有运行机会。

`asyncio.gather` 可并发等待多个 awaitable：

```python
a, b = await asyncio.gather(ask("A"), ask("B"))
```

返回值顺序与传入顺序一致，不代表完成顺序。这里的“并发”仍不等于多核并行。

## 3. await 到底让出了什么

`await` 是否真正切换任务，取决于被等待对象。已经完成的 Future 等路径可能不切换，因此不能假定每个 `await` 都必然让其他任务运行。

```python
await asyncio.sleep(0)
```

是显式让出一次事件循环的常用写法。它适合本 lab 在第一次 Engine tick 前让其他请求提交，以及在 tick 之间处理其他就绪任务。

它不是：

- 定时等待；
- 锁；
- 请求一定完成入队的全局保证；
- CPU 计算并发工具。

若 coroutine 在 `sleep(0)` 前后执行的都是同步代码，这段同步代码内部不会插入其他 coroutine。

## 4. async iterator 与 async generator

### 普通生成器回顾

含 `yield` 的普通函数返回 generator：

```python
def numbers():
    yield 1
    yield 2

for value in numbers():
    print(value)
```

每次执行到 `yield`，产生一个值并挂起；下次迭代从挂起处继续。

### 异步生成器

同时含 `async def` 和 `yield` 的函数是 async generator function：

```python
async def numbers():
    yield 1
    await asyncio.sleep(0)
    yield 2
```

消费方式：

```python
async for value in numbers():
    print(value)
```

也可以手动取下一条：

```python
stream = numbers()
first = await anext(stream)
```

`async for` 会持续等待 `__anext__`，遇到 `StopAsyncIteration` 正常结束。

特别注意：调用异步生成器函数返回的是 **async generator，不是 coroutine**，所以不能这样等待整个生成过程：

```python
# 错误：numbers() 不是 coroutine
await numbers()
```

本项目中的类型关系：

```python
ChatBackend.stream(...) -> AsyncGenerator[ChatEvent, None]
```

意味着调用 `stream` 后，通过 `async for` 或 `anext` 逐个获得 `ChatEvent`，最终没有返回值。

### aclose 与 finally

调用方可能只消费一部分就停止：

```python
stream = backend.stream(request)
first = await anext(stream)
await stream.aclose()
```

`aclose` 在生成器当前挂起的 `yield` 处请求关闭，生成器中的 `finally` 因而获得执行清理的机会：

```python
async def stream():
    try:
        yield event
        await wait_for_more()
    finally:
        cleanup()
```

从消费端 `break` 与显式关闭不是同一概念。本 lab 要确保不再消费时生成器确实被关闭，才能触发 Engine 取消。

## 5. async with

异步资源需要在进入和退出时执行异步操作，使用 `async with`：

```python
async with httpx.AsyncClient(...) as client:
    response = await client.post(...)
```

退出时会释放 client 管理的连接等资源。对象实现 `__aenter__` 和 `__aexit__` 后可作为异步上下文管理器。

对于 async generator，可以使用 `contextlib.aclosing` 保证关闭：

```python
async with aclosing(backend.stream(request)) as stream:
    async for event in stream:
        ...
```

即使消费过程中异常或提前退出，上下文退出时也会调用 `aclose()`。

## 6. 常用同步原语

### asyncio.Event

Event 只表示“某件事是否发生”，不承载事件数据：

```python
started = asyncio.Event()

await started.wait()  # 未 set 时挂起
started.set()         # 唤醒所有等待者
```

项目测试用 `started` 等到第一条请求已进入 backend，再发送第二条请求验证 429；用 `release` 控制第一条请求何时结束。

### asyncio.Queue

Queue 用于在 coroutine 之间传递数据：

```python
queue = asyncio.Queue()
queue.put_nowait(item)
item = await queue.get()
```

没有数据时，`get` 挂起而不忙等。私有队列可以把 Engine tick 的事件生产者与单路 HTTP stream 的消费者解耦，避免不同请求共用一个输出缓冲区。

`asyncio.Queue` 不是线程安全队列；跨线程应使用其他机制。本 lab 全程单事件循环。

### asyncio.Lock

```python
async with lock:
    ...
```

用于避免多个 coroutine 在关键异步区间交错执行。它不是自动让 Engine 线程安全的工具，也不应在不必要的大范围内持有。

若一段同步代码中没有 `await`，它在单事件循环里不会被其他 coroutine 插入。这是同步 Engine 单 tick 可保持状态一致的基础。

## 7. 超时与取消

### wait_for

```python
result = await asyncio.wait_for(coro, timeout=2)
```

超时会取消正在等待的任务并抛出超时异常。测试用它防止实现死锁后无限挂起。

### CancelledError 与 finally

任务被取消时，会在当前挂起点收到取消。清理应写在 `finally`：

```python
try:
    await work()
finally:
    release()
```

不要吞掉取消后继续正常执行；若显式捕获 `asyncio.CancelledError` 做补充处理，通常应重新 `raise`。现代 Python 中它不是普通业务异常，因此只写 `except Exception` 不能替代完整的取消设计。

### 为什么清理中的 await 也可能被取消

任务已经处于取消过程时，清理代码里的下一次异步等待仍可能立即被打断。AnyIO 测试中的写法：

```python
with anyio.CancelScope(shield=True):
    await anyio.sleep(0)
    self.closed = True
```

表示在这个作用域内屏蔽外部取消，以验证生成器确实进入了清理逻辑。它不是要求业务代码屏蔽所有取消。实际实现应优先让必要清理短且可靠，不能借 shield 无限延迟取消。

`anyio.sleep_forever()` 表示一直异步等待，直到被取消，测试用它模拟永不自行结束的流。

## 8. FastAPI、StreamingResponse 与 ASGI

FastAPI 应用本身是一个 ASGI 应用。可把它理解为：

```python
await app(scope, receive, send)
```

- `scope`：这次连接的静态信息，例如 method、path、headers。
- `receive`：等待客户端事件，例如 `http.request`、`http.disconnect`。
- `send`：发送 `http.response.start` 和若干次 `http.response.body`。

普通 JSON 响应可一次发送完整 body；流式响应则多次发送 body。

`StreamingResponse` 接收同步或异步迭代器，每迭代出一段内容，就作为响应体的一部分发送：

```python
async def body():
    yield "data: {...}\n\n"
    yield "data: [DONE]\n\n"
```

这段 `body` 是 HTTP 编码层；`backend.stream` 是业务事件层。二者通过 async generator 串接。

Uvicorn 是运行 ASGI app 的服务器。HTTPX 的 `ASGITransport` 则直接调用 app，不经过真实 socket，适合接口测试，但会缓冲流式响应，不能单独证明客户端中途断开的处理正确。因此项目还使用原始 ASGI 测试主动注入 `http.disconnect`。

## 9. 阅读本项目测试时的语法对照

```python
first = asyncio.create_task(client.post(...))
```

把第一条 HTTP 请求交给事件循环独立运行，否则测试本身会一直等待第一条完成，无法构造第二条并发请求。

```python
await asyncio.wait_for(backend.started.wait(), 2)
```

等待 backend 确认第一条已经启动，最多等两秒。

```python
events = [event async for event in backend.stream(request)]
```

异步推导式：逐个消费 async generator，最终收集为 list。

```python
stream = backend.stream(...)
await anext(stream)
await stream.aclose()
```

只取首 token 后主动关闭，模拟客户端提前结束。

## 10. 高频错误

- 调用 coroutine 后没有 `await` 或 `create_task`，导致函数体根本没执行。
- 把 coroutine 对象放进 `if`，误把“对象为真”当成“异步操作已经成功”；例如 `if lock.acquire():` 并未获取锁。
- 直接访问 `queue.get().attribute`；`queue.get()` 返回 coroutine，等待它之后得到的才是队列元素。
- 对整个 async generator 使用 `await`，而不是 `async for` / `anext`。
- 在 async generator 上用同步 `for`。
- 提前停止消费却没有关闭 generator，导致取消和回收没有发生。
- 在事件循环中执行长同步计算，误以为其他请求仍能及时运行。
- 用 `asyncio.sleep(0)` 当锁，忽略跨多个 `await` 的状态交错。
- 把不同请求的事件放入同一个累计文本状态。
- 在 `finally` 中释放资源后又执行可能覆盖清理结果的返回逻辑。

## 11. 锁所有权与 Queue 的同步/异步方法

`lock.locked()` 只表示锁处于持有状态，不表示当前 coroutine 是持有者。只有成功获取锁的任务才能释放它。常规写法：

```python
async with lock:
    ...
```

`asyncio.Queue` 同时提供两套方法：

| 异步方法 | 同步方法 |
| --- | --- |
| `await queue.put(item)` | `queue.put_nowait(item)` |
| `await queue.get()` | `queue.get_nowait()` |

同步版本不是“自动等待版本”的简写，它们在无法立即完成时会抛出 `QueueFull` 或 `QueueEmpty`。选择哪套方法取决于是否允许挂起，而不是取决于哪套写起来更短。

## 12. 单事件循环中什么时候需要 asyncio.Lock

判断依据不是“变量是否共享”，而是“一组必须保持一致的复合操作是否可能被 await 切断”。

```text
同步检查 + 同步修改，中间无 await
→ 单事件循环不会在中途切换 coroutine
→ 通常不需要 asyncio.Lock

检查 → await → 根据旧状态修改
→ await 期间其他 coroutine 可能改变状态
→ 需要锁、重新校验或调整设计
```

锁只保护遵守同一把锁的代码路径。如果 step 加锁而 submit/cancel 不加锁，不能声称 Engine 的全部访问已经互斥。

`asyncio.Lock` 只协调事件循环中的 coroutine，不是跨线程锁。若引入 `to_thread` 或独立 worker thread，应重新设计 Engine owner 和线程间通信，而不是沿用单事件循环结论。

不必要的锁还会带来 async 染色：原本同步的 dict 辅助函数为了获取锁变成 async，所有调用方被迫 await，于是出现更多交错点。优先减少共享状态和临界区，再考虑增加锁。

## 与 Lab 07 的最小关系图

```text
async def + await        请求协程与事件循环切换
create_task / gather     构造并发请求
async generator          backend.stream 和 SSE body
async for / anext        消费 ChatEvent
aclose / finally         断流取消与资源回收
Event / Queue            测试同步、请求事件路由
StreamingResponse        把业务事件编码为 HTTP SSE
原始 ASGI disconnect     验证真实清理路径
```
