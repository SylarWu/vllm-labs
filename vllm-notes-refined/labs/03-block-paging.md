# Lab 03 · Block Pool 与 Block Table

**CPU 验收 `3 passed`。** 跨非连续物理页、整页释放、只增不减、同页扩容和失败后状态保留均已复验。

## 1. 分页处理哪类空间问题

连续 KV 给每个请求固定容量，可能同时出现请求内部闲置和池内空闲区分散。分页池把容量切成等大的物理块，逻辑 token 仍按 `0,1,2,…` 排列，相邻逻辑块可以分配到不相邻的物理块。若仅给每个请求分配当前长度所需的块，尾块最多空 `B-1` 个 slot，属于内部碎片。预先分配的物理内存量和池里还有多少可用块是两个独立指标。[PagedAttention 论文第 4 节](https://arxiv.org/html/2309.06180)

| 对象 | 管理量 |
|---|---|
| `BlockPool` | 合法物理 ID `0..capacity-1`、每个 ID 的使用权。 |
| `BlockTable` | 单请求 `reserved_tokens` 以及按逻辑顺序排列的物理块 ID。 |
| `PagedKV` | 各层实际已写 K/V 长度，可能短于保留长度。[Lab 05](05-engine.md) |

## 2. 地址翻译与增长

对逻辑位置 `p`，`logical_block, offset = divmod(p,B)`，再取 `flat_slot = block_ids[logical_block] * B + offset`。块表必须保留分配后的逻辑顺序，物理地址排序会打乱历史。分页不改变 RoPE、causal mask 或 `start_pos` 使用的逻辑坐标。

`B=4`、物理块 0/2 已占、该请求的表为 `[3,1]` 时，逻辑位置 `2,3,4,5,6` 映射成 `(14,15,4,5,6)`。两块共有 8 个 slot，若只保留 7 个位置，`slots(7,1)` 仍非法。

| 从 `reserved_tokens=7`、`block_ids=(3,1)`、池满出发 | 结果 |
|---|---|
| `reserve(5)` | 长度仍为 7，映射稳定。 |
| `reserve(8)` | 无需新块，更新长度为 8，`slots(7,1)` 可映射到物理 slot 7。 |
| `reserve(9)` | 需要第 3 块，分配失败，长度、表与池状态都保留。 |

总块数向上取整 `(T+B-1)//B`，本次只申请总数减已有块数。先向池申请差额，再接到表尾并提交新长度。`slots(start,count)` 整段检查 `[start,start+count)`。`release()` 根据实际上持有的块归还，再清空表和保留长度，因此可重复调用。直接向池重复 free 仍报错。

## 3. BlockPool 的事务与复盘

- 单 owner 下采用先校验再提交。`allocate(n)` 容量不足时整批失败，`allocate(0)` 为空操作。`free(Iterable)` 先消费整批，校验重复、越界和当前已分配状态，再归还。实测生成器给第一个 ID 后抛异常时也保持原状态。
- 始终满足 `空闲 + 已分配 = capacity`，每个合法 ID 恰有一种状态，空闲 ID 唯一，每块最多一个请求持有。分配顺序由实现选择。可以只维护空闲集合并用 `len` 计数，也可用空闲列表加 allocated 布尔表。
- 初版布尔表加缓存 count 时，`allocate(-1)` 曾让三块池的 `free_count=4`，随后 `allocate(4)` 只返回三块。另一版负 count 虽未改状态，负 capacity 仍可构造。当前非法负数均拒绝，普通不足不改变状态。源码现用 `CapacityError` 同时表达非法参数和真不足，调用者需要区分可等待性时可改为非法参数 `ValueError`、合法但暂缺容量 `CapacityError`。
- 列表切片 `[:count]` 与 `[count:]` 共复制原空闲列表的 `F` 项，`list(ret)` 再复制一次。当前从尾部 `pop()` 取得所需块，避开整表复制。第二版用参考集合独立对拍 1000 轮分配、归还、非法操作。

## 4. BlockTable 的边界与复验

首次随机失败为 `reserve(16)` 实际记录 12。根因是 `T % B=0` 表示末页已填满，旧实现把它当成空，`reserve(4)` 可占一页却记录长度 0，还掩盖了释放。当前保留 `slot_counts`，末页用 `T-(块数-1)*B`，正长度时范围 `1..B`。较小或相同的目标直接返回。也可以只存一个逻辑长度和块 ID 列表，由它们推导占用。构造要求 `B>0`，翻译拒绝负 count。外部 `block_ids` 返回 tuple 快照，内部逐 slot 直接索引列表，避免循环复制整张表。

```bash
.venv/bin/python -m pytest -q tests/test_lab03_paging.py -k allocation_is_atomic
.venv/bin/python -m pytest -q tests/test_lab03_paging.py
```

公开随机测试固定种子跑 150 轮，检查计数、唯一归属、失败回滚。另验整页映射后连续两次释放、满池内成功增至 8 后对 9 的失败不会倒退。代码 `src/tinyserve/paging.py`，原笔记 `learning_notes/lab03_paging.md`。
