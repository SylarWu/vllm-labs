# vLLM 学习笔记

这里记录学习路线中的可复用知识、实验结论和易错点。每个 lab 使用独立文档；环境配置单独维护。

## 学习进度

| Lab | 主题 | 状态 | 笔记 |
| --- | --- | --- | --- |
| 01 | Qwen3 算子、权重映射与完整 forward | 已完成，`6 passed` | [Lab 01](learning_notes/lab01_model.md) |
| 02 | 连续 KV cache、prefill/decode 与贪心生成 | CPU toy 验收完成，Lab 01+02 合计 `12 passed`；真权重对拍待选做 | [Lab 02](learning_notes/lab02_cache.md) |
| 03 | Block pool 与 block table | 已完成，`3 passed`，整页释放和保留范围边界已复验 | [Lab 03](learning_notes/lab03_paging.md) |
| 04 | 迭代级调度 | 已完成，`5 passed`；Lab 01～04 回归 `20 passed, 1 skipped`，准入、取消和退休边界已复验 | [Lab 04](learning_notes/lab04_scheduler.md) |
| 05 | 分页 KV 与端到端 Engine | 未开始 | 待创建 |
| 06 | 真 batch decode 与 chunked prefill | 未开始；RoPE 扩展已提前完成 | 待创建 |
| 07 | FastAPI、SSE 与取消 | 未开始 | 待创建 |
| 08 | 指标、压测与官方 vLLM 对照 | 未开始 | 待创建 |

## 通用资料

- [Python 环境与 uv](learning_notes/environment.md)
- [项目学习计划](README.md)

## 维护原则

- 记录可复用的原理、形状推导、实验结论和排错方法，不保留逐句对话。
- 每个 lab 开头维护学习成果和验收状态，细节按主题归类。
- 跨 lab 知识放在首次建立该概念的文档中，并在相关 lab 中链接，不重复复制。
- 真模型与性能实验记录运行环境、命令、结果和结论，避免只记分数而缺少上下文。
