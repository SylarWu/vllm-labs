# 附录 · Python 环境与 uv

| 工具 | 定位 |
|---|---|
| pip | Python 包安装与依赖解析，环境由 venv 等工具创建。 |
| Anaconda / Conda | Python 发行版与环境管理，也可管理 Python 版本和部分非 Python 系统依赖。 |
| uv | Python、环境、包与项目管理。项目选用 uv、`.venv`、setuptools 构建和 `src` 布局，同一环境按主要工具维护。 |

## 1. 建环境与安装

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[lab]'
```

第一行创建项目内隔离环境，第二行让当前 shell 选用其中的解释器和包。`-e` 是 editable，通常改源码即可看到效果。`.` 安装 `pyproject.toml` 描述的当前项目，`[lab]` 启用可选依赖组，引号避免 shell 把方括号当通配。可在激活后 `uv pip install numpy`，也可显式指定 `uv pip install --python .venv/bin/python numpy`。

## 2. 临时安装、项目声明和锁定

| 命令 | 作用 |
|---|---|
| `uv pip install ...` | 装进环境，依赖声明与锁文件保持原样。 |
| `uv add numpy` | 写入核心依赖、更新 `uv.lock` 并安装。源码直接 import 的包应声明核心依赖。 |
| `uv add --optional lab numpy` | 写入 lab extra，可用于仅测试或实验的包。 |
| `uv sync --extra lab` | 按项目声明与锁文件同步环境，启用 lab。 |
| `uv run pytest ...` | 直接使用项目环境执行，通常可省手动激活。 |

原配置示例中核心 `torch>=2.7,<3`、`numpy>=2.0`，lab 组为 `pytest>=8,<10`、`fastapi>=0.115,<1`、`httpx>=0.28,<1`、`uvicorn>=0.30,<1`。`pyproject.toml` 保存项目元数据、依赖、构建与工具配置。`uv.lock` 锁定完整依赖，适合纳入版本管理。隔离环境 `.venv/` 留在本机。`src/tinyserve/` 是实际包，另有 `tests/` 与 `scripts/`。

```bash
source .venv/bin/activate
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
python -m pytest --collect-only -q
```

原笔记 `learning_notes/environment.md`。真实模型所需依赖及入口见 [总索引](../README.md#下一步真实-qwen3)。
