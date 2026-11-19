# Python 环境与 uv

## pip、Anaconda 和 uv 的区别

| 工具 | 核心定位 |
| --- | --- |
| pip | Python 包安装工具，主要负责从 PyPI 安装包和解析依赖，不负责创建隔离环境。 |
| Anaconda / Conda | Python 发行版和环境管理工具，可以管理 Python 版本、环境及部分非 Python 系统依赖。 |
| uv | 高速 Python 包、环境和项目管理工具，可替代许多 pip、venv 等命令的用法。 |

同一个环境尽量固定一种主要安装方式，避免依赖管理混乱。本实验使用 uv 和 `.venv`，不需要 Anaconda。

## 创建和激活虚拟环境

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

- `uv venv`：创建虚拟环境。
- `--python 3.12`：指定 Python 3.12。
- `.venv`：将环境放在当前项目的 `.venv/` 目录中。
- 创建不等于激活；激活后，包与系统 Python 和其他项目隔离。

## 安装当前实验项目

```bash
uv pip install -e '.[lab]'
```

- `uv pip install`：以 pip 风格安装依赖，但由 uv 执行。
- `-e`：editable（可编辑）安装，源码修改后通常无需重新安装。
- `.`：安装当前目录中由 `pyproject.toml` 描述的项目。
- `[lab]`：同时安装项目的 `lab` 可选依赖组。
- 引号用于避免 shell 对方括号进行路径匹配。

## 在指定环境中补装依赖

激活环境后：

```bash
uv pip install numpy
```

或者不激活环境，显式指定 Python：

```bash
uv pip install --python .venv/bin/python numpy
```

## `uv pip` 与 `uv add` / `uv sync`

- `uv pip install ...`：只安装，不修改项目依赖声明，也不生成锁文件。
- `uv add numpy`：把 NumPy 写入 `pyproject.toml`，更新 `uv.lock` 并安装。
- `uv add --optional lab numpy`：把 NumPy 写入 `lab` 可选依赖组。
- `uv sync --extra lab`：按照项目声明和锁文件同步环境，同时启用 `lab` 依赖组。
- `uv run pytest ...`：直接使用项目环境执行命令，通常不必手动激活环境。

如果源码直接使用 NumPy，应声明为核心依赖；如果只用于测试或实验，可放入 `lab`：

```toml
[project]
dependencies = [
    "torch>=2.7,<3",
    "numpy>=2.0",
]

[project.optional-dependencies]
lab = [
    "pytest>=8,<10",
    "fastapi>=0.115,<1",
    "httpx>=0.28,<1",
    "uvicorn>=0.30,<1",
]
```

本项目采用 uv 管理依赖、setuptools 构建和 src 布局：

```text
vllm-labs/
├── pyproject.toml       # 项目元数据、依赖、构建和工具配置
├── uv.lock              # 锁定完整依赖版本，建议纳入版本管理
├── .venv/               # 本地虚拟环境，不纳入版本管理
├── src/tinyserve/       # 实际 Python 包
├── tests/
└── scripts/
```

## 常用验证命令

```bash
source .venv/bin/activate
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
python -m pytest --collect-only -q
```
