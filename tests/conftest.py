"""测试环境准备：把仓库根的 `src/` 放进 sys.path，让 tests 能 import chainworth。
Test bootstrap: put the repository's `src/` on sys.path so tests can import chainworth.

中文：
    用途：仓库根目录只允许五个文件（README 许可等），因此没有 `pyproject.toml`/`setup.py`，
          也不能 `pip install -e .`（P0 冻结的三个环境里不该出现本仓库的 egg-info）。
          所以在测试会话开始前把 `src/` 手工加进 `sys.path`，一条 pytest 就能从仓库根跑。
    依赖：标准库 + pytest。
    配置：无。
    对外：无（pytest 自动加载）。
    被谁用：tests/ 下全部测试。

English:
    Purpose: the repository root may only hold five files, so there is no pyproject/setup.py and
    no `pip install -e .` (the three frozen P0 environments should not carry this repo's
    egg-info). Adding `src/` to sys.path here keeps `pytest` runnable from the repository root.
    Depends on: standard library and pytest.
    Config: none.
    Public API: none (pytest loads it automatically).
    Callers: every test under tests/.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
