"""Test bootstrap for imports that pytest cannot resolve on its own.

pytest 単体では解決できない import のためのブートストラップ。
"""

from __future__ import annotations

import sys
from pathlib import Path

from yoshidacomp.geometry.graycode_path import ensure_on_syspath

# GrayCode is a git submodule, not an installed package.
# GrayCode は git submodule でありインストール済みパッケージではない。
ensure_on_syspath()

# ``examples`` is a plain directory in the repository root, not a
# distribution, so ``from examples.python import ...`` only resolves when the
# repository root is on sys.path. pytest inserts the *test* directory
# (``tests/``), not the root, so without this several test modules import
# ``examples.*`` successfully only by accident: ``tests/test_inverse_lut.py``
# sorts early and inserts the root as a side effect, and ``python -m pytest``
# prepends the working directory. Run one of those modules on its own with the
# ``pytest`` console script and it fails with ``ModuleNotFoundError: No module
# named 'examples'``. Bootstrapping here removes the ordering and invocation
# dependency for every affected module at once.
# ``examples`` はリポジトリルート直下のただのディレクトリであり配布物では
# ないため、``from examples.python import ...`` はリポジトリルートが
# sys.path にある場合のみ解決する。pytest が挿入するのは*テスト*
# ディレクトリ（``tests/``）でありルートではないので、これが無いと複数の
# テストモジュールは偶然に頼ることになる（``tests/test_inverse_lut.py`` が
# 先に並んで副作用でルートを挿入する／``python -m pytest`` が作業
# ディレクトリを前置する）。``pytest`` コンソールスクリプトで単体実行すると
# ``ModuleNotFoundError: No module named 'examples'`` で落ちる。ここで
# 一括してブートストラップし、順序と起動方法への依存を取り除く。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
