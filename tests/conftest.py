"""Test bootstrap for imports that pytest cannot resolve on its own.

pytest 単体では解決できない import のためのブートストラップ。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from yoshidacomp.config import reload_config
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

_REAL_CONFIG_PATH = _REPO_ROOT / "config.toml"


@pytest.fixture(autouse=True)
def _restore_real_config() -> None:
    """Force the shared ``yoshidacomp.config`` singleton back to the real
    ``config.toml`` (1920x1080 projector) before *and* after every test.

    ``yoshidacomp.config`` is a process-wide singleton (``get_config`` /
    ``reload_config``). ``sample.main()`` (``examples/python/sample.py:626``)
    calls ``reload_config`` internally with a temporary config carrying a
    tiny 4x3 test projector, and never restores it -- neither does
    ``sample_inverse_lut.main()``. Any test in ``tests/test_sample_mocked.py``
    or ``tests/test_sample_inverse_lut_mocked.py`` that drives one of those
    entry points therefore leaves the singleton poisoned for whatever test
    happens to run next, and the outcome (e.g. whether a matrix classifies
    as ``"projector"`` or ``"camera"`` space) silently depends on collection
    order. ``tests/test_inverse_lut.py`` works around this locally by calling
    ``reload_config`` back to the real path at the end of its own two
    offending tests (``:74``, ``:744``); the sample-script tests do not.

    Restoring is done on *both* sides of the test, not just one:

    - Before (setup): guarantees every test starts from the real config
      regardless of what an earlier test left behind, which is what
      actually eliminates the order-dependence -- a test run right after a
      polluting one, and a test run on its own via ``pytest
      tests/test_web_views.py``, see the identical starting state.
    - After (teardown): leaves the singleton clean for anything that reads
      it after the test body has finished but is not itself a test in this
      autouse fixture's rotation (e.g. session-scoped fixtures elsewhere,
      or a REPL importing the test modules interactively), and keeps the
      invariant "the real config is loaded whenever no test is currently
      overriding it" true at all times rather than only between tests that
      happen to both carry this fixture.

    ``yoshidacomp.config`` はプロセス全体で共有されるシングルトンである。
    ``sample.main()``（``examples/python/sample.py:626``）は幅4x高さ3の
    テスト用一時configで内部的に ``reload_config`` を呼び出し、後始末を
    しない。``sample_inverse_lut.main()`` も同様。これらを経由する
    ``tests/test_sample_mocked.py`` / ``tests/test_sample_inverse_lut_mocked.py``
    のテストは、次に実行されるテストのためにシングルトンを汚染したまま
    残し、結果（行列が ``"projector"``/``"camera"`` のどちらに判定される
    か等）が実行順序に暗黙に依存してしまう。``tests/test_inverse_lut.py``
    は該当2テストの末尾（``:74``、``:744``）で個別に実config
    へ戻して対処しているが、sample系の2ファイルは対処していない。

    前後どちらでも戻す理由: 前（setup）は直前のテストが何を残していよう
    と必ず実configから始まることを保証し、実行順序依存を実際に解消する
    （汚染テストの直後に実行されるテストも、``pytest
    tests/test_web_views.py`` のように単体実行されるテストも同じ初期状態
    を見る）。後（teardown）は、このautouseフィクスチャの回転に含まれない
    何か（他所のセッションスコープfixtureや、テストモジュールを対話的に
    importするREPLなど）がテスト本体終了後に読む場合にもシングルトンを
    綺麗に保ち、「テストが上書き中でない限り実configがロードされている」
    という不変条件を常に真に保つ。
    """
    reload_config(_REAL_CONFIG_PATH)
    yield
    reload_config(_REAL_CONFIG_PATH)
