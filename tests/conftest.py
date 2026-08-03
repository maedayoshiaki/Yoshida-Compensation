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


@pytest.fixture
def stand_in_projector_config(tmp_path):
    """Return a callable that loads a throwaway projector config for one test.

    Any test whose *assertions* depend on how a matrix classifies
    (``"projector"`` vs ``"camera"``, and therefore whether the active-region
    NaN mask runs) must not read those numbers out of the repository's live
    ``config.toml``. That file is an experiment log edited constantly -- it has
    eight sibling variants in the working tree right now -- and this project's
    projector has been 1280x720 in its own history, so "1080x1920 is projector
    space" is a fact about today's experiment, not about the web layer.

    A stale ``active_*`` rectangle is worse: ``resolve_active_region`` raises
    when the rectangle does not fit the frame, which turns a web-layer 200 into
    a 400 and reds the suite for a reason that has nothing to do with the code
    under test.

    The autouse ``_restore_real_config`` fixture above is still the isolation
    backstop -- it puts the real config back after the test either way. This
    fixture only stops assertions from depending on the live file's contents.

    行列の座標空間判定に依存するテストは、実験のたびに編集される実
    ``config.toml`` の値を当てにしてはならない。代用configをテスト内で
    読み込み、後始末は上の autouse フィクスチャに任せる。
    """

    def _load(
        width: int,
        height: int,
        *,
        active_x: int = 0,
        active_y: int = 0,
        active_width: int = 0,
        active_height: int = 0,
        gamma: float | None = None,
    ) -> Path:
        """Write and activate a minimal ``[projector]`` config; return its path.

        Non-positive ``active_width``/``active_height`` keep the full frame,
        matching ``resolve_active_region``'s own default.

        *gamma* is omitted from the file unless given, so callers that do not
        care keep ``ProjectorConfig``'s compiled-in 2.2. A caller that asserts
        on a recorded gamma must pass one: 2.2 is also ``--target-gamma``'s
        default and ``decode_target``'s, so an expected value of 2.2 cannot
        tell a gamma read from the projector config from one read out of the
        wrong place -- the third of this plan's recurring shapes ("an expected
        value that coincides with a real default").
        *gamma* は省略時にはファイルに書かない（既定の 2.2 のまま）。記録された
        gamma を assert する側は必ず明示すること。2.2 は他の既定値とも一致する
        ため、期待値が 2.2 だと判別力が無い。
        """
        path = tmp_path / "stand_in_projector_config.toml"
        gamma_line = "" if gamma is None else f"gamma = {gamma}\n"
        path.write_text(
            f"""
[projector]
width = {width}
height = {height}
active_x = {active_x}
active_y = {active_y}
active_width = {active_width}
active_height = {active_height}
{gamma_line}""",
            encoding="utf-8",
        )
        reload_config(path)
        return path

    return _load
