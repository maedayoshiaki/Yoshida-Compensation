"""Frontend behaviour tests: the shipped JavaScript, driven under node.

The viewer's request ordering, view selection and button state are plain
JavaScript control flow with no Python counterpart, so ``pytest`` could not see
them at all: an edit to ``static/js/main.js`` could undo every one of those
guarantees with the whole suite still green. That is the same gap the golden
shader test exists to close on the GLSL side, and this module closes it on the
JavaScript side.

``tests/js/frontend_harness.mjs`` stubs a DOM and a WebGL context, stands up a
fake backend, and then **imports the shipped ``static/js/main.js`` itself** --
not a copy, not a transcription. Everything it asserts, it asserts about the
file the browser loads. Pointing it at a deliberately mutated copy is how
``test_harness_catches_the_regressions_it_exists_for`` proves these checks can
actually fail; a frontend test that cannot fail is worse than none, because it
implies coverage that is not there.

Requires ``node`` (no npm, no ``package.json``, no dependency of any kind --
the harness is one file using only node built-ins). Where node is unavailable
the module skips rather than fails.

ビューアのリクエスト順序・ビュー選択・ボタン状態は純粋なJSの制御フローで、
Python側からは見えない。``static/js/main.js`` を書き換えれば全滅しうるのに
テストは緑のままだった。ハーネスは出荷物の ``main.js`` をそのまま import し、
写しを作らない。変異させたコピーに向けて失敗することも検証する。
node が無い環境ではスキップする（npm も package.json も不要）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = _REPO_ROOT / "tests" / "js" / "frontend_harness.mjs"
_STATIC_DIR = _REPO_ROOT / "packages" / "web" / "yoshidacomp_web" / "static"


def _node_executable() -> str:
    """Return the node interpreter, or skip the test.

    ``YOSHIDACOMP_NODE`` overrides the lookup, for a machine where node is
    installed but not on ``PATH`` (and, in turn, is how the skip path itself
    gets exercised).
    ``YOSHIDACOMP_NODE`` で明示指定できる。未設定なら PATH から探し、
    見つからなければスキップする。
    """
    candidate = os.environ.get("YOSHIDACOMP_NODE") or shutil.which("node")
    if not candidate or not Path(candidate).is_file():
        pytest.skip(
            "node was not found (set YOSHIDACOMP_NODE or put node on PATH); "
            "the frontend JavaScript tests need it"
        )
    return candidate


def _run_harness(static_dir: Path, scenario: str) -> list[dict]:
    """Run one harness scenario against *static_dir* and return its checks."""
    node = _node_executable()
    completed = subprocess.run(
        [node, str(_HARNESS), str(static_dir), scenario],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        cwd=str(_REPO_ROOT),
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise AssertionError(
            f"harness scenario {scenario!r} did not produce a result "
            f"(exit {completed.returncode})\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)["checks"]
    except (ValueError, KeyError) as exc:  # pragma: no cover - diagnostic path
        raise AssertionError(
            f"harness scenario {scenario!r} printed unparseable output: {exc}\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        ) from exc


def _assert_all_passed(checks: list[dict]) -> None:
    """Fail with every failing check spelled out, not just the first."""
    assert checks, "the harness reported no checks at all"
    failed = [c for c in checks if not c["ok"]]
    if failed:
        detail = "\n".join(f"  - {c['name']}: {c['detail']}" for c in failed)
        raise AssertionError(f"{len(failed)} frontend check(s) failed:\n{detail}")


# --------------------------------------------------------------------- F2/F4


def test_out_of_order_responses_cannot_cross_metadata_and_payload() -> None:
    """A late response must never publish over a newer one (F2, spec §7).

    Two loads are issued and resolved in the opposite order, once with the slow
    await in ``reload()`` (the binary view fetch) and once with it in
    ``load()`` (``POST /session/matrix``) -- the two have separate guards. The
    failure this prevents is not a crash: the header describes matrix B while
    ``state.payload``, which backs all 12 tiles *and* the hover readout, still
    holds matrix A. Two clicks, no warning, coefficients read off a matrix the
    user is not looking at.
    """
    _assert_all_passed(_run_harness(_STATIC_DIR, "ordering"))


def test_view_selector_reaches_a_second_view_and_refuses_unknown_kinds() -> None:
    """A registered view must be reachable, and an unknown kind must fail (F4).

    ``REGISTRY`` is insertion-ordered with ``coef_tiles`` first, so before the
    selector existed a newly registered view could not be selected at all and
    the spec's "register in Python, no JavaScript change" promise was false on
    its first use. The ``kind`` half matters differently: this frontend
    implements ``scalar`` only, and drawing ``rgb`` or ``categorical`` data
    with the scalar colormap would produce plausible pixels for a view it does
    not understand.
    """
    _assert_all_passed(_run_harness(_STATIC_DIR, "views"))


def test_an_rgb_view_draws_one_colour_tile_and_keeps_a_three_channel_readout() -> None:
    """An ``rgb`` view must be one colour picture, not three greyscale ones.

    Nothing about this fails loudly on its own, which is why it is here. The
    grid is derived from the channel labels, and ``surface``'s are ``R``/``G``/
    ``B`` -- not the ``"a -> b"`` form ``axesFromLabels`` parses -- so the
    label-derived fallback yields a 1x3 grid and each component goes through
    the scalar path on its own tile. That draws. It draws three greyscale
    pictures of a colour, from a viewer that looks like it is working.

    The upload is checked at the same time, because the wire format is planar
    and WebGL textures are interleaved: the one pack this path needs has to
    actually happen, and the harness's recorded ``head`` is what separates a
    correct ``R,G,B,R,G,B`` upload from the planar buffer passed through
    unchanged (identical length, first and last).
    """
    _assert_all_passed(_run_harness(_STATIC_DIR, "rgb"))


def test_load_button_is_released_after_a_reload_during_a_load() -> None:
    """Touching a server-side control mid-load must not wedge 読み込み.

    ``reload()`` bumps the shared request counter, so a button keyed off that
    counter is never released by the ``load()`` that disabled it -- changing
    正規化 during exactly the long transfer the ordering guard was written for
    left the button dead until a page reload, taking Enter in the path box with
    it. Exercised through both entry points that call ``reload()``: the
    server-parameter control and the view selector.
    """
    _assert_all_passed(_run_harness(_STATIC_DIR, "button"))


# ------------------------------------------------------------- discrimination

# (id, scenario, [(find, replace), ...], substring of a check that must fail)
#
# Each entry removes exactly one guard, reproducing the behaviour that shipped
# before it existed. If any of these stops failing, the corresponding test above
# has stopped testing anything.
# 各項目はガードを1つだけ外し、修正前の挙動を再現する。ここが落ちなくなったら
# 対応するテストは何も検証していない。
_MUTATIONS = [
    pytest.param(
        "ordering",
        [("  if (isStale(req)) return;   // a newer request owns the screen now\n", "")],
        "slow view fetch",
        id="reload_drops_its_stale_check",
    ),
    pytest.param(
        "ordering",
        [("    if (isStale(req)) return;\n    state.matrixId = res.id;", "    state.matrixId = res.id;")],
        "slow session POST",
        id="load_drops_its_stale_check",
    ),
    pytest.param(
        "button",
        [
            ("  setLoadBusy(true);", "  $(\"btn-load\").disabled = true;"),
            ("    setLoadBusy(false);", "    if (!isStale(req)) $(\"btn-load\").disabled = false;"),
        ],
        "released after a reload",
        id="button_borrows_the_request_counter",
    ),
    pytest.param(
        "views",
        [("  if (!SUPPORTED_KINDS.includes(kind)) {", "  if (false) {")],
        "unsupported kind is refused",
        id="unsupported_kind_renders_as_scalar",
    ),
    pytest.param(
        "views",
        [("    buildViewSelect(specs);\n", "")],
        "view selector is populated",
        id="view_selector_removed",
    ),
    # The rgb view's two halves, removed one at a time. Both mutations still
    # *render* -- that is the point of testing them. One draws three greyscale
    # tiles where one colour tile belongs; the other draws one tile of channel 0
    # only. Neither throws, neither logs, and both look like a working viewer.
    # rgbビューの2つの要素を1つずつ外す。どちらも描画自体は成功してしまう。
    pytest.param(
        "rgb",
        [('  if (payload.kind === "rgb") {', "  if (false) {")],
        "draws one tile",
        id="rgb_view_falls_back_to_one_tile_per_channel",
    ),
    pytest.param(
        "rgb",
        [('        mode: payload.kind === "rgb" ? "rgb" : "scalar",', '        mode: "scalar",')],
        "packed interleaved RGB texture",
        id="rgb_view_uploads_only_its_first_channel",
    ),
]


@pytest.mark.parametrize("scenario,edits,expected_failure", _MUTATIONS)
def test_harness_catches_the_regressions_it_exists_for(
    tmp_path, scenario: str, edits: list[tuple[str, str]], expected_failure: str
) -> None:
    """Each guard, removed from a copy, must make its scenario fail.

    Without this, the stubs above could quietly decay into a harness that
    passes no matter what the frontend does -- a real hazard with ~200 lines of
    fake DOM, where one over-permissive stub is enough to swallow the failure.
    スタブが緩むとどんな実装でも通るハーネスになりうるので、外したガードごとに
    実際に落ちることを確認する。
    """
    mutated_static = tmp_path / "static"
    shutil.copytree(_STATIC_DIR, mutated_static)
    main_js = mutated_static / "js" / "main.js"

    source = main_js.read_text(encoding="utf-8")
    for find, replace in edits:
        assert find in source, f"mutation anchor no longer present in main.js: {find!r}"
        source = source.replace(find, replace, 1)
    main_js.write_text(source, encoding="utf-8", newline="")

    checks = _run_harness(mutated_static, scenario)
    failed = [c["name"] for c in checks if not c["ok"]]
    assert failed, (
        f"removing this guard changed nothing: scenario {scenario!r} still passes, "
        "so the test that relies on it proves nothing"
    )
    assert any(expected_failure in name for name in failed), (
        f"expected a failing check mentioning {expected_failure!r}; "
        f"got {failed}"
    )
