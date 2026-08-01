"""Frontend behaviour tests: the shipped JavaScript, driven under node.

The viewer's request ordering, view selection and button state are plain
JavaScript control flow with no Python counterpart, so ``pytest`` could not see
them at all: an edit to ``static/js/main.js`` could undo every one of those
guarantees with the whole suite still green. That is the same gap the golden
shader test exists to close on the GLSL side, and this module closes it on the
JavaScript side.

``tests/js/frontend_harness.mjs`` stubs a DOM and a WebGL context, stands up a
fake backend, and then **imports the shipped ``static/js/main.js`` itself** --
not a copy, not a transcription, and with it the whole module graph the browser
loads (``glview.js`` included, which is where the planar upload lives).
Everything it asserts, it asserts about the files the browser loads. Pointing it
at a deliberately mutated copy of that tree is how
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

# (id, scenario, [(js file, find, replace), ...], the checks that must fail)
#
# The last field is the *complete* set of failing check names, matched as
# substrings. Anything else failing is itself a finding; see the assertion.
# 最後の項目は落ちるべきチェック名の集合（部分一致）で、それ以外が落ちたら
# それ自体が発見である。
#
# Each entry removes exactly one guard, reproducing the behaviour that shipped
# before it existed. If any of these stops failing, the corresponding test above
# has stopped testing anything.
#
# The file is named per edit because the guards do not all live in ``main.js``:
# the planar upload is ``glview.js``'s ``setImage``, and a mutation table that
# could only reach one file would have left that one permanently unguarded.
# 各項目はガードを1つだけ外し、修正前の挙動を再現する。ここが落ちなくなったら
# 対応するテストは何も検証していない。編集ごとにファイル名を持つのは、ガードが
# ``main.js`` だけにあるわけではないため（planar のアップロードは ``glview.js``）。
_MUTATIONS = [
    pytest.param(
        "ordering",
        [("main.js", "  if (isStale(req)) return;   // a newer request owns the screen now\n", "")],
        ("slow view fetch", "state.payload is the newest matrix's data"),
        id="reload_drops_its_stale_check",
    ),
    pytest.param(
        "ordering",
        [("main.js", "    if (isStale(req)) return;\n    state.matrixId = res.id;", "    state.matrixId = res.id;")],
        ("slow session POST", "displayed metadata describes the newest matrix (D)"),
        id="load_drops_its_stale_check",
    ),
    pytest.param(
        "button",
        [
            ("main.js", "  setLoadBusy(true);", "  $(\"btn-load\").disabled = true;"),
            ("main.js", "    setLoadBusy(false);", "    if (!isStale(req)) $(\"btn-load\").disabled = false;"),
        ],
        ("released after a reload", "released after a view switch"),
        id="button_borrows_the_request_counter",
    ),
    pytest.param(
        "views",
        [("main.js", "  if (!SUPPORTED_KINDS.includes(kind)) {", "  if (false) {")],
        ("unsupported kind is refused", "leaves state.payload identity unchanged"),
        id="unsupported_kind_renders_as_scalar",
    ),
    pytest.param(
        "views",
        [("main.js", "    buildViewSelect(specs);\n", "")],
        (
            # Removing the selector removes the only route into every view
            # switch the scenario makes, so the blast radius really is this
            # wide. Listed in full rather than narrowed, because that breadth
            # is the finding: nothing after the first load is reachable.
            # セレクタを外すと以降のビュー切替の入口が全て消えるため、影響は
            # 実際にこれだけ広い。その広さ自体が事実なので省略せず並べる。
            "view selector is populated",
            "a second registered view is reachable",
            "unsupported kind is refused with a message naming it",
            "a view without a `normalize` param is fetched without one",
            "its per-tile control is gone from the DOM",
            "its ranges are shared, not per-tile",
            "a response without X-Layout falls back to interleaved",
        ),
        id="view_selector_removed",
    ),
    # The rgb view's two halves, removed one at a time. Both mutations still
    # *render* -- that is the point of testing them. One draws three greyscale
    # tiles where one colour tile belongs; the other draws one tile of channel 0
    # only. Neither throws, neither logs, and both look like a working viewer.
    # rgbビューの2つの要素を1つずつ外す。どちらも描画自体は成功してしまう。
    pytest.param(
        "rgb",
        [("main.js", '  if (payload.kind === "rgb" && payload.channels === 3) {', "  if (false) {")],
        (
            "draws one tile",
            # The three tiles share one range object, so applyDisplay reads
            # state.ranges[1] === undefined and throws: the "renders" check
            # fails too, loudly, for a reason downstream of the layout.
            # 3枚が1組のレンジを共有するため applyDisplay が undefined を読んで
            # 例外になる。レイアウトの下流の理由で「描けた」も落ちる。
            "an rgb view renders instead of being refused",
            "packed interleaved RGB texture",
        ),
        id="rgb_view_falls_back_to_one_tile_per_channel",
    ),
    pytest.param(
        "rgb",
        [("main.js", '        mode: payload.kind === "rgb" ? "rgb" : "scalar",', '        mode: "scalar",')],
        ("packed interleaved RGB texture", "told it is in rgb mode", "u_mode = 1 for the rgb tile"),
        id="rgb_view_uploads_only_its_first_channel",
    ),
    # ---- the planar wire format's three consumers, one mutation each
    #
    # These were demonstrated to be discriminating by hand when the planar
    # format landed, and that demonstration went into a report rather than into
    # CI -- which is to say it stopped being true the moment nobody re-ran it.
    # Every one of the three reads the same body through a different index
    # calculation, and every one of them fails *silently*: a wrong plane still
    # draws a picture, a wrong probe index still prints numbers, a wrong
    # histogram still yields twelve finite ranges. Only the correspondence
    # between a tile's label and its contents is lost, and no exception marks it.
    # planar なワイヤフォーマットを読む3箇所に1つずつ。実装時に手で判別力を
    # 確認したがCIには残らなかった＝誰も再実行しなくなった時点で保証が消えた。
    # 3箇所とも同じ本体を別の添字計算で読み、いずれも静かに壊れる。
    pytest.param(
        "views",
        [(
            "glview.js",
            "      plane = img.data.subarray(channel * n, (channel + 1) * n);",
            "      plane = img.data.subarray(((channel + 1) % channels) * n,"
            " (((channel + 1) % channels) + 1) * n);",
        )],
        ("planar body: tile c uploads",),
        id="planar_upload_slices_the_neighbouring_channel",
    ),
    pytest.param(
        "views",
        [("main.js", "  const planar = p.layout === \"planar\";", "  const planar = true;")],
        ("interleaved body: the hover readout",),
        id="probe_assumes_every_body_is_planar",
    ),
    pytest.param(
        "views",
        [("main.js", "  const planar = layout === \"planar\";", "  const planar = false;")],
        ("per-tile ranges follow each channel",),
        id="per_tile_ranges_read_a_planar_body_as_interleaved",
    ),
    # ---- the two halves of "how many tiles", and the rgb channel contract
    pytest.param(
        "rgb",
        [(
            "main.js",
            '  if (payload.kind === "rgb" && payload.channels === 3) {',
            "  if (payload.channels === 3) {",
        )],
        ("kind is scalar still draws one tile per channel",),
        id="single_tile_layout_keyed_on_the_channel_count_alone",
    ),
    pytest.param(
        "rgb",
        [(
            "main.js",
            "  requireRgbHasThreeChannels(state.spec.kind, state.spec.channels, specName);\n",
            "",
        )],
        ("refused before the transfer",),
        id="rgb_channel_count_checked_only_after_the_transfer",
    ),
]


@pytest.mark.parametrize("scenario,edits,expected_failures", _MUTATIONS)
def test_harness_catches_the_regressions_it_exists_for(
    tmp_path,
    scenario: str,
    edits: list[tuple[str, str, str]],
    expected_failures: tuple[str, ...],
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

    for filename, find, replace in edits:
        target = mutated_static / "js" / filename
        source = target.read_text(encoding="utf-8")
        assert find in source, f"mutation anchor no longer present in {filename}: {find!r}"
        target.write_text(source.replace(find, replace, 1), encoding="utf-8", newline="")

    checks = _run_harness(mutated_static, scenario)
    failed = [c["name"] for c in checks if not c["ok"]]
    assert failed, (
        f"removing this guard changed nothing: scenario {scenario!r} still passes, "
        "so the test that relies on it proves nothing"
    )
    missing = [p for p in expected_failures if not any(p in name for name in failed)]
    assert not missing, (
        f"expected a failing check mentioning each of {missing}; got {failed}"
    )
    # The failure set is pinned exactly, not just "at least one of them".
    #
    # Detection alone says "something broke". What makes the table useful is
    # that each entry names the checks that watch that one guard, so a future
    # failure points at the line to look at rather than at the scenario. A
    # mutation that begins tripping a check nobody listed means either the
    # guard grew a second job or two checks stopped being independent -- both
    # worth knowing, and neither visible if unexpected failures are tolerated.
    # 落ちるチェックの集合まで固定する。検出だけなら「何かが壊れた」で終わる。
    # どのガードをどのチェックが見ているかを表に書いておけば、将来の失敗から
    # 見るべき行へ辿れる。想定外のチェックが落ち始めたら、ガードが仕事を増やした
    # か、チェック同士が独立でなくなったかのどちらかで、どちらも知る価値がある。
    unexpected = [n for n in failed if not any(p in n for p in expected_failures)]
    assert not unexpected, (
        f"this mutation also broke checks nobody expected: {unexpected}. "
        f"Expected exactly the ones matching {list(expected_failures)}."
    )
