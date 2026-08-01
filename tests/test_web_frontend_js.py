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
    """A registered view must be reachable; unknown kinds and layouts must fail.

    ``REGISTRY`` is insertion-ordered with ``coef_tiles`` first, so before the
    selector existed a newly registered view could not be selected at all and
    the spec's "register in Python, no JavaScript change" promise was false on
    its first use.

    The two contract guards matter differently. This frontend implements
    ``scalar`` and ``rgb``; drawing ``categorical`` data with the scalar
    colormap would produce plausible pixels for a view it does not understand.
    And ``X-Layout`` decides four separate index calculations, so a value
    outside ``{planar, interleaved}`` -- a header a proxy rewrote, or "Planar"
    with a capital P -- would send all four to the interleaved walk at once and
    have the tiles, the hover readout and the per-tile ranges corroborate one
    another about the wrong channels.
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


def test_the_hover_readout_reads_the_matrix_through_the_point_query() -> None:
    """The readout must show the matrix on every view, one pixel at a time.

    It used to read ``state.payload``, which is the *displayed view* -- so
    hovering ``surface`` printed three projection-surface colours under
    headings that said ``in_R -> out_R``. The matrix itself never crosses the
    wire (a second 99 MB copy would undo the heap the planar transfer bought),
    so the table is filled from ``GET /api/matrix/{id}/probe``.

    Two of the checks are about timing rather than arithmetic and neither can
    be observed against a fetch stub that settles immediately: the coalescing
    (a mousemove fires dozens of times a second, and every one of them would
    otherwise be a round trip) and the staleness guard (a response that comes
    back for a pixel the pointer has already left must not be painted under
    the newer position's heading). The harness holds point-query responses
    open so both states are reachable.

    読み取り欄は表示中のビューではなく行列を出す。行列本体は線に乗せないので、
    1画素ずつの点クエリで取る。合流とスタール判定は応答を保留できないと
    観測できないため、ハーネス側で点クエリの応答を止められるようにしてある。
    """
    _assert_all_passed(_run_harness(_STATIC_DIR, "probe"))


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

# (id, scenario, [(file under static/, find, replace), ...], the checks that must fail)
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
# The file is named per edit, as a path relative to ``static/``, because the
# guards do not all live in ``main.js``: the planar upload is ``glview.js``'s
# ``setImage``, and *where an element sits* is ``index.html`` -- a table that
# could only reach ``static/js/`` would have left the page's own structure
# permanently unguarded, which is precisely where the readout's placement lives.
# 各項目はガードを1つだけ外し、修正前の挙動を再現する。ここが落ちなくなったら
# 対応するテストは何も検証していない。編集ごとに ``static/`` からの相対パスを
# 持つのは、ガードが ``main.js`` だけにあるわけではないため（planar の
# アップロードは ``glview.js``、要素の置き場所は ``index.html``）。
_MUTATIONS = [
    pytest.param(
        "ordering",
        [("js/main.js", "  if (isStale(req)) return;   // a newer request owns the screen now\n", "")],
        ("slow view fetch", "state.payload is the newest matrix's data"),
        id="reload_drops_its_stale_check",
    ),
    pytest.param(
        "ordering",
        [("js/main.js", "    if (isStale(req)) return;\n    state.matrixId = res.id;", "    state.matrixId = res.id;")],
        ("slow session POST", "displayed metadata describes the newest matrix (D)"),
        id="load_drops_its_stale_check",
    ),
    pytest.param(
        "button",
        [
            ("js/main.js", "  setLoadBusy(true);", "  $(\"btn-load\").disabled = true;"),
            ("js/main.js", "    setLoadBusy(false);", "    if (!isStale(req)) $(\"btn-load\").disabled = false;"),
        ],
        ("released after a reload", "released after a view switch"),
        id="button_borrows_the_request_counter",
    ),
    pytest.param(
        "views",
        [("js/contract.js", "  if (!SUPPORTED_KINDS.includes(kind)) {", "  if (false) {")],
        ("unsupported kind is refused", "leaves state.payload identity unchanged"),
        id="unsupported_kind_renders_as_scalar",
    ),
    pytest.param(
        "views",
        [("js/main.js", "    buildViewSelect(specs);\n", "")],
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
            "a body declared interleaved is read as interleaved",
            "interleaved body: per-tile ranges follow each channel",
            "an unknown X-Layout is refused with a message naming it",
            "a missing X-Layout is refused rather than assumed to be interleaved",
            "a server refusal is shown as its detail message",
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
        [("js/main.js", "  if (traits.singleTile && payload.channels === 3) {", "  if (false) {")],
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
        [("js/main.js", "        kind: payload.kind,", '        kind: "scalar",')],
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
            "js/glview.js",
            "      plane = img.data.subarray(channel * n, (channel + 1) * n);",
            "      plane = img.data.subarray(((channel + 1) % channels) * n,"
            " (((channel + 1) % channels) + 1) * n);",
        )],
        ("planar body: tile c uploads",),
        id="planar_upload_slices_the_neighbouring_channel",
    ),
    pytest.param(
        "views",
        [("js/main.js", "  const planar = p.layout === LAYOUT_PLANAR;", "  const planar = true;")],
        ("interleaved body: the hover readout",),
        id="probe_assumes_every_body_is_planar",
    ),
    pytest.param(
        "views",
        [("js/main.js", "  const planar = layout === LAYOUT_PLANAR;", "  const planar = false;")],
        ("planar body: per-tile ranges follow each channel",),
        id="per_tile_ranges_read_a_planar_body_as_interleaved",
    ),
    # The same line forced the other way. Both walks of percentileAbsPerChannel
    # need their own entry, and until `legacy_tiles` was registered the
    # interleaved one had no execution anywhere: forcing it to the planar walk
    # broke nothing at all. A branch no test reaches is a branch free to rot,
    # and this one is the fallback kept for endpoints that were never converted
    # -- exactly the code least likely to be noticed when it stops working.
    # 同じ行を逆向きに固定する。`legacy_tiles` を足すまで、インタリーブ側の
    # 走査はどこからも実行されておらず、planar 側に置き換えても何も落ちなかった。
    pytest.param(
        "views",
        [("js/main.js", "  const planar = layout === LAYOUT_PLANAR;", "  const planar = true;")],
        ("interleaved body: per-tile ranges follow each channel",),
        id="per_tile_ranges_read_an_interleaved_body_as_planar",
    ),
    # ---- the two halves of "how many tiles", and the rgb channel contract
    pytest.param(
        "rgb",
        [(
            "js/main.js",
            "  if (traits.singleTile && payload.channels === 3) {",
            "  if (payload.channels === 3) {",
        )],
        ("kind is scalar still draws one tile per channel",),
        id="single_tile_layout_keyed_on_the_channel_count_alone",
    ),
    pytest.param(
        "rgb",
        [(
            "js/main.js",
            "  requireRgbHasThreeChannels(state.spec.kind, state.spec.channels, specName);\n",
            "",
        )],
        ("refused before the transfer",),
        id="rgb_channel_count_checked_only_after_the_transfer",
    ),
    # ---- the X-Layout contract, which used to fail open
    #
    # ``api.js`` defaulted an absent header to "interleaved" and four consumers
    # then compared against a bare ``"planar"`` literal with no validation
    # anywhere. A header that is absent, or present but saying something else --
    # "Planar", or whatever a proxy rewrote it to -- failed all four comparisons
    # at once and routed every one of them to the interleaved walk. Nothing
    # throws: every channel is attributed to the wrong tile, the hover readout
    # agrees with the wrong tiles, and the per-tile ranges agree with both.
    # Three independent readings corroborating a single mistake.
    #
    # Both halves fail together under one mutation because both reach the same
    # guard: ``mangled_layout`` sends ``"Planar"`` and ``headerless_tiles``
    # sends nothing, and ``LAYOUTS.includes`` is what refuses each.
    #
    # ヘッダが無い場合の既定が "interleaved" で、4つの読み手が素の文字列比較を
    # していた。欠落でも、値が有って未知（"Planar" など）でも、4つとも一斉に
    # インタリーブ側へ落ちる。例外は出ず、タイル・読み取り値・レンジが同じ
    # 間違い方で一致する。両方が同じガードに来るので変異は1つで足りる。
    pytest.param(
        "views",
        [("js/contract.js", "  if (!LAYOUTS.includes(layout)) {", "  if (false) {")],
        (
            "an unknown X-Layout is refused with a message naming it",
            "an unknown X-Layout leaves state.payload identity unchanged",
            "a missing X-Layout is refused rather than assumed to be interleaved",
            "a missing X-Layout leaves state.payload identity unchanged",
        ),
        id="unknown_layout_silently_read_as_interleaved",
    ),
    # ---- the colormap control, which was tested nowhere at all
    #
    # ``$("cmap-select").disabled`` was one of the five independent places that
    # dispatched on ``kind``, and the only one with no test and no mutation
    # entry: ``grep -n "cmap" tests/js/frontend_harness.mjs`` returned nothing.
    # Leaving it enabled for an rgb view offers a dropdown that does nothing --
    # the shader's rgb branch never reaches a colormap -- and implies the
    # picture is a false-colour render of a number rather than a colour.
    # kind で分岐する5か所のうち、これだけテストも変異項目も無かった。rgb で
    # 有効のままだと、何も起きないドロップダウンを見せることになる。
    pytest.param(
        "rgb",
        [("js/main.js", "  $(\"cmap-select\").disabled = !traits.colormap;",
          "  $(\"cmap-select\").disabled = false;")],
        ("the colormap control is disabled for an rgb view",),
        id="colormap_stays_enabled_for_an_rgb_view",
    ),
    # ---- the layout fallback that could inherit another kind's axes
    pytest.param(
        "rgb",
        [(
            "js/main.js",
            "  const inheritable = current.corner === MATRIX_CORNER\n"
            "    && current.rows.length * current.cols.length === payload.channels;",
            "  const inheritable ="
            " current.rows.length * current.cols.length === payload.channels;",
        )],
        ("does not inherit the rgb view's column label",),
        id="layout_fallback_inherits_axes_from_any_grid_of_the_right_size",
    ),
    # ---- one render per gesture, not one per control
    #
    # Each server-side control used to fetch on its own ``change``, so moving
    # the projection command from (1,1,1) to (0,0,0) cost three full renders --
    # ~740 ms and ~390 MB peak each on the 1080p matrix, two thirds discarded by
    # ``reqId`` after the fact, and computed concurrently because
    # ``render_view`` is a sync ``def`` whose NumPy kernels release the GIL.
    # Nothing about that is visible from the screen: the right picture appears,
    # eventually, having cost three times what it needed to.
    # コントロールごとに change で取得していたため、投影コマンドを3つ動かすと
    # フルレンダが3回走っていた（1080pで各約740ms・ピーク約390MB、うち2回分は
    # reqId が事後に捨てる）。画面からは見えない——正しい絵は最後には出る。
    pytest.param(
        "rgb",
        [("js/main.js", "    el.addEventListener(\"change\", () => { refreshParamsPending(); });",
          "    el.addEventListener(\"change\", () => { applyServerParams(); });")],
        (
            "changing three server params issues no fetch on its own",
            "適用 submits the whole group as exactly one fetch",
            # Staging now fetches, so nothing is ever left staged to show.
            # 置いた時点で送ってしまうので「未適用」が存在しなくなる。
            "staged-but-unsent changes are shown as pending",
            # And Enter's `change` fires a second fetch behind the keydown's.
            # Enter では keydown の1回に続いて change がもう1回撃つ。
            "Enter in a server-param field submits exactly one fetch",
        ),
        id="each_server_param_control_fetches_on_its_own",
    ),
    # ---- the "not applied" indicator, which used to be armed by events
    #
    # A `change`-armed / reload-cleared pair cannot be ordered correctly,
    # because the browser fires `keydown` *before* it commits the edit and fires
    # `change`. Enter therefore ran apply -> clear -> re-arm, and the ring stayed
    # lit over a value that had just been applied, with the status line still
    # asking for 適用. Forcing the derived comparison to a constant reproduces
    # exactly that: an indicator that reports its own event history rather than
    # the state it claims to describe.
    # `change` で立て reload で降ろす方式は順序では直せない。ブラウザは編集を
    # 確定して `change` を出す*前*に `keydown` を出すため、Enter では
    # 適用→解除→再点灯となる。比較を定数に潰すと、まさにその「状態ではなく
    # 自分のイベント履歴を報告する指標」が再現する。
    pytest.param(
        "rgb",
        [(
            "js/main.js",
            "  const pending = !!state.matrixId\n"
            "    && JSON.stringify(serverParams()) !== JSON.stringify(state.sentParams);",
            "  const pending = !!state.matrixId;",
        )],
        (
            "適用 clears the pending marker once the values are sent",
            "Enter does not leave 適用 claiming the value was never applied",
            "Enter does not leave the status line asking for 適用",
        ),
        id="pending_indicator_reports_events_not_state",
    ),
    # ---- a server refusal shown as its envelope instead of its message
    #
    # ``fetchView`` used to ``throw new Error(await res.text())``. The
    # camera-space refusal that ``views/surface.py`` exists to give therefore
    # reached the status bar as the literal string
    # ``{"detail":"view \"surface\" needs a projector-space matrix; …"}``.
    # The information was all there and the sentence was unreadable, which for
    # a message whose entire job is "load this other matrix instead" is most of
    # the way to not having sent one.
    # 拒否の本文をそのまま出していたため、`surface` が返す説明が JSON の外側
    # ごとステータス行に出ていた。情報はあるが文としては読めず、「代わりに
    # この行列を読め」という用途にはほとんど役に立たない。
    pytest.param(
        "views",
        [("js/api.js", '    if (parsed && typeof parsed.detail === "string") {', "    if (false) {")],
        ("a server refusal is not shown as a raw JSON envelope",),
        id="server_refusal_shown_as_its_json_envelope",
    ),
    # ---- the hover point query's two timing guards
    #
    # Both are invisible from a screenshot and invisible from a stub that
    # settles every fetch immediately, which is why the harness can hold a
    # point-query response open. Without the coalescing a mousemove is a round
    # trip: a drag across the image issues one request per event, and the
    # server answers every one of them. Without the staleness comparison the
    # answer for a pixel the pointer has already left is painted anyway --
    # under a ``#probe-pos`` that was updated synchronously and now names a
    # different pixel, so the readout states twelve coefficients that were
    # never true of the position it is captioned with.
    #
    # ホバーの点クエリが持つ2つの時間的ガード。どちらも画面写真からは見えず、
    # 即座に解決するスタブからも見えない（ゆえにハーネスは応答を保留できる）。
    # 合流が無ければ mousemove ごとに往復が発生し、スタール判定が無ければ
    # 「ポインタが去った画素の係数が、別の画素の見出しの下に」出る。
    pytest.param(
        "probe",
        [("js/main.js", "  if (state.probeInFlight) return;", "  if (false) return;")],
        ("a fast drag costs one point query in flight",),
        id="hover_probe_fetches_once_per_mousemove",
    ),
    pytest.param(
        "probe",
        [(
            "js/main.js",
            "        if (want.seq !== state.probeSeq) continue;   // the pointer has moved on",
            "        if (false) continue;   // the pointer has moved on",
        )],
        ("answered after its claim was retired",),
        id="hover_probe_paints_a_superseded_response",
    ),
    # ---- and the point query's coordinates, which are not symmetric
    #
    # The failure this repository has now shipped twice is the fixture that
    # cannot tell two arguments apart. ``probeMatrix(matrixId, row, col)``
    # feeds a query string whose two parameters are both plain integers, and a
    # square test matrix -- or a probe point where row happens to equal col --
    # accepts them in either order. The harness's matrix is 5x4 and the pixel
    # is (col 3, row 2), so a transposition stays in range and answers,
    # silently, for a different pixel.
    # 取り違えても範囲内に収まるため、黙って別の画素を答える。行と列を区別
    # できないフィクスチャこそ、このリポジトリが繰り返してきた失敗である。
    pytest.param(
        "probe",
        [(
            "js/main.js",
            "        const body = await probeMatrix(matrixId, want.row, want.col);",
            "        const body = await probeMatrix(matrixId, want.col, want.row);",
        )],
        (
            # Listed in full rather than narrowed: this one guard is upstream of
            # every value the table ever shows, so the breadth is the finding.
            # What survives is what does not depend on *which* pixel was asked
            # for -- the view-value panel (which never leaves the client), the
            # refusal path (out of range either way round) and the view
            # transfer. The null fixture is deliberately at (row 1, col 3)
            # rather than on the diagonal, so it fails here too instead of
            # being indifferent to the swap.
            # 1つのガードが表に出る全ての値の上流にあるため、この広さ自体が事実。
            # 生き残るのは「どの画素を訊いたか」に依存しないもの。null の
            # フィクスチャは対角線上ではなく (row 1, col 3) に置いてあるので、
            # 入れ替えに無関心ではなくここでも落ちる。
            "a point query for the hovered pixel",
            "the readout fills all twelve cells from the point query, row-major",
            "a null coefficient renders as a masked NaN cell",
            "a fast drag costs one point query in flight",
            "answered after its claim was retired",
            # Whether a pixel is inside the active region is per-pixel, so it
            # transposes with everything else: the fixture's out-of-region pixel
            # is at (row 0, col 2), whose transpose (row 2, col 0) is inside.
            # The marker therefore goes missing rather than surviving the swap,
            # which is the whole reason it is not on the diagonal.
            # 領域の内外は画素ごとの性質なので、これも一緒に転置する。
            # フィクスチャの領域外画素 (row 0, col 2) の転置 (row 2, col 0) は
            # 領域内なので、印は付いたままにならず消える（対角線上に置かない
            # 理由がこれである）。
            "outside the active region is marked as such",
        ),
        id="hover_probe_transposes_row_and_col",
    ),
    # ---- the hover series merged back into the view series
    #
    # The one property Task 2's brief states in bold, and the one nothing
    # asserted until this entry existed: replacing every live use of
    # ``probeSeq`` with ``state.reqId`` passed all 76 checks in all five
    # scenarios, because every other hover in the harness follows a
    # ``settled()`` and so never overlaps a transfer.
    #
    # The damage is silent and it is not a corner case. ``reload()`` drops its
    # payload when ``isStale(req)``, so one mousemove during a transfer retires
    # the transfer that is still arriving: the tiles stay on the previous
    # matrix and the status line stays on 「ビュー取得中…」 for good. On the
    # 1080p matrix the transfer takes seconds with the pointer over the image
    # the whole time, so this is every load, not an unlucky one.
    #
    # 読み取り側の系列をビュー側の ``reqId`` に統合する変異。ブリーフが太字で
    # 述べていた唯一の性質でありながら、この項目ができるまで5シナリオ76件の
    # どれも検出しなかった（他のホバーはすべて ``settled()`` のあとで転送と
    # 重ならないため）。転送中の mousemove 1回でその転送が失効し、タイルは前の
    # 行列のまま、ステータス行は「ビュー取得中…」のまま固まる。
    pytest.param(
        "probe",
        [
            (
                "js/main.js",
                "  state.probeSeq += 1;\n"
                "  state.probeWanted = { col: hit.col, row: hit.row, seq: state.probeSeq };",
                "  state.reqId += 1;\n"
                "  state.probeWanted = { col: hit.col, row: hit.row, seq: state.reqId };",
            ),
            (
                "js/main.js",
                "        if (want.seq !== state.probeSeq) continue;   // the pointer has moved on",
                "        if (want.seq !== state.reqId) continue;   // the pointer has moved on",
            ),
            (
                "js/main.js",
                "        if (want.seq !== state.probeSeq) continue;\n        clearMatrixCells();",
                "        if (want.seq !== state.reqId) continue;\n        clearMatrixCells();",
            ),
            (
                "js/main.js",
                "  clearMatrixCells();\n  state.probeSeq += 1;\n  state.probeWanted = null;",
                "  clearMatrixCells();\n  state.reqId += 1;\n  state.probeWanted = null;",
            ),
        ],
        ("hovering during a view transfer does not stall the tiles",),
        id="hover_series_shares_the_view_request_counter",
    ),
    # ---- the view-value panel's layout branch, forced the other way
    #
    # ``probe_assumes_every_body_is_planar`` pins one direction; this pins the
    # other, matching how ``percentileAbsPerChannel``'s two walks each have
    # their own entry. Both directions are needed for the same reason there:
    # forcing the branch either way still prints twelve plausible numbers, and
    # only the correspondence between a label and its channel is lost.
    # もう一方の向きを固定する（``percentileAbsPerChannel`` と同じ扱い）。
    # どちらへ倒してももっともらしい数値は出て、ラベルと中身の対応だけが失われる。
    pytest.param(
        "views",
        [("js/main.js", "  const planar = p.layout === LAYOUT_PLANAR;", "  const planar = false;")],
        ("planar body: the hover readout",),
        id="view_values_read_a_planar_body_as_interleaved",
    ),
    # ---- the readout's own shape, which is a behaviour and not a decoration
    #
    # Both of these were the state that shipped, and neither produces an error
    # of any kind: the viewer draws, the numbers are right, and the only thing
    # wrong is that they cannot be read together. Twelve seven-character cells
    # stretched across a 1600px window put one number per 500px column, and the
    # display range and the three timings sat below the controls in a sidebar
    # that has ``overflow-y: auto`` -- off the bottom of a short window, while
    # the strip they belong in had room to spare.
    # どちらも実際に出荷されていた状態であり、エラーは一切出ない。描画は
    # 成功し、数値も正しく、ただ同時に読めないだけである。
    pytest.param(
        "probe",
        [(
            "js/main.js",
            "    `${PROBE_HEAD_PX}px repeat(${cols.length}, ${PROBE_CELL_PX}px)`;",
            "    `var(--rowhead) repeat(${cols.length}, minmax(0, 1fr))`;",
        )],
        ("the readout table's columns are fixed width",),
        id="readout_table_stretches_across_the_window",
    ),
    # Relocated rather than deleted: removing the elements would make
    # ``$("range-info")`` null and throw on the first render, which is a loud
    # failure and not the one this pins. The sidebar is where they used to be,
    # so this reproduces the layout exactly as it shipped.
    # 消すのではなく戻す。要素ごと消すと ``$("range-info")`` が null になって
    # 描画時に例外になり、それは派手な失敗であってここで見たいものではない。
    pytest.param(
        "probe",
        [
            (
                "index.html",
                "\n    <div class=\"rd-panel\">\n"
                "      <div class=\"rd-title mono\">表示レンジ</div>\n"
                "      <div class=\"rd-rows mono\">\n"
                "        <div><span id=\"range-info\" class=\"val\">—</span></div>\n"
                "      </div>\n"
                "    </div>\n"
                "\n    <div class=\"rd-panel\">\n"
                "      <div class=\"rd-title mono\">計測</div>\n"
                "      <div class=\"rd-rows mono\">\n"
                "        <div>転送 <span id=\"perf-fetch\" class=\"val\">—</span></div>\n"
                "        <div>GPU転送 <span id=\"perf-upload\" class=\"val\">—</span></div>\n"
                "        <div>JSヒープ <span id=\"perf-heap\" class=\"val\">—</span></div>\n"
                "      </div>\n"
                "    </div>\n",
                "",
            ),
            (
                "index.html",
                "      <div class=\"group legend\">",
                "      <div class=\"group readouts mono\">\n"
                "        <div>表示レンジ <span id=\"range-info\" class=\"val\">—</span></div>\n"
                "        <div>転送 <span id=\"perf-fetch\" class=\"val\">—</span></div>\n"
                "        <div>GPU転送 <span id=\"perf-upload\" class=\"val\">—</span></div>\n"
                "        <div>JSヒープ <span id=\"perf-heap\" class=\"val\">—</span></div>\n"
                "      </div>\n\n"
                "      <div class=\"group legend\">",
            ),
        ],
        ("the display range and the timings sit in the readout dashboard",),
        id="the_range_and_timings_go_back_into_the_sidebar",
    ),
    # ---- the heading that says whose numbers the second panel holds
    pytest.param(
        "probe",
        [(
            "js/main.js",
            "  $(\"view-title\").textContent = state.spec.label || state.spec.key;\n",
            "",
        )],
        ("names the view whose values it is showing",),
        id="view_value_panel_never_names_its_view",
    ),
    # ---- and the readout's last remaining lie
    #
    # ``inside_active_region`` was shipped by the point query and consumed by
    # nobody. The mask ``coef_tiles.py`` and ``surface.py`` paint is applied at
    # *render* time, so outside the active region the tile is grey while the
    # stored array still holds a finite coefficient -- and the readout, which
    # reads the array, printed it in the same type, colour and precision as a
    # real one. On data/linear_pipeline_1080/matrices_linear_gamma.npy (active
    # region [613, 201, 722, 722]) hovering row 5, col 5 gave
    # ``13.440 / 1.197 / 0.017`` under a grey tile, with nothing on screen to
    # say those numbers mean nothing. Dropping the call restores exactly that:
    # the position line is still written synchronously by ``showProbe``, so the
    # readout keeps working and keeps lying.
    # 点クエリが返している ``inside_active_region`` を誰も消費していなかった
    # 状態に戻す。位置は ``showProbe`` が同期的に書くので、読み取り欄は
    # 動いたまま嘘をつく。
    pytest.param(
        "probe",
        [(
            "js/main.js",
            "        setProbePos(want.col, want.row, body.inside_active_region === false);\n",
            "",
        )],
        ("outside the active region is marked as such",),
        id="readout_never_says_a_pixel_is_outside_the_active_region",
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
        target = mutated_static / filename
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
