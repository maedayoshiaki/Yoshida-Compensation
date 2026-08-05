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


def test_a_categorical_view_draws_a_discrete_palette_with_a_legend() -> None:
    """A ``categorical`` view must be class indices, not small numbers.

    Nothing about the wrong reading fails loudly, which is the whole reason
    this scenario exists. A class index is a perfectly ordinary float: the
    scalar path uploads it as R32F, the diverging colormap gives it a colour,
    exposure and gamma move that colour, and the tile comes out as four flat
    regions in four shades of one ramp under a ``±`` range. It renders, and it
    renders something that looks like a working view of a classification.

    Four properties are pinned and each fails on its own:

    - **One tile, headed as one.** ``gamut``'s labels are class names with no
      ``"->"`` in them, so ``axesFromLabels`` returns null and the fallback
      heads the tile ``labels[0]`` -- the name of class 0, over a picture
      containing all four.
    - **R32F/RED.** RGB32F uploads the same twenty numbers, so a contents
      check cannot see it; the index lands in the red component with two
      undefined components beside it.
    - **The legend's labels come from ``payload.labels``.** The palette is
      JavaScript and the class names are the server's, and writing the names
      into ``index.html`` makes those two sources of truth for one mapping.
      ``views/gamut.py`` says in as many words that the *order* of its
      ``LABELS`` is the class index and part of the wire contract, so a
      relabelling there would leave the page explaining the previous meaning
      of the same four colours.
    - **Exposure, γ and the colormap do not apply to a name.** The shader's
      categorical branch reads none of the tone uniforms and the controls are
      disabled; either half alone is insufficient (a uniform can still be
      pushed from code; a live slider that moves nothing is the same lie the
      colormap dropdown on an rgb view was).

    What this scenario cannot see is what the pixels look like -- the stub has
    no GPU. That is measured separately, by compiling and drawing on
    ANGLE/D3D11 and reading the framebuffer back; see the task report.

    ``categorical`` はクラス番号の絵であり、小さな数値の絵ではない。誤った
    読み方でも描画は成功する（1本のランプ上の4段階として「それらしく」出る）
    ため、静かに壊れる4つの性質を個別に固定する。画素そのものはスタブでは
    見えないので、実GPUでの読み戻しは別に行う。
    """
    _assert_all_passed(_run_harness(_STATIC_DIR, "gamut"))


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


def test_the_target_loader_makes_the_compensation_view_drivable() -> None:
    """The ``compensation`` view must be reachable from the page, not curl.

    Its ``target`` parameter is free text naming an id that only exists after
    a ``POST /api/target`` (``views/base.py::ParamSpec.type`` carries the
    argument for why it cannot be a ``select``), and until this task no
    element on the page issued that POST. The view shipped complete in plan
    B3 Task 3 and every browser request it could receive came back "needs a
    target image" -- a view in the dropdown that could not be made to draw,
    with nothing failing anywhere.

    Three things have to hold and each fails silently on its own: the id the
    server returns has to reach the parameter box (a client that only
    remembered it internally looks identical from the outside and leaves the
    user an empty box), choosing another image of a directory has to re-issue
    the load *with that index* rather than derive an id locally, and a refused
    load has to be legible and leave the loader usable.

    ``compensation`` ビューをブラウザから駆動できること。ターゲットIDは
    ``POST /api/target`` の後にしか存在しないidであり、それを発行する要素が
    ページに無かったため、このビューは「選べるが描けない」状態で出荷されて
    いた。
    """
    _assert_all_passed(_run_harness(_STATIC_DIR, "target"))


def test_the_lut_loader_plots_three_curves_against_the_analytic_gamma() -> None:
    """The measured inverse LUT must be visible, and visible as itself.

    Three properties, all of which a plausible-looking plot can violate:

    - Each polyline has to be at *its own channel's* coordinates. Counting
      lines cannot see a channel swap, and three curves of the same shape
      cannot either -- so the fixture's three differ in shape and in endpoint,
      and every point of every line is compared against the values it claims
      to be.
    - The vertical axis has to come from the data. A measured LUT does not end
      at 1.0 (``data/full_recalibration_20260730_iso400`` stops at 0.813 /
      0.876 / 0.751), and an axis pinned to [0, 1] draws the same picture for
      a LUT that reaches 1.0 and one that stops at 0.69.
    - The analytic gamma has to be a fourth line at the exponent the *server*
      sent, not a hard-coded 2.2 and not one of the measured curves redrawn.

    Plus the backing store scaling with ``devicePixelRatio``, without which
    the whole plot is drawn into a quarter of the canvas and stretched on the
    150% display this project is used on.

    実測逆LUTが「それ自身として」見えること。本数を数えるだけでは
    チャンネルの入れ替えも、縦軸を 1.0 に固定した実装も検出できない。
    """
    _assert_all_passed(_run_harness(_STATIC_DIR, "lut"))


def test_the_generate_button_saves_the_settings_that_are_on_screen() -> None:
    """The one control that writes to disk must write what is displayed.

    Every other check in this module is about a picture, and a wrong picture is
    wrong until the next render. This one is about a file: a PNG written from a
    stale target id, a defaulted linearizer or an unapplied decode mode stays
    wrong on the user's disk, under a name saying it is what they were looking
    at, with nothing on screen afterwards to contradict it. Five plans have
    been spent removing exactly that class of divergence between what is shown
    and what is reported; this is the first place where the consequence is a
    file rather than a pixel.

    "Pressing the button issues a fetch" cannot see any of it -- a body of
    hard-coded defaults satisfies it as well as a correct one -- so the request
    body is compared field for field, against a fixture in which **every value
    differs from that parameter's own declared default** (target = image 2 of a
    directory against ``""``, linearizer = ``inverse_lut`` against ``"none"``,
    decode = ``gamma`` against ``"srgb"``, exponent 1.7 against 2.2). The key
    set is compared too, so a field the endpoint does not declare is a
    difference rather than something nobody looks at.

    Four more properties, each of which a plausible-looking build violates:

    - **"On screen" is the picture, not the boxes.** A staged-but-unapplied
      control is ahead of what is displayed, so generating there writes a file
      matching neither the picture nor the previous state -- only the boxes,
      which is the one thing nobody is reading. Both directions are checked;
      "always refuse" passes the first half alone.
    - **One press is one generation.** The call takes seconds on a real matrix,
      which is long enough for a second click, and two overlapping writes into
      one directory make the second meet the first one's files and come back
      409 -- a conflict about a directory the user is in the middle of filling.
    - **A 409 is a question.** ``generate.py`` writes nothing at all for a
      non-empty directory because a mistyped path here can name captures that
      cost hardware and a day to reproduce. A frontend that resent with
      ``overwrite: true`` on its own, or offered it as the default action,
      turns a refusal that stops every typo into a dialog people click through.
      Task 3's review found a partial-write path that left debris and thereby
      trained the user to reach for the override; the guard only survives if
      the UI keeps the answer deliberate.
    - **A view with no linearizer does not generate.** See the harness's own
      note, and the task report, for why refusing is the only answer that keeps
      "what is saved is what is displayed" true under ``gamut``. The fixture
      declares the same three parameters ``views/gamut.py`` does and omits only
      ``linearizer``, so a refusal keyed on how many parameters a view declares
      -- or on "any view that is not ``compensation``" -- fails here rather than
      passing against a shape nobody ships.
    - **The screen includes the matrix.** ``state.matrixId`` moves seconds
      before the transfer that draws it, and no control changes in between, so
      the staged-vs-applied guard passes and a press goes through. The body has
      to carry the matrix on screen, which is the one field of it whose source
      cannot be seen once a load has settled.
    - **The refusal is discoverable before the press.** A disabled button
      dispatches no click, so what a person meets under ``gamut`` is the
      button's appearance and the line beside it -- both asserted, after both
      shipped broken: the button was pixel-identical to a live one (no
      ``button:disabled`` rule existed) and the line said "this view cannot
      generate" on every view, from the first paint onward.
    - **The 上書き confirmation is about one directory.** Bound to the path the
      409 named, retired when the box is edited, and re-checked at press time so
      that a value set without an event cannot spend an agreement about A on B.

    唯一ディスクへ書くコントロールが、画面に出ている設定で書くこと。誤りは
    絵ではなくファイルとして残り、しかも「見ていた絵」の名前で残る。「押すと
    fetch が飛ぶ」では既定値だけで組んだ本文も通ってしまうので、本文を
    フィールド単位で、しかも全項目を既定値から動かした状態で突き合わせる。
    """
    _assert_all_passed(_run_harness(_STATIC_DIR, "generate"))


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


def test_the_browse_button_fills_the_box_the_way_typing_would() -> None:
    """``参照`` がパス欄を埋め、手打ちと区別が付かない状態にすること。

    ファイラの3つの性質はどれも静かに壊れる。フィルタをサーバへ渡さずに
    クライアント側で絞れば、2000件で切ったあとに絞ることになり件数表示が嘘に
    なる。一覧をクライアントで並べ替え直せば、``LoadTargetRequest.index`` の
    基準である ``target.py::_listing`` の casefold 順とずれ、「3枚目」が別の
    画像を指す。そして確定を代入だけで済ませれば、``gen-output-dir`` の編集
    リスナ（上書き合意を失効させる）を迂回する欄が1つできる。

    拒否の出し先も同じ種類の問題である。ページのステータス行は背景の裏に
    あるので、そこへ出した 403 は読めない。
    """
    _assert_all_passed(_run_harness(_STATIC_DIR, "picker"))


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
    # The refusal itself, removed: the viewer no longer says which kind it
    # cannot draw.
    #
    # Only one check fails, and the one that does not is worth reading. The
    # payload's identity survives this mutation because the *table* still
    # refuses: `traitsFor` finds no row for "flow", returns `undefined`, and
    # the first read of it throws before `state.payload = payload` runs. That
    # is the trait table doing its job with the guard gone -- which is why the
    # entry below, which makes the lookup answer `scalar` instead of nothing,
    # is a separate mutation rather than a variant of this one. Narrowing the
    # expectation here rather than making that read defensive: a `traits &&`
    # would turn "there is no row for this kind" back into a silent default.
    # 拒否の文言だけを外す。ペイロードが差し替わらないのは、ガードが無くても
    # 表が拒んでいるためである（行が無く `undefined` を読んだ時点で例外になり、
    # `state.payload` の代入まで到達しない）。既定行へ落とす変異は下に別項目
    # として置く。ここで `traits &&` のように守ると、それこそ静かな既定値に戻る。
    pytest.param(
        "views",
        [("js/contract.js", "  if (!SUPPORTED_KINDS.includes(kind)) {", "  if (false) {")],
        ("unsupported kind is refused",),
        id="unsupported_kind_renders_as_scalar",
    ),
    # ...and the failure the table was built to make impossible: an unlisted
    # kind reaching a default row.
    #
    # This is the pre-table behaviour exactly, restated in one line. Five sites
    # used to answer "scalar" for anything they did not recognise, and this
    # mutation reinstates all five at once from a single place: `vector_field`
    # (kind "flow") is then drawn as two greyscale coefficient tiles through
    # the diverging colormap, with per-channel ranges and a live colormap
    # dropdown. Nothing throws, nothing is logged, and the viewer paints
    # plausible pixels for a view it does not understand -- including
    # overwriting the payload that was on screen, which is why both checks fail
    # here while only one does above.
    # 表が防いでいる当の失敗（未登録の kind が既定行へ落ちる）。以前は5か所が
    # 独立に「知らなければ scalar」と答えていたが、それを1行で再現する。例外も
    # ログも出ず、理解していないビューがそれらしく描かれ、画面に出ていた
    # ペイロードも差し替わる。
    #
    # Two edits, one guard. The rule -- "unknown has exactly one meaning,
    # refused" -- is enforced at the check *and* at the only way to read the
    # table, and reinstating the fall-through means undoing both: with the
    # lookup alone defaulted, `reload()`'s own pre-flight `requireSupportedKind`
    # still refuses and the mutation changes nothing (measured -- the views
    # scenario stayed at 29/29). That the rule needs two lines to remove is a
    # property of the design and not of this entry.
    # 2つの編集で1つのガード。既定行へ落とす変更だけでは `reload()` 側の
    # 事前検査が依然として断り、何も変わらない（実測 29/29 のまま）。
    pytest.param(
        "views",
        [
            ("js/contract.js", "  if (!SUPPORTED_KINDS.includes(kind)) {", "  if (false) {"),
            (
                "js/contract.js",
                "  requireSupportedKind(kind, where);\n  return KIND_TRAITS[kind];",
                "  return KIND_TRAITS[kind] ?? KIND_TRAITS.scalar;",
            ),
        ],
        ("unsupported kind is refused", "leaves state.payload identity unchanged"),
        id="an_unlisted_kind_falls_through_to_the_scalar_row",
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
            # The heading is correct throughout -- it is still naming the only
            # view that ever rendered. It fails because the third view, the one
            # this check reads the heading on, is now unreachable.
            # 見出しは終始正しい（唯一描かれたビューを名乗り続ける）。落ちるのは
            # このチェックが見る3つ目のビューへ到達できなくなるためである。
            "the readout's heading names the view that rendered",
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
        [(
            "js/main.js",
            "  if (traits.singleTile && payload.channels === traits.tileChannels) {",
            "  if (false) {",
        )],
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
            "  if (traits.singleTile && payload.channels === traits.tileChannels) {",
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
            # The text control stages through the same listener as the number
            # ones, so it observes this guard too -- listed rather than allowed
            # to be an "unexpected" failure, which is the point of the pinned
            # set. It is not a duplicate: the three above are number controls,
            # and a change that special-cased `type === "number"` would leave
            # them staging while the text box fetched on every keystroke.
            # テキスト欄も同じリスナで置かれるためこのガードを観測する。
            # 上の3つは number 欄なので重複ではない（number だけを特別扱い
            # する変更なら、テキスト欄だけが1文字ごとに取得することになる）。
            "typing a target id stages it without fetching",
        ),
        id="each_server_param_control_fetches_on_its_own",
    ),
    # ---- the text-input branch of buildServerParams, narrowed
    #
    # ``ParamSpec(type="text")`` exists because a stored target or lut id only
    # comes into being at runtime and cannot be a static ``choices``. The
    # frontend serves it with the ``else`` branch that every non-number,
    # non-select type falls into -- a branch that had no execution anywhere
    # until ``views/compensation.py`` declared the first one.
    #
    # Narrowing it to ``"hidden"`` is the shape this fails in: the element is
    # still created, still registered, still submitted, so every other check in
    # every scenario passes and the Python suite cannot see the frontend at all.
    # In a browser the ターゲットID box is simply not there, ``serverParams()``
    # sends the default ``""``, and the view answers "needs a target image"
    # forever with no way to give it one.
    #
    # ``type="text"`` を出す ``else`` 枝を ``"hidden"`` に狭める。要素は作られ
    # 登録もされ送信もされるため、他のチェックもPython側も一切気付かない。
    # ブラウザ上ではターゲットID欄が消え、既定の ``""`` が送られ続けて
    # ビューは永遠に「ターゲットが要る」と断る。
    pytest.param(
        "rgb",
        [("js/main.js", 'el.type = p.type === "number" ? "number" : "text";',
          'el.type = p.type === "number" ? "number" : "hidden";')],
        ("the ターゲットID control is a text input",),
        id="text_param_renders_as_a_hidden_field",
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
            # And the check that watches *when* the marker moves needs a marked
            # pixel to watch; with the coordinates swapped there is not one.
            # 印がいつ動くかを見るチェックは、印の付く画素が要る。転置すると
            # そのような画素が無くなる。
            "the marker stays with the numbers it disclaims",
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
    # failure and not the one this pins. The sidebar block it writes is where
    # these lines used to live -- not a byte-exact restoration of the pre-branch
    # page, since ``#perf-ranges`` did not exist before this branch and is
    # carried along here so the id survives the move.
    # 消すのではなく戻す。要素ごと消すと ``$("range-info")`` が null になって
    # 描画時に例外になり、それは派手な失敗であってここで見たいものではない。
    # 書き戻す先はこれらの行が元々あった場所だが、ブランチ以前のページの
    # 完全な復元ではない（``#perf-ranges`` はこのブランチで増えた id であり、
    # 移動先で消えないよう一緒に運んでいる）。
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
                "      <!-- In the order the work happens. レンジ came up from the status line,\n"
                "           which was printing 転送 and GPU転送 again 20px below this panel.\n"
                "           作業の順。レンジはステータス行から移した（同じ行がこの板の20px下で\n"
                "           転送とGPU転送を繰り返していた）。 -->\n"
                "      <div class=\"rd-rows mono\">\n"
                "        <div>転送 <span id=\"perf-fetch\" class=\"val\">—</span></div>\n"
                "        <div>レンジ <span id=\"perf-ranges\" class=\"val\">—</span></div>\n"
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
                "        <div>レンジ <span id=\"perf-ranges\" class=\"val\">—</span></div>\n"
                "        <div>GPU転送 <span id=\"perf-upload\" class=\"val\">—</span></div>\n"
                "        <div>JSヒープ <span id=\"perf-heap\" class=\"val\">—</span></div>\n"
                "      </div>\n\n"
                "      <div class=\"group legend\">",
            ),
        ],
        ("the readout's panels all sit in the dashboard, not the sidebar",),
        id="the_range_and_timings_go_back_into_the_sidebar",
    ),
    # ---- the readout table, fixed but no longer compact
    #
    # `fr` is not the only way to give twelve seven-character cells the whole
    # window; a fixed 600px column does it too, and satisfies a check that only
    # forbids fractions. The bound the check carries is what makes "fixed width"
    # mean what the task meant by it.
    # 全幅に伸ばす方法は `fr` だけではない。固定600pxでも同じことが起き、
    # 分数だけを禁じるチェックは通ってしまう。
    pytest.param(
        "probe",
        [("js/main.js", "const PROBE_CELL_PX = 66;", "const PROBE_CELL_PX = 600;")],
        ("the readout table's columns are fixed width",),
        id="readout_table_columns_are_fixed_but_enormous",
    ),
    # ---- the strip stacked again, in each of the two places it can be written
    #
    # `display: block` on `.readout` is the pre-task layout exactly: the panels
    # sit one above another, the strip is as tall as their sum, and the tiles
    # give up the difference. Nothing throws, every number is right, and the
    # placement and width checks are both indifferent to it -- they constrain
    # ancestry and the absence of `fr`, neither of which is a layout. The
    # stylesheet and an inline attribute are separate entries because they are
    # separate edits and the sheet-only check missed the second.
    # `.readout` の `display: block` はタスク以前のレイアウトそのもので、板は
    # 縦に積まれ、その分だけタイルが低くなる。例外は出ず、数値は正しく、
    # 置き場所と列幅のチェックはどちらも無関心である（祖先と `fr` の不在を
    # 見ているだけで、どちらもレイアウトではない）。シートとインラインは別の
    # 編集なので項目も分ける。
    pytest.param(
        "probe",
        [("style.css", ".readout {\n  display: flex;", ".readout {\n  display: block;")],
        ("the readout lays its panels out in a row",),
        id="readout_panels_stack_via_the_stylesheet",
    ),
    pytest.param(
        "probe",
        [("index.html", "<section class=\"readout\">",
          "<section class=\"readout\" style=\"display: block\">")],
        ("the readout lays its panels out in a row",),
        id="readout_panels_stack_via_an_inline_style",
    ),
    # ---- the duplicate panel, put back
    #
    # `ビュー値` beside `行列` is the same twelve numbers twice on any view whose
    # picture *is* the matrix: 974px of a 1600px strip -- the widest element in
    # a bar whose purpose is to stop wasting width -- and at 900px the strip
    # doubles to 224px, taking 113px off the tiles to restate what is 20px to
    # its left. Forcing it visible is the state that shipped, and it fails
    # nothing else: the panel's contents are still written and still correct.
    # 描かれているものが行列そのものであるビューでは、`ビュー値` は `行列` と
    # 同じ12個である（1600pxの帯で974px＝最も広い要素、900pxでは帯が224pxに
    # なりタイルから113px）。常に出す実装は出荷されていた状態であり、中身は
    # 書かれ続けているので他のチェックは落ちない。
    pytest.param(
        "probe",
        [(
            "js/main.js",
            "  $(\"view-panel\").hidden = state.spec.values_are_matrix === true;",
            "  $(\"view-panel\").hidden = false;",
        )],
        ("the view-value panel is dropped where it repeats the matrix",),
        id="the_duplicate_view_value_panel_is_always_shown",
    ),
    # ---- and the two ways to ignore the declaration while still passing
    #
    # `views/base.py` argues the default must be the showing direction,
    # `routers/view.py` says the answer is "not derivable from ``kind``", and
    # `main.js` repeats both -- and until `tri_scalar`'s panel was checked, all
    # three sentences were unasserted prose. `coef_tiles` (scalar, declares
    # true) and `surface` (rgb, declares false) cannot tell any of these
    # implementations apart. A view that is `scalar` and declares nothing --
    # a residual-error view, say -- loses its only numbers under both.
    # 宣言を無視したまま通ってしまう2つの実装。`coef_tiles`（scalar・真）と
    # `surface`（rgb・偽）だけでは、宣言を読む実装と区別が付かない。
    # scalar で無宣言のビュー（残差ビューなど）は、どちらでも唯一の数値を失う。
    pytest.param(
        "probe",
        [(
            "js/main.js",
            "  $(\"view-panel\").hidden = state.spec.values_are_matrix === true;",
            "  $(\"view-panel\").hidden = state.spec.kind === \"scalar\";",
        )],
        ("a view that declares nothing about its values keeps its panel",),
        id="view_value_panel_hidden_by_kind_instead_of_the_declaration",
    ),
    pytest.param(
        "probe",
        [(
            "js/main.js",
            "  $(\"view-panel\").hidden = state.spec.values_are_matrix === true;",
            "  $(\"view-panel\").hidden = state.spec.values_are_matrix !== false;",
        )],
        ("a view that declares nothing about its values keeps its panel",),
        id="an_undeclared_view_defaults_to_hiding_its_own_values",
    ),
    # ---- the readout left describing a matrix that is no longer on screen
    #
    # `clearReadout()` at the end of `reload()`. Deleting it broke no check at
    # all: after loading matrix B the strip kept matrix A's twelve coefficients
    # and A's out-of-region marker, under the picture of B, until the pointer
    # moved.
    # `reload()` 末尾の `clearReadout()`。消しても何も落ちなかった——行列Bを
    # 読み込んだあとも、帯にはAの12個とAの「領域外」の印が、Bの絵の下に、
    # ポインタが動くまで残る。
    pytest.param(
        "probe",
        [(
            "js/main.js",
            "  clearReadout();\n\n  const bytes = payload.data.byteLength;",
            "\n  const bytes = payload.data.byteLength;",
        )],
        ("a load clears the readout instead of leaving the previous matrix's numbers",),
        id="a_load_leaves_the_previous_matrix_in_the_readout",
    ),
    # ---- the marker cleared while the numbers it disclaims were still shown
    #
    # `showProbe` writes the heading synchronously and the twelve cells are not
    # touched until the answer lands, so writing `false` here cleared the marker
    # over the previous pixel's out-of-region values -- printed in the same
    # type, colour and precision as real coefficients, for one round trip,
    # every time the pointer left a masked pixel.
    # 見出しは同期的に書かれ、セルは応答が来るまで変わらない。ここで `false`
    # を書くと、前の画素の領域外の値が本物と同じ見た目で残ったまま印だけが
    # 消える——マスク画素からポインタを動かすたび、1往復ぶん。
    pytest.param(
        "probe",
        [(
            "js/main.js",
            "  setProbePos(hit.col, hit.row, $(\"probe-pos\").classList.contains(\"outside\"));",
            "  setProbePos(hit.col, hit.row, false);",
        )],
        ("the marker stays with the numbers it disclaims",),
        id="the_marker_is_cleared_before_the_numbers_it_describes",
    ),
    # ---- the readout bound to the newest matrix instead of the drawn one
    #
    # `state.matrixId` changes when `POST /session/matrix` answers, seconds
    # before that matrix's pixels arrive. Reading it here answers hovers about
    # the matrix still in transfer while the tiles show the previous one --
    # and `inside_active_region` differs between matrices, so the marker and
    # the picture disagree for the whole transfer.
    # `state.matrixId` は POST が返った時点で変わり、その行列の画素が届くのは
    # 数秒後である。ここでそれを読むと、タイルが前の行列を映している間の
    # ホバーに転送中の行列が答える。有効領域は行列ごとに違うので、印と絵が
    # 転送のあいだずっと食い違う。
    pytest.param(
        "probe",
        [("js/main.js", "      const matrixId = state.shownMatrixId;",
          "      const matrixId = state.matrixId;")],
        ("a hover during a load asks the matrix on screen",),
        id="the_readout_probes_the_arriving_matrix_not_the_drawn_one",
    ),
    # ---- the named grid rows, whose absence is invisible everywhere else
    #
    # `body`'s template is positional and `#space-note` is `hidden` for a
    # projector-space matrix, so without these five lines every band after it
    # moves up one row and the `1fr` lands on the readout: a blank band under
    # the strip, and a tile height equal to the sidebar's content -- i.e.
    # decided by how many controls the current view declares. Deleting them
    # passed every check in every scenario.
    # `body` のテンプレートは位置で決まり、`#space-note` は投影座標の行列では
    # `hidden` になるため、この5行が無いと以降の段が1行ずつ繰り上がり `1fr` が
    # 読み取り欄に載る（帯の下の空白と、サイドバーの内容で決まるタイル高）。
    # 消しても全シナリオが通っていた。
    pytest.param(
        "probe",
        [(
            "style.css",
            ".toolbar { grid-row: 1; }\n"
            "#space-note { grid-row: 2; }\n"
            "main { grid-row: 3; }\n"
            ".readout { grid-row: 4; }\n"
            "footer#status { grid-row: 5; }\n",
            "",
        )],
        ("each band of the page is placed in a named grid row",),
        id="the_page_bands_fall_back_to_document_order",
    ),
    # ---- the status line printing what is already on screen
    #
    # Restores both halves at once, because that is how it would come back: the
    # line regains 転送 / レンジ / GPU転送 -- two of the three verbatim copies of
    # the 計測 panel 20px above it -- and the panel loses the one figure that
    # lived nowhere else. The check pins the line to the dimension summary *and*
    # レンジ to the panel, so either half alone fails it.
    # 戻るときは両方まとめて戻るので、変異も両方を戻す。行は 計測 パネルの
    # 20px下でその2つを逐語的に繰り返し、パネルはここにしか無い数値を失う。
    pytest.param(
        "probe",
        [
            (
                "js/main.js",
                "  $(\"perf-ranges\").textContent = `${perf.rangesMs} ms`;\n",
                "",
            ),
            (
                "js/main.js",
                "  status(`${payload.width}×${payload.height}×${payload.channels} を表示中`);",
                "  status(\n"
                "    `${payload.width}×${payload.height}×${payload.channels} を表示中 — ` +\n"
                "    `転送 ${perf.fetchMs} ms / レンジ ${perf.rangesMs} ms / ` +\n"
                "    `GPU転送 ${perf.uploadMs}+${perf.uploadFinishMs} ms`\n"
                "  );",
            ),
        ],
        ("the status line after a render is the dimensions alone",),
        id="the_status_line_repeats_the_timings_panel",
    ),
    # ---- the heading written when the selector changes rather than when a view
    #      renders
    #
    # The implementation the comment in ``reload()`` exists to forbid, and the
    # one nothing caught: moving the write into the ``change`` handler passed
    # every check in every scenario. It only shows on a view that is *refused* --
    # unsupported kind, unknown layout, server 400 -- where the heading and the
    # panel then describe a view that never rendered, over the numbers of the
    # one that did.
    # ``reload()`` のコメントが禁じている当の実装であり、何も検出しなかった
    # もの（`change` 側へ移しても全シナリオが通っていた）。差が出るのは
    # 拒否されたビューのときだけで、そこでは見出しと板が「描かれなかった
    # ビュー」を説明し、その下には別のビューの数値が並ぶ。
    pytest.param(
        "views",
        [
            (
                "js/main.js",
                "  $(\"view-title\").textContent = state.spec.label || state.spec.key;\n",
                "",
            ),
            (
                "js/main.js",
                "    state.spec = spec;\n    buildServerParams(spec);",
                "    state.spec = spec;\n"
                "    $(\"view-title\").textContent = spec.label || spec.key;\n"
                "    buildServerParams(spec);",
            ),
        ],
        ("a refused view leaves the readout naming the view still on screen",),
        id="view_heading_written_when_the_selector_changes",
    ),
    # ---- and the heading inferred from the payload instead of read off the spec
    #
    # The other way to pass a heading check without having one: two of the three
    # views in play are 12-channel scalar views, so a label chosen by channel
    # count is right for `coef_tiles` and for `surface` and wrong only for the
    # third. Registering a view in Python and getting its name on screen is the
    # promise this whole registry exists for.
    # 見出しらしきものを持たずにチェックを通すもう一つの方法。12チャンネルの
    # scalar ビューが2つあるため、チャンネル数で選んだラベルは `coef_tiles`
    # でも `surface` でも正しく、3つ目でだけ誤る。
    pytest.param(
        "views",
        [(
            "js/main.js",
            "  $(\"view-title\").textContent = state.spec.label || state.spec.key;",
            "  $(\"view-title\").textContent ="
            " state.payload.channels === 3 ? \"投影面の色\" : \"12係数タイル\";",
        )],
        ("the readout's heading names the view that rendered, not its shape",),
        id="view_heading_inferred_from_the_payload_shape",
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
        (
            "outside the active region is marked as such",
            # The marker never comes on at all, so the check that watches it
            # survive a pending answer has nothing to watch either.
            # 印がそもそも一度も付かないので、応答待ちのあいだ印が残ることを
            # 見るチェックにも見るものが無くなる。
            "the marker stays with the numbers it disclaims",
        ),
        id="readout_never_says_a_pixel_is_outside_the_active_region",
    ),
    # ---- the categorical row of KIND_TRAITS, one cell at a time
    #
    # The table exists so that adding a kind means writing down what it does at
    # every decision, and these three are the cells whose wrong value still
    # draws. Each is a one-token edit to ``contract.js`` or to the legend, and
    # each changes exactly one thing on screen -- which is why each is expected
    # to fail exactly one check. A cell whose wrong value broke five checks
    # would not need a mutation entry; it would announce itself.
    #
    # 表の意味は「kind を足すとは全ての判断を書き下すこと」であり、この3つは
    # 「誤った値でも描けてしまう」セルである。いずれも1トークンの変更で、
    # 画面上の変化もちょうど1つずつになる。
    #
    # RGB32F on a one-channel plane uploads the identical twenty floats, so
    # every contents assertion passes: the class index goes into the red
    # component and green and blue are whatever the driver leaves there. The
    # tile still draws. Only the recorded internal format tells them apart,
    # which is why the harness stopped answering ``1`` for every GL constant.
    # 1チャンネルの平面を RGB32F で載せても数値は同一なので中身の検査は全て
    # 通り、番号が赤成分に入るだけである。区別できるのは形式の記録だけ。
    pytest.param(
        "gamut",
        [(
            "js/contract.js",
            'texture: Object.freeze({ internal: "R32F", format: "RED" }),\n'
            "    // shaders.js: 2 selects the discrete-palette branch.",
            'texture: Object.freeze({ internal: "RGB32F", format: "RGB" }),\n'
            "    // shaders.js: 2 selects the discrete-palette branch.",
        )],
        ("class indices are uploaded as R32F/RED",),
        id="categorical_uploads_class_indices_as_a_colour",
    ),
    # The legend written from a hard-coded list instead of from the payload.
    #
    # The obvious build, and the one that is right on the machine it was
    # written on: these *are* ``views/gamut.py``'s LABELS, so against the real
    # server the legend would be correct and would stay correct until somebody
    # renamed a class or reordered the list -- at which point the page keeps
    # explaining the previous meaning of the same four colours, with nothing
    # failing. The row count is untouched (four hard-coded names, four
    # classes), so exactly one check sees it, and that is the point of having
    # the count and the contents as separate checks.
    # 実装側の LABELS をそのまま書いたもの。実サーバに対しては正しく、名前や
    # 順序を変えた瞬間だけ嘘になる（そして何も失敗しない）。行数は変わらない
    # ので、落ちるのは内容のチェック1つだけである。
    pytest.param(
        "gamut",
        [(
            "js/main.js",
            "    text.textContent = payload.labels[i];",
            '    text.textContent = ["範囲内", "負クリップ", "1超クリップ", "両方"][i];',
        )],
        (
            "the legend's labels are the payload's",
            # The hard-coded list has four entries, so on the three-class view
            # it is a row count error as well as a wrong label.
            # 直書きの一覧は4つなので、3クラスのビューでは行数の誤りにもなる。
            "the legend's row count is the payload's class count",
        ),
        id="the_class_legend_hardcodes_the_shipped_labels",
    ),
    # The colormap control left live over a discrete palette.
    #
    # Same shape as ``colormap_stays_enabled_for_an_rgb_view`` and kept as its
    # own entry for the same reason the two ``percentileAbsPerChannel`` walks
    # are: one kind passing is not the other kind passing. The dropdown does
    # nothing here -- the shader's categorical branch never reaches a colormap
    # -- and offering it says the four colours are a false-colour render of a
    # quantity, which is exactly the reading this view exists to prevent.
    # rgb 版と同じ形だが別項目にする（一方が通ることは他方が通ることではない）。
    # ここでドロップダウンは何もせず、しかも「4色は量の疑似カラーである」と
    # 主張してしまう。
    pytest.param(
        "gamut",
        [(
            "js/contract.js",
            "    colormap: false,\n"
            "    // One plane, uploaded as one float channel",
            "    colormap: true,\n"
            "    // One plane, uploaded as one float channel",
        )],
        ("the colormap control is disabled for a categorical view",),
        id="colormap_stays_enabled_for_a_categorical_view",
    ),
    # The legend put back down with the two standing legends at the foot of the
    # sidebar, which is where it was first written.
    #
    # It renders there. Every other check in the scenario passes: the rows are
    # built, the labels are the payload's, the colours are distinct. The
    # sidebar is ``overflow-y: auto``, though, and measured in the browser at
    # 1600x1000 with a target loaded, its foot is below the fold -- so the only
    # statement of what the four colours mean is off screen beside the picture
    # they explain. Same failure as ``the_range_and_timings_go_back_into_the_
    # sidebar`` above, and worse: those two restated numbers that were on
    # screen elsewhere, and this restates nothing.
    # 凡例を下端の凡例2つのところへ戻す。描画も中身も正しく、他のチェックは
    # 全て通る。ただしサイドバーは縦に溢れるので、実測（1600x1000・ターゲット
    # 読み込み済み）ではそこは画面外であり、4色の意味を述べる唯一の場所が
    # 絵の隣から消える。
    pytest.param(
        "gamut",
        [
            ("index.html", '      <div id="class-legend" class="group class-legend" hidden></div>\n', ""),
            (
                "index.html",
                '      <div class="group legend">\n'
                '        <span class="legend-swatch nan"></span>',
                '      <div id="class-legend" class="group class-legend" hidden></div>\n\n'
                '      <div class="group legend">\n'
                '        <span class="legend-swatch nan"></span>',
            ),
        ],
        ("the class legend sits above the sidebar's first divider",),
        id="the_class_legend_goes_back_below_the_fold",
    ),
    # The display range recomputed from the pixels instead of read off the wire.
    #
    # ``views/gamut.py`` takes its range from ``LABELS`` rather than from the
    # frame, and says why: on a frame that is entirely in gamut a data-derived
    # range gives ``vmin == vmax == 0``, every class collapses onto one colour,
    # and the view breaks exactly when the answer is "everything is
    # representable". This is that decision undone from the client side, and it
    # is the obvious thing to write if ``sharedRange`` is ever generalised.
    #
    # It could not be seen until the fixture stopped holding all four classes:
    # with classes 0-3 present the data-derived range is ``{0, 3}``, the same
    # numbers the header sends, and this mutation left the scenario 24/24 green
    # (measured). The fill is ``(row * 2 + col) % 3`` against four labels for
    # that reason -- see the note in the fake server.
    #
    # Only ``state.ranges`` moves; ``#range-info`` is written from
    # ``payload.vmin``/``vmax`` a line later, so exactly one check sees this.
    # レンジを画素から再計算する変異。`views/gamut.py` が LABELS から取っている
    # 判断をクライアント側で打ち消すもので、全画素が範囲内のフレームでは
    # `vmin == vmax == 0` になり全クラスが1色に潰れる。フィクスチャが4クラス
    # 揃っていた間は再計算しても同じ {0, 3} になり、24/24 のまま通っていた。
    pytest.param(
        "gamut",
        [(
            "js/main.js",
            "    state.ranges = [{ vmin: p.vmin, vmax: p.vmax }];",
            "    state.ranges = [(() => { let lo = Infinity, hi = -Infinity;\n"
            "      for (const value of p.data) {\n"
            "        if (!Number.isFinite(value)) continue;\n"
            "        if (value < lo) lo = value;\n"
            "        if (value > hi) hi = value;\n"
            "      }\n"
            "      return { vmin: lo, vmax: hi };\n"
            "    })()];",
        )],
        ("the categorical range is the class index range from the payload",),
        id="categorical_range_recomputed_from_the_pixels",
    ),
    # The two strings that say the values are names rather than quantities.
    #
    # ``const classes = traits.labelsAreClasses`` looks like a special case with
    # one caller, which is what makes it a candidate for a ``computeRanges``
    # cleanup. Forced false, the sidebar reads 表示レンジ ``[0.0000, 3.0000]`` --
    # the "1.5 is a value this view can hold" reading the comment on that
    # ternary exists to prevent -- and the tile is captioned
    # ``色域外 1超クリップ 両方 範囲内, 補正可  [0.0000, 3.0000]``, which is the
    # legend restated in the one place with no room for it. Both are correct for
    # ``rgb``, whose labels are R/G/B and whose values are a measurement.
    #
    # Measured before the two checks existed: this left the scenario 24/24.
    # 値が量ではなく名前であることを述べている2つの文字列。`classes` の三項は
    # 呼び出し1箇所の特別扱いに見えるため整理の対象になりやすい。false に倒すと
    # レンジは `[0.0000, 3.0000]`、見出しは4つのクラス名の羅列になる（どちらも
    # rgb では正しい）。この2件の検査が無い間は 24/24 で通っていた。
    pytest.param(
        "gamut",
        [(
            "js/main.js",
            "    const classes = traits.labelsAreClasses;",
            "    const classes = false;",
        )],
        (
            "the display range is worded as a class count, not as an interval",
            "the tile caption does not restate the legend under the tile",
            # The same two strings, read again on the three-class view: both
            # are built from the class count, so both are wrong there too.
            # 同じ2つの文字列を3クラスのビューでも読んでいる。
            "the caption and range of a three-class view count three classes",
        ),
        id="the_categorical_range_and_caption_lose_their_class_wording",
    ),
    # ``sharedRange`` flipped for ``categorical`` -- an *old* column whose answer
    # for the new kind was unpinned.
    #
    # Invisible in ``state.ranges``: with one channel both branches of
    # ``computeRanges`` build the same single-entry array, so the display range
    # the tiles are drawn against does not change at all. What changes is the
    # same two strings above, through the other branch: the caption becomes
    # ``色域外  ±3.0000`` (``labels[0]`` again -- the per-channel axis) and
    # ``#range-info`` becomes ``±3.0000``. Before the two checks existed,
    # flipping this column failed **nothing** (measured), which is the report's
    # §9.1 question answered from the other side.
    # 古い列 `sharedRange` の categorical における答えを固定する。1チャンネルでは
    # どちらの分岐でも `state.ranges` は同じになるため、レンジ自体は変わらない。
    # 変わるのは上と同じ2つの文字列で、見出しは `labels[0]` を使った
    # `色域外  ±3.0000` になる。検査が無い間は、この列を反転しても何も落ちなかった。
    pytest.param(
        "gamut",
        [(
            "js/contract.js",
            "    sharedRange: true,\n"
            "    // **`labels[c]` is class c's name, not channel c's.**",
            "    sharedRange: false,\n"
            "    // **`labels[c]` is class c's name, not channel c's.**",
        )],
        (
            "the display range is worded as a class count, not as an interval",
            "the tile caption does not restate the legend under the tile",
            # The same two strings, read again on the three-class view: both
            # are built from the class count, so both are wrong there too.
            # 同じ2つの文字列を3クラスのビューでも読んでいる。
            "the caption and range of a three-class view count three classes",
        ),
        id="categorical_takes_a_per_channel_range_instead_of_a_shared_one",
    ),
    # The GLSL palette typed out beside CLASS_COLORS instead of generated from
    # it, with two classes given the same colour.
    #
    # The drift this catches is silent in every other check and in the browser:
    # the legend still lists four rows in four distinct colours (it reads
    # ``CLASS_COLORS``), the picture still shows flat regions, and the only
    # symptom is that two of the four classes are now one region on screen while
    # the legend insists they are two. That is the exact failure the view exists
    # to avoid, arrived at through the palette rather than through the data.
    # Nothing textual can see it -- ``test_web_golden.py`` reads ``shaders.js``
    # as text and sees ``${CLASS_PALETTE_GLSL}``, not what it expands to -- so
    # the check that catches it evaluates the module under node.
    # GLSL のパレットを生成ではなく手書きに戻し、2クラスを同じ色にする変異。
    # 凡例は `CLASS_COLORS` を読むので4色のまま、絵も平坦な領域のままで、
    # 「凡例は4クラスと言うが画面には3領域しかない」という状態だけが残る。
    # テキストとして読むゴールデンテストからは見えない（展開前の
    # `${CLASS_PALETTE_GLSL}` しか見えない）ため、node 側で評価して突き合わせる。
    pytest.param(
        "gamut",
        [(
            "js/shaders.js",
            "vec3 classColor(int i) {\n${CLASS_PALETTE_GLSL}\n    return u_nanColor;\n}",
            "vec3 classColor(int i) {\n"
            "    if (i == 0) return vec3(0.0000, 0.6196, 0.4510);\n"
            "    if (i == 1) return vec3(0.0000, 0.4471, 0.6980);\n"
            "    if (i == 2) return vec3(0.9020, 0.6235, 0.0000);\n"
            "    if (i == 3) return vec3(0.9020, 0.6235, 0.0000);\n"
            "    return u_nanColor;\n}",
        )],
        ("the legend's colours are the palette the shader compiles",),
        id="the_glsl_palette_is_typed_out_beside_the_array_it_should_come_from",
    ),
    # The class readout's own index arithmetic, transposed.
    #
    # The failure this repository has now shipped twice, in the one place this
    # task added a new one. It is here as much to keep the *fixture* honest as
    # the code. Measured, three runs of this same mutation against three
    # fixtures:
    #
    #   ``(row * 3 + col) % 4``, NaN at (2,1)   2 checks fail
    #   ``(row + col) % 4``,     NaN at (2,1)   2 checks fail
    #   ``(row + col) % 4``,     NaN at (2,2)   0 checks fail -- 24/24 green
    #
    # ``(row + col) % 4`` is symmetric under transposition, so it catches this
    # only because the swapped index lands on the masked pixel; move the mask
    # onto the diagonal, where it is its own transpose, and the bug becomes
    # invisible to the whole scenario. The shipped fill is
    # ``(row * 3 + col) % 4``, and this entry is what stops it quietly going
    # back to something symmetric.
    # 行と列の取り違え。このリポジトリが繰り返してきた失敗を、本タスクが増やした
    # 唯一の添字計算に対して固定する。同じ変異を3つのフィクスチャで実測した
    # 結果が上表で、対称な埋め方かつマスクを対角へ置くと24件すべて通る。
    # この項目は埋め方が黙って対称なものへ戻らないようにするためでもある。
    pytest.param(
        "gamut",
        [(
            "js/main.js",
            "    const v = p.data[hit.row * p.width + hit.col];",
            "    const v = p.data[hit.col * p.width + hit.row];",
        )],
        (
            "the readout names the class the pixel is",
            "a pixel with no class reads as NaN rather than as class 0",
        ),
        id="the_class_readout_transposes_row_and_col",
    ),
    # ---- the LUT plot's vertical axis, pinned back to 1.0
    #
    # The state a plot arrives in if nobody thinks about it, and the one the
    # spec calls out by name. A measured inverse LUT does not end at 1.0:
    # data/full_recalibration_20260730_iso400/inverse_lut/inverse_lut.npz stops
    # at 0.813 / 0.876 / 0.751, and that gap -- the projector never gets there
    # -- is one of the few things the plot exists to show. With the axis fixed
    # at [0, 1] the picture is identical for a LUT that reaches 1.0 and one
    # that stops at 0.69: three curves, right colours, right shapes, ending
    # somewhere below the top, and nothing on screen saying which.
    #
    # The coordinate check does *not* fail here, deliberately: it reads the
    # mapping back off `drawLutCurve`'s return value, so a pinned axis stays
    # internally consistent with what it drew. Pinning the range is a separate
    # question from drawing it faithfully, and the two checks are separate for
    # that reason -- the axis one compares against the fixture's own maximum,
    # which no internally-consistent wrong answer can satisfy.
    #
    # 縦軸を 1.0 に固定する変異。実測LUTの終端は 1.0 ではなく、[0, 1] 固定では
    # 1.0 まで届くLUTと 0.69 で止まるLUTが同じ絵になる。座標のチェックは
    # 意図的に落ちない（戻り値の写像と描画は整合したままなので）。範囲を
    # データから取ることと、その範囲へ忠実に描くことは別の問いであり、
    # だからチェックも別になっている。
    pytest.param(
        "lut",
        [(
            "js/plot.js",
            "  const vMax = vMaxRaw > vMin ? vMaxRaw : vMin + 1;",
            "  const vMax = 1;",
        )],
        ("the plot's vertical axis comes from the data, not pinned to 1.0",),
        id="lut_plot_axis_pinned_to_one",
    ),
    # ---- the three curves drawn for the wrong channels
    #
    # Rotated by one, so every polyline is a real curve of the fixture -- just
    # not the one its colour claims. Nothing throws, three lines appear in
    # three colours, the axis is right, and the only thing wrong is which
    # channel each line is. On a real LUT that is the difference between "blue
    # tops out lowest" and "red does", which is what the plot gets read for.
    # Only the correspondence check sees it: the count, the colours, the axis
    # and the overlay are all indifferent to a permutation.
    # 1つずらす。どの折れ線もフィクスチャの実在の曲線であり、ただ色が名乗って
    # いるチャンネルと中身が違う。本数・色・軸・重ね描きのどれも順列に無関心で、
    # 座標対応のチェックだけが見ている。
    pytest.param(
        "lut",
        [(
            "js/plot.js",
            "    polyline(ctx, xs, rows[c].map((v) => mapY(Number(v))),",
            "    polyline(ctx, xs, rows[(c + 1) % rows.length].map((v) => mapY(Number(v))),",
        )],
        ("each of the three curves is drawn at its own channel's coordinates",),
        id="lut_plot_draws_the_channels_in_the_wrong_order",
    ),
    # ---- the canvas backing store, sized in CSS pixels
    #
    # Without the devicePixelRatio term the figure is drawn into a quarter of
    # the canvas on a 200% display and then stretched to fit -- three curves
    # and a scale at half resolution, on the 150% Windows display this project
    # is used on. It renders; it is simply blurred, which is not a failure any
    # assertion about content can reach.
    # DPR項が無いと 200% 表示で図がcanvasの1/4に描かれて引き伸ばされる。描画は
    # 成功し、ただ半分の解像度になる（内容についてのどの検査にも掛からない）。
    pytest.param(
        "lut",
        [
            ("js/plot.js", "  canvas.width = Math.max(1, Math.round(cssW * dpr));",
             "  canvas.width = Math.max(1, Math.round(cssW));"),
            ("js/plot.js", "  canvas.height = Math.max(1, Math.round(cssH * dpr));",
             "  canvas.height = Math.max(1, Math.round(cssH));"),
        ],
        ("the canvas backing store scales with devicePixelRatio",),
        id="lut_plot_ignores_the_device_pixel_ratio",
    ),
    # ---- the loaded id thrown away by a view switch
    #
    # `buildServerParams` rebuilds every `sp-*` from the spec's declared
    # default, and for a runtime id that default is "". Without the line that
    # re-applies the loaded id, looking at another view and coming back empties
    # the ターゲットID box for a target that is still loaded server-side -- and
    # `compensation` then refuses a request the user has no reason to think
    # incomplete, naming a parameter they can see they filled in.
    # コントロールはビュー切替のたびに既定値（空文字）から作り直される。読み込み
    # 済みのIDを書き戻す行が無いと、サーバには残っているターゲットについて欄
    # だけが空に戻り、利用者から見て理由の無い拒否になる。
    pytest.param(
        "target",
        [("js/main.js", "    if (loadedId) el.value = loadedId;\n", "")],
        ("coming back re-fills the ターゲットID box with the loaded id",),
        id="a_view_switch_empties_the_loaded_target_id",
    ),
    # ---- the image picker that changes the label and not the image
    #
    # Forced to index 0: the dropdown moves, a request goes out, an id comes
    # back, and the box is filled -- with the id of the image that was already
    # loaded. Everything about the interaction looks like it worked. This is
    # why the check asserts the *index that was posted* and the id that came
    # back, rather than that "a request happened".
    # index を 0 に固定する。ドロップダウンは動き、要求は飛び、IDは返り、欄は
    # 埋まる——既に読み込まれていた画像のIDで。操作は成功したように見える。
    # だからチェックは「要求が飛んだ」ではなく「送られたindexと返ったID」を見る。
    pytest.param(
        "target",
        [(
            "js/main.js",
            "    loadTargetImage(meta.path, Number($(\"target-images\").value) || 0)",
            "    loadTargetImage(meta.path, 0)",
        )],
        ("choosing another image re-fetches that index and re-fills the id",),
        id="the_image_picker_always_reloads_the_first_image",
    ),
    # ---- a loader that renders on its own
    #
    # "They stage; they do not render" is the one rule of the server-parameter
    # group (see the note above `refreshParamsPending`), and until the `target`
    # and `lut` scenarios loaded a matrix these two entries could not exist: with
    # `state.matrixId` null `reload()` returns at its first line, so the loaders
    # could have called `applyServerParams()` on every load with the whole suite
    # green. That is what the review measured -- 12/12 in both scenarios with the
    # call inserted -- and it is why those scenarios now start with `clickLoad`.
    #
    # The cost is the point. Loading a target and then a LUT would run two extra
    # full compensation renders of the 1080p matrix (~740 ms and ~390 MB peak
    # each), both discarded afterwards by `reqId`, so the screen shows the right
    # picture and nothing on it says the machine did the work three times.
    #
    # 「読み込みは置くだけで描かない」規則の変異。行列を読まないシナリオでは
    # `reload()` が先頭で打ち切るため、この2件はそもそも成立しなかった
    # （レビューが実測: 挿入しても両シナリオ 12/12）。だから両シナリオは
    # `clickLoad` から始まる。
    pytest.param(
        "target",
        [(
            "js/main.js",
            '    setServerParam("target", res.id);',
            '    setServerParam("target", res.id);\n    applyServerParams();',
        )],
        ("loading a target does not fetch the view on its own",),
        id="the_target_loader_renders_on_load",
    ),
    pytest.param(
        "lut",
        [(
            "js/main.js",
            '    setServerParam("lut", res.id);',
            '    setServerParam("lut", res.id);\n    applyServerParams();',
        )],
        ("loading a LUT does not fetch the view on its own",),
        id="the_lut_loader_renders_on_load",
    ),
    # ---- the LUT id published before its curve is in hand
    #
    # `loadInverseLut` issues two requests and publishes nothing until both have
    # returned. That was stated in a comment and asserted nowhere: the fake
    # server had no LUT whose POST succeeded and whose curve did not, so the two
    # orders were indistinguishable and moving the assignments up left the
    # scenario at 12/12 (measured). `LUTS["data/curveless.npz"]` exists to make
    # the state reachable.
    #
    # What the order prevents is not an exception -- both builds show the same
    # error sentence -- but a disagreement: the LUT ID box and `state.lutId`
    # naming LUT B while the canvas still holds LUT A's curves. Press 適用 and
    # the compensation is computed with B against a plot of A, which is the
    # "the heading and the numbers describe different things" failure this
    # branch has closed everywhere else.
    #
    # 公開順の変異。POSTは成功しcurveだけが失敗するLUTが無かったため、順序を
    # 入れ替えても12/12だった（実測）。防いでいるのは例外ではなく食い違いで
    # ある——欄と `state.lutId` は新しいLUTを名乗り、canvas には前のLUTが残る。
    pytest.param(
        "lut",
        [(
            "js/main.js",
            """    const res = await loadLut(path);
    const curve = await fetchLutCurve(res.id);
    state.lutId = res.id;
    state.lutCurve = curve;
    setServerParam("lut", res.id);""",
            """    const res = await loadLut(path);
    state.lutId = res.id;
    setServerParam("lut", res.id);
    const curve = await fetchLutCurve(res.id);
    state.lutCurve = curve;""",
        )],
        ("a LUT whose curve could not be fetched is not named as loaded",),
        id="the_lut_id_is_published_before_its_curve",
    ),
    # ---- the three info lines, which nothing read until the final fix wave
    #
    # All three are text beside a control, so none of them can break anything:
    # the id still reaches the parameter box, the curve is still drawn at the
    # right coordinates, and the picture is still correct. What is lost is the
    # page's account of *what it is showing* -- the image's size, which LUT
    # this is and how it was measured, and which exponent the dashed line is.
    # 3行とも表示だけなので壊れても絵は正しいままで、失われるのは「いま何を
    # 見ているか」の説明である。
    pytest.param(
        "target",
        [(
            "js/main.js",
            '$("target-info").textContent = `${res.id} / ${width}×${height}`;',
            '$("target-info").textContent = `${res.id} / ${height}×${width}`;',
        )],
        ("the target info line names the id and the image's width x height",),
        id="the_target_info_line_swaps_width_and_height",
    ),
    pytest.param(
        "lut",
        [(
            "js/main.js",
            '$("lut-info").textContent = `${res.id} / ${channels}×${samples} / ${source}`;',
            '$("lut-info").textContent = "";',
        )],
        ("the LUT info line names the id, the table's shape and its source",),
        id="the_lut_info_line_says_nothing",
    ),
    pytest.param(
        "lut",
        [(
            "js/main.js",
            '$("lut-gamma-value").textContent = `γ=${curve.gamma}`;',
            '$("lut-gamma-value").textContent = "γ=2.2";',
        )],
        ("the analytic gamma's exponent is shown, and it is the server's",),
        id="the_gamma_legend_hardcodes_the_usual_exponent",
    ),
    # ---- the two guards on the one control that writes to disk
    #
    # The double-submit guard, removed from the function and left only on the
    # button's `disabled` attribute -- which is the build this looks like from
    # the outside, and the reason the check asserts both halves. `disabled` is
    # a statement to the user, not a guarantee: an already-dispatched click,
    # Enter in the path box (which delegates to `.click()`), or anything else
    # that reaches `runGenerate` does so regardless of it. On the 1080p matrix
    # the call runs the pipeline and encodes two PNGs, so the window for a
    # second press is seconds wide, and two overlapping writes into one
    # directory do not merely duplicate work: the second meets the first one's
    # files and comes back 409, reporting a conflict about a directory that is
    # being filled by the request it conflicts with -- and the only in-GUI
    # answer to a 409 is the override this whole design keeps deliberate.
    #
    # ボタンの `disabled` だけを残して関数側のガードを外す（外から見ると同じに
    # 見える実装であり、チェックが両方を見ている理由でもある）。`disabled` は
    # 表示であって保証ではなく、既に発行されたクリックも Enter も届く。実機では
    # 数秒かかるため2度目を押す余地があり、同じディレクトリへの二重書き込みは
    # 「遅い重複」では済まない——2つ目は1つ目のファイルに衝突して 409 になる。
    pytest.param(
        "generate",
        [("js/main.js", "  if (generatesInFlight > 0) return;\n", "")],
        ("a second press sends nothing",),
        id="generate_relies_on_the_disabled_attribute_alone",
    ),
    # ...and the body carrying something other than what is on screen.
    #
    # The one field is enough to make the point and is deliberately set to a
    # *real* id for the wrong image -- `target_dir_0` is the first entry of the
    # same directory, which is what a client that ignored the image picker
    # would send. The saved PNG is then a complete, correct compensation of an
    # image the user was not looking at, filed under the index of the one they
    # were. Nothing on screen changes, no request fails, and the file is wrong
    # for as long as it exists.
    #
    # This is the entry that says the body assertion is worth its length. A
    # check that only counted requests, or that compared against values which
    # happened to be the parameters' declared defaults, would pass here --
    # which is why the `compensation` fixture in the harness carries the
    # shipped defaults and the scenario moves every field off them first.
    #
    # Exactly one check fails, and the one that does not is the interesting
    # half: `confirming 上書き resends the same request` compares the resent
    # body against the *refused* one rather than against the expectation, so a
    # body that is consistently wrong stays consistent. That is intentional --
    # that check is about the resend being the same request, not about the
    # request being right, and the two properties fail independently.
    #
    # 本文が画面と違うものを運ぶ変異。1フィールドで足り、値は「別の画像の実在
    # するID」にしてある（同じディレクトリの1枚目＝選択を無視した実装が送る値）。
    # 保存されるのは「見ていない画像の、完全に正しい補償画像」であり、見ていた
    # 画像の番号で綴じられる。画面は何も変わらず、要求も失敗しない。落ちるのは
    # 1件で、上書き再送の検査が落ちないことには意味がある——あちらは「断られた
    # 本文と同じものを送り直したか」を見ており、本文が正しいかは見ていない。
    pytest.param(
        "generate",
        [("js/main.js", "    target_id: params.target,", '    target_id: "target_dir_0",')],
        ("the generate call sends the settings on screen",),
        id="the_generate_body_carries_a_fixed_target_id",
    ),
    # ...and the matrix, which the stale-screen guard does not cover.
    #
    # ``state.matrixId`` moves when ``POST /session/matrix`` returns; the
    # transfer that draws it takes seconds on a 1.56 GB matrix. In that window
    # every control still says what it said, so the guard passes and a press
    # goes through -- and a body built from ``state.matrixId`` compensates the
    # picture on screen with a matrix that has never been drawn. This is the one
    # field of the body whose source cannot be checked at rest: with the load
    # settled the two ids are equal and the mutation is invisible.
    # 行列のIDにも同じ性質がある。`POST /session/matrix` はすぐ返るが描画までは
    # 数秒あり、その間コントロールは変わらないのでガードは通す。
    pytest.param(
        "generate",
        [("js/main.js", "    matrix_id: state.shownMatrixId,", "    matrix_id: state.matrixId,")],
        ("a press while a matrix is still loading",),
        id="the_generate_body_carries_the_newest_matrix_not_the_shown_one",
    ),
    # ---- the three guards on the destructive confirmation
    #
    # The line beside 生成, left unwritten when the refusal stops applying.
    #
    # This is the shape the defect shipped in: an ``if (reason)`` with no
    # ``else``. It runs once before the first render -- when the reason is
    # "nothing is loaded", not "this view" -- and then never clears, so
    # ``compensation`` carried 「このビューでは生成できません」 beside an
    # *enabled* 生成 button, in every state including the screenshot captioned
    # as the settings about to be sent. The sentence is the only textual signal
    # that ``gamut`` genuinely cannot generate; one that appears everywhere
    # signals nothing anywhere, and a permanently false statement stands next to
    # the one control on this page that writes to the user's disk.
    # `else` の無い `if (reason)`。初回描画では理由が「未読み込み」なのにビューの
    # 文が出て、以後消えない——生成できるビューでも、有効なボタンの脇に出続ける。
    pytest.param(
        "generate",
        [(
            "js/main.js",
            '  $("gen-info").textContent = reason ? reason.short : generateOutcome;',
            '  if (reason) $("gen-info").textContent = reason.short;',
        )],
        (
            "the line beside 生成 names the obstacle",
            # The success check makes the same round trip through `gamut`, so
            # the missing `else` strands the refusal there too. Two checks watch
            # this one branch from its two sides, which is the point of both.
            # 成功側のチェックも `gamut` を往復するので、同じ `else` の欠落で
            # 落ちる。1つの枝を両側から見ている。
            "a successful save is what the line beside 生成 says",
        ),
        id="the_reason_line_beside_generate_is_never_cleared",
    ),
    # The disabled button, left looking exactly like a live one.
    #
    # ``button`` sets a background, ``cursor: pointer`` and a hover brightening,
    # and none of them stop applying when it is disabled. A disabled button also
    # dispatches no click, so the view-naming refusal ``runGenerate`` prints is
    # unreachable from the page: pressing 生成 under ``gamut`` produced full
    # accent blue, a pointer cursor, a hover highlight -- and silence. The
    # button's appearance is the whole pre-press answer to "why can I not save
    # this view", which is the half of the ``gamut`` decision that has to be
    # visible rather than merely true of the DOM.
    # 無効なボタンが生きているボタンと同じ見た目になる変異。クリックは配送
    # されないので、押した人には何の答えも返らない。
    pytest.param(
        "generate",
        [(
            "style.css",
            "input:disabled, select:disabled, button:disabled { opacity: 0.45; cursor: not-allowed; }",
            "input:disabled, select:disabled { opacity: 0.45; cursor: not-allowed; }",
        )],
        ("a disabled button looks disabled",),
        id="a_disabled_button_is_styled_like_a_live_one",
    ),
    # The confirmation, unbound from the directory it was given for.
    #
    # 上書きして生成 is the only control here that destroys data, and what makes
    # it safe is that the sentence being agreed to names a place. Without the
    # comparison at press time, a value set **without an event** -- autofill, an
    # extension, a driver, a refactor that drops a listener -- spends an
    # agreement about ``data/out_A`` on whatever the box says when the button is
    # finally pressed, and in this repository that can be
    # ``data/full_recalibration_20260730_iso400``: a real 99 GB tree that cost
    # hardware, a subject and a day and cannot be regenerated.
    # 確認を安全にしているのは「同意する文が場所を名指ししていること」である。
    # イベントを伴わない設定（自動入力・拡張機能・自動操作）でも、Aについての
    # 同意がBに使われてはならない。
    pytest.param(
        "generate",
        [(
            "js/main.js",
            '    if (overwriteOfferedFor === null\n'
            '      || overwriteOfferedFor !== $("gen-output-dir").value.trim()) {',
            "    if (false) {",
        )],
        ("上書き agreed for one directory cannot be spent on another",),
        id="the_overwrite_confirmation_is_not_bound_to_a_directory",
    ),
    # ...and the same guard's other half: the offer left standing over an edit.
    #
    # Either listener alone covers the other in this scenario, so the entry
    # removes both -- but each was measured on its own as well and each fails
    # this same check, which is the reason both are wired rather than whichever
    # one the stub happens to fire.
    # 片方だけでもこのシナリオは覆えるので両方外す（個別にも測っており、
    # どちらを外してもこのチェックが落ちる）。
    pytest.param(
        "generate",
        [(
            "js/main.js",
            '  $("gen-output-dir").addEventListener("input", () => { hideOverwriteOffer(); });\n'
            '  $("gen-output-dir").addEventListener("change", () => { hideOverwriteOffer(); });\n',
            "",
        )],
        ("editing the output directory retires the 上書き offer",),
        id="editing_the_output_directory_leaves_the_offer_standing",
    ),
    # The confirmation turned into a setting.
    #
    # A remembered ``overwrite`` is the difference between answering a question
    # and turning something on: every later press writes over whatever the box
    # names, with no 409 and nothing to answer. Three checks fail and the order
    # is the point -- the first names the constraint, and the other two are the
    # 409 path disappearing from the rest of the scenario, which is exactly what
    # made this regression previously surface under the name of the ``gamut``
    # refusal and nowhere else.
    # 上書きを記憶する実装は「問いに答えること」を「設定を入れること」に変える。
    # 落ちる3件のうち先頭が制約を名指しし、残り2件は 409 の経路が以降のシナリオ
    # から消えたことの現れである。
    pytest.param(
        "generate",
        [
            (
                "js/main.js",
                "let overwriteOfferedFor = null;",
                "let overwriteOfferedFor = null;\nlet stickyOverwrite = false;",
            ),
            (
                "js/main.js",
                "    overwrite: !!overwrite,",
                "    overwrite: (stickyOverwrite = stickyOverwrite || !!overwrite),",
            ),
        ],
        (
            "the press after a confirmed 上書き starts from overwrite: false",
            "a view that declares no linearizer cannot generate",
            "the line beside 生成 names the obstacle",
        ),
        id="a_confirmed_overwrite_is_remembered_for_every_later_press",
    ),
    # The other half of the line beside 生成: what it says after a save.
    #
    # The refusal half was fixed by making `refreshGenerateButton` write on both
    # branches; what `generateOutcome` *holds* after a success went unasserted in
    # the same edit, which is where the second instance of a two-branch defect
    # hides -- the fix reads as coverage. Removing the recording produces the
    # exact inverse of the defect it was written for: three absolute paths
    # appear under a line still reading 未生成, because the `finally`'s
    # `refreshGenerateButton()` restores the stale outcome over 「生成中…」.
    # 断りの側を直した編集で、成功の側は誰も読んでいなかった。記録を外すと、
    # 3つの絶対パスの脇に 未生成 と出る（直した欠陥のちょうど裏返し）。
    pytest.param(
        "generate",
        [(
            "js/main.js",
            "    setGenerateOutcome(`${(result.files || []).length} 件を保存`);\n",
            "",
        )],
        ("a successful save is what the line beside 生成 says",),
        id="a_successful_save_is_not_recorded_as_the_outcome",
    ),
    # ---- the class legend, driven off the palette instead of the payload
    #
    # `renderClassLegend` loops `payload.labels.length`; the swatches come from
    # `CLASS_COLORS`. Looping the palette instead is invisible on `gamut`, which
    # has four classes against a four-entry palette -- the fixture was symmetric
    # in exactly this dimension until `gamut_lite` (three classes) was added.
    #
    # `routers/view.py` admits any categorical view at `len(labels) >= 2` and
    # names the consumer that will break the four-class assumption: "the second
    # categorical view". Against a three-class view this build draws a fourth
    # row whose label is the string `undefined`, beside a colour no pixel in the
    # picture carries -- in the one place the class-to-colour mapping is stated
    # in words.
    # 凡例をパレット側から作る変異。`gamut`（4クラス）だけでは4===4で見えない。
    # 3クラスのビューでは「undefined」という行が、絵に無い色の隣に並ぶ。
    pytest.param(
        "gamut",
        [(
            "js/main.js",
            "  for (let i = 0; i < payload.labels.length; i++) {",
            "  for (let i = 0; i < CLASS_COLORS.length; i++) {",
        )],
        ("the legend's row count is the payload's class count",),
        id="the_class_legend_is_drawn_from_the_palette_not_the_payload",
    ),
    # ---- the picker's four silent failure modes
    #
    # None of these throws, and none of them is visible in a screenshot of a
    # working dialog. ここに挙げた4つはどれも例外を出さず、動いている
    # ダイアログの画面写真からも見えない。
    #
    # The filter, moved to the client: the server cuts at MAX_ENTRIES and *then*
    # the client would drop the non-matching files, so a directory of 12800
    # captures shows "12800件中2件を表示" over a list that is neither.
    # フィルタをクライアント側へ移す。サーバが2000件で切ったあとに絞ることに
    # なり、件数表示が実際の一覧と対応しなくなる。
    pytest.param(
        "picker",
        [(
            "js/filepicker.js",
            '  if (session.suffixes.length) params.set("suffixes", session.suffixes.join(","));',
            "",
        )],
        ("asks the server for .npy only",),
        id="picker_filters_on_the_client_instead_of_the_server",
    ),
    # The listing, re-sorted locally. `LoadTargetRequest.index` is an index into
    # the *server's* casefold order (`target.py::_listing`), so a client-side
    # sort makes "the third image" name a different file in the box than in the
    # request. Bytewise sort puts `Delta` before `data`.
    #
    # Four more checks fail alongside the order check, and the breadth is the
    # finding rather than something to narrow away: `picker-entry-N` is an
    # index into whichever render is currently on screen, and every check
    # after the order check reuses the "/repo" listing that check renders --
    # entering a directory (`picker-entry-0`, now `Delta` instead of `data`),
    # picking a file and confirming it (both keyed off the directory that
    # click was supposed to open), and the refusal check (which revisits
    # `Delta`, now reached by a different route). Only the two checks whose
    # own click reopens the dialog from scratch (取消 with a fresh "untouched"
    # path, and the truncated listing under `/repo/data/shots`) are
    # independent of the corrupted render and keep passing.
    #
    # クライアント側で並べ替え直す。index の基準はサーバ側の casefold 順なので、
    # 「3枚目」が欄と要求とで別のファイルを指すようになる。並び替え検査の他に
    # 4件が道連れで落ちるのは、`picker-entry-N` が「今画面にある一覧」への
    # 添字であり、並び替え検査より後のほとんどが、その並び替えられた
    # 「/repo」の一覧を使い回しているためである。ダイアログを開き直す2件
    # （取消・打ち切り表示）だけは影響を受けない。
    pytest.param(
        "picker",
        [(
            "js/filepicker.js",
            "  body.entries.forEach((entry, i) => {",
            "  [...body.entries].sort((a, b) => (a.name < b.name ? -1 : 1)).forEach((entry, i) => {",
        )],
        (
            "keeps the order the server sent",
            "entering a directory costs exactly one /api/browse",
            "選択 is dead until a file is picked, and names exactly what it will return",
            "choosing writes the box and fires change on it",
            "a refusal is shown inside the dialog, which stays where it was",
        ),
        id="picker_re_sorts_the_listing_on_the_client",
    ),
    # The confirmation, assigned instead of dispatched. `gen-output-dir`'s two
    # edit listeners retire the 上書き agreement, and this is the one way into
    # that box that would bypass them.
    # 確定を代入で済ませる。`gen-output-dir` の編集リスナ2つを迂回する唯一の
    # 経路になる。
    pytest.param(
        "picker",
        [(
            "js/main.js",
            '        box.dispatchEvent(new Event("input", { bubbles: true }));\n'
            '        box.dispatchEvent(new Event("change", { bubbles: true }));',
            "",
        )],
        ("fires change on it",),
        id="picker_sets_the_box_without_firing_its_events",
    ),
    # The cut, made silent. The list is right; the sentence over it is the lie.
    # 切り詰めを黙って行う。一覧そのものは正しく、その上の文だけが嘘になる。
    pytest.param(
        "picker",
        [(
            "js/filepicker.js",
            "  $(\"picker-count\").textContent = body.truncated\n"
            "    ? `${body.total}件中${body.entries.length}件を表示`\n"
            "    : `${body.total}件`;",
            "  $(\"picker-count\").textContent = `${body.entries.length}件`;",
        )],
        ("says how much of it is shown",),
        id="picker_truncates_the_listing_silently",
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
