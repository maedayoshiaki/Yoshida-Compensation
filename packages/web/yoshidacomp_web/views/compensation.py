"""The target image with the matrix applied: what the projector should emit.

ターゲット画像に行列を適用した結果を描くビュー。線形補償画像
（``output="linear"``）と、線形化を通したあとの投影コード値
（``output="code"``）のどちらかを返す。

**このモジュールに補償の数式は無い。** 計算は
``yoshidacomp.compensation.pipeline.generate_projection_image()`` -- CLI
（``examples/python/rebuild_compensation_with_map.py``）が呼ぶのと同じ関数、
同じ引数 -- を1回呼ぶだけである。このモジュールがするのは (a) 引数の組み立てと
検証、(b) その1回の呼び出し、(c) 戻り値2つのうち片方を名前で選ぶこと、
(d) 有効領域の外を NaN にすること、(e) ``vmin``/``vmax`` を決めること の5つで、
そのそれぞれに名指しのテストがある。
``tests/test_web_compensation_view.py::
test_view_matches_generate_projection_image_exactly`` が、ビューの画素が
その関数の戻り値と**完全一致**することを linearizer × output の4通りで
確認している。ここに数式を書き足せばその1件が落ちる。

``pipeline`` を import してよい理由（と、してはならないこと）:
計画B3 Task 1 が ``_linearize_gamma`` と ``calc_compensation_image`` を
numpy 実装へ切り替えたため、``generate_projection_image`` の呼び出しグラフ
全体が torch を引かない（import 0.29s、初回呼び出し 0.015s）。ただし
``pipeline.prepare_target_image_for_compensation`` は今も ``geometry.warp``
経由で torch を引き初回に約7秒かかる。このビューはそれを呼ばない
（ターゲットは ``embed_in_active_region`` で投影フレームへ収まるだけで、
座標系のワープは計画B4の仕事である）。``color_mixing_matrix`` の直接
import は引き続き禁止で、ガンマは ``LINEARIZERS["gamma"]`` 経由でのみ使う。

``target``/``lut`` が ``type="text"`` である理由（``base.ParamSpec.type`` が
参照している議論の本体）: 有効なidは実行時に増える。``ParamSpec.choices`` は
静的な ``list[str]`` であり ``ViewSpec`` は frozen なので、``/api/views`` を
叩いた時点の一覧を返す形にすると、レジストリを毎回組み直すか frozen を
破ることになる。しかもフロントは ``buildServerParams`` の1回でコントロールを
作るため、その後に読み込んだターゲットは再読込まで選べず、「一覧に無いidは
使えない」という失敗はどのみち残る。空の ``choices`` を持つ ``select`` は
``static/js/main.js::buildServerParams`` で**選択肢ゼロのドロップダウン**に
なり、値を入れる手段が無い。``"text"`` は JavaScript の変更を要さない
（同関数の ``else`` 枝が既に ``<input type="text">`` を作り、
``routers/view.py`` は number/select 以外を生の文字列として渡す）。
不正なidは ``ViewNotApplicable`` にする -- ``select`` の ``choices`` 外を
router が 400 にするのと、利用者から見て同じ結果になる。
"""

from __future__ import annotations

import math
from typing import NoReturn, get_args

import numpy as np

from yoshidacomp.compensation.pipeline import (
    LINEARIZERS,
    TargetDecodeMode,
    generate_projection_image,
)
from yoshidacomp.config import get_config

from ..matrix import MatrixContext, classify_space
from ..store import targets
from .base import ParamSpec, ViewNotApplicable, ViewResult, ViewSpec, register

LABELS = ["R", "G", "B"]

# The two arrays ``generate_projection_image`` returns, in the order it returns
# them. ``output`` names one of them rather than being a boolean flag: a boolean
# has to be inverted somewhere between the control and the tuple, and an
# inverted one still returns a full, plausible frame -- silently the wrong one.
# Naming both ends the same way, and pairing them with ``dict(zip(...))`` below,
# makes the mapping one substitution;
# ``test_view_matches_generate_projection_image_exactly`` then pins it for each
# name separately. (That parametrization is what catches a swap: the
# "linear differs from code" test passes under one, since both are still
# different from each other.)
#
# 戻り値2つを名前で選ぶ。真偽値だとどこかで反転させることになり、反転した
# ままでも「もっともらしいフレーム」が返るため誤りが表に出ない。
_OUTPUTS = ("linear", "code")

# Every parameter's default, in one place.
#
# Each of these used to be written twice -- once in the ``ParamSpec`` the GUI
# reads and once as the fallback in ``params.get(name, ...)`` -- and only the
# declared copy was pinned by a test. Over HTTP the router always fills the
# declared default, so the fallback is unreachable there and the two could drift
# apart silently; a direct ``render()`` call, which this module supports (see
# the ``output`` guard), would then get the drifted value. One dict, read by
# both, removes the question.
#
# 既定値は1箇所にまとめる。以前は ``ParamSpec`` と ``params.get`` の
# フォールバックに二重に書いており、routerが常に宣言側の既定を埋めるため
# フォールバック側は HTTP からは到達不能で、静かにずれても誰も気付かない。
# ``render()`` を直接呼ぶ経路だけがそのずれた値を受け取ることになる。
_DEFAULTS: dict[str, object] = {
    "target": "",
    "linearizer": "none",
    "lut": "",
    "output": _OUTPUTS[0],
    "target_decode": "srgb",
    "target_gamma": 2.2,
}


def _param(params: dict, name: str) -> object:
    """Read *name* from *params*, falling back to its declared default."""
    value = params.get(name)
    return _DEFAULTS[name] if value is None else value


# Why a camera-space matrix is refused, in one place so the router's 400 and
# this module's docstring cannot drift apart. Same policy, and mostly the same
# reasons, as ``views/surface.py::_CAMERA_SPACE_REFUSAL``.
#
# Correctness first here, unlike ``surface`` where cost came first: outside
# projector space ``active_region`` is ``None``, and ``active_region`` is not
# optional for this view -- ``generate_projection_image`` needs it to place the
# target inside the frame at all. There is no meaningful answer to substitute,
# because "the target image resized into the projector's active rectangle" is
# not a thing a camera-space matrix has. Cost is the same argument as
# ``surface``'s and applies unchanged: the work is set by the pixel count, and
# ``data/color_mixing_matrices.npy`` (4660x6984 = 32.5M pixels) is 15.7x the
# 1080p projector frame.
#
# カメラ座標では ``active_region`` が ``None`` になり、このビューにとって
# それは省略可能な情報ではない（``generate_projection_image`` はターゲットを
# フレームのどこへ置くかをそれで決める）。「投影領域へリサイズしたターゲット」
# という概念自体がカメラ座標の行列には無いので、代わりに描けるものが無い。
# コスト面の理由は ``surface`` と同じで、画素数だけで決まる。
_CAMERA_SPACE_REFUSAL = (
    'view "compensation" needs a projector-space matrix; this one is in camera '
    "space, where there is no active region to place the target image in. "
    "Warping a camera-space matrix into projector space is planned for B4 -- "
    "until then, load a matrix whose frame matches the configured projector. "
    "（compensation はプロジェクタ座標の行列を必要とします。カメラ座標では"
    "ターゲットを収める有効領域が存在しません。カメラ座標からのワープは"
    "計画B4の予定です。）"
)


def _identifier(params: dict, name: str) -> str:
    """Read a runtime id parameter as a stripped string.

    ``""`` means "not given". The router supplies the declared default for a
    parameter absent from the query string, so an omitted ``target`` and an
    empty box arrive here identically -- and both mean the same thing.
    ``.strip()`` for the same reason: a box the user cleared with the space bar
    is an empty box, and telling them "no target with id '  '" instead of "pass
    target=<id>" answers a question they did not ask.
    未指定と空欄は同じ意味なので同じ ``""`` に落とす。空白だけの入力も同様で、
    「id '  ' は存在しない」ではなく「target=<id> を渡せ」と言うべきである。
    """
    return str(_param(params, name)).strip()


def _require_target(params: dict) -> np.ndarray:
    """Return the selected target image, or refuse.

    Raises:
        ViewNotApplicable: ``target`` was not given, or names a target that is
            not loaded. Both are the client asking for something this server
            does not have, not a bug -- ``routers/view.py`` turns them into a
            400 carrying this message, which is where the user is told to POST
            the image first.
    """
    target_id = _identifier(params, "target")
    if not target_id:
        raise ViewNotApplicable(
            'view "compensation" needs a target image: pass target=<id>, the '
            "id returned by POST /api/target. "
            "（compensation は target=<id> を必要とします。先に "
            "POST /api/target で画像を読み込んでください。）"
        )
    try:
        return targets.get(target_id).image
    except KeyError:
        raise ViewNotApplicable(
            f'view "compensation": no target image with id {target_id!r} is '
            "loaded. POST /api/target to load one and use the id it returns. "
            f"（target {target_id!r} は読み込まれていません。）"
        ) from None


def _require_lut(params: dict) -> NoReturn:
    """Resolve ``lut`` for ``linearizer="inverse_lut"`` -- today, always refuse.

    ``NoReturn`` is the honest annotation *today* and is meant to change: every
    path raises because there is nothing to look a lut id up in yet. Task 4
    turns this into ``-> InverseLut``, at which point the ``lut=lut`` argument
    at the call site -- provably ``None`` while this is ``NoReturn``, and kept
    deliberately -- starts carrying a value.
    ``NoReturn`` は今日の事実であり、変わることを前提にした注釈である。

    There is no LUT store yet: ``POST /api/lut`` and ``store.luts`` arrive in
    plan B3 Task 4, so *every* lut id is unknown at the moment. Refusing with
    a message that says so is the whole point -- without this,
    ``_linearize_inverse_lut`` raises ``ValueError`` for ``lut=None``, which
    ``routers/view.py`` deliberately leaves as a 500 (a ``ValueError`` from a
    view is a server bug, see ``views/base.py::ViewNotApplicable``), and the
    user gets a blank failure for a request that was merely early.

    **Task 4:** replace this body with a ``store.luts.get(lut_id)`` lookup that
    keeps both refusals -- the empty-id one and the unknown-id one, the latter
    then being a genuine ``KeyError`` rather than an unconditional raise.

    LUTストアは計画B3 Task 4 で入る。それまで LUT id は全て未知なので、
    理由の分かる 400 で断る。断らないと ``_linearize_inverse_lut`` の
    ``ValueError`` が 500 になり、単に手順が早かっただけの要求が
    「サーバが壊れた」として返る。
    """
    lut_id = _identifier(params, "lut")
    if not lut_id:
        raise ViewNotApplicable(
            'view "compensation": linearizer="inverse_lut" needs a measured '
            "LUT: pass lut=<id>, or choose linearizer=gamma for the analytic "
            "inverse gamma. "
            "（inverse_lut は lut=<id> を必要とします。）"
        )
    raise ViewNotApplicable(
        f'view "compensation": no LUT with id {lut_id!r} is loaded. Loading '
        "inverse LUTs (POST /api/lut) is plan B3 Task 4; until then use "
        "linearizer=none or linearizer=gamma. "
        f"（LUT {lut_id!r} は読み込まれていません。LUTの読み込みは"
        "計画B3 Task 4 で入ります。）"
    )


def _target_decode(params: dict) -> tuple[str, float]:
    """Read the target's transfer function and its exponent, or refuse.

    This view owns the decode mode, and deliberately so. A target ``.png`` is
    stored as raw uint8 by ``target.py`` because ``decode_target`` needs a
    ``mode`` and a ``gamma`` that are a property of *this request*, not of the
    file: the same image is a different compensation input depending on how it
    was authored. The names, the choices and the defaults are the CLI's
    (``rebuild_compensation_with_map.py``'s ``--target-decode`` /
    ``--target-gamma``, defaulting to ``srgb`` / ``2.2``), so a GUI reading and
    a command line are comparable without a translation table.

    復号方式はこのビューが持つ。``target.py`` が uint8 の生画素だけを保持して
    いるのは、``mode``/``gamma`` がファイルの性質ではなく要求ごとの選択だから
    である。名前・選択肢・既定値は CLI の ``--target-decode`` /
    ``--target-gamma`` に揃えてあるので、GUIの見た目とCLIの出力を突き合わせ
    られる。

    Raises:
        ViewNotApplicable: ``target_decode="gamma"`` with a non-positive or
            non-finite exponent. ``decode_target`` raises ``ValueError`` for
            those, which would be a 500; typing 0 into a number box is a
            routine mistake, not a server bug.
    """
    mode = str(_param(params, "target_decode"))
    gamma = float(_param(params, "target_gamma"))  # type: ignore[arg-type]
    if mode == "gamma" and not (math.isfinite(gamma) and gamma > 0.0):
        raise ViewNotApplicable(
            f'view "compensation": target_gamma must be a finite positive '
            f"number, got {gamma!r}. "
            "（target_gamma は正の有限値である必要があります。）"
        )
    return mode, gamma


def _render(context: MatrixContext, params: dict) -> ViewResult:
    """Run the CLI's compensation for one target image and mask it.

    Raises:
        ViewNotApplicable: the matrix is in camera space, or one of the
            runtime ids (``target``, ``lut``) cannot be resolved, or
            ``target_gamma`` is unusable. ``routers/view.py`` maps this -- and
            only this -- to a 400 with the message intact. Deliberately not a
            ``ValueError``, which numpy and the pipeline also raise for genuine
            bugs; those must stay 500s.
    """
    projector = get_config().projector
    height, width = int(context.array.shape[0]), int(context.array.shape[1])
    region = context.active_region
    # Two clauses, both load-bearing, and deliberately not three.
    #
    # ``region is None`` *is* "the matrix is in camera space": ``matrix.py``
    # sets ``active_region`` from ``resolve_active_region`` only when the space
    # is ``"projector"``, and to ``None`` otherwise, so testing
    # ``context.space == "camera"`` as well says the same thing a second time.
    # This condition used to carry that third clause; it was removed after a
    # review measured that reducing the whole condition to ``region is None``
    # left every test green. ``region is None`` is the clause kept because it
    # is also what narrows ``region`` to a tuple for everything below.
    #
    # The ``classify_space`` call is the one that adds behaviour, and it is a
    # *different* question: ``context.space`` was decided when the matrix was
    # loaded, against the config in force **then**. Re-deciding it against the
    # config in force **now** is what keeps a mismatched frame from reaching
    # ``calc_compensation_image`` as a numpy shape ``ValueError`` -- a 500
    # blaming the server for a configuration the user changed under a matrix id
    # that is still perfectly valid. ``reload_config`` is a public API and this
    # repository's ``config.toml`` has eight sibling variants in the working
    # tree right now. Pinned by
    # ``test_a_matrix_loaded_under_another_projector_config_is_refused_not_500``.
    #
    # 2条件であって3条件ではない。``region is None`` は「カメラ座標である」
    # ことそのもの（``matrix.py`` がプロジェクタ座標のときだけ領域を入れる）
    # なので、``context.space == "camera"`` を併記しても同じことを2度言うだけ
    # である。実際レビューが「全条件を ``region is None`` だけに縮めても
    # 全テストが通る」ことを実測したため削除した。``classify_space`` の方は
    # 別の問いで、読み込み時ではなく**現在の** config との一致を見る。
    # 食い違ったまま進むと numpy の形状 ``ValueError``（＝500）になるが、
    # それはサーバのバグではない。
    if region is None or classify_space(height, width, projector) == "camera":
        raise ViewNotApplicable(_CAMERA_SPACE_REFUSAL)

    target_image = _require_target(params)
    linearizer = str(_param(params, "linearizer"))
    lut = _require_lut(params) if linearizer == "inverse_lut" else None
    decode_mode, decode_gamma = _target_decode(params)

    # The one call. ``dtype=np.float32`` rather than the CLI's ``np.uint8``:
    # the viewer wants the compensation in normalized units, and the pipeline
    # documents that float dtype as the viewer's case. Nothing else differs
    # from what the CLI passes -- including the exponent trap, which is
    # ``_linearize_gamma``'s to handle (it applies ``1 / projector.gamma``
    # itself), so this module never passes an exponent of its own.
    # 唯一の呼び出し。CLI と違うのは ``dtype`` だけ（ビューアは正規化された
    # ままの値を必要とする）。ガンマの指数を反転する罠は
    # ``_linearize_gamma`` 側の責務であり、ここでは指数を渡さない。
    linear, code = generate_projection_image(
        target_image,
        context.array,
        projector=projector,
        active_region=region,
        linearizer=linearizer,  # type: ignore[arg-type]
        lut=lut,  # type: ignore[arg-type]
        target_decode=decode_mode,  # type: ignore[arg-type]
        target_gamma=decode_gamma,
        dtype=np.float32,
    )
    # ``linear is code`` when ``linearizer="none"`` -- ``_linearize_none``
    # returns the object it was given. That is a true statement about the
    # pipeline, not a defect, and the copy below means neither is written to.
    # ``linearizer="none"`` では両者は同一オブジェクトである。
    by_name = dict(zip(_OUTPUTS, (linear, code)))
    output = str(_param(params, "output"))
    if output not in by_name:
        # Unreachable over HTTP -- ``routers/view.py`` rejects a ``select``
        # value outside ``choices`` with its own 400 -- but not unreachable
        # from a direct ``render()`` call, which is how another view or a later
        # task would reuse this. Without it, a misspelled ``output`` silently
        # falls through to whichever image the ``else`` happens to name and
        # returns a full, plausible frame of the wrong one. ``linearizer`` needs
        # no such guard: an unknown one reaches ``generate_projection_image``
        # and raises there, loudly.
        # HTTP経由では router が弾くので到達しないが、``render()`` を直接
        # 呼ぶ経路では到達する。無いと綴り間違いが黙って別の画像に落ち、
        # 「もっともらしい別の絵」が返る。
        raise ViewNotApplicable(
            f'view "compensation": output must be one of {list(_OUTPUTS)}, got '
            f"{output!r}. "
            f"（output は {list(_OUTPUTS)} のいずれかです。）"
        )
    # A lookup, not an ``if/else``: with a third entry in ``_OUTPUTS`` an
    # ``else`` would silently hand back the second image for it, which is the
    # exact class of failure the named-select decision exists to prevent.
    # ``if/else`` ではなく引き当てにする。``_OUTPUTS`` が3つになったとき
    # ``else`` は黙って2番目を返すが、それは名前で選ぶ設計が避けたい失敗そのもの。
    data = np.array(by_name[output], dtype=np.float32)

    # Outside the active region there is no answer, and it must not be black.
    # ``generate_projection_image`` already returns 0 there
    # (``mask_to_active_region``), which the shader would draw as a projector
    # commanded to emit nothing -- a plausible picture of a wrong thing.
    # ``coef_tiles`` and ``surface`` both mark unanswerable pixels NaN and the
    # hover readout's out-of-region marker is aligned with that convention, so
    # this view uses it too.
    # 有効領域の外は「答えが無い」であって「黒を投影する」ではない。
    # パイプラインは 0 を書くのでそのままだと黒として描かれ、読み取り欄の
    # 領域外マーカーとも食い違う。``coef_tiles``/``surface`` と同じ規約に揃える。
    x, y, w, h = region
    inside = np.zeros((height, width), dtype=bool)
    inside[y : y + h, x : x + w] = True
    data[~inside] = np.nan

    # No ``if inside.any()`` guard around this, unlike ``surface``, because here
    # it cannot be false: ``resolve_active_region`` returns the full frame when
    # the configured active width or height is non-positive and otherwise
    # validates a rectangle with ``w, h >= 1``, so at least one pixel is always
    # inside. ``surface``'s equivalent branch is genuinely reachable -- an
    # all-singular matrix leaves nothing usable even inside the region -- and
    # that difference is why it has one and this does not.
    # ``surface`` と違いここには ``inside.any()`` の分岐を置かない。
    # ``resolve_active_region`` が幅・高さ1以上を保証するため常に真になる。
    # ``surface`` 側は全画素が特異な行列で偽になりうるので分岐が要る。
    low, high = (float(v) for v in np.nanpercentile(data, [1, 99]))

    # ``surface``'s compromise, for ``surface``'s reason: the rgb branch maps
    # ``[vmin, vmax]`` onto ``[0, 1]``, so ``vmin`` is the black point, and all
    # three channels share one range -- lifting the floor subtracts the same
    # constant from R, G and B, which desaturates rather than darkens.
    #
    # ``min(0.0, low)`` is 0 for every request this view can serve today,
    # because ``generate_projection_image`` clips the compensation to [0, 1]
    # (``clip`` defaults to ``True`` and this view does not expose it). It is
    # written as the ``min`` anyway, not as a literal 0: ``clip=False`` is a
    # documented option of the same call -- it is how the planned ``gamut``
    # view classifies pixels the matrix drives below 0 -- and a hard-coded
    # floor would silently clip exactly the values that option exists to show.
    #
    # ``surface`` と同じ折衷を同じ理由で踏襲する。3チャンネルが1つのレンジを
    # 共有するため、床を浮かせると露出ではなく彩度が変わる。``clip=True``
    # である以上 ``min`` は今日どの要求でも 0 になるが、``clip=False`` を
    # 使う日（計画中の ``gamut``）に備えて 0 の直書きにはしない。
    vmin = min(0.0, low)
    return ViewResult(data=data, vmin=vmin, vmax=max(high, 1e-6), labels=LABELS)


SPEC = register(
    ViewSpec(
        key="compensation",
        label="補償画像",
        kind="rgb",
        channels=3,
        # The picture is the target with the matrix applied -- not the matrix --
        # so the readout's second panel says something its 4x3 table cannot.
        # 描いているのは行列そのものではないので、読み取り欄の2枚目の板は残す。
        values_are_matrix=False,
        # Every ``default`` below is read from ``_DEFAULTS``, which ``_render``
        # also reads. See that dict for why they are not written twice.
        # 既定値はすべて ``_DEFAULTS`` から引く（``_render`` も同じ辞書を読む）。
        params=[
            # Runtime ids: free text, no choices. See ``base.ParamSpec.type``
            # for why a ``select`` cannot express them.
            ParamSpec(
                name="target",
                label="ターゲットID",
                type="text",
                default=_DEFAULTS["target"],
            ),
            ParamSpec(
                name="linearizer",
                label="線形化",
                type="select",
                default=_DEFAULTS["linearizer"],
                # Taken from the pipeline's own registry rather than written
                # out again: a linearizer added there appears here without a
                # second edit, and cannot be offered here without existing
                # there. ``dict`` preserves insertion order, so the order the
                # dropdown shows is the order ``pipeline`` declares.
                # パイプライン側の登録をそのまま使う。書き写すとGUIだけが
                # 古いまま静かにずれる。
                choices=list(LINEARIZERS),
            ),
            ParamSpec(
                name="lut", label="LUT ID", type="text", default=_DEFAULTS["lut"]
            ),
            ParamSpec(
                name="output",
                label="出力",
                type="select",
                default=_DEFAULTS["output"],
                choices=list(_OUTPUTS),
            ),
            ParamSpec(
                name="target_decode",
                label="ターゲット復号",
                type="select",
                default=_DEFAULTS["target_decode"],
                # Same source-of-truth argument as ``linearizer`` above; this
                # one is a ``Literal`` rather than a registry.
                choices=list(get_args(TargetDecodeMode)),
            ),
            ParamSpec(
                name="target_gamma",
                label="ターゲットγ",
                type="number",
                default=_DEFAULTS["target_gamma"],
            ),
        ],
        render=_render,
    )
)
