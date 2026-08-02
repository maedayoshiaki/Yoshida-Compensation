"""The target image with the matrix applied: what the projector should emit.

ターゲット画像に行列を適用した結果を描くビュー。線形補償画像
（``output="linear"``）と、線形化を通したあとの投影コード値
（``output="code"``）のどちらかを返す。

**このモジュールに補償の数式は無い。** 計算は
``yoshidacomp.compensation.pipeline.generate_projection_image()`` -- CLI
（``examples/python/rebuild_compensation_with_map.py``）が呼ぶのと同じ関数、
同じ引数 -- を1回呼ぶだけで、このモジュールがするのは引数を組み立てること、
戻り値2つのうち片方を選ぶこと、有効領域の外を NaN にすることの3つである。
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
"""

from __future__ import annotations

import math
from typing import get_args

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
# them. ``output`` indexes this tuple by name rather than by a boolean flag:
# a boolean has to be inverted somewhere between the control and the tuple, and
# an inverted one still returns a full, plausible frame -- silently the wrong
# one. Naming both ends the same way means the mapping is one substitution, and
# the equality test above pins it for each name separately.
#
# 戻り値2つを名前で選ぶ。真偽値だとどこかで反転させることになり、反転した
# ままでも「もっともらしいフレーム」が返るため誤りが表に出ない。
_OUTPUTS = ("linear", "code")


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
    未指定と空欄は同じ意味なので同じ ``""`` に落とす。
    """
    value = params.get(name)
    return "" if value is None else str(value).strip()


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


def _require_lut(params: dict) -> object:
    """Resolve ``lut`` for ``linearizer="inverse_lut"`` -- today, always refuse.

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
    mode = str(params.get("target_decode", "srgb"))
    gamma = float(params.get("target_gamma", 2.2))
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
    # ``context.space`` was decided when the matrix was loaded, against the
    # config in force *then*. Re-deciding it against the config in force *now*
    # is what keeps a mismatched frame from reaching ``calc_compensation_image``
    # as a numpy shape ``ValueError`` -- a 500 blaming the server for a
    # configuration that changed under a still-valid matrix id. The two agree
    # in the shipped app (nothing reloads the config in-process); they do not
    # agree in a test suite that swaps projector configs per test.
    # ``context.space`` は読み込み時の判定なので、現在の config でもう一度
    # 判定する。食い違ったまま進むと numpy の形状 ``ValueError``（＝500）に
    # なるが、それはサーバのバグではない。
    if (
        region is None
        or context.space == "camera"
        or classify_space(height, width, projector) == "camera"
    ):
        raise ViewNotApplicable(_CAMERA_SPACE_REFUSAL)

    target_image = _require_target(params)
    linearizer = str(params.get("linearizer", "none"))
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
    output = str(params.get("output", _OUTPUTS[0]))
    if output not in _OUTPUTS:
        # Unreachable over HTTP -- ``routers/view.py`` rejects a ``select``
        # value outside ``choices`` with its own 400 -- but not unreachable
        # from a direct ``render()`` call, which is how another view or a later
        # task would reuse this. Without it, a misspelled ``output`` silently
        # falls through to ``code`` and returns a full, plausible frame of the
        # wrong image. ``linearizer`` needs no such guard: an unknown one
        # reaches ``generate_projection_image`` and raises there, loudly.
        # HTTP経由では router が弾くので到達しないが、``render()`` を直接
        # 呼ぶ経路では到達する。無いと綴り間違いが黙って ``code`` に落ち、
        # 「もっともらしい別の画像」が返る。
        raise ViewNotApplicable(
            f'view "compensation": output must be one of {list(_OUTPUTS)}, got '
            f"{output!r}. "
            f"（output は {list(_OUTPUTS)} のいずれかです。）"
        )
    data = np.array(linear if output == _OUTPUTS[0] else code, dtype=np.float32)

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

    if inside.any():
        low, high = (float(v) for v in np.nanpercentile(data, [1, 99]))
    else:
        low, high = 0.0, 1.0

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
        params=[
            # Runtime ids: free text, no choices. See ``base.ParamSpec.type``.
            ParamSpec(name="target", label="ターゲットID", type="text", default=""),
            ParamSpec(
                name="linearizer",
                label="線形化",
                type="select",
                default="none",
                # Taken from the pipeline's own registry rather than written
                # out again: a linearizer added there appears here without a
                # second edit, and cannot be offered here without existing
                # there. ``dict`` preserves insertion order, so the order the
                # dropdown shows is the order ``pipeline`` declares.
                # パイプライン側の登録をそのまま使う。書き写すとGUIだけが
                # 古いまま静かにずれる。
                choices=list(LINEARIZERS),
            ),
            ParamSpec(name="lut", label="LUT ID", type="text", default=""),
            ParamSpec(
                name="output",
                label="出力",
                type="select",
                default=_OUTPUTS[0],
                choices=list(_OUTPUTS),
            ),
            ParamSpec(
                name="target_decode",
                label="ターゲット復号",
                type="select",
                default="srgb",
                # Same source-of-truth argument as ``linearizer`` above; this
                # one is a ``Literal`` rather than a registry.
                choices=list(get_args(TargetDecodeMode)),
            ),
            ParamSpec(
                name="target_gamma", label="ターゲットγ", type="number", default=2.2
            ),
        ],
        render=_render,
    )
)
