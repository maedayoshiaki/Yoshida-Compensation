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
全体が torch を引かない。**担保しているのは所要時間ではなく依存の不在**で
ある（``sys.modules`` に ``torch``/``psutil``/``color_mixing_matrix`` が
現れないこと）。所要時間はファイルキャッシュ次第で動くので数値は参考値と
して扱うこと -- 計画B3 当時の実測は ``import pipeline`` 0.29s、計画B4 で
測り直すと同じものが 0.45〜0.54s だった（初回呼び出しは 0.015s のまま）。
ただし
``pipeline.prepare_target_image_for_compensation`` は今も ``geometry.warp``
経由で torch を引き初回に約7秒かかる。このビューはそれを呼ばない
（ターゲットは ``embed_in_active_region`` で投影フレームへ収まるだけで、
座標系のワープは計画B5の仕事である）。``color_mixing_matrix`` の直接
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

ターゲットの解決・復号方式の読み取り・カメラ座標の断り文・有効領域マスクは
``views/target_view.py`` にあり、``gamut`` と共有している（この2つは同じ
問いの前半を共有し、違うのは最後の一手だけである）。なぜ写しではなく共有な
のかはそちらの冒頭を参照。
"""

from __future__ import annotations

import numpy as np

from yoshidacomp.compensation.inverse_lut import InverseLut
from yoshidacomp.compensation.pipeline import (
    LINEARIZERS,
    generate_projection_image,
)
from yoshidacomp.config import get_config

from ..matrix import MatrixContext
from ..store import luts
from . import target_view
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

# Every parameter's default, in one place -- the three target-related ones from
# ``target_view`` (shared with ``gamut``) plus the three this view adds.
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
    **target_view.TARGET_DEFAULTS,
    "linearizer": "none",
    "lut": "",
    "output": _OUTPUTS[0],
}

# Reads this view's parameters, and refuses in this view's name -- every
# message below says "compensation" because ``ViewParams`` is told to.
# ``target`` / ``target_decode`` / ``target_gamma`` are resolved by it end to
# end; ``lut`` borrows only its id reader (``_require_lut`` below writes its
# own refusals, which talk about a LUT -- something ``gamut`` does not have).
_PARAMS = target_view.ViewParams("compensation", _DEFAULTS)


def _require_lut(params: dict) -> InverseLut:
    """Resolve ``lut`` for ``linearizer="inverse_lut"``, or refuse.

    Both refusals are deliberate and neither may become a 500. Without them
    ``_linearize_inverse_lut`` raises ``ValueError`` for ``lut=None``, which
    ``routers/view.py`` leaves as a 500 (a ``ValueError`` from a view is a
    server bug, see ``views/base.py::ViewNotApplicable``) -- so a request that
    merely named a LUT that was never loaded would come back as "the server
    broke".

    This used to be annotated ``NoReturn``, because until plan B3 Task 4 there
    was nothing to look an id up in and *every* id was unknown. Task 4 added
    ``POST /api/lut`` and ``store.luts``; the unknown-id branch is now a
    genuine ``KeyError`` from the store rather than an unconditional raise, and
    the ``lut=lut`` argument at the call site -- kept through three tasks while
    it was provably ``None`` -- carries the loaded ``InverseLut``.

    Returns:
        The stored ``InverseLut``, passed to ``generate_projection_image``
        unchanged so the GUI linearizes through the same object the CLI does.

    Raises:
        ViewNotApplicable: ``lut`` was not given, or names a LUT that is not
            loaded. Both are the client asking for something this server does
            not have, exactly as in ``_require_target``.

    未指定と未知のid、どちらも 400 で断る。断らないと ``lut=None`` の
    ``ValueError`` が 500 になり、「読み込んでいないLUTを指定した」だけの
    要求が「サーバが壊れた」として返る。
    """
    lut_id = _PARAMS.identifier(params, "lut")
    if not lut_id:
        raise ViewNotApplicable(
            'view "compensation": linearizer="inverse_lut" needs a measured '
            "LUT: pass lut=<id>, the id returned by POST /api/lut, or choose "
            "linearizer=gamma for the analytic inverse gamma. "
            "（inverse_lut は lut=<id> を必要とします。）"
        )
    try:
        return luts.get(lut_id).lut
    except KeyError:
        raise ViewNotApplicable(
            f'view "compensation": no LUT with id {lut_id!r} is loaded. '
            "POST /api/lut to load one and use the id it returns, or choose "
            "linearizer=none / linearizer=gamma. "
            f"（LUT {lut_id!r} は読み込まれていません。POST /api/lut で"
            "読み込んでください。）"
        ) from None


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
    # The camera-space refusal, its two-clause condition and the ~20 lines of
    # reasoning behind it live in ``target_view`` -- ``gamut`` asks the same
    # question for the same reasons, and this module's copy of that reasoning
    # was verbatim. See ``ViewParams.require_projector_space``.
    height, width, region = _PARAMS.require_projector_space(context, projector)

    target_image = _PARAMS.require_target(params)
    linearizer = str(_PARAMS.value(params, "linearizer"))
    lut = _require_lut(params) if linearizer == "inverse_lut" else None
    decode_mode, decode_gamma = _PARAMS.target_decode(params)

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
        lut=lut,
        target_decode=decode_mode,  # type: ignore[arg-type]
        target_gamma=decode_gamma,
        dtype=np.float32,
    )
    # ``linear is code`` when ``linearizer="none"`` -- ``_linearize_none``
    # returns the object it was given. That is a true statement about the
    # pipeline, not a defect, and the copy below means neither is written to.
    # ``linearizer="none"`` では両者は同一オブジェクトである。
    by_name = dict(zip(_OUTPUTS, (linear, code)))
    output = str(_PARAMS.value(params, "output"))
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
    inside = target_view.active_region_mask(height, width, region)
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
    # No test holds this to anything: every request this view accepts today
    # produces `low >= 0`, so `vmin = 0.0` passes all 26 of
    # `tests/test_web_compensation_view.py` (measured, task-4 final review
    # Minor 1). Kept anyway, and recorded rather than pinned -- the reason is
    # in the comment above, and a fixture that made it discriminating would
    # have to be a request this view does not currently offer.
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
            target_view.target_spec(_DEFAULTS),
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
            # Same source-of-truth argument as ``linearizer`` above, one level
            # down: ``target_view`` reads the choices off the pipeline's own
            # ``TargetDecodeMode``.
            target_view.target_decode_spec(_DEFAULTS),
            target_view.target_gamma_spec(_DEFAULTS),
        ],
        render=_render,
    )
)
