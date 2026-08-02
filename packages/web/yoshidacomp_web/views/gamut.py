"""Which pixels the compensation cannot represent.

補償が表現できない画素を描くビュー。``compensation`` の双子であり、違うのは
最後の一手だけ -- あちらは補償値をそのまま描き、こちらはそれを4クラスに
分ける。ターゲットの解決、カメラ座標の拒否、コスト上限、有効領域マスク、
パラメータの持ち方はすべて ``views/target_view.py`` を共有している。

| クラス | 条件 | ラベル |
| ---: | --- | --- |
| 0 | どのチャンネルも ``0 <= c <= 1`` | ``範囲内`` |
| 1 | いずれかが ``c < 0``、かつ 1超は無い | ``負クリップ`` |
| 2 | いずれかが ``c > 1``、かつ 負は無い | ``1超クリップ`` |
| 3 | 負と1超の両方がある | ``両方`` |

**このモジュールに補償の数式は無い。** 計算は
``pipeline.generate_projection_image()`` -- CLI が呼ぶのと同じ関数 -- を1回
呼ぶだけである。このモジュールがするのは (a) 引数の組み立てと検証、(b) その
1回の呼び出し、(c) 戻り値の分類、(d) 答えの無い画素（有効領域の外と、領域内の
非有限値）を NaN にすること、(e) ``vmin``/``vmax`` を決めることの5つで、
そのそれぞれに名指しのテストがある。

``clip=False`` がこのビューの存在理由である。``generate_projection_image`` の
既定は ``clip=True`` で、それは [0, 1] へ丸めてしまう -- このビューが見せ
ようとしている情報そのものを消す。``tests/test_web_gamut_view.py::
test_the_view_classifies_the_unclipped_values_not_the_clipped_ones`` が、
クリップすると色域外が1画素も残らないことと、このビューがそちらを返して
いないことの両方を測っている。

``pipeline`` を import してよい理由: 計画B3 Task 1 が numpy 実装へ
切り替えたため、``generate_projection_image`` の呼び出しグラフ全体が torch を
引かない。**担保しているのは所要時間ではなく依存の不在**であり、import 後・
レンダ後のいずれでも ``sys.modules`` に ``torch``・``psutil``・
``yoshidacomp.compensation.color_mixing_matrix`` が現れないことで確認する
（Task 1 レビューが実測）。所要時間はファイルキャッシュ次第で動くため参考値
であり、``import yoshidacomp_web.views.gamut`` は温状態 0.45〜0.52s、冷状態
2.88s、内訳の大半は cv2（0.21s）と numpy である。``pipeline`` 単体は
0.45〜0.54s（``compensation.py`` の docstring にある 0.29s は計画B3当時の値）。

``pipeline.prepare_target_image_for_compensation`` は今も ``geometry.warp``
経由で torch を引き初回に約7秒かかるが、このビューはそれを呼ばない
（座標系のワープは計画B4の仕事である）。レンダ実測 0.015s。
"""

from __future__ import annotations

import numpy as np

from yoshidacomp.compensation.pipeline import generate_projection_image
from yoshidacomp.config import get_config

from ..matrix import MatrixContext
from . import target_view
from .base import ViewResult, ViewSpec, register

# Class index -> label. The index into this list *is* the value written into
# the pixel, so the order is part of the wire contract and not a display
# choice: ``routers/view.py`` sends these as ``X-Labels-B64`` and the frontend
# reads the n-th label for class n. Swapping two entries relabels a region of
# the picture without changing a single pixel -- the most plausible-looking
# wrong answer this view can give.
#
# 並びがクラス番号そのものである（表示上の都合ではなくワイヤ契約の一部）。
# 入れ替えても画素は1つも変わらないまま、画面の意味だけが変わる。
LABELS = ["範囲内", "負クリップ", "1超クリップ", "両方"]

# This view's parameters are exactly the target-related ones, and deliberately
# no more.
#
# No ``linearizer``, no ``lut``, no ``output``: whether a pixel is out of gamut
# is a property of the *linear* compensation value, decided before any
# linearization happens. Offering those controls here would suggest that
# changing the LUT changes which pixels are out of gamut, which is false --
# and a control that appears to do something it does not is worse than no
# control. ``generate_projection_image`` agrees, from the other side: it
# refuses ``clip=False`` for any linearizer but ``"none"``, because
# ``_apply_inverse_gamma_numpy`` and ``apply_inverse_lut`` both assume input in
# [0, 1] and would return garbage for exactly the out-of-range values this view
# exists to show.
#
# ``linearizer``/``lut``/``output`` は持たない。色域外かどうかは線形補償値の
# 性質であり、その後の線形化とは無関係である。持たせると「LUTを変えると色域外
# が変わる」という誤った印象を与える。``generate_projection_image`` 側も同じ
# 立場で、``clip=False`` は ``linearizer="none"`` 以外を拒否する。
_DEFAULTS: dict[str, object] = dict(target_view.TARGET_DEFAULTS)

_PARAMS = target_view.ViewParams("gamut", _DEFAULTS)


def _render(context: MatrixContext, params: dict) -> ViewResult:
    """Classify the unclipped compensation into four classes, and mask it.

    Raises:
        ViewNotApplicable: the matrix is in camera space, or ``target`` cannot
            be resolved, or ``target_gamma`` is unusable. ``routers/view.py``
            maps this -- and only this -- to a 400 with the message intact.
            Deliberately not a ``ValueError``, which numpy and the pipeline
            also raise for genuine bugs; those must stay 500s.
    """
    projector = get_config().projector
    # Shared with ``compensation``, which refuses for the same reasons: see
    # ``target_view.ViewParams.require_projector_space`` for the two-clause
    # condition and why it is two and not three.
    height, width, region = _PARAMS.require_projector_space(context, projector)

    target_image = _PARAMS.require_target(params)
    decode_mode, decode_gamma = _PARAMS.target_decode(params)

    # The one call. Two arguments differ from ``compensation``'s and both are
    # forced by ``clip=False``: a float ``dtype`` (integer output scales by 255
    # and casts, which is undefined outside [0, 1]) and ``linearizer="none"``
    # (see the ``_DEFAULTS`` comment above). The second return value -- the
    # projector codes -- is the same object as the first under
    # ``linearizer="none"``, so there is nothing to choose between here and no
    # ``output`` parameter to choose it with.
    # 唯一の呼び出し。``compensation`` と違う2つの引数はどちらも
    # ``clip=False`` が要求するものである。``linearizer="none"`` では戻り値の
    # 2つは同一オブジェクトなので、選ぶものが無い。
    linear, _ = generate_projection_image(
        target_image,
        context.array,
        projector=projector,
        active_region=region,
        linearizer="none",
        lut=None,
        target_decode=decode_mode,  # type: ignore[arg-type]
        target_gamma=decode_gamma,
        dtype=np.float32,
        clip=False,
    )

    # The classification, and the only arithmetic in this module.
    #
    # **The bounds are closed**: exactly 0.0 and exactly 1.0 are 範囲内, so the
    # comparisons are strict (``< 0``, ``> 1``) and not ``<=`` / ``>=``.
    #
    # Read the next paragraph before touching
    # ``tests/test_web_gamut_view.py``'s fixture, because it says the opposite
    # of what one would guess. **Real data never reaches these bounds on this
    # view's path.** The pinning at exactly 0.0 and exactly 1.0 that everyone
    # has seen in this codebase is *produced by* ``calc_compensation_image``'s
    # clip -- and ``clip=False`` above is precisely what turns that clip off.
    # Measured over 12 real matrix x target pairs
    # (``data/comp_test_20260724/matrices_{gamma,none,lut_gray,lin_none}.npy``
    # x ``data/comp_test_targets/{nature_1,nature_2,car}.jpg``, 2,764,800
    # channel values each):
    #
    #   clip=True  (``compensation``'s path)  1,149,991 exactly 0.0, 649,022 exactly 1.0
    #   clip=False (**this view's path**)     0 exactly 0.0, 0 exactly 1.0
    #                                         (one pair produced a single 1.0; 1 value in 33M)
    #
    # So a ``<`` -> ``<=`` regression would misclassify **zero** pixels on real
    # data, and no amount of looking at the running GUI would reveal it. The
    # only thing standing between this rule and a silent regression is the
    # synthetic fixture in ``test_pixels_at_exactly_zero_and_exactly_one_are_in_range``,
    # which constructs the boundary exactly (via a black target band, where the
    # compensation is the matrix's bias row with no rounding at all). That is
    # what ``_assert_boundary_pixels_exist`` exists to protect: weaken the
    # fixture and this rule has no check anywhere, in test or in production.
    #
    # The rule itself is still right -- 0.0 and 1.0 are the two values the
    # projector reproduces exactly, so they are in gamut by definition, and a
    # ``<=`` would report the darkest and brightest reproducible pixels as
    # unrepresentable.
    #
    # 境界は閉区間である。ただし**実データはこの経路では境界を踏まない**。
    # 「白飛びが 1.0 に張り付く」現象を作っているのは
    # ``calc_compensation_image`` のクリップであり、このビューは ``clip=False``
    # でそれを止めている（実測: 実行列×実写12組、1組あたり276万チャンネル値で
    # ちょうど 0.0 / 1.0 はいずれも 0 件、1組だけ 1.0 が1件）。つまり ``<=`` へ
    # 退行しても実データでは1画素も誤分類されず、GUI を眺めても気付けない。
    # この規則を守っているのはテストの合成フィクスチャだけである。
    below = (linear < 0.0).any(axis=2)
    above = (linear > 1.0).any(axis=2)
    classes = np.where(
        below & above, 3.0, np.where(above, 2.0, np.where(below, 1.0, 0.0))
    )
    data = classes[:, :, None].astype(np.float32)

    # Two kinds of pixel have no class, and neither may come out as class 0.
    #
    # **After the classification, never before.** Every comparison against a
    # non-finite value is False, so such a pixel has neither ``< 0`` nor
    # ``> 1`` true and falls through to class 0 -- "in range" -- which is the
    # most reassuring possible wrong answer. Masking here instead makes it NaN,
    # which is what ``coef_tiles``, ``surface`` and ``compensation`` all write
    # for an unanswerable pixel and what the hover readout's out-of-region
    # marker looks for.
    #
    # 1. Outside the active region. ``generate_projection_image`` writes
    #    exactly 0.0 there (``mask_to_active_region``), and 0.0 is a perfectly
    #    good class 0 -- so without this line three quarters of a frame would
    #    report "the compensation is representable" for pixels the projector
    #    never lit.
    # 2. Non-finite inside it. ``calc_compensation_image`` is a forward product
    #    and cannot manufacture a NaN from finite input, but the matrix is
    #    whatever ``POST /api/session/matrix`` was pointed at, and a NaN or inf
    #    coefficient propagates straight through. Sampling this repository's
    #    own ``data/**/*.npy`` found none, so this is a guard against a bad
    #    input file rather than against the pipeline -- but the failure it
    #    prevents is the same one the region mask prevents, in the one place
    #    the region mask does not reach. An inf is masked with the NaN, and
    #    deliberately: ``+inf > 1`` would classify as 1超クリップ, which reads
    #    as a measurement of a pixel that has no measurement. ``surface`` marks
    #    its singular pixels NaN for the same reason.
    #
    # 分類の**後**に潰す。非有限値との比較はすべて False になるため、先に
    # やると領域外も NaN 画素も黙ってクラス0（範囲内）に落ちる。領域外は
    # パイプライン自身が 0.0 を書くのでやはりクラス0になる。行列側の NaN/inf
    # は入力ファイル由来でありうる（``calc_compensation_image`` は前進積なので
    # 有限入力からは作れない）。inf も NaN と同じく潰す -- ``+inf > 1`` は
    # 「1超クリップ」と読めてしまうが、それは測定値の無い画素の測定値である。
    inside = target_view.active_region_mask(height, width, region)
    answerable = inside & np.isfinite(linear).all(axis=2)
    data[~answerable] = np.nan

    # The class index range, not a percentile and not the data's own extent.
    #
    # ``compensation`` and ``surface`` take ``nanpercentile`` because their
    # values are a measurement whose useful range is unknown in advance. Here
    # the range is known exactly and is a property of ``LABELS``, not of the
    # frame -- and the two are easy to confuse, because a frame containing all
    # four classes has ``nanmin == 0`` and ``nanmax == 3`` too.
    #
    # They come apart on a frame that is entirely in gamut. Every pixel is then
    # class 0, so a data-derived range gives ``vmin == vmax == 0``, every class
    # collapses onto one colour, and ``routers/view.py``'s categorical check
    # turns the response into a 500 -- the view breaking precisely when the
    # answer is "everything is representable". That frame is not exotic: it is
    # what a dim or low-saturation target produces. Real data goes the other
    # way just as hard (47-85% out of gamut across the 12 measured pairs
    # above), so neither extreme is "the common case" and neither should set
    # the scale.
    #
    # Pinned by ``test_the_display_range_survives_an_entirely_in_gamut_frame``,
    # which renders such a frame; the four-class fixture alone cannot fail for
    # this, since its own min and max happen to be 0 and 3.
    #
    # Derived from ``LABELS`` rather than written as 3.0: adding a class means
    # adding a label, and the display range has to follow it in the same edit.
    #
    # クラス番号の範囲は既知であり ``LABELS`` の性質である（フレームの性質では
    # ない）。4クラスが揃ったフレームでは ``nanmin``/``nanmax`` も 0/3 になる
    # ため両者は紛らわしいが、全画素が範囲内のフレームで分かれる。そこでは
    # データ由来だと ``vmin == vmax == 0`` になり、router の categorical 規則が
    # 500 にする -- 「全部表現できています」という最も出したい答えのときだけ
    # ビューが壊れる。
    return ViewResult(
        data=data, vmin=0.0, vmax=float(len(LABELS) - 1), labels=list(LABELS)
    )


SPEC = register(
    ViewSpec(
        key="gamut",
        label="色域外",
        # One channel of class indices. ``routers/view.py`` knows that a
        # ``categorical`` view's labels name the classes rather than the
        # channels, and checks the declared range against them instead of
        # against ``channels``.
        kind="categorical",
        channels=1,
        # The picture is a property of the target-through-the-matrix, not the
        # matrix, so the readout's second panel says something its 4x3 table
        # cannot.
        values_are_matrix=False,
        params=[
            target_view.target_spec(_DEFAULTS),
            target_view.target_decode_spec(_DEFAULTS),
            target_view.target_gamma_spec(_DEFAULTS),
        ],
        render=_render,
    )
)
