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
1回の呼び出し、(c) 戻り値の分類、(d) 有効領域の外を NaN にすること、
(e) ``vmin``/``vmax`` を決めることの5つで、そのそれぞれに名指しのテストがある。

``clip=False`` がこのビューの存在理由である。``generate_projection_image`` の
既定は ``clip=True`` で、それは [0, 1] へ丸めてしまう -- このビューが見せ
ようとしている情報そのものを消す。``tests/test_web_gamut_view.py::
test_clipping_the_compensation_would_erase_every_out_of_gamut_class`` が、
クリップすると色域外が1画素も残らないことを測っている。

``pipeline`` を import してよい理由: 計画B3 Task 1 が numpy 実装へ
切り替えたため、``generate_projection_image`` の呼び出しグラフ全体が torch を
引かない（import 0.29s）。``pipeline.prepare_target_image_for_compensation``
は今も ``geometry.warp`` 経由で torch を引き初回に約7秒かかるが、このビューは
それを呼ばない（座標系のワープは計画B4の仕事である）。
"""

from __future__ import annotations

import numpy as np

from yoshidacomp.compensation.pipeline import generate_projection_image
from yoshidacomp.config import get_config

from ..matrix import MatrixContext, classify_space
from . import target_view
from .base import ViewNotApplicable, ViewResult, ViewSpec, register

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

_CAMERA_SPACE_REFUSAL = target_view.camera_space_refusal("gamut")


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
    height, width = int(context.array.shape[0]), int(context.array.shape[1])
    region = context.active_region
    # Two clauses, both load-bearing, exactly as in ``views/compensation.py``.
    # ``region is None`` *is* "the matrix is in camera space" (``matrix.py``
    # fills ``active_region`` only for projector space) and is also what
    # narrows ``region`` to a tuple below. ``classify_space`` asks a
    # *different* question: ``context.space`` was decided against the config in
    # force when the matrix was loaded, and re-deciding it against the config in
    # force **now** is what keeps a mismatched frame from reaching
    # ``calc_compensation_image`` as a numpy shape ``ValueError`` -- a 500
    # blaming the server for a configuration the user changed under a matrix id
    # that is still perfectly valid. ``reload_config`` is a public API and this
    # repository's ``config.toml`` has eight sibling variants in the working
    # tree right now.
    #
    # 2条件であって3条件ではない。``region is None`` は「カメラ座標である」
    # ことそのもの。``classify_space`` の方は別の問いで、読み込み時ではなく
    # **現在の** config との一致を見る。
    if region is None or classify_space(height, width, projector) == "camera":
        raise ViewNotApplicable(_CAMERA_SPACE_REFUSAL)

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
    # comparisons are strict (``< 0``, ``> 1``) and not ``<=`` / ``>=``. This
    # is not a detail. On real data the highlights sit pinned at exactly 1.0
    # -- ``calc_compensation_image``'s own clip produces that value by the
    # million -- so confusing the two misclassifies a large, contiguous and
    # entirely plausible-looking region as "clipped" when it is the one value
    # the projector can still reproduce exactly.
    # 境界は閉区間である。実データでは白飛び付近が 1.0 ちょうどに張り付く
    # ため、``<`` と ``<=`` の取り違えは広く連続した領域を丸ごと誤分類する。
    below = (linear < 0.0).any(axis=2)
    above = (linear > 1.0).any(axis=2)
    classes = np.where(
        below & above, 3.0, np.where(above, 2.0, np.where(below, 1.0, 0.0))
    )
    data = classes[:, :, None].astype(np.float32)

    # Outside the active region there is no answer, and it must not be class 0.
    #
    # **After the classification, never before.** NaN compares False against
    # everything, so a NaN pixel has neither ``< 0`` nor ``> 1`` true and falls
    # through to class 0 -- "in range" -- which is the most reassuring possible
    # wrong answer, painted over the three quarters of the frame that were
    # never projected onto. Masking here instead makes it NaN, which is what
    # ``coef_tiles``, ``surface`` and ``compensation`` all write for an
    # unanswerable pixel and what the hover readout's out-of-region marker
    # looks for.
    #
    # 有効領域の外は「答えが無い」であって「範囲内」ではない。マスクは分類の
    # **後**に置く。先にやると NaN の比較がすべて False になり、領域外が黙って
    # クラス0（範囲内）に落ちる。パイプライン自身は領域外に 0.0 を書くので、
    # 上書きしなければやはりクラス0になる。
    inside = target_view.active_region_mask(height, width, region)
    data[~inside] = np.nan

    # The class index range, not a percentile of the data.
    #
    # ``compensation`` and ``surface`` take ``nanpercentile`` because their
    # values are a measurement whose useful range is unknown in advance. Here
    # the range is known exactly and is a property of ``LABELS``, not of the
    # frame: a percentile would collapse to [0, 0] on a frame that happens to
    # be entirely in gamut -- the common case, and the one where a viewer most
    # needs the few out-of-gamut pixels to stand out rather than to become the
    # whole scale.
    #
    # Derived from ``LABELS`` rather than written as 3.0: adding a class means
    # adding a label, and the display range has to follow it in the same edit.
    # クラス番号の範囲は既知であり ``LABELS`` の性質である（フレームの性質では
    # ない）。パーセンタイルにすると全画素が範囲内のフレームで [0, 0] に潰れる。
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
