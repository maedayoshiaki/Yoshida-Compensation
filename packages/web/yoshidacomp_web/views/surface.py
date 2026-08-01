"""The forward model: what the camera sees for a given projector command.

順方向モデル。行列 M は [C_r, C_g, C_b, 1] @ M = P を満たすので、
M = [A; b]（A は 3x3、b は 1x3）と書けば C·A + b = P。したがって
任意の投影コマンド P に対するカメラ応答は C = (P - b)·A^-1 である。

実装は A^-1 を作らず、A^T x = (P - b)^T をクラメルの公式で直接解く。
逆行列は画素ごとに1本のベクトルとしか縮約しないため9要素を持つ必要が無く、
実測で2.8倍速・メモリ4割減になる（詳細は _render 内のコメント）。

P = (1,1,1) が投影面の色、P = (0,0,0) が環境光のみの応答になる。

このビューはプロジェクタ座標の行列のみを受け付ける。理由は
_CAMERA_SPACE_REFUSAL のコメントを参照。
"""

from __future__ import annotations

import numpy as np

from ..matrix import MatrixContext
from .base import ParamSpec, ViewNotApplicable, ViewResult, ViewSpec, register

LABELS = ["R", "G", "B"]

# Below this ratio of |det(A)| to ||A||_F**3, the affine has no usable
# inverse. This used to be a fixed |det(A)| > 1e-12 test, which is not
# scale-invariant: det scales as the cube of A's entries, so the same
# physical degeneracy can sit on either side of a fixed cutoff depending on
# the units the matrix happens to be stored in. That was a real bug, not a
# hypothetical one -- a pixel with one input channel attenuated to 1e-8 of
# normal produced |det(A)| ~= 1.6e-9, a thousand times above 1e-12, and was
# treated as invertible, painting a plausible-looking but meaningless answer.
#
# Comparing |det(A)| against a small multiple of ||A||_F**3 fixes this: both
# sides scale as s**3 under A -> sA, so the test means the same thing
# regardless of the matrix's units. 1e-6 sits comfortably above float32's
# ~1.19e-7 relative precision floor (with margin, since A is cast to float64
# before this comparison but originates as float32), while every
# well-conditioned matrix seen in this codebase's real data -- including a
# plain 2*I, where the ratio is ~0.19 -- sits many orders of magnitude above
# it.
#
# 以前は固定の |det(A)| > 1e-12 という基準だったが、これはスケール不変では
# ない。行列式は成分の3乗でスケールするため、同じ物理的な特異性でも行列の
# 単位次第でしきい値のどちら側にも転びうる。これは仮定の話ではなく実際の
# 不具合で、ある入力チャンネルが正常の1e-8まで減衰した画素は
# |det(A)| ~= 1.6e-9 となり、1e-12 の千倍でも「正則」として通り、もっとも
# らしいが無意味な値を描いていた。
#
# |det(A)| を ||A||_F**3 の定数倍と比較すればこれは解消する。A -> sA で
# 両辺とも s**3 倍されるため、行列の単位によらず同じ意味を持つ。1e-6 は
# float32 の相対精度限界（約1.19e-7、A はここに来る前に float64 化される
# が元は float32 由来）に十分な余裕を持たせた値で、このコードベースの実データ
# にある正則な画素（例えば単純な 2*I ではこの比は約0.19）はこれより何桁も
# 大きい。
_DET_REL_EPS = 1e-6


# The refusal for a camera-space matrix, in one place so the router's 400 and
# this module's docstring cannot drift apart.
#
# This is a cost guard *and* a correctness one. Cost: the work is one 3x3
# solve per pixel and is set by the pixel count alone -- measured at 738 ms and
# 390 MB peak for the 1080p projector-space matrix (2.07M pixels), which
# extrapolates by a factor of 15.7 to ~12 s and ~6 GB for
# ``data/color_mixing_matrices.npy`` (4660x6984 = 32.5M pixels, listed in spec
# section 9 as a first-class input and offered ``surface`` by the view
# selector). Correctness: ``active_region`` is
# ``None`` outside projector space, so the NaN mask that keeps this view honest
# does not run at all, and camera pixels that never saw the projector would be
# painted a "surface colour" derived from an affine fitted to nothing.
#
# コスト面と正しさの両方の理由で断る。コストは画素数だけで決まり、1080p
# （207万画素）で 738 ms・ピーク 390 MB の実測から、カメラ座標行列
# （4660x6984 = 3250万画素、仕様書9節が第一級の入力として挙げる）へ外挿すると
# 15.7倍で約12秒・約6GBになる。正しさの面では、プロジェクタ座標でなければ
# ``active_region`` が ``None`` になり、このビューを正直に保っている NaN
# マスクが一切走らない。投影を受けていないカメラ画素にまで「投影面の色」を
# 描いてしまう。
_CAMERA_SPACE_REFUSAL = (
    'view "surface" needs a projector-space matrix; this one is in camera '
    "space. Warping a camera-space matrix into projector space is planned for "
    "B4 -- until then, load a matrix whose frame matches the configured "
    "projector. "
    "（surface はプロジェクタ座標の行列を必要とします。カメラ座標からの"
    "ワープは計画B4の予定です。）"
)


def _render(context: MatrixContext, params: dict) -> ViewResult:
    """Solve C·A = P - b per pixel by Cramer's rule, NaN where A is singular.

    Raises:
        ViewNotApplicable: the matrix is in camera space. ``routers/view.py``
            maps this to a 400 with the message intact; see
            ``_CAMERA_SPACE_REFUSAL``. Deliberately not a ``ValueError``, which
            numpy also raises for shape mistakes inside this function -- those
            are bugs and must stay 500s.
    """
    if context.space == "camera":
        raise ViewNotApplicable(_CAMERA_SPACE_REFUSAL)

    array = np.asarray(context.array, dtype=np.float32)
    height, width = int(array.shape[0]), int(array.shape[1])

    # A's nine entries, each as its own contiguous float64 plane.
    #
    # Not ``a_mat.reshape(-1, 3, 3)`` any more, and not ``np.linalg.inv``.
    # Because the inverse is only ever contracted with a single vector per
    # pixel, Cramer's rule on ``A**T x = (P - b)**T`` gives the same answer
    # without ever materialising the nine-element inverse, and it is both
    # faster and smaller. Measured A/B on the real 1080p matrix
    # (``data/linear_pipeline_1080/matrices_linear_gamma.npy``, 1920x1080,
    # 2.07M pixels, 4 runs after a warm-up), whole ``_render`` either way:
    #
    #   np.linalg.inv form   2091-2154 ms   650.6 MB peak
    #   this form             732- 755 ms   389.6 MB peak
    #
    # -- 2.8x faster and 40% less memory, for a 24 MB response.
    #
    # Agreement, stated precisely, because the loose version invites the wrong
    # conclusion. The two algorithms genuinely differ in float64 -- they sum the
    # same products in a different order -- at about 1e-15 relative. The float32
    # cast then discards 29 of those bits, so a difference survives only when a
    # value happens to straddle a rounding boundary, which is rare enough that
    # the modal outcome is zero differing floats. Measured over 6,220,800
    # floats per command:
    #
    #   P = (1,1,1)        0 differ        P = (0,0,0)          0 differ
    #   P = (0.9,0.1,0.5)  0 differ        P = (0.2,0.6,0.9)    1 differs, by 1 ULP
    #
    # So: **identical at the benchmarked command, and never worse than one ULP
    # on one pixel** -- not "bit-identical" as a property of the change. The NaN
    # masks are equal at every command. The ``vmin``/``vmax`` header hints move
    # by up to 1.88e-8 relative (largest seen at P = (0,0,0)), because
    # ``np.nanpercentile`` interpolates in float64 where ``np.percentile`` over
    # a float32 copy did so in float32; that is well under this view's own
    # float32 spacing at the values involved.
    #
    # The ``np.linalg.inv`` form allocated four full-size arrays -- the float64
    # ``a_mat``, ``zeros_like``, the ``flat[invertible]`` boolean-index copy and
    # ``inv``'s result, 149 MB each at 1080p.
    #
    # 逆行列は画素ごとに1本のベクトルとしか縮約しないので、``A**T x = (P-b)**T``
    # をクラメルの公式で直接解けば9要素の逆行列を作らずに済む。1080p の実行列で
    # A/B を取り、``np.linalg.inv`` 版が 2091-2154 ms・ピーク650.6MB、こちらが
    # 732-755 ms・389.6MB（2.8倍速・メモリ4割減）。
    #
    # 一致の程度は正確に書く。両者は float64 では実際に食い違う（積の加算順が
    # 違うため、相対 1e-15 程度）。float32 化でそのうち29ビットが落ちるので、
    # 差が残るのは値が丸め境界をまたぐ場合だけであり、最頻値は「差ゼロ」である。
    # 622万要素あたりの実測は上表のとおりで、**基準コマンドでは完全一致、
    # 最悪でも1画素が1ULP** である（「この変更はビット一致する」ではない）。
    # NaN マスクはどのコマンドでも一致する。ヘッダの vmin/vmax は相対 1.88e-8
    # まで動く（P=(0,0,0) で最大）。``np.nanpercentile`` が float64 で補間する
    # のに対し、旧実装は float32 の複製上で補間していたためで、該当値域における
    # float32 の刻みより十分小さい。
    a_mat = array[:, :, :3, :]

    def plane(row: int, column: int) -> np.ndarray:
        return np.asarray(a_mat[:, :, row, column], dtype=np.float64)

    a00, a01, a02 = plane(0, 0), plane(0, 1), plane(0, 2)
    a10, a11, a12 = plane(1, 0), plane(1, 1), plane(1, 2)
    a20, a21, a22 = plane(2, 0), plane(2, 1), plane(2, 2)

    # ``||A||_F ** 3`` without the square root: ``(sum of squares) ** 1.5``.
    frobenius_cubed = (
        a00 * a00 + a01 * a01 + a02 * a02
        + a10 * a10 + a11 * a11 + a12 * a12
        + a20 * a20 + a21 * a21 + a22 * a22
    ) ** 1.5

    # The first row of cofactors serves twice: once for ``det`` (expansion
    # along A's first row) and once for the R component of the answer.
    # 1行目の余因子は2度使う（det の第1行展開と、答えのR成分）。
    cof_r0 = a11 * a22 - a12 * a21
    cof_r1 = a12 * a20 - a10 * a22
    cof_r2 = a10 * a21 - a11 * a20
    det = a00 * cof_r0 + a01 * cof_r1 + a02 * cof_r2
    invertible = np.abs(det) > _DET_REL_EPS * frobenius_cubed

    def command_minus_bias(index: int, name: str) -> np.ndarray:
        """``P_i - b_i`` as one plane, subtracted in place to avoid a copy.

        ``.astype`` rather than ``np.asarray``: it copies unconditionally, so
        the in-place subtraction below is guaranteed to be writing into our own
        buffer. ``np.asarray`` can return a *view*, and a view here would
        overwrite the loaded matrix's bias row -- silently, and for every later
        request against the same ``matrix_id``.
        ``np.asarray`` ではなく ``.astype`` を使う。無条件に複製するため、
        続くインプレース減算が自分のバッファへの書き込みであることが保証される。
        ``np.asarray`` はビューを返しうるが、ここでビューを掴むと読み込み済み
        行列の bias 行を破壊し、同じ ``matrix_id`` への以降の要求すべてに残る。
        """
        bias = array[:, :, 3, index].astype(np.float64)
        return np.subtract(float(params.get(name, 1.0)), bias, out=bias)

    q_r = command_minus_bias(0, "p_r")
    q_g = command_minus_bias(1, "p_g")
    q_b = command_minus_bias(2, "p_b")

    # ``det`` doubles as the divisor. A singular pixel's quotient is thrown away
    # by the NaN mask below, so putting 1 there only keeps the division from
    # producing a warning and an inf on the way to being discarded.
    # ``det`` をそのまま除数に使う。特異な画素の商は下の NaN マスクで捨てられる
    # ので、1 を置くのは警告と inf を出さないためだけである。
    det[~invertible] = 1.0

    def component(cof0: np.ndarray, cof1: np.ndarray, cof2: np.ndarray) -> np.ndarray:
        """``(P - b) · (column of adj(A)) / det`` -- one channel of C."""
        return (q_r * cof0 + q_g * cof1 + q_b * cof2) / det

    # Written straight into the float32 array that becomes the body: the old
    # float64 ``response`` plus its ``ascontiguousarray`` copy were another
    # 75 MB at 1080p for no gain.
    # 本体になる float32 配列へ直接書く。
    data = np.empty((height, width, 3), dtype=np.float32)
    data[:, :, 0] = component(cof_r0, cof_r1, cof_r2)
    data[:, :, 1] = component(
        a02 * a21 - a01 * a22, a00 * a22 - a02 * a20, a01 * a20 - a00 * a21
    )
    data[:, :, 2] = component(
        a01 * a12 - a02 * a11, a02 * a10 - a00 * a12, a00 * a11 - a01 * a10
    )

    # A singular pixel has no answer. Painting a plausible number there would
    # mislead; NaN makes the shader draw the mask colour instead.
    # 特異な画素には答えが無い。もっともらしい数字を描くと誤解を招くので
    # NaN にしてシェーダにマスク色を塗らせる。
    usable = invertible
    if context.active_region is not None:
        x, y, w, h = context.active_region
        inside = np.zeros((height, width), dtype=bool)
        inside[y : y + h, x : x + w] = True
        usable = usable & inside
    data[~usable] = np.nan

    # ``np.nanpercentile`` over the whole array, rather than
    # ``np.percentile`` twice over a ``data[np.isfinite(data)]`` copy: one pass
    # and one partition instead of one boolean-index copy and two partitions.
    # Measured on the 1080p response, 77.6 ms -> 40.5 ms.
    # 一度の走査で1と99を取る。実測で 77.6 ms -> 40.5 ms。
    if usable.any():
        low, high = (float(v) for v in np.nanpercentile(data, [1, 99]))
    else:
        low, high = 0.0, 1.0
    vmax = high

    # The black point is 0 whenever the data allows it, and only then.
    #
    # The frontend's rgb branch maps [vmin, vmax] onto [0, 1] and tone-maps
    # what comes out, so vmin *is* the black level of the picture. Because all
    # three channels share one range, moving vmin off zero subtracts the same
    # constant from R, G and B -- which is not a brightness change but a
    # desaturation. This view's question is "what colour is the projection
    # surface", so a floating floor would answer it with the wrong hue.
    #
    # But the answer is legitimately negative: P = (0,0,0) asks "what does the
    # camera see with the projector dark" and yields -b·A^-1, negative wherever
    # there is ambient light. Pinned at 0, every one of those clips to the
    # bottom and the active region renders as one flat black frame --
    # indistinguishable from "no data" and from "exactly zero".
    #
    # So: keep 0 unless the 1st *percentile* is itself below it, and drop to
    # that percentile when it is. The percentile mirrors the 99th used for vmax
    # and is robust to one wild pixel in the same way a bare min() would not be.
    #
    # Read that literally, because it is not the same as "drop whenever any
    # value is negative": a response whose negative values are a sub-1% tail
    # keeps vmin = 0 and clips them. That is the intended trade. A handful of
    # pixels is not the ambient-light regime this exists for, and letting them
    # move the black point would hand the whole picture a hue it does not have
    # on the strength of a rounding artefact.
    #
    # Note this also keeps the span strictly positive: vmin <= 0 < 1e-6 <= vmax.
    #
    # 黒レベルは可能な限り0に固定する。rgb経路は [vmin, vmax] を [0,1] へ写す
    # ため vmin が黒レベルそのものであり、3チャンネル共通のレンジをずらすと
    # 明るさではなく彩度が変わる（「投影面が何色か」を問うビューで色相が狂う）。
    # 一方で P=(0,0,0) の応答 -b·A^-1 は環境光があれば負であり、0固定では
    # 一様な黒になって「データ無し」とも「ちょうど0」とも区別がつかない。
    # よって1パーセンタイル（vmax の99の鏡像で、外れ値1画素に引きずられない）が
    # 負のときだけそこまで下げる。「負値があれば下げる」ではない点に注意。
    # 負値が全体の1%未満しかなければ vmin は0のままクリップされる。少数の画素で
    # 黒レベルを動かし、丸め誤差程度のもので画面全体の色相を変えないためである。
    # これで span も必ず正になる。
    vmin = min(0.0, low)

    return ViewResult(data=data, vmin=vmin, vmax=max(vmax, 1e-6), labels=LABELS)


def _command_param(name: str, label: str) -> ParamSpec:
    return ParamSpec(name=name, label=label, type="number", default=1.0)


SPEC = register(
    ViewSpec(
        key="surface",
        label="投影面の色",
        kind="rgb",
        channels=3,
        params=[
            _command_param("p_r", "投影 R"),
            _command_param("p_g", "投影 G"),
            _command_param("p_b", "投影 B"),
        ],
        render=_render,
    )
)
