"""The forward model: what the camera sees for a given projector command.

順方向モデル。行列 M は [C_r, C_g, C_b, 1] @ M = P を満たすので、
M = [A; b]（A は 3x3、b は 1x3）と書けば C·A + b = P。したがって
任意の投影コマンド P に対するカメラ応答は C = (P - b)·A^-1 である。

P = (1,1,1) が投影面の色、P = (0,0,0) が環境光のみの応答になる。
"""

from __future__ import annotations

import numpy as np

from ..matrix import MatrixContext
from .base import ParamSpec, ViewResult, ViewSpec, register

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


def _render(context: MatrixContext, params: dict) -> ViewResult:
    """Solve C = (P - b) A^-1 per pixel, NaN where A is singular."""
    array = np.asarray(context.array, dtype=np.float32)
    height, width = array.shape[0], array.shape[1]

    a_mat = np.asarray(array[:, :, :3, :], dtype=np.float64)
    b_vec = np.asarray(array[:, :, 3, :], dtype=np.float64)

    command = np.array(
        [
            float(params.get("p_r", 1.0)),
            float(params.get("p_g", 1.0)),
            float(params.get("p_b", 1.0)),
        ],
        dtype=np.float64,
    )

    flat = a_mat.reshape(-1, 3, 3)
    det = np.linalg.det(flat)
    frobenius_norm = np.linalg.norm(flat, ord="fro", axis=(1, 2))
    invertible = np.abs(det) > _DET_REL_EPS * frobenius_norm**3
    inverse = np.zeros_like(flat)
    if invertible.any():
        inverse[invertible] = np.linalg.inv(flat[invertible])
    inverse = inverse.reshape(height, width, 3, 3)

    response = np.einsum("hwk,hwkc->hwc", command[None, None, :] - b_vec, inverse)

    # A singular pixel has no answer. Painting a plausible number there would
    # mislead; NaN makes the shader draw the mask colour instead.
    # 特異な画素には答えが無い。もっともらしい数字を描くと誤解を招くので
    # NaN にしてシェーダにマスク色を塗らせる。
    usable = invertible.reshape(height, width)
    if context.active_region is not None:
        x, y, w, h = context.active_region
        inside = np.zeros((height, width), dtype=bool)
        inside[y : y + h, x : x + w] = True
        usable &= inside
    response[~usable] = np.nan

    data = np.ascontiguousarray(response, dtype=np.float32)
    finite = data[np.isfinite(data)]
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0

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
    # So: keep 0 when the data is non-negative, and drop to the data's own 1st
    # percentile when it is not. The percentile mirrors the 99th used for vmax
    # and is robust to one wild pixel in the same way a bare min() would not be.
    # Note this also keeps the span strictly positive: vmin <= 0 < 1e-6 <= vmax.
    #
    # 黒レベルは可能な限り0に固定する。rgb経路は [vmin, vmax] を [0,1] へ写す
    # ため vmin が黒レベルそのものであり、3チャンネル共通のレンジをずらすと
    # 明るさではなく彩度が変わる（「投影面が何色か」を問うビューで色相が狂う）。
    # 一方で P=(0,0,0) の応答 -b·A^-1 は環境光があれば負であり、0固定では
    # 一様な黒になって「データ無し」とも「ちょうど0」とも区別がつかない。
    # よって負値があるときだけ1パーセンタイル（vmax の99の鏡像で、外れ値1画素に
    # 引きずられない）まで下げる。これで span も必ず正になる。
    vmin = min(0.0, float(np.percentile(finite, 1))) if finite.size else 0.0

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
