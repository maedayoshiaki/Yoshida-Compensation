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

# Below this |det(A)| the affine has no usable inverse. The value is a floor on
# float32 round-off, not a physical threshold.
# これ未満の |det(A)| では逆行列が使えない。物理的な閾値ではなく float32 の
# 丸め誤差に対する下限。
_DET_EPS = 1e-12


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
    invertible = np.abs(np.linalg.det(flat)) > _DET_EPS
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
    return ViewResult(data=data, vmin=0.0, vmax=max(vmax, 1e-6), labels=LABELS)


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
