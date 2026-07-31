"""The 12 affine coefficients laid out as the matrix itself.

行列の形（4行×3列）そのままに係数マップを並べるビュー。
"""

from __future__ import annotations

import numpy as np

from ..matrix import MatrixContext
from .base import ParamSpec, ViewResult, ViewSpec, register

_INPUT_LABELS = ("in_R", "in_G", "in_B", "bias")
_OUTPUT_LABELS = ("out_R", "out_G", "out_B")

# Channel order is row-major over (input, output): c = i * 3 + j.
# チャンネル順は (入力, 出力) の行優先。
LABELS = [f"{i} -> {j}" for i in _INPUT_LABELS for j in _OUTPUT_LABELS]


def _render(context: MatrixContext, params: dict) -> ViewResult:
    """Flatten (H, W, 4, 3) into (H, W, 12) and pick a symmetric range."""
    array = np.asarray(context.array, dtype=np.float32)
    height, width = array.shape[0], array.shape[1]
    data = array.reshape(height, width, 12).copy()

    if context.active_region is not None:
        x, y, w, h = context.active_region
        mask = np.zeros((height, width), dtype=bool)
        mask[y : y + h, x : x + w] = True
        # Outside the active region the fit is physically meaningless; NaN so
        # the shader paints it as a mask colour instead of a plausible value.
        # 有効領域外のfitは物理的に無意味なので NaN にしてシェーダでマスク色にする。
        data[~mask] = np.nan

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        vmax = 1.0
    elif params.get("normalize", "shared") == "per_tile":
        # Per-tile normalisation is applied client-side; report the global
        # range so the caller still has a sane fallback.
        vmax = float(np.percentile(np.abs(finite), 99))
    else:
        vmax = float(np.percentile(np.abs(finite), 99))
    vmax = max(vmax, 1e-6)

    # Symmetric range keeps zero at the white point of the diverging map.
    # 対称にすることで発散カラーマップの0が白に保たれる。
    return ViewResult(data=data, vmin=-vmax, vmax=vmax, labels=LABELS)


SPEC = register(
    ViewSpec(
        key="coef_tiles",
        label="12係数タイル",
        kind="scalar",
        channels=12,
        params=[
            ParamSpec(
                name="normalize",
                label="正規化",
                type="select",
                default="shared",
                choices=["shared", "per_tile"],
            )
        ],
        render=_render,
    )
)
