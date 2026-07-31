"""Color mixing matrix loading and coordinate-space classification.

色変換行列の読み込みと座標空間の判定。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from yoshidacomp.compensation.projection_region import resolve_active_region
from yoshidacomp.config import ProjectorConfig

MatrixSpace = Literal["projector", "camera"]


@dataclass(frozen=True)
class MatrixContext:
    """A loaded matrix plus what the viewer needs to interpret it.

    Attributes:
        array: Per-pixel affine matrices, shape ``(H, W, 4, 3)``, float32.
            画素ごとのアフィン行列。
        space: Which coordinate space ``array`` lives in.
            行列が住んでいる座標空間。
        path: Where it was loaded from.
        active_region: ``(x, y, w, h)`` of the valid projection area, or None
            when the matrix is not in projector space.
            有効投影領域。プロジェクタ座標でない場合は None。
    """

    array: np.ndarray
    space: MatrixSpace
    path: Path
    active_region: tuple[int, int, int, int] | None


def classify_space(height: int, width: int, projector: ProjectorConfig) -> MatrixSpace:
    """Return "projector" when the matrix matches the configured frame.

    行列の縦横がプロジェクタのフレームと一致すればプロジェクタ座標とみなす。
    一致しなければカメラ座標として扱う。カメラ解像度は機種依存で固定値を
    持たないため、こちらを既定にする。
    """
    if width == projector.width and height == projector.height:
        return "projector"
    return "camera"


def load_matrix(path: Path, projector: ProjectorConfig) -> MatrixContext:
    """Load a ``(H, W, 4, 3)`` float matrix and classify its space.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: the array is not shaped ``(H, W, 4, 3)``.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    array = np.load(path, mmap_mode="r")
    if array.ndim != 4 or array.shape[2] != 4 or array.shape[3] != 3:
        raise ValueError(
            f"expected a matrix shaped (H, W, 4, 3), got {tuple(array.shape)}"
        )

    height, width = int(array.shape[0]), int(array.shape[1])
    space = classify_space(height, width, projector)
    region = resolve_active_region(projector) if space == "projector" else None
    return MatrixContext(array=array, space=space, path=path, active_region=region)
