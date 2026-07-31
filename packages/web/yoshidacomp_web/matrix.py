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

    Every way this can fail is one of exactly two exception types, because the
    router maps exactly two (``routers/session.py``) and anything else reaches
    the user as a bare 500 whose plain-text body ``jsonOrThrow``
    (``static/js/api.js``) cannot even parse.
    失敗経路は必ずこの2種類のどちらかにする。それ以外はルータの対応表から
    漏れて素の500になり、フロントは理由を表示できない。

    Raises:
        FileNotFoundError: the path does not exist. 存在しないパス。
        ValueError: the path exists but cannot become a single
            ``(H, W, 4, 3)`` array -- a directory, an ``.npz`` archive rather
            than a bare ``.npy`` (this repository stores inverse LUTs that
            way, so it is a routine paste), an unreadable or corrupt file, or
            the right kind of file with the wrong shape. The message always
            names the path, since the router hands it straight to the user.
            存在するが単一の ``(H, W, 4, 3)`` 配列にならない場合。ディレクトリ、
            ``.npz``、破損ファイル、形状違いが該当。メッセージには必ずパスを含める。
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not path.is_file():
        raise ValueError(
            f"not a file: {path} -- expected a .npy holding one (H, W, 4, 3) float array"
        )

    try:
        loaded = np.load(path, mmap_mode="r")
    except Exception as exc:  # noqa: BLE001 - any read failure is user-facing input
        raise ValueError(f"could not read {path} as a .npy array: {exc}") from exc

    if not isinstance(loaded, np.ndarray):
        # np.load returns a lazy NpzFile for an archive; it has no ``.ndim``,
        # so without this the next line would AttributeError into a 500.
        # .npz は NpzFile として返り ``.ndim`` を持たないため、そのままだと
        # AttributeError から500になる。
        keys = list(getattr(loaded, "files", []))
        close = getattr(loaded, "close", None)
        if close is not None:
            close()
        raise ValueError(
            f"{path} is an .npz archive (contains {keys}), not a single .npy "
            "array; expected one (H, W, 4, 3) float array"
        )

    array = loaded
    if array.ndim != 4 or array.shape[2] != 4 or array.shape[3] != 3:
        raise ValueError(
            f"expected a matrix shaped (H, W, 4, 3), got {tuple(array.shape)} "
            f"from {path}"
        )

    height, width = int(array.shape[0]), int(array.shape[1])
    space = classify_space(height, width, projector)
    region = resolve_active_region(projector) if space == "projector" else None
    return MatrixContext(array=array, space=space, path=path, active_region=region)
