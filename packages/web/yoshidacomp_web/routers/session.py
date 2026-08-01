"""Matrix session endpoints: load a matrix file and report its metadata.

行列ファイルを読み込みIDを発行するエンドポイントと、そのメタ情報を返す
エンドポイント。行列そのもの(H, W, 4, 3)は決してレスポンスに乗せない。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from yoshidacomp.config import get_config

from ..deps import require_matrix
from ..matrix import MatrixContext, load_matrix
from ..models import LoadMatrixRequest
from ..store import store

router = APIRouter()


def _build_meta(matrix_id: str, context: MatrixContext) -> dict[str, Any]:
    """Build the JSON-serialisable metadata for a stored matrix."""
    projector = get_config().projector
    height, width = context.array.shape[0], context.array.shape[1]
    return {
        "id": matrix_id,
        "shape": [height, width, 4, 3],
        "dtype": str(context.array.dtype),
        "space": context.space,
        "path": str(context.path),
        "active_region": list(context.active_region) if context.active_region else None,
        "projector": {"width": projector.width, "height": projector.height},
    }


@router.post("/session/matrix")
def load_session_matrix(request: LoadMatrixRequest) -> dict[str, Any]:
    """Load a matrix file from disk and register it in the store."""
    projector = get_config().projector
    # ``Path("")`` is ``Path(".")`` -- a directory -- so an empty box would
    # otherwise be reported as "not a file: .", which explains nothing.
    # ``Path("")`` は ``Path(".")`` になるため、空欄のまま送ると
    # 「. はファイルではない」という無意味な説明になる。ここで先に弾く。
    if not request.path.strip():
        raise HTTPException(
            status_code=400,
            detail="path is empty; expected the path of a .npy matrix file",
        )
    try:
        context = load_matrix(Path(request.path), projector)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    matrix_id = store.put(context)
    return {"id": matrix_id, "meta": _build_meta(matrix_id, context)}


@router.get("/matrix/{matrix_id}/meta")
def get_matrix_meta(matrix_id: str) -> dict[str, Any]:
    """Return the stored metadata for a previously loaded matrix."""
    context = require_matrix(matrix_id)
    return _build_meta(matrix_id, context)


def _finite_or_none(value: float) -> float | None:
    """Return ``value`` as a float, or ``None`` when it is not finite.

    Python の ``json`` は NaN と Infinity を JSON として不正なリテラルで
    書き出し、ブラウザの ``JSON.parse`` はそれを受け取れない。行列は有効投影
    領域の外に NaN を持ちうるので、線に乗せる前に ``null`` へ落とす。
    """
    v = float(value)
    return v if math.isfinite(v) else None


def _inside_active_region(context: MatrixContext, row: int, col: int) -> bool:
    """Return whether ``(row, col)`` falls inside ``context.active_region``.

    ``active_region`` が ``None``（プロジェクタ座標でない行列）なら常に True。
    """
    if context.active_region is None:
        return True
    x, y, w, h = context.active_region
    return x <= col < x + w and y <= row < y + h


@router.get("/matrix/{matrix_id}/probe")
def probe_matrix(
    matrix_id: str,
    row: int = Query(..., description="行（0 起点、画像の上から）"),
    col: int = Query(..., description="列（0 起点、画像の左から）"),
) -> dict[str, Any]:
    """Return the 4x3 coefficients at one pixel.

    ビューが何であっても行列そのものを読めるようにするための点クエリ。
    ``surface`` のような非係数ビューではクライアントに行列が無く、
    99MB の2つ目のコピーを送るとヒープが計画B2以前に戻ってしまう。
    mmap からの1画素読みなので転送量は12個の float に収まる。
    """
    context = require_matrix(matrix_id)
    height, width = context.array.shape[0], context.array.shape[1]
    if not (0 <= row < height and 0 <= col < width):
        raise HTTPException(
            status_code=400,
            detail=(
                f"pixel (row={row}, col={col}) is outside the matrix; "
                f"expected 0 <= row < {height} and 0 <= col < {width}"
            ),
        )
    block = context.array[row, col]
    return {
        "row": row,
        "col": col,
        "values": [[_finite_or_none(v) for v in coefficient_row] for coefficient_row in block],
        "inside_active_region": _inside_active_region(context, row, col),
    }
