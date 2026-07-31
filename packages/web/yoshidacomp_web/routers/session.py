"""Matrix session endpoints: load a matrix file and report its metadata.

行列ファイルを読み込みIDを発行するエンドポイントと、そのメタ情報を返す
エンドポイント。行列そのもの(H, W, 4, 3)は決してレスポンスに乗せない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

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
