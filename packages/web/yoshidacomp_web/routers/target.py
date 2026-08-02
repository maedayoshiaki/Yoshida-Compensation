"""Target image session endpoints: load an image (or a directory of them).

行列と同じ作法（``routers/session.py``）: パスを受け取りIDを発行する
エンドポイントと、そのメタ情報を返すエンドポイント。画素データ本体は
このタスクの応答には乗せない（ビューが必要になるのは計画B3 Task 3から）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..deps import require_target
from ..models import LoadTargetRequest
from ..store import targets
from ..target import TargetContext, load_target

router = APIRouter()


def _build_meta(target_id: str, context: TargetContext) -> dict[str, Any]:
    """Build the JSON-serialisable metadata for a stored target."""
    height, width = context.image.shape[0], context.image.shape[1]
    return {
        "id": target_id,
        "shape": [height, width, 3],
        "dtype": str(context.image.dtype),
        "path": str(context.path),
        "is_directory": context.is_directory,
        "images": list(context.names),
        "index": context.index,
    }


@router.post("/target")
def load_session_target(request: LoadTargetRequest) -> dict[str, Any]:
    """Load a target image (or a directory of them) from disk."""
    # ``Path("")`` is ``Path(".")`` -- a directory -- so an empty box would
    # otherwise be reported as "no supported images found in .", which
    # explains nothing. Same precedent as ``routers/session.py``.
    # ``Path("")`` は ``Path(".")`` になるため、空欄のまま送ると
    # 意味の無い説明になる。``routers/session.py`` と同じ扱い。
    if not request.path.strip():
        raise HTTPException(
            status_code=400,
            detail="path is empty; expected the path of an image file or a "
            "directory of images",
        )
    try:
        context = load_target(Path(request.path), index=request.index)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    target_id = targets.put(context)
    return {"id": target_id, "meta": _build_meta(target_id, context)}


@router.get("/target/{target_id}/meta")
def get_target_meta(target_id: str) -> dict[str, Any]:
    """Return the stored metadata for a previously loaded target."""
    context = require_target(target_id)
    return _build_meta(target_id, context)
