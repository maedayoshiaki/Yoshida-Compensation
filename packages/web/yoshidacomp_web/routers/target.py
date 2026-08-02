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
from ..target import TargetContext, load_resolved, resolve_target

router = APIRouter()


def _build_meta(target_id: str, context: TargetContext) -> dict[str, Any]:
    """Build the JSON-serialisable metadata for a stored target."""
    height, width, channels = context.image.shape
    return {
        "id": target_id,
        "shape": [height, width, channels],
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
    # Resolved before it is read, so an image already in the store is answered
    # without decoding it a second time.
    #
    # This endpoint is not only the 読み込み button: the sidebar's directory
    # picker re-POSTs on every selection, so flipping through a directory used
    # to decode all of its images once per click and leave one more full-size
    # copy in the store each time -- unbounded in clicks, and O(N) decodes per
    # click (measured: a full pass over a 3-image directory called
    # ``cv2.imread`` 9 times and grew the store to 3). Now the identity says
    # whether this is a thing already loaded, so a pass costs one decode per
    # image and a second pass costs none at all. What is stored is bounded by
    # the images actually selected rather than by how often they were.
    #
    # The identity carries the selected file's size and mtime, so a file
    # replaced on disk is a different target and is read again -- serving a
    # stale picture from a cache would be a far worse failure than the leak
    # this replaces.
    #
    # 一覧から選び直すたびにこのエンドポイントが呼ばれる。以前は1クリックごとに
    # ディレクトリ内の全画像を復号し、店には毎回1枚ずつ積み上がっていた
    # （実測: 3枚を一巡して ``cv2.imread`` 9回、店は3件）。同一性で引き当てる
    # ことで、一巡は1枚1回の復号になり、2周目は復号ゼロになる。
    # 同一性にはサイズと更新時刻が入るので、差し替えられたファイルは別物として
    # 読み直される（古い絵をキャッシュから出すほうが害が大きい）。
    try:
        source = resolve_target(Path(request.path), index=request.index)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    loaded = targets.get_by_key(source.identity)
    if loaded is not None:
        return {"id": loaded, "meta": _build_meta(loaded, targets.get(loaded))}

    try:
        context = load_resolved(source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    target_id = targets.put(context, key=source.identity)
    return {"id": target_id, "meta": _build_meta(target_id, context)}


@router.get("/target/{target_id}/meta")
def get_target_meta(target_id: str) -> dict[str, Any]:
    """Return the stored metadata for a previously loaded target."""
    context = require_target(target_id)
    return _build_meta(target_id, context)
