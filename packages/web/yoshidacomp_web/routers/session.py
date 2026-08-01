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

    Non-finite coefficients must leave this endpoint as ``null``. Two
    framework layers happen to agree today -- Pydantic's ``TypeAdapter``
    (reached because this route has a return annotation) maps NaN/Inf to
    ``null`` by default, and Starlette's ``JSONResponse`` uses
    ``json.dumps(allow_nan=False)`` so a NaN that got past it would 500
    rather than emit an invalid literal, never a malformed body reaching
    ``JSON.parse``. Neither is this endpoint's own contract, and both
    disappear if the route ever returns a ``Response`` directly or loses its
    return annotation; then a NaN coefficient turns the hover into an opaque
    500 instead of a readable masked value. Converting here makes ``null``
    ours.
    ``tests/test_web_matrix_probe.py::test_finite_or_none_maps_nan_and_infinities_to_none_directly``
    is what holds this; no HTTP-level test can, because the framework
    launders it either way.

    This is a contract for matrices that *do* carry NaN, not a description of
    the ones this viewer has been pointed at. Measured on
    ``data/linear_pipeline_1080/matrices_linear_gamma.npy`` (1080x1920x4x3):
    **zero** non-finite values, while 74.86% of pixels are outside the active
    region -- every one of them finite. The NaN in that frame is written by
    ``views/coef_tiles.py`` at *render* time and never reaches the stored
    array, which is exactly why the readout needs an out-of-region marker of
    its own rather than letting NaN speak for it. But nothing in ``.npy``
    forbids NaN and other pipelines in this repository produce masked arrays,
    so a matrix that holds one must be answerable rather than a 500.

    非有限係数は必ず ``null`` としてこのエンドポイントを出る。今日は2つの
    フレームワーク層がたまたま同じ結果を作っている――戻り値注釈つきの
    ルートなので Pydantic の ``TypeAdapter`` が既定で NaN/Inf を ``null`` にし、
    それをすり抜けても Starlette の ``JSONResponse`` は
    ``json.dumps(allow_nan=False)`` を使うため不正なリテラルではなく500になる
    （壊れた本文が ``JSON.parse`` に届くことはない）。しかしどちらもこの
    エンドポイント自身の契約ではなく、``Response`` を直接返す・戻り値注釈を
    外すといった変更で消える。そうなればNaNの係数は読み取り可能なマスク値では
    なく素の500になる。ここで変換することで ``null`` をこのエンドポイント自身の
    契約にする。

    これは「NaNを持ちうる行列」に対する契約であって、このビューアが実際に
    読んでいる行列の説明ではない。実測（``data/linear_pipeline_1080/
    matrices_linear_gamma.npy``、1080x1920x4x3）: 非有限値は**0個**。
    有効領域の外は全画素の74.86%あるが、そのすべてが有限である。あの画面の
    NaNは ``views/coef_tiles.py`` が**描画時**に書いているもので、保存された
    配列には入っていない――読み取り欄が独自の「領域外」の印を必要とするのは
    まさにこのためである。とはいえ ``.npy`` がNaNを禁じているわけではなく、
    このリポジトリの他の経路はマスク済み配列を作るので、NaNを持つ行列も
    500ではなく答えられなければならない。
    """
    v = float(value)
    return v if math.isfinite(v) else None


def _inside_active_region(context: MatrixContext, row: int, col: int) -> bool:
    """Return whether ``(row, col)`` is unmasked by ``context.active_region``.

    This answers "is this pixel excluded from the NaN mask that
    ``coef_tiles.py``/``surface.py`` apply", not literally "is there an
    active region and is this point inside it". When ``active_region`` is
    ``None`` (a camera-space matrix, which has no active-region concept at
    all) this returns ``True`` for every pixel -- callers must not read a
    bare ``True`` as "yes, there is a region and you are in it".

    これが答えているのは「``coef_tiles.py``/``surface.py`` が適用するNaNマスク
    から、この画素が除外されているか」であり、文字通りの「有効領域が存在し、
    かつその内側か」ではない。``active_region`` が ``None``（カメラ座標の行列。
    有効領域という概念自体を持たない）のときは全画素で ``True`` を返す。
    呼び出し側はこの ``True`` を「領域が存在してその内側にいる」と読んではならない。
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
