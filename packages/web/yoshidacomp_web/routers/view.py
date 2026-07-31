"""View endpoints: the registry listing and the binary view payload."""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response

from ..deps import require_matrix
from ..views import REGISTRY

router = APIRouter()


def encode_labels(labels: list[str]) -> str:
    """Encode channel labels as base64 of a UTF-8 JSON array.

    Starlette encodes header values as latin-1, so a Japanese label -- the
    planned ``gamut`` view's 「色域外」, say, and every UI string in this
    codebase is already Japanese -- raises on the way out and turns the whole
    response into a 500. Joining with commas fails differently and more
    quietly: a label that itself contains a comma splits into two on the
    client, so there are more labels than channels and every tile caption
    after it is wrong. base64 of UTF-8 JSON has neither problem and the list
    survives the round trip exactly as written, commas and all.

    ヘッダ値はlatin-1なので日本語ラベルは500になり、カンマ連結はラベル内の
    カンマで分割されてタイル名が静かにずれる。UTF-8 JSONのbase64なら
    どちらも起きず、ラベルはそのまま往復する。
    """
    payload = json.dumps(list(labels), ensure_ascii=False).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


@router.get("/views")
def list_views() -> dict[str, Any]:
    """Return every registered view and its parameter declarations.

    フロントはこれを読んでコントロールを自動生成する。
    """
    return {
        "views": [
            {
                "key": spec.key,
                "label": spec.label,
                "kind": spec.kind,
                "channels": spec.channels,
                "params": [
                    {
                        "name": p.name,
                        "label": p.label,
                        "type": p.type,
                        "default": p.default,
                        "choices": p.choices,
                    }
                    for p in spec.params
                ],
            }
            for spec in REGISTRY.values()
        ]
    }


@router.get("/matrix/{matrix_id}/view/{key}")
def render_view(matrix_id: str, key: str, request: Request) -> Response:
    """Render a view as a raw float32 binary body.

    Shape and display range travel in ``X-*`` headers, readable because the
    frontend is served same-origin. Pixel data is never JSON-encoded.
    形状と表示レンジは ``X-*`` ヘッダで返す。画素データはJSONに載せない。
    """
    context = require_matrix(matrix_id)
    spec = REGISTRY.get(key)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown view: {key}")

    params: dict[str, Any] = {p.name: p.default for p in spec.params}
    for p in spec.params:
        raw = request.query_params.get(p.name)
        if raw is None:
            continue
        if p.type == "number":
            try:
                params[p.name] = float(raw)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"{p.name} must be a number, got {raw!r}",
                ) from None
        elif p.type == "select":
            if p.choices and raw not in p.choices:
                raise HTTPException(
                    status_code=400,
                    detail=f"{p.name} must be one of {p.choices}, got {raw!r}",
                )
            params[p.name] = raw
        else:
            params[p.name] = raw

    result = spec.render(context, params)
    height, width = result.data.shape[0], result.data.shape[1]
    channels = result.data.shape[2] if result.data.ndim == 3 else 1

    # Checked before serialising: ``.tobytes()`` copies the whole array, which
    # is 1.5 GB for a camera-space matrix, and there is no point paying that
    # for a response that is about to fail.
    # 直列化の前に検査する。``.tobytes()`` は全体を複製するため、失敗する
    # 応答のために払う必要はない。
    #
    # ``/api/views`` advertises ``spec.channels`` and the frontend builds that
    # many tiles and readout cells from it, while ``X-Channels`` is derived
    # from the array. A view that declares 99 and renders 3 used to get a
    # 200 OK and a silently wrong grid; fail here instead.
    # 宣言値と実配列が食い違うと200のまま表示だけが壊れるので明示的に失敗させる。
    if channels != spec.channels or len(result.labels) != channels:
        raise HTTPException(
            status_code=500,
            detail=(
                f"view {key!r} declares channels={spec.channels} but rendered "
                f"{channels} with {len(result.labels)} label(s)"
            ),
        )

    body = np.ascontiguousarray(result.data, dtype="<f4").tobytes()
    headers = {
        "X-Height": str(height),
        "X-Width": str(width),
        "X-Channels": str(channels),
        "X-Kind": spec.kind,
        "X-Vmin": repr(float(result.vmin)),
        "X-Vmax": repr(float(result.vmax)),
        "X-Labels-B64": encode_labels(result.labels),
    }
    return Response(
        content=body, media_type="application/octet-stream", headers=headers
    )
