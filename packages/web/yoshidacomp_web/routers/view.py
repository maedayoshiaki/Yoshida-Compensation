"""View endpoints: the registry listing and the binary view payload."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response

from ..deps import require_matrix
from ..views import REGISTRY

router = APIRouter()


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
            params[p.name] = float(raw)
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
    body = np.ascontiguousarray(result.data, dtype="<f4").tobytes()
    height, width = result.data.shape[0], result.data.shape[1]
    channels = result.data.shape[2] if result.data.ndim == 3 else 1
    headers = {
        "X-Height": str(height),
        "X-Width": str(width),
        "X-Channels": str(channels),
        "X-Kind": spec.kind,
        "X-Vmin": repr(float(result.vmin)),
        "X-Vmax": repr(float(result.vmax)),
        "X-Labels": ",".join(result.labels),
    }
    return Response(
        content=body, media_type="application/octet-stream", headers=headers
    )
