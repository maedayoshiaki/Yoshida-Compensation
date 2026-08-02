"""View endpoints: the registry listing and the binary view payload."""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response

from ..deps import require_matrix
from ..views import REGISTRY, ViewNotApplicable

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
                # Whether the picture is the matrix itself. The frontend drops
                # its second readout panel where this is true, because there the
                # panel repeats the 4x3 table beside it. Not derivable from
                # ``kind``: ``coef_tiles`` and ``plain_tiles`` are both
                # ``scalar`` and so would be a residual-error view, which is not
                # the matrix at all.
                # 描いているものが行列そのものかどうか。真なら読み取り欄の
                # 2枚目の板は隣の表の繰り返しになるのでフロントが落とす。
                # ``kind`` からは導けない（残差ビューも ``scalar`` である）。
                "values_are_matrix": spec.values_are_matrix,
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

    # A view may refuse the matrix it was handed. ``surface`` does, for a
    # camera-space one: the work is one 3x3 solve per pixel and camera space is
    # ~16x the pixels of the projector frame, so it would be tens of seconds and
    # gigabytes for an answer whose active-region mask cannot even run (see
    # ``views/surface.py``). Refusing with a message beats both a bare 500 and a
    # thrashing machine -- ``static/js/api.js::fetchView`` reads the ``detail``
    # out of this body and shows it as the status line, so the user is told
    # which matrix to load instead.
    #
    # ``ViewNotApplicable`` only, deliberately not ``ValueError``. Views are
    # numpy code and numpy raises ``ValueError`` for a broadcast or reshape
    # mismatch, which is a server bug; mapping that to 400 would blame the
    # request and hand over a raw numpy message as the reason. A bug stays a
    # 500. See ``views/base.py::ViewNotApplicable``.
    #
    # ビューが行列そのものを断ることがある（``surface`` はカメラ座標を断る）。
    # 理由付きで断るほうが、素の500よりも、実行して機械を殴るよりもよい。
    # ``static/js/api.js::fetchView`` はこの本文から ``detail`` を取り出して
    # ステータス行に出すので、利用者は「どの行列を読めばよいか」を知れる。
    # ``ValueError`` ではなく ``ViewNotApplicable`` だけを写すのは、numpy が
    # 形状の不整合（＝サーバのバグ）にも ``ValueError`` を使うためである。
    # バグは500のままにする。
    try:
        result = spec.render(context, params)
    except ViewNotApplicable as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    if channels != spec.channels:
        raise HTTPException(
            status_code=500,
            detail=(
                f"view {key!r} declares channels={spec.channels} but rendered "
                f"{channels} with {len(result.labels)} label(s)"
            ),
        )

    # What ``labels`` counts is decided by ``kind``, and exactly one kind
    # differs.
    #
    # For ``scalar`` and ``rgb`` a label names a channel -- they are the tile
    # captions and the readout's column headings -- so there must be one per
    # channel. For ``categorical`` the single channel carries a class *index*
    # and the labels name the classes, so there are as many labels as classes
    # and ``len(labels) == channels`` would reject every such view on sight
    # (``views/gamut.py``: one channel, four class names). The class names have
    # nowhere else to travel: ``X-Labels-B64`` is the only string-carrying
    # header in this contract.
    #
    # The check kept for that kind is the one that still has teeth. A
    # categorical view's values are indices into ``labels``, so its display
    # range is not a choice either -- it is ``[0, len(labels) - 1]`` by
    # definition, and the frontend maps exactly that range onto its colours.
    # Adding a class and forgetting ``vmax`` (or the reverse) leaves every
    # class drawn in the wrong colour, with no error anywhere; this fails
    # instead. It is O(1) and does not touch the pixels.
    #
    # ``labels`` が何を数えているかは ``kind`` が決める。``scalar``/``rgb`` は
    # チャンネル名なのでチャンネル数と一致しなければならないが、
    # ``categorical`` の1チャンネルはクラス番号であり、ラベルはクラス名なので
    # 数はクラス数になる（``gamut`` は1チャンネル・4クラス）。クラス名を運べる
    # ヘッダは他に無い。代わりに固定するのは表示レンジで、クラス番号の
    # ビューではそれは選択の余地が無く ``[0, len(labels) - 1]`` である。
    # クラスを増やして ``vmax`` を忘れると、全クラスが誤った色で描かれたまま
    # どこにも異常が出ない。
    if spec.kind == "categorical":
        labels_ok = (result.vmin, result.vmax) == (0.0, float(len(result.labels) - 1))
        expected = (
            "a categorical view's labels name its classes, so vmin/vmax must "
            f"be (0.0, {float(len(result.labels) - 1)}) for "
            f"{len(result.labels)} label(s)"
        )
    else:
        labels_ok = len(result.labels) == channels
        expected = (
            f"a {spec.kind!r} view's labels name its channels, so it must "
            "render one label per channel"
        )
    if not labels_ok:
        raise HTTPException(
            status_code=500,
            detail=(
                f"view {key!r} rendered {len(result.labels)} label(s) with "
                f"vmin={result.vmin!r} vmax={result.vmax!r}: {expected}"
            ),
        )

    # Serialise as planar (C, H, W). The client uploads one channel per texture,
    # so a planar body lets it take a zero-copy subarray instead of walking the
    # array with a stride. Measured: the strided de-interleave was 93% of the
    # client's "GPU upload" time for a 95 MB payload.
    # planar (C, H, W) で直列化する。クライアントはチャンネルごとにテクスチャを
    # 作るので、planar ならストライド走査ではなく subarray のゼロコピーで済む。
    # 実測では 95MB のペイロードでストライド走査がアップロード時間の93%を占めた。
    #
    # ``np.moveaxis`` returns a view; ``ascontiguousarray`` performs the single
    # copy. The reorder happens once here rather than as twelve strided walks on
    # the client.
    # ``np.moveaxis`` は view を返し、実際の複製は ``ascontiguousarray`` が行う。
    #
    # This line is not free, and is in fact the most expensive operation in the
    # request: the transpose is ~130 ms of a ~590 ms 1080p load, against ~25 ms
    # for the old contiguous copy. Measured A/B (1920x1080x12 float32, 94.9 MB,
    # RTX 3060 Ti / D3D11, median of 4 warm runs):
    #
    #   uploadMs  165 -> 13 ms      fetchMs  480 -> 577 ms
    #   totalMs   651 -> 590 ms     jsHeap   381 -> 191 MB
    #
    # So the de-interleave did not disappear, it moved here; wall clock improved
    # by only 9%. What justifies it is the heap: the client no longer allocates
    # twelve scratch planes, which is 95 MB at 1080p and ~1.5 GB for the
    # camera-space matrix a later plan warps -- and that one already used 71% of
    # the JS heap before this change. Do not "optimise" this back to interleaved
    # on the strength of fetchMs alone.
    # この行はリクエスト中で最も高い（転置に約130ms、従来の連続コピーは約25ms）。
    # 実測は上記の通りで、総時間の改善は9%にとどまる。正当化しているのは
    # ヒープ側で、クライアントが12枚の作業用平面を確保しなくなる。1080pで95MB、
    # 後続計画が扱うカメラ座標行列では約1.5GBに相当する（同行列は本変更前で
    # JSヒープの71%を使っていた）。fetchMs だけを見て戻してはならない。
    body = np.ascontiguousarray(np.moveaxis(result.data, 2, 0), dtype="<f4").tobytes()
    headers = {
        "X-Height": str(height),
        "X-Width": str(width),
        "X-Channels": str(channels),
        "X-Layout": "planar",
        "X-Kind": spec.kind,
        "X-Vmin": repr(float(result.vmin)),
        "X-Vmax": repr(float(result.vmax)),
        "X-Labels-B64": encode_labels(result.labels),
    }
    return Response(
        content=body, media_type="application/octet-stream", headers=headers
    )
