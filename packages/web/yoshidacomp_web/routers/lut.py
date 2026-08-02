"""Inverse LUT endpoints: load a measured ``.npz`` and return its curves.

行列・ターゲットと同じ作法（``routers/session.py`` / ``routers/target.py``）:
パスを受け取りIDを発行するエンドポイントと、中身を返すエンドポイント。

LUTだけは中身そのものを JSON に載せる。3x256 の float は 768 個 -- 画面に
描くカーブの全点であり、行列 (H, W, 4, 3) のように「送ると計画B2で半減させた
ヒープが元に戻る」量ではない。点クエリ（``routers/session.py::probe_matrix``）
にしなかった理由もそこで、カーブは全点そろって初めて意味を持つ。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from yoshidacomp.config import get_config

from ..deps import require_lut
from ..lut import LutContext, load_lut
from ..models import LoadLutRequest
from ..store import luts

router = APIRouter()


def _build_meta(lut_id: str, context: LutContext) -> dict[str, Any]:
    """Build the JSON-serialisable metadata for a stored LUT.

    ``meta`` is the LUT file's *own* metadata -- the dict
    ``inverse_lut.save_inverse_lut`` wrote as a JSON string and
    ``load_inverse_lut`` parsed back (``source``, ``trust_low``,
    ``trust_high``, ...). It is nested rather than merged into this dict so a
    future key in the file cannot collide with, or silently overwrite, one of
    the endpoint's own.
    ``meta`` は LUT ファイル自身のメタデータ（``source``/``trust_low`` など）。
    平坦化して混ぜると、ファイル側にキーが増えたときエンドポイント側のキーを
    黙って上書きしうるので入れ子にする。
    """
    channels, samples = context.lut.lut.shape
    return {
        "id": lut_id,
        "path": str(context.path),
        "shape": [int(channels), int(samples)],
        "dtype": str(context.lut.lut.dtype),
        "meta": dict(context.lut.meta),
    }


@router.post("/lut")
def load_session_lut(request: LoadLutRequest) -> dict[str, Any]:
    """Load a measured inverse LUT from disk."""
    # ``Path("")`` is ``Path(".")`` -- a directory -- so an empty box would
    # otherwise be reported as "not a file: .", which explains nothing. Same
    # precedent as ``routers/session.py`` and ``routers/target.py``.
    # 空欄のまま送ると意味の無い説明になるので先に弾く。
    if not request.path.strip():
        raise HTTPException(
            status_code=400,
            detail="path is empty; expected the path of an inverse LUT .npz file",
        )
    try:
        context = load_lut(Path(request.path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    lut_id = luts.put(context)
    return {"id": lut_id, "meta": _build_meta(lut_id, context)}


@router.get("/lut/{lut_id}/curve")
def get_lut_curve(lut_id: str) -> dict[str, Any]:
    """Return the stored LUT's sample points, ready to plot.

    ``lut`` is ``[R, G, B]`` in that order -- the row order
    ``apply_inverse_lut`` indexes with ``lut.lut[channel]``, so the plot and
    the compensation agree about which curve is which channel.

    ``gamma`` is the *projector's configured* gamma, not the LUT's: it is what
    ``LINEARIZERS["gamma"]`` would use instead of this LUT, so ``t ** (1 /
    gamma)`` is the analytic curve the measured one replaces. Sent with the
    curve because the frontend has no other way to know it, and overlaying the
    wrong exponent would make a comparison that looks quantitative and is not.
    ``gamma`` は LUT のものではなく**プロジェクタ設定**の値である。
    ``LINEARIZERS["gamma"]`` がこのLUTの代わりに使う指数であり、
    ``t ** (1 / gamma)`` が「測定版が置き換えている解析曲線」そのものになる。
    フロントには他に知る手段が無く、違う指数を重ねれば定量的に見えて実は
    そうでない比較になるので、カーブと一緒に返す。

    The file's own path is **not** here. It was, and nothing read it: the
    frontend takes it from the ``POST /api/lut`` meta, which is the response
    that answers "what did I just load", and no test asserted it (task-4 final
    review, Minor 6). A field that is sent, documented and never read is a
    field the next reader has to decide about; it is easier to add back than to
    find out whether anything depended on it.
    ファイルのパスはここには入れない。以前は入っていたが誰も読まず、フロントは
    ``POST /api/lut`` の meta から取っている。
    """
    context = require_lut(lut_id)
    return {
        "t_grid": [float(v) for v in context.lut.t_grid],
        "lut": [[float(v) for v in row] for row in context.lut.lut],
        "meta": dict(context.lut.meta),
        "gamma": float(get_config().projector.gamma),
    }
