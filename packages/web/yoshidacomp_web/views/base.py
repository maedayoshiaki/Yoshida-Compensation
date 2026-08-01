"""View registry.

拡張の中心。ビューを1つ登録すればGUIへ現れ、``kind`` が標準的なものであれば
JavaScript の変更を要しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from ..matrix import MatrixContext

ViewKind = Literal["scalar", "rgb", "categorical"]


class ViewNotApplicable(Exception):
    """A view refuses the matrix it was handed, as policy rather than as a bug.

    ``routers/view.py`` maps this -- and only this -- to a 400 with the message
    intact. It exists so that "this input is not for me" is distinguishable
    from "I crashed".

    A bare ``ValueError`` cannot make that distinction. Views are numpy code,
    and numpy raises ``ValueError`` for a broadcast or reshape mismatch, i.e.
    for a *server bug*. Catching ``ValueError`` at the router turned every such
    bug into a 400 -- a client error, telling the user their request was wrong
    when it was not -- carrying a raw numpy message as the explanation. A
    dedicated type lets genuine bugs stay 500s, where a 500 is the truth.

    ビューが渡された行列を「自分の対象ではない」として断るための例外。
    ``routers/view.py`` はこれだけを400へ写す。素の ``ValueError`` では
    「対象外」と「壊れた」を区別できない。ビューの中身は numpy であり、
    ブロードキャストや reshape の不整合——すなわちサーバ側のバグ——も
    ``ValueError`` を投げる。それを400にすると、利用者の要求が誤っていた
    かのように伝えたうえで、説明として numpy の生のメッセージを渡すことに
    なる。専用の型にすれば、本物のバグは500のままでいられる。
    """


@dataclass(frozen=True)
class ParamSpec:
    """One GUI control, rendered automatically by the frontend.

    Attributes:
        name: Parameter key passed back as a query string.
        label: Human-readable label.
        type: "number" | "select" | "rgb".
        default: Default value.
        choices: Options when ``type == "select"``.
    """

    name: str
    label: str
    type: Literal["number", "select", "rgb"]
    default: object
    choices: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ViewResult:
    """What a view renders.

    Attributes:
        data: ``(H, W, C)`` float32 -- the natural shape on the Python side.
            ``routers/view.py`` moves it to planar ``(C, H, W)`` little-endian
            for the wire and says so in ``X-Layout``; a view never does that
            itself.
            Python 側は ``(H, W, C)`` のまま。ワイヤ上の planar ``(C, H, W)``
            への並べ替えは ``routers/view.py`` が一括で行う。
        vmin, vmax: Suggested display range (the frontend may override).
        labels: One label per channel, shown as tile captions.
    """

    data: np.ndarray
    vmin: float
    vmax: float
    labels: list[str]


@dataclass(frozen=True)
class ViewSpec:
    """A registered view."""

    key: str
    label: str
    kind: ViewKind
    channels: int
    params: list[ParamSpec]
    render: Callable[[MatrixContext, dict], ViewResult]


REGISTRY: dict[str, ViewSpec] = {}


def register(spec: ViewSpec) -> ViewSpec:
    """Add a view to the registry. Raises on a duplicate key."""
    if spec.key in REGISTRY:
        raise ValueError(f"duplicate view key: {spec.key}")
    REGISTRY[spec.key] = spec
    return spec
