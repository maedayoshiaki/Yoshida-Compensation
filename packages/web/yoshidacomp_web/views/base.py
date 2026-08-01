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
