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
        type: "number" | "select" | "text" | "rgb".

            ``"text"`` is for a value whose valid set only exists at runtime --
            a stored ``target`` or ``lut`` id, which is minted by a POST after
            this registry was built. ``choices`` is a static ``list[str]`` and
            ``ViewSpec`` is frozen, so those cannot be offered as a
            ``"select"``: an empty ``choices`` renders an empty ``<select>`` in
            ``static/js/main.js::buildServerParams``, a control that can never
            be given a value at all. The view validates the string it receives
            and raises ``ViewNotApplicable`` for an id it cannot resolve, which
            is the same 400-with-a-reason the router already produces for a
            value outside a ``select``'s ``choices``.

            No JavaScript change was needed for it: ``buildServerParams``
            already renders every non-``"number"``, non-``"select"`` type as
            ``<input type="text">``, and ``routers/view.py`` already passes
            such a parameter through as the raw string.

            ``"text"`` は「有効な値の集合が実行時にしか存在しない」もの
            （``target``/``lut`` の id）のための型。``choices`` は静的な
            ``list[str]`` で ``ViewSpec`` は frozen なので、これらを
            ``"select"`` にすると空のドロップダウンになり値を入れる手段が
            無くなる。ビュー側が受け取った文字列を検証し、解決できない id は
            ``ViewNotApplicable``（router が 400 に写す）にする。
        default: Default value.
        choices: Options when ``type == "select"``.
    """

    name: str
    label: str
    type: Literal["number", "select", "text", "rgb"]
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
    """A registered view.

    Attributes:
        key, label, kind, channels, params, render: as used by
            ``routers/view.py`` and by the frontend's automatic controls.
        values_are_matrix: Whether the picture this view draws *is* the matrix.

            The readout along the bottom of the page always shows the hovered
            pixel's own 4x3 coefficients, fetched a pixel at a time, on every
            view. Beside it sits a second panel holding the *displayed view's*
            values at that pixel -- which for a view of the coefficients
            themselves is the same twelve numbers a second time, in a second
            format, twenty pixels away. Declaring it here lets the frontend drop
            that panel where it would only repeat the readout, and keep it where
            the picture is something else (``surface``'s three projected
            colours, say).

            Defaults to ``False``, which is the *showing* direction, and
            deliberately so: a view that does not answer this question is one
            whose values the readout has no reason to duplicate, and hiding them
            would silently blank the only numbers that view produces. Wrong in
            this direction costs a redundant panel; wrong in the other costs the
            data.

            このビューが描いているものが行列そのものかどうか。画面下の読み取り
            欄はどのビューでもホバー画素の4x3係数を出す。その隣の板は「表示中の
            ビューの値」であり、係数そのものを見ているビューでは同じ12個が別の
            書式で20px隣に並ぶだけになる。ここで宣言しておけば、重複する場合
            だけ板を落とし、``surface`` のように別のものを描いているビューでは
            残せる。既定は「出す」側の ``False`` にする。答えていないビューの値
            は重複と決めつけられず、消せばそのビューが出す唯一の数値が黙って
            消える。誤ったときの損失が小さいのはこちら。
    """

    key: str
    label: str
    kind: ViewKind
    channels: int
    params: list[ParamSpec]
    render: Callable[[MatrixContext, dict], ViewResult]
    values_are_matrix: bool = False


REGISTRY: dict[str, ViewSpec] = {}


def register(spec: ViewSpec) -> ViewSpec:
    """Add a view to the registry. Raises on a duplicate key."""
    if spec.key in REGISTRY:
        raise ValueError(f"duplicate view key: {spec.key}")
    REGISTRY[spec.key] = spec
    return spec
