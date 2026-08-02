"""In-memory context stores (one process, thread-safe).

読み込み済みの行列・ターゲット画像をサーバのRAMに置き、不透明なIDで参照する。
線を渡るのは2次元のビュー結果だけで、(H, W, 4, 3) の行列そのものは決して
送らない。FastAPI に依存させないので単体でテストできる。

``MatrixStore`` と ``TargetStore`` は id 採番・ロック・``get``/``put``/
``clear`` が完全に同じ形なので、その共通部分を ``_IdStore`` に切り出して
両方から使う。id の接頭辞（``"matrix"``/``"target"``）だけが違う。
"""

from __future__ import annotations

import itertools
import threading
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from .matrix import MatrixContext
    from .target import TargetContext

_T = TypeVar("_T")


class _IdStore(Generic[_T]):
    """Thread-safe id -> context map, ids prefixed by *kind*.

    ``MatrixStore``/``TargetStore`` の共通実装。公開する振る舞い
    （id の形式・ロック・例外）はどちらの元実装とも変えていない。
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, _T] = {}
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def put(self, context: _T) -> str:
        """Store a context and return its new id."""
        with self._lock:
            item_id = f"{self._kind}{next(self._counter)}"
            self._items[item_id] = context
            return item_id

    def get(self, item_id: str) -> _T:
        """Return the context for ``item_id`` or raise ``KeyError``."""
        with self._lock:
            context = self._items.get(item_id)
        if context is None:
            raise KeyError(item_id)
        return context

    def clear(self) -> None:
        """Drop everything (used by tests)."""
        with self._lock:
            self._items.clear()


class MatrixStore(_IdStore["MatrixContext"]):
    """Thread-safe id -> MatrixContext map. Ids look like ``"matrix1"``."""

    def __init__(self) -> None:
        super().__init__("matrix")


class TargetStore(_IdStore["TargetContext"]):
    """Thread-safe id -> TargetContext map. Ids look like ``"target1"``."""

    def __init__(self) -> None:
        super().__init__("target")


store = MatrixStore()
targets = TargetStore()
