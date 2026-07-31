"""In-memory matrix store (one process, thread-safe).

読み込み済みの行列をサーバのRAMに置き、不透明なIDで参照する。線を渡るのは
2次元のビュー結果だけで、(H, W, 4, 3) の行列そのものは決して送らない。
FastAPI に依存させないので単体でテストできる。
"""

from __future__ import annotations

import itertools
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .matrix import MatrixContext


class MatrixStore:
    """Thread-safe id -> MatrixContext map."""

    def __init__(self) -> None:
        self._items: dict[str, MatrixContext] = {}
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def put(self, context: MatrixContext) -> str:
        """Store a matrix context and return its new id."""
        with self._lock:
            matrix_id = f"matrix{next(self._counter)}"
            self._items[matrix_id] = context
            return matrix_id

    def get(self, matrix_id: str) -> MatrixContext:
        """Return the context for ``matrix_id`` or raise ``KeyError``."""
        with self._lock:
            context = self._items.get(matrix_id)
        if context is None:
            raise KeyError(matrix_id)
        return context

    def clear(self) -> None:
        """Drop everything (used by tests)."""
        with self._lock:
            self._items.clear()


store = MatrixStore()
