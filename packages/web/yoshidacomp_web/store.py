"""In-memory context stores (one process, thread-safe).

読み込み済みの行列・ターゲット画像・逆LUTをサーバのRAMに置き、不透明なIDで
参照する。線を渡るのは2次元のビュー結果だけで、(H, W, 4, 3) の行列そのものは
決して送らない。FastAPI に依存させないので単体でテストできる。

``MatrixStore``/``TargetStore``/``LutStore`` は id 採番・ロック・``get``/
``put``/``clear`` が完全に同じ形なので、その共通部分を ``_IdStore`` に切り出して
3つとも使う。id の接頭辞（``"matrix"``/``"target"``/``"lut"``）だけが違う。
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Hashable
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from .lut import LutContext
    from .matrix import MatrixContext
    from .target import TargetContext

_T = TypeVar("_T")


class _IdStore(Generic[_T]):
    """Thread-safe id -> context map, ids prefixed by *kind*.

    ``MatrixStore``/``TargetStore``/``LutStore`` の共通実装。公開する振る舞い
    （id の形式・ロック・例外）は元の3実装とも変えていない。

    An id, once issued, resolves for the life of the process: there is no
    eviction and no bound. Nothing here has to guess when the user is finished
    with a loaded thing, and ``put`` accepts an optional *key* so a caller who
    can say "this is the same thing I stored before" gets the same id back
    instead of a second copy -- see ``get_by_key`` and
    ``routers/target.py::load_session_target``, which is the one caller that
    can be driven repeatedly by a single click.
    発行したidは寿命の間ずっと解決できる（追い出しも上限も無い）。同じものを
    再び読み込んだ場合に増やさないための鍵が ``put(key=...)`` である。
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, _T] = {}
        # Cache keys -> the id issued for them. Emptied by ``clear`` together
        # with ``_items``: a key surviving a clear would name an id that no
        # longer resolves, which is worse than a cache miss.
        # ``clear`` で ``_items`` と一緒に空にする（鍵だけ残ると解決できない
        # idを指すことになる）。
        self._ids_by_key: dict[Hashable, str] = {}
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def put(self, context: _T, key: Hashable | None = None) -> str:
        """Store a context and return its new id.

        *key*, when given, is remembered so ``get_by_key`` can find this id
        again. Storing a context under a key already present replaces the
        mapping; the previously stored context keeps its own id and stays
        resolvable.
        """
        with self._lock:
            item_id = f"{self._kind}{next(self._counter)}"
            self._items[item_id] = context
            if key is not None:
                self._ids_by_key[key] = item_id
            return item_id

    def get_by_key(self, key: Hashable) -> str | None:
        """Return the id stored under *key*, or ``None``.

        An id this returns always resolves, so a caller may follow it with
        ``get`` -- but that is a property of ``clear`` emptying both dicts
        together, not of a second check here. There was one
        (``item_id if item_id in self._items else None``) and it was dead:
        nothing removes from ``_items`` except ``clear``, which removes the
        key at the same moment, so neither guard could be removed on its own
        without the other still answering ``None`` (measured -- both
        single-guard mutations left ``tests/test_web_store.py`` green). Two
        clauses where either alone suffices is two clauses no test can hold to
        anything; the invariant now lives in one place and
        ``test_clear_drops_the_cache_keys_with_the_contexts`` can fail.
        If eviction is ever added, it must drop the key with the item.
        ここに ``_items`` の再確認は置かない。``clear`` が両方を同時に空に
        することが保証であり、二重にすると片方ずつでは落ちないテストになる
        （両方の単独変異で緑のままであることを実測した）。
        """
        with self._lock:
            return self._ids_by_key.get(key)

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
            self._ids_by_key.clear()


class MatrixStore(_IdStore["MatrixContext"]):
    """Thread-safe id -> MatrixContext map. Ids look like ``"matrix1"``."""

    def __init__(self) -> None:
        super().__init__("matrix")


class TargetStore(_IdStore["TargetContext"]):
    """Thread-safe id -> TargetContext map. Ids look like ``"target1"``."""

    def __init__(self) -> None:
        super().__init__("target")


class LutStore(_IdStore["LutContext"]):
    """Thread-safe id -> LutContext map. Ids look like ``"lut1"``."""

    def __init__(self) -> None:
        super().__init__("lut")


store = MatrixStore()
targets = TargetStore()
luts = LutStore()
