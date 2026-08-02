"""Tests for the shared id-issuing store base (``store.py``'s ``_IdStore``)
and its two subclasses, ``MatrixStore``/``TargetStore``.

これらは task-2 のレビューで指摘された欠落を埋める:

- id の採番規則（``clear()`` はカウンタを戻さない）を、プロセス全体で
  共有される singleton ではなく**自前のインスタンス**で固定する。singleton
  越しに検証すると、他のテストファイルの実行順で ``matrix1``/``target1``
  という具体値が変わってしまい、検証対象（採番規則）と無関係な理由で
  テストが赤くなる（task-2-review.md Important 4）。
- ``MatrixStore``/``TargetStore`` が状態を共有していないことを直接固定する。
  共通基底 ``_IdStore`` への切り出しは、``_items``/``_lock``/``_counter`` を
  誤ってクラス変数へ格上げする一行の変更を「もっともらしい単純化」に
  見せてしまう。それが起きると2つの singleton が1つのid空間を共有し、
  ``GET /api/matrix/target1/meta`` が ``TargetContext`` を解決して
  ``.array`` 読み取りで素の500になる（task-2-review.md Important 5）。
- スレッド安全性（before/afterどちらにもテストが無かった、Minor 7）。
"""

from __future__ import annotations

import threading

from yoshidacomp_web.store import LutStore, MatrixStore, TargetStore


def test_fresh_stores_issue_kind_prefixed_ids_starting_at_one() -> None:
    """Pins the numbering *rule* on instances this test fully controls.

    Asserting this against the process-wide singletons instead would make
    the literal depend on how many other tests already called
    ``store.put``/``targets.put``/``luts.put`` before this one runs -- exactly
    the order-fragility this test exists to avoid (task-2-review.md
    Important 4).
    """
    matrix_store = MatrixStore()
    target_store = TargetStore()
    lut_store = LutStore()

    assert matrix_store.put(object()) == "matrix1"
    assert matrix_store.put(object()) == "matrix2"
    assert target_store.put(object()) == "target1"
    assert target_store.put(object()) == "target2"
    assert lut_store.put(object()) == "lut1"
    assert lut_store.put(object()) == "lut2"


def test_clear_does_not_reset_the_counter() -> None:
    """Documents and pins existing, deliberately preserved behaviour.

    The store refactor (task-2-report.md) must not change this: ``clear()``
    only empties ``_items``, so an id already issued is never reissued after
    a clear, even though it is no longer resolvable.
    """
    store = TargetStore()
    assert store.put(object()) == "target1"
    store.clear()
    assert store.put(object()) == "target2"


def test_the_stores_do_not_share_state() -> None:
    """A ``MatrixStore``, a ``TargetStore`` and a ``LutStore`` must never
    resolve each other's ids. Fails immediately if a future edit moves
    ``_items``, ``_lock``, or ``_counter`` from ``_IdStore.__init__``
    (per-instance) to class scope (shared across every instance, including the
    three live process-wide singletons ``store``/``targets``/``luts``).

    All three pairs are checked, not only matrix-vs-target: the third
    subclass arrived in plan B3 Task 4 and a pair-wise check that stopped at
    two would have left it unguarded, which is the whole failure mode this
    test exists for (``GET /api/lut/{a matrix id}/curve`` resolving a
    ``MatrixContext`` and 500ing on ``.lut`` instead of 404ing).
    """
    stores = {
        "matrix": MatrixStore(),
        "target": TargetStore(),
        "lut": LutStore(),
    }

    for a_kind, a in stores.items():
        for b_kind, b in stores.items():
            if a_kind == b_kind:
                continue
            assert a._items is not b._items, (a_kind, b_kind)
            assert a._lock is not b._lock, (a_kind, b_kind)
            assert a._counter is not b._counter, (a_kind, b_kind)
            # The cache-key index is per-instance for the same reason, and is
            # listed here rather than left implicit: a shared one would hand a
            # ``TargetStore`` key an id issued by another store, and
            # ``get_by_key`` would answer with it.
            assert a._ids_by_key is not b._ids_by_key, (a_kind, b_kind)

            issued = a.put(object())
            assert issued not in b._items, (a_kind, b_kind)
            try:
                b.get(issued)
            except KeyError:
                pass
            else:
                raise AssertionError(
                    f"the {b_kind} store resolved {issued!r}, an id it never "
                    f"issued -- it is sharing state with the {a_kind} store"
                )


def test_a_keyed_put_is_found_again_and_an_unkeyed_one_is_not() -> None:
    """``put(key=...)``/``get_by_key`` must reuse, and only where asked.

    This is what stops ``POST /api/target`` from storing another copy of an
    image it already holds every time the sidebar's picker is clicked. Both
    directions matter: a key must find its id, and a context stored without a
    key must not be reachable by any key at all (a store that answered
    ``get_by_key`` from ``_items`` would hand back whatever it last stored,
    and the router would then serve one target's pixels under another's
    request).
    """
    store = TargetStore()
    first = store.put(object(), key=("a.png", 1, 100))
    assert store.get_by_key(("a.png", 1, 100)) == first
    # A different key is a different thing, even one that only differs in the
    # part that says "which file version".
    assert store.get_by_key(("a.png", 1, 101)) is None
    assert store.get_by_key(("b.png", 1, 100)) is None

    unkeyed = store.put(object())
    assert store.get(unkeyed) is not None
    assert store.get_by_key(("a.png", 1, 100)) == first  # not the newest put


def test_clear_drops_the_cache_keys_with_the_contexts() -> None:
    """A key surviving ``clear()`` would name an id that no longer resolves.

    ``get_by_key`` would then return an id whose ``get`` raises, and
    ``routers/target.py``, which calls the two in sequence, would answer a
    routine reload with a 500 instead of reading the file again.

    This is the *only* guard: ``get_by_key`` deliberately does not re-check
    ``_items``. It used to, and the two clauses masked each other -- removing
    either on its own left this file green (measured), so nothing held either
    of them to anything. See ``_IdStore.get_by_key``.

    Both halves are asserted, because the first alone passes against a store
    that never stores keys at all.
    """
    store = TargetStore()
    key = ("a.png", 0, 7)
    assert store.put(object(), key=key) == "target1"
    assert store.get_by_key(key) == "target1"  # it was really stored
    store.clear()
    assert store.get_by_key(key) is None
    # And the next load stores it afresh, under a new id (the counter does not
    # rewind -- see test_clear_does_not_reset_the_counter).
    assert store.put(object(), key=key) == "target2"
    assert store.get_by_key(key) == "target2"


def test_put_is_thread_safe_and_issues_no_duplicate_or_lost_ids() -> None:
    """No test, before or after the store refactor, exercised the lock under
    real concurrency (task-2-review.md Minor 7). 8 threads x 500 puts each
    must yield exactly 4000 ids, all unique, all resolvable.
    """
    store = TargetStore()
    threads_n, puts_per_thread = 8, 500
    ids: list[str] = []
    ids_lock = threading.Lock()

    def worker() -> None:
        local_ids = [store.put(object()) for _ in range(puts_per_thread)]
        with ids_lock:
            ids.extend(local_ids)

    threads = [threading.Thread(target=worker) for _ in range(threads_n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(ids) == threads_n * puts_per_thread
    assert len(set(ids)) == threads_n * puts_per_thread  # no duplicates
    for item_id in ids:
        store.get(item_id)  # every id must resolve; raises KeyError if lost
