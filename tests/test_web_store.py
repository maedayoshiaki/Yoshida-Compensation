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
