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

from yoshidacomp_web.store import MatrixStore, TargetStore


def test_fresh_stores_issue_kind_prefixed_ids_starting_at_one() -> None:
    """Pins the numbering *rule* on instances this test fully controls.

    Asserting this against the process-wide singletons instead would make
    the literal depend on how many other tests already called
    ``store.put``/``targets.put`` before this one runs -- exactly the
    order-fragility this test exists to avoid (task-2-review.md Important 4).
    """
    matrix_store = MatrixStore()
    target_store = TargetStore()

    assert matrix_store.put(object()) == "matrix1"
    assert matrix_store.put(object()) == "matrix2"
    assert target_store.put(object()) == "target1"
    assert target_store.put(object()) == "target2"


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


def test_matrix_and_target_stores_do_not_share_state() -> None:
    """A ``MatrixStore`` and a ``TargetStore`` must never resolve each
    other's ids. Fails immediately if a future edit moves ``_items``,
    ``_lock``, or ``_counter`` from ``_IdStore.__init__`` (per-instance) to
    class scope (shared across every instance, including the two live
    process-wide singletons ``store``/``targets``).
    """
    matrix_store = MatrixStore()
    target_store = TargetStore()

    assert matrix_store._items is not target_store._items
    assert matrix_store._lock is not target_store._lock
    assert matrix_store._counter is not target_store._counter

    marker = object()
    matrix_id = matrix_store.put(marker)
    assert matrix_id not in target_store._items
    try:
        target_store.get(matrix_id)
    except KeyError:
        pass
    else:
        raise AssertionError(
            f"TargetStore resolved {matrix_id!r}, an id it never issued -- "
            "the two stores are sharing state"
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
