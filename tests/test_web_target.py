"""Target image loading and metadata for the web layer.

``tests/test_web_matrix.py`` の作法（クライアントの作り方、``store.clear()``
の扱い、一時ファイルの置き方）に合わせる。

判別力について: 正方形の画像では縦横の取り違えが見えない
（計画Cで実際に出た欠陥の形）。ここでは常に高さ ``_HEIGHT`` != 幅
``_WIDTH`` の画像を書く。
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store, targets  # noqa: E402

# Deliberately non-square. A square fixture cannot distinguish a correct
# (H, W, 3) report from one that silently swapped H and W.
_HEIGHT, _WIDTH = 3, 5


def _write_image(path, height: int, width: int, seed: int = 0) -> np.ndarray:
    """Write a non-uniform RGB PNG (so pixel shuffles would be visible too)."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok, f"failed to write test fixture image: {path}"
    return image


def _write_matrix(path, height: int, width: int) -> None:
    """Write a minimal valid ``(H, W, 4, 3)`` identity-ish matrix (see
    ``tests/test_web_matrix.py``'s helper of the same name).
    """
    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    for channel in range(3):
        matrices[:, :, channel, channel] = 1.0
    np.save(path, matrices)


def test_single_file_reports_matching_shape(tmp_path) -> None:
    targets.clear()
    path = tmp_path / "target.png"
    _write_image(path, _HEIGHT, _WIDTH)

    client = TestClient(create_app())
    response = client.post("/api/target", json={"path": str(path)})
    assert response.status_code == 200, response.text
    body = response.json()
    # Not a literal "target1": the id counter is a process-wide singleton
    # shared with every other test in this file (and, since `clear()` does
    # not reset it, never rewound) -- see task-2-review.md Important 4 and
    # tests/test_web_store.py, which pins the actual numbering rule against
    # a store instance this test controls instead of the shared singleton.
    assert body["id"].startswith("target")
    assert body["id"] == body["meta"]["id"]
    assert body["meta"]["shape"] == [_HEIGHT, _WIDTH, 3]
    assert body["meta"]["path"] == str(path)
    assert body["meta"]["is_directory"] is False


def test_directory_lists_all_images_with_correct_count_and_order(tmp_path) -> None:
    """Verifies the *reported* order and count are casefold-sorted.

    This does **not**, by itself, prove the product's ``sorted(...)`` call is
    what produces that order: on NTFS, ``os.listdir`` already returns
    entries pre-sorted by name, so writing the files in ``b, a, c`` creation
    order here is laundered away by the filesystem before the sort ever
    runs -- confirmed independently (task-2-review.md Important 1). Removing
    the product's ``sorted(...)`` call therefore leaves *this* test green.
    ``test_directory_listing_is_sorted_even_when_the_filesystem_already_sorts_it``
    below is the one with actual judgment power over the sort's existence,
    via a monkeypatched ``os.listdir`` that the filesystem cannot normalise.

    The sort under test is ``yoshidacomp_web.target._listing``'s. It was
    ``pipeline.load_target_images``' until the web layer stopped decoding
    every image in a directory to find out their names (see
    ``test_a_directory_pass_decodes_one_image_per_selection``); the order it
    reports is unchanged, and is still the same ``sorted(os.listdir(...),
    key=str.casefold)`` filtered by ``SUPPORTED_TARGET_IMAGE_SUFFIXES``.
    """
    targets.clear()
    directory = tmp_path / "targets"
    directory.mkdir()
    on_disk_creation_order = ["b_target.png", "a_target.png", "c_target.png"]
    for i, name in enumerate(on_disk_creation_order):
        _write_image(directory / name, _HEIGHT, _WIDTH, seed=i)

    client = TestClient(create_app())
    response = client.post("/api/target", json={"path": str(directory)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["meta"]["is_directory"] is True
    assert body["meta"]["images"] == ["a_target.png", "b_target.png", "c_target.png"]
    assert len(body["meta"]["images"]) == 3
    assert body["meta"]["index"] == 0  # first is default
    assert body["meta"]["shape"] == [_HEIGHT, _WIDTH, 3]


def test_directory_listing_is_sorted_even_when_the_filesystem_already_sorts_it(
    tmp_path, monkeypatch
) -> None:
    """The genuine mutation-detection test for directory ordering.

    This is the brief's Step 5 mutation 2, run as written (remove
    ``sorted(..., key=str.casefold)`` from the listing), made able to
    actually catch it. The test above cannot: NTFS's ``os.listdir`` already
    returns entries in name order regardless of creation order, so no
    on-disk write pattern can present the product with genuinely scrambled
    input. Intercepting ``os.listdir`` itself is the only way to do that.

    Removing the product's ``sorted()`` call makes this test fail while
    leaving the test above green -- confirmed (task-2-report.md addendum),
    and re-confirmed against ``yoshidacomp_web.target._listing``, which is
    where that call moved when the web layer stopped decoding a whole
    directory to list it. The patch is on ``os.listdir`` itself, so it
    reaches the listing wherever it lives -- which is why this test kept its
    judgment power across that move rather than quietly losing it.
    """
    targets.clear()
    directory = tmp_path / "targets"
    directory.mkdir()
    names = ["a_target.png", "b_target.png", "c_target.png"]
    for i, name in enumerate(names):
        _write_image(directory / name, _HEIGHT, _WIDTH, seed=i)

    real_listdir = os.listdir
    target_dir_str = str(directory)

    def scrambled_listdir(path):
        result = real_listdir(path)
        if os.path.abspath(str(path)) == os.path.abspath(target_dir_str):
            result = list(reversed(list(result)))  # c, b, a -- not sorted
        return result

    monkeypatch.setattr(os, "listdir", scrambled_listdir)

    client = TestClient(create_app())
    response = client.post("/api/target", json={"path": str(directory)})
    assert response.status_code == 200, response.text
    # Must come back re-sorted (a, b, c), not in the scrambled order the
    # (patched) filesystem call handed to the product.
    assert response.json()["meta"]["images"] == names


def test_an_unreadable_file_stays_in_the_listing_and_says_why_when_selected(
    tmp_path,
) -> None:
    """A file the decoder cannot read must not vanish from the directory.

    ``pipeline.load_target_images`` drops an entry ``cv2.imread`` returns
    ``None`` for, silently. The web layer stopped using it when the listing
    became name-only (see ``target.py::_listing``), and that changed this
    behaviour deliberately: a file the user can see in Explorer now stays in
    the picker and answers "could not read ... as an image" when chosen,
    rather than disappearing with no message anywhere. Silently omitting an
    entry is the failure class this repository keeps closing; the change was
    documented and, until this test, pinned by nothing -- ``_listing`` could
    have been given the filter back with all 292 tests green (measured, both
    with a decode call and with a reader-availability probe that never
    decodes).

    ``test_corrupt_image_file_is_400_with_a_readable_message`` above covers
    only the single-file path, which never consults a listing at all.

    The entry *after* the unreadable one is asserted too, and that is the half
    a listing-length check would miss: dropping the broken file does not merely
    shorten the list, it renumbers everything past it, so index 2 would come
    back as ``c_good.png``'s neighbour or as "out of range" while the picker's
    labels still said otherwise -- the dropdown naming one image and the server
    loading another.

    復号できないファイルを一覧から黙って落とさない。選んだときに理由を言う。
    落とすと**後ろの index が全部ずれる**ので、長さだけを見るテストでは
    「一覧の見出しと実際に読まれる画像が食い違う」ほうを取り逃がす。
    """
    targets.clear()
    directory = tmp_path / "targets"
    directory.mkdir()
    _write_image(directory / "a_good.png", _HEIGHT, _WIDTH, seed=1)
    # Right extension, contents ``cv2.imread`` cannot decode -- the same
    # fixture the single-file test uses, placed inside a directory.
    (directory / "b_broken.png").write_bytes(b"this is not actually a png file")
    third_height, third_width = _HEIGHT + 2, _WIDTH + 3
    _write_image(directory / "c_good.png", third_height, third_width, seed=2)

    client = TestClient(create_app())
    listing = client.post("/api/target", json={"path": str(directory)})
    assert listing.status_code == 200, listing.text
    assert listing.json()["meta"]["images"] == [
        "a_good.png", "b_broken.png", "c_good.png"
    ]

    # Selecting it is refused, and the message names the file and the reason.
    refused = client.post("/api/target", json={"path": str(directory), "index": 1})
    assert refused.status_code == 400, refused.text
    detail = refused.json()["detail"]
    assert "b_broken.png" in detail
    assert "could not read" in detail.lower()
    assert "out of range" not in detail.lower()  # not the index-bounds branch

    # And the indices past it are unshifted: index 2 is still c_good.png.
    third = client.post("/api/target", json={"path": str(directory), "index": 2})
    assert third.status_code == 200, third.text
    assert third.json()["meta"]["index"] == 2
    assert third.json()["meta"]["shape"] == [third_height, third_width, 3]


def test_directory_index_selects_a_specific_image(tmp_path) -> None:
    """``index`` must actually change which image is selected, not just be echoed.

    Uses two differently-shaped images so that selecting the wrong one is
    visible in ``shape``, not just in some opaque identity check.
    """
    targets.clear()
    directory = tmp_path / "targets"
    directory.mkdir()
    _write_image(directory / "a_first.png", _HEIGHT, _WIDTH, seed=1)
    second_height, second_width = _HEIGHT + 1, _WIDTH + 2
    _write_image(directory / "b_second.png", second_height, second_width, seed=2)

    client = TestClient(create_app())
    default_response = client.post("/api/target", json={"path": str(directory)})
    assert default_response.json()["meta"]["shape"] == [_HEIGHT, _WIDTH, 3]

    selected_response = client.post(
        "/api/target", json={"path": str(directory), "index": 1}
    )
    assert selected_response.status_code == 200, selected_response.text
    body = selected_response.json()
    assert body["meta"]["index"] == 1
    assert body["meta"]["shape"] == [second_height, second_width, 3]


def test_a_directory_pass_decodes_one_image_per_selection(tmp_path, monkeypatch) -> None:
    """Choosing an image must cost one decode, and choosing it again none.

    The sidebar's image picker re-POSTs here on every selection, so this
    endpoint's cost per click is the picker's cost per click. It used to call
    ``pipeline.load_target_images``, which decodes **every** image in the
    directory and keeps one, and then stored the result under a fresh id: a
    pass over an N-image directory was N**2 decodes and N retained
    full-resolution copies, growing without bound in clicks. On the shipped
    default (5 x 512x512) that is 0.08 s and 3.8 MB and nobody notices; on a
    directory of 4K captures it is seconds and hundreds of megabytes per
    click (task-4 final review, Minor 7).

    Counting ``cv2.imread`` calls is what makes this able to fail. Asserting
    only the ids would pass against an implementation that decoded the whole
    directory and then threw it away, which is precisely the implementation
    being replaced.

    一覧から選び直す操作1回あたりの復号回数を数える。以前はディレクトリの
    全画像を復号して1枚だけ残していたため、一巡で N**2 回の復号と N 枚の
    恒久保持になっていた。IDだけを見るテストでは、全部復号して捨てる実装と
    区別が付かない。
    """
    targets.clear()
    directory = tmp_path / "targets"
    directory.mkdir()
    names = ["a.png", "b.png", "c.png"]
    for i, name in enumerate(names):
        _write_image(directory / name, _HEIGHT + i, _WIDTH, seed=i)

    import yoshidacomp_web.target as web_target

    real_imread = web_target.cv2.imread
    decodes: list[str] = []

    def counting_imread(filename, *args, **kwargs):
        decodes.append(str(filename))
        return real_imread(filename, *args, **kwargs)

    monkeypatch.setattr(web_target.cv2, "imread", counting_imread)

    client = TestClient(create_app())
    first_pass = []
    for index in range(3):
        response = client.post(
            "/api/target", json={"path": str(directory), "index": index}
        )
        assert response.status_code == 200, response.text
        first_pass.append(response.json())

    # One decode per selection, and the file decoded is the one selected.
    assert len(decodes) == 3, decodes
    assert [os.path.basename(d) for d in decodes] == names, decodes
    # Each selection really did select its own image (different heights).
    assert [b["meta"]["shape"][0] for b in first_pass] == [
        _HEIGHT, _HEIGHT + 1, _HEIGHT + 2
    ]
    assert [b["meta"]["index"] for b in first_pass] == [0, 1, 2]
    ids = [b["id"] for b in first_pass]
    assert len(set(ids)) == 3, ids

    # A second pass costs nothing: same ids, same metadata, no further decode.
    decodes.clear()
    for index in range(3):
        body = client.post(
            "/api/target", json={"path": str(directory), "index": index}
        ).json()
        assert body["id"] == ids[index]
        assert body["meta"] == first_pass[index]["meta"]
    assert decodes == [], decodes


def test_a_replaced_file_is_read_again_rather_than_served_from_the_cache(
    tmp_path,
) -> None:
    """Reusing a loaded target must key on the file, not merely on its path.

    The reuse above is what stops the picker from growing the store without
    bound; keyed on the path alone it would also stop an edited file from ever
    being re-read, and the viewer would keep drawing a picture that no longer
    exists on disk -- a silent wrong answer, which is worse than the leak the
    reuse exists to fix. The identity therefore carries the file's size and
    modification time.

    パスだけを鍵にすると、差し替えられたファイルが二度と読み直されず、
    ディスク上に無い絵を描き続けることになる。同一性にサイズと更新時刻を
    含める理由がこれである。
    """
    targets.clear()
    path = tmp_path / "target.png"
    _write_image(path, _HEIGHT, _WIDTH, seed=1)

    client = TestClient(create_app())
    first = client.post("/api/target", json={"path": str(path)}).json()
    assert first["meta"]["shape"] == [_HEIGHT, _WIDTH, 3]

    # Same path, different content *and* a different size, written far enough
    # ahead in time that no filesystem timestamp granularity can hide it.
    replaced_height, replaced_width = _HEIGHT + 4, _WIDTH + 6
    _write_image(path, replaced_height, replaced_width, seed=2)
    os.utime(path, ns=(os.stat(path).st_atime_ns, os.stat(path).st_mtime_ns + 10**9))

    second = client.post("/api/target", json={"path": str(path)}).json()
    assert second["id"] != first["id"]
    assert second["meta"]["shape"] == [replaced_height, replaced_width, 3]
    # The first id still resolves, and still describes what it loaded.
    still_there = client.get(f"/api/target/{first['id']}/meta")
    assert still_there.status_code == 200
    assert still_there.json()["shape"] == [_HEIGHT, _WIDTH, 3]


def test_out_of_range_index_is_400(tmp_path) -> None:
    targets.clear()
    directory = tmp_path / "targets"
    directory.mkdir()
    _write_image(directory / "only.png", _HEIGHT, _WIDTH)

    client = TestClient(create_app())
    response = client.post("/api/target", json={"path": str(directory), "index": 5})
    assert response.status_code == 400
    assert str(directory) in response.json()["detail"]


def test_nonzero_index_on_a_single_file_is_400(tmp_path) -> None:
    """A single file has exactly one selectable entry (index 0). Any other
    value must be rejected, not silently ignored -- see
    ``models.py::LoadTargetRequest`` and ``target.py::resolve_target``'s
    docstrings, which this pins so the three no longer disagree
    (task-2-review.md Important 3).
    """
    targets.clear()
    path = tmp_path / "target.png"
    _write_image(path, _HEIGHT, _WIDTH)

    client = TestClient(create_app())
    for bad_index in (1, -7, 99):
        response = client.post(
            "/api/target", json={"path": str(path), "index": bad_index}
        )
        assert response.status_code == 400, bad_index
        assert str(path) in response.json()["detail"]


def test_zero_index_on_a_single_file_is_accepted(tmp_path) -> None:
    targets.clear()
    path = tmp_path / "target.png"
    _write_image(path, _HEIGHT, _WIDTH)

    client = TestClient(create_app())
    response = client.post("/api/target", json={"path": str(path), "index": 0})
    assert response.status_code == 200
    assert response.json()["meta"]["index"] == 0


def test_empty_directory_is_400(tmp_path) -> None:
    targets.clear()
    directory = tmp_path / "empty"
    directory.mkdir()

    client = TestClient(create_app())
    response = client.post("/api/target", json={"path": str(directory)})
    assert response.status_code == 400
    assert str(directory) in response.json()["detail"]


def test_missing_path_is_404(tmp_path) -> None:
    targets.clear()
    missing = tmp_path / "nope.png"
    client = TestClient(create_app())
    response = client.post("/api/target", json={"path": str(missing)})
    assert response.status_code == 404
    assert str(missing) in response.json()["detail"]


def test_empty_path_is_400() -> None:
    """``Path("")`` is ``Path(".")``; say "empty" rather than "'.' is not a file"."""
    targets.clear()
    client = TestClient(create_app())
    for value in ("", "   "):
        response = client.post("/api/target", json={"path": value})
        assert response.status_code == 400, value
        assert "empty" in response.json()["detail"]


def test_non_image_extension_is_400_with_a_readable_message(tmp_path) -> None:
    targets.clear()
    path = tmp_path / "notes.txt"
    path.write_text("this is not an image", encoding="utf-8")

    client = TestClient(create_app())
    response = client.post("/api/target", json={"path": str(path)})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert str(path) in detail
    # The reason, not just the path: this is the extension-rejection branch
    # specifically (before any attempt to decode), so the message must name
    # what is supported.
    assert "supported" in detail.lower()
    assert ".png" in detail


def test_corrupt_image_file_is_400_with_a_readable_message(tmp_path) -> None:
    """Right extension, unreadable content -- ``cv2.imread`` returns ``None``.

    Distinct branch from the extension-rejection test above: this file's
    extension *is* accepted, so reaching 400 here proves the decode-failure
    path fired, not the extension check. Asserting a substring unique to
    this branch (``"could not read"``), and asserting the other branch's
    substring is *absent*, is what makes the two tests able to tell the
    branches apart, rather than both merely reaching some shared generic
    "path is bad" message (task-2-review.md Important 2).
    """
    targets.clear()
    path = tmp_path / "broken.png"
    path.write_bytes(b"this is not actually a png file")

    client = TestClient(create_app())
    response = client.post("/api/target", json={"path": str(path)})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert str(path) in detail
    assert "could not read" in detail.lower()
    assert "supported" not in detail.lower()  # not the extension-check message


def test_get_meta_matches_post_meta(tmp_path) -> None:
    targets.clear()
    path = tmp_path / "target.png"
    _write_image(path, _HEIGHT, _WIDTH)

    client = TestClient(create_app())
    post_body = client.post("/api/target", json={"path": str(path)}).json()
    meta_response = client.get(f"/api/target/{post_body['id']}/meta")
    assert meta_response.status_code == 200
    assert meta_response.json() == post_body["meta"]


def test_unknown_id_is_404() -> None:
    targets.clear()
    client = TestClient(create_app())
    response = client.get("/api/target/nosuch/meta")
    assert response.status_code == 404
    # Not just "some 404": StaticFiles(html=True) mounted at "/" would also
    # 404 an unmatched path if this route were never registered at all
    # (task-2-review.md Minor 5). The message is what proves *this* route,
    # not the static-file fallback, answered.
    assert "target not found" in response.json()["detail"]
    assert "nosuch" in response.json()["detail"]


def test_matrix_and_target_ids_do_not_cross_resolve(tmp_path) -> None:
    """The store refactor (task-2-report.md) shares one base class between
    ``MatrixStore``/``TargetStore``. If a future edit moved their
    per-instance state to class scope, the two live process-wide singletons
    (``store``/``targets``) would merge into one id space, and
    ``GET /api/matrix/{a target id}/meta`` would resolve a ``TargetContext``
    and 500 on ``.array`` instead of 404ing (task-2-review.md Important 5).
    """
    store.clear()
    targets.clear()

    matrix_path = tmp_path / "m.npy"
    _write_matrix(matrix_path, 4, 4)
    target_path = tmp_path / "t.png"
    _write_image(target_path, _HEIGHT, _WIDTH)

    client = TestClient(create_app())
    matrix_id = client.post(
        "/api/session/matrix", json={"path": str(matrix_path)}
    ).json()["id"]
    target_id = client.post(
        "/api/target", json={"path": str(target_path)}
    ).json()["id"]
    assert matrix_id.startswith("matrix")
    assert target_id.startswith("target")

    # A target id must not resolve through the matrix endpoint, and vice
    # versa -- if the two stores shared `_items`, one of these would 200
    # (or 500, reading the wrong context's attributes) instead of 404ing.
    assert client.get(f"/api/matrix/{target_id}/meta").status_code == 404
    assert client.get(f"/api/target/{matrix_id}/meta").status_code == 404
