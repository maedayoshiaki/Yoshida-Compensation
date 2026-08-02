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
    """Verifies the *reported* order and count match ``load_target_images``'s
    casefold-sorted contract.

    This does **not**, by itself, prove the product's ``sorted(...)`` call is
    what produces that order: on NTFS, ``os.listdir`` already returns
    entries pre-sorted by name, so writing the files in ``b, a, c`` creation
    order here is laundered away by the filesystem before the sort ever
    runs -- confirmed independently (task-2-review.md Important 1). Removing
    the product's ``sorted(...)`` call therefore leaves *this* test green.
    ``test_directory_listing_is_sorted_even_when_the_filesystem_already_sorts_it``
    below is the one with actual judgment power over the sort's existence,
    via a monkeypatched ``os.listdir`` that the filesystem cannot normalise.
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
    ``sorted(..., key=str.casefold)`` from
    ``yoshidacomp.compensation.pipeline.load_target_images``), made able to
    actually catch it. The test above cannot: NTFS's ``os.listdir`` already
    returns entries in name order regardless of creation order, so no
    on-disk write pattern can present the product with genuinely scrambled
    input. Intercepting ``os.listdir`` itself is the only way to do that.

    Removing the product's ``sorted()`` call makes this test fail while
    leaving the test above green -- confirmed (task-2-report.md addendum).
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
    ``models.py::LoadTargetRequest`` and ``target.py::load_target``'s
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
