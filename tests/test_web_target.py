"""Target image loading and metadata for the web layer.

``tests/test_web_matrix.py`` の作法（クライアントの作り方、``store.clear()``
の扱い、一時ファイルの置き方）に合わせる。

判別力について: 正方形の画像では縦横の取り違えが見えない
（計画Cで実際に出た欠陥の形）。ここでは常に高さ ``_HEIGHT`` != 幅
``_WIDTH`` の画像を書く。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import targets  # noqa: E402

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


def test_single_file_reports_matching_shape(tmp_path) -> None:
    targets.clear()
    path = tmp_path / "target.png"
    _write_image(path, _HEIGHT, _WIDTH)

    client = TestClient(create_app())
    response = client.post("/api/target", json={"path": str(path)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == "target1"
    assert body["meta"]["shape"] == [_HEIGHT, _WIDTH, 3]
    assert body["meta"]["path"] == str(path)
    assert body["meta"]["is_directory"] is False


def test_directory_lists_all_images_with_correct_count_and_order(tmp_path) -> None:
    targets.clear()
    directory = tmp_path / "targets"
    directory.mkdir()
    # Deliberately out of alphabetical order on disk. ``load_target_images``
    # sorts by casefold, so the reported order must be a, b, c regardless of
    # creation order -- an unsorted listing would leave this in b, a, c.
    on_disk_order = ["b_target.png", "a_target.png", "c_target.png"]
    for i, name in enumerate(on_disk_order):
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
    client = TestClient(create_app())
    response = client.post("/api/target", json={"path": str(tmp_path / "nope.png")})
    assert response.status_code == 404


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
    assert str(path) in response.json()["detail"]


def test_corrupt_image_file_is_400_with_a_readable_message(tmp_path) -> None:
    """Right extension, unreadable content -- ``cv2.imread`` returns ``None``."""
    targets.clear()
    path = tmp_path / "broken.png"
    path.write_bytes(b"this is not actually a png file")

    client = TestClient(create_app())
    response = client.post("/api/target", json={"path": str(path)})
    assert response.status_code == 400
    assert str(path) in response.json()["detail"]


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
    assert client.get("/api/target/nosuch/meta").status_code == 404
