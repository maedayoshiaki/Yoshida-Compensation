"""Matrix loading and metadata for the web layer."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store  # noqa: E402

# The real config.toml (1920x1080 projector) is restored before/after every
# test by the autouse fixture in tests/conftest.py, so this module does not
# need its own copy. See that fixture's docstring for why both sides matter.
# 実 config.toml（1920x1080プロジェクタ）は tests/conftest.py の autouse
# フィクスチャが毎テストの前後で復元するため、本ファイルは独自コピーを
# 持たない。前後両方が必要な理由は当該フィクスチャのdocstring参照。


def _write_matrix(path, height: int, width: int) -> None:
    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    for channel in range(3):
        matrices[:, :, channel, channel] = 1.0
    np.save(path, matrices)


def test_projector_space_matrix_is_accepted(tmp_path) -> None:
    store.clear()
    path = tmp_path / "projector.npy"
    _write_matrix(path, 1080, 1920)

    client = TestClient(create_app())
    response = client.post("/api/session/matrix", json={"path": str(path)})
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["space"] == "projector"
    assert body["meta"]["shape"] == [1080, 1920, 4, 3]

    meta = client.get(f"/api/matrix/{body['id']}/meta")
    assert meta.status_code == 200
    assert meta.json()["space"] == "projector"


def test_camera_space_matrix_is_reported_not_rejected(tmp_path) -> None:
    """カメラ座標も読み込めるが space="camera" と申告する（ワープは計画B2）。"""
    store.clear()
    path = tmp_path / "camera.npy"
    _write_matrix(path, 60, 80)

    client = TestClient(create_app())
    response = client.post("/api/session/matrix", json={"path": str(path)})
    assert response.status_code == 200
    assert response.json()["meta"]["space"] == "camera"


def test_wrong_shape_is_rejected(tmp_path) -> None:
    store.clear()
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((4, 4, 3), dtype=np.float32))

    client = TestClient(create_app())
    response = client.post("/api/session/matrix", json={"path": str(path)})
    assert response.status_code == 400
    assert "(H, W, 4, 3)" in response.json()["detail"]


def test_missing_file_is_404(tmp_path) -> None:
    store.clear()
    client = TestClient(create_app())
    response = client.post("/api/session/matrix", json={"path": str(tmp_path / "nope.npy")})
    assert response.status_code == 404


def test_unknown_id_is_404() -> None:
    store.clear()
    client = TestClient(create_app())
    assert client.get("/api/matrix/nosuch/meta").status_code == 404
