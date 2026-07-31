"""Matrix loading and metadata for the web layer."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store  # noqa: E402

# Space classification compares the matrix against the *configured* projector
# frame, so every test below that asserts on ``space`` loads its own stand-in
# projector config (``stand_in_projector_config``, tests/conftest.py) instead of
# borrowing whatever the developer's live config.toml happens to say today.
# The autouse fixture in conftest.py remains the isolation backstop.
# 座標空間の判定は設定されたプロジェクタ枠との比較なので、``space`` を検証
# するテストは実 config.toml ではなく代用configを自前で読み込む。

_STAND_IN_WIDTH, _STAND_IN_HEIGHT = 64, 48


def _write_matrix(path, height: int, width: int) -> None:
    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    for channel in range(3):
        matrices[:, :, channel, channel] = 1.0
    np.save(path, matrices)


def test_projector_space_matrix_is_accepted(tmp_path, stand_in_projector_config) -> None:
    stand_in_projector_config(width=_STAND_IN_WIDTH, height=_STAND_IN_HEIGHT)
    store.clear()
    path = tmp_path / "projector.npy"
    _write_matrix(path, _STAND_IN_HEIGHT, _STAND_IN_WIDTH)

    client = TestClient(create_app())
    response = client.post("/api/session/matrix", json={"path": str(path)})
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["space"] == "projector"
    assert body["meta"]["shape"] == [_STAND_IN_HEIGHT, _STAND_IN_WIDTH, 4, 3]
    assert body["meta"]["projector"] == {
        "width": _STAND_IN_WIDTH,
        "height": _STAND_IN_HEIGHT,
    }

    meta = client.get(f"/api/matrix/{body['id']}/meta")
    assert meta.status_code == 200
    assert meta.json()["space"] == "projector"


def test_camera_space_matrix_is_reported_not_rejected(
    tmp_path, stand_in_projector_config
) -> None:
    """カメラ座標も読み込めるが space="camera" と申告する（ワープは計画B2）。"""
    stand_in_projector_config(width=_STAND_IN_WIDTH, height=_STAND_IN_HEIGHT)
    store.clear()
    path = tmp_path / "camera.npy"
    # Deliberately neither the configured frame nor its transpose.
    _write_matrix(path, _STAND_IN_HEIGHT + 12, _STAND_IN_WIDTH + 16)

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


def test_npz_archive_is_400_with_a_readable_message(tmp_path) -> None:
    """An ``.npz`` must fail as a bad request, not as an unexplained 500.

    ``np.load`` hands back a lazy ``NpzFile`` for an archive, which has no
    ``.ndim``; before this fix the ``AttributeError`` escaped ``load_matrix``
    and Starlette answered with a plain-text 500 that ``jsonOrThrow``
    (``static/js/api.js``) cannot parse, so the user saw only
    "500: Internal Server Error". This repository stores inverse LUTs as
    ``.npz``, so pasting one into the path box is a routine mistake, not a
    hypothetical.

    ``.npz`` は ``NpzFile`` として読み込まれ ``.ndim`` を持たないため、
    以前は 500 になり画面には理由が出なかった。本リポジトリは逆LUTを
    ``.npz`` で保存しており、取り違えは日常的に起こる。
    """
    store.clear()
    path = tmp_path / "inverse_lut.npz"
    np.savez(path, curve=np.zeros((4, 3), dtype=np.float32))

    client = TestClient(create_app())
    response = client.post("/api/session/matrix", json={"path": str(path)})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "npz" in detail
    assert str(path) in detail


def test_directory_path_is_400(tmp_path) -> None:
    """A directory exists, so it is not a 404; it is a bad request."""
    store.clear()
    directory = tmp_path / "matrices"
    directory.mkdir()

    client = TestClient(create_app())
    response = client.post("/api/session/matrix", json={"path": str(directory)})
    assert response.status_code == 400
    assert str(directory) in response.json()["detail"]


def test_empty_path_is_400() -> None:
    """``Path("")`` is ``Path(".")``; say "empty" rather than "'.' is not a file"."""
    store.clear()
    client = TestClient(create_app())
    for value in ("", "   "):
        response = client.post("/api/session/matrix", json={"path": value})
        assert response.status_code == 400, value
        assert "empty" in response.json()["detail"]


def test_corrupt_npy_is_400(tmp_path) -> None:
    """A file that is not a ``.npy`` at all still fails as a bad request."""
    store.clear()
    path = tmp_path / "not_really.npy"
    path.write_bytes(b"this is not a numpy file")

    client = TestClient(create_app())
    response = client.post("/api/session/matrix", json={"path": str(path)})
    assert response.status_code == 400
    assert str(path) in response.json()["detail"]
