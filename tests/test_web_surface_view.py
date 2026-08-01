"""The forward-model view: what the camera sees for a given projector command."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store  # noqa: E402


def _load(client, tmp_path, matrices):
    path = tmp_path / "m.npy"
    np.save(path, matrices)
    return client.post("/api/session/matrix", json={"path": str(path)}).json()["id"]


def test_surface_inverts_a_known_affine(tmp_path, stand_in_projector_config) -> None:
    """M が [C,1]M = P を満たすとき、surface は C = (P - b)A^-1 を返す。

    A = 2I, b = (0.1, 0.2, 0.3) を置くと、P=(1,1,1) に対する答えは
    ((1-0.1)/2, (1-0.2)/2, (1-0.3)/2) = (0.45, 0.40, 0.35) と手計算できる。
    """
    store.clear()
    stand_in_projector_config(width=5, height=4)
    client = TestClient(create_app())

    matrices = np.zeros((4, 5, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 2.0
    matrices[:, :, 3, :] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    matrix_id = _load(client, tmp_path, matrices)

    response = client.get(f"/api/matrix/{matrix_id}/view/surface")
    assert response.status_code == 200
    assert response.headers["X-Kind"] == "rgb"
    assert response.headers["X-Channels"] == "3"
    assert response.headers["X-Layout"] == "planar"

    planar = np.frombuffer(response.content, dtype="<f4").reshape(3, 4, 5)
    np.testing.assert_allclose(planar[0], 0.45, atol=1e-5)
    np.testing.assert_allclose(planar[1], 0.40, atol=1e-5)
    np.testing.assert_allclose(planar[2], 0.35, atol=1e-5)


def test_surface_honours_the_projection_command(tmp_path, stand_in_projector_config) -> None:
    """P=(0,0,0) は環境光のみの応答 -b·A^-1 を返す。"""
    store.clear()
    stand_in_projector_config(width=5, height=4)
    client = TestClient(create_app())

    matrices = np.zeros((4, 5, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 2.0
    matrices[:, :, 3, :] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    matrix_id = _load(client, tmp_path, matrices)

    response = client.get(
        f"/api/matrix/{matrix_id}/view/surface?p_r=0&p_g=0&p_b=0"
    )
    planar = np.frombuffer(response.content, dtype="<f4").reshape(3, 4, 5)
    np.testing.assert_allclose(planar[0], -0.05, atol=1e-5)
    np.testing.assert_allclose(planar[1], -0.10, atol=1e-5)
    np.testing.assert_allclose(planar[2], -0.15, atol=1e-5)


def test_surface_marks_singular_pixels_nan(tmp_path, stand_in_projector_config) -> None:
    """A が特異な画素は逆行列を持たないので NaN にする。

    もっともらしい値を描くとユーザーを誤らせる。NaN ならシェーダが
    マスク色で塗る。
    """
    store.clear()
    stand_in_projector_config(width=5, height=4)
    client = TestClient(create_app())

    matrices = np.zeros((4, 5, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 2.0
    matrices[2, 3, :3, :] = 0.0          # この画素だけ A = 0 で特異
    matrix_id = _load(client, tmp_path, matrices)

    response = client.get(f"/api/matrix/{matrix_id}/view/surface")
    planar = np.frombuffer(response.content, dtype="<f4").reshape(3, 4, 5)
    assert np.isnan(planar[:, 2, 3]).all()
    assert np.isfinite(planar[:, 0, 0]).all()


def test_surface_is_listed_with_its_params() -> None:
    store.clear()
    client = TestClient(create_app())
    views = client.get("/api/views").json()["views"]
    spec = next(v for v in views if v["key"] == "surface")
    assert spec["kind"] == "rgb"
    assert spec["channels"] == 3
    assert {p["name"] for p in spec["params"]} == {"p_r", "p_g", "p_b"}
    assert all(p["type"] == "number" for p in spec["params"])
