"""View registry and the coefficient-tile view."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp.config import reload_config  # noqa: E402
from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _default_projector_config():
    """Ensure the real ``config.toml`` (1920x1080 projector) is loaded.

    See ``tests/test_web_matrix.py`` for why this is necessary: the global
    config singleton can be left poisoned by unrelated tests that run
    earlier in the same session, and this file reads it indirectly via
    ``classify_space``.
    """
    reload_config(_REPO_ROOT / "config.toml")
    yield


def _load(client, tmp_path, height=1080, width=1920):
    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)[None, :]
    # 空間的に変化させる。一様だと座標の取り違えを検出できない。
    matrices[:, :, 0, 0] = xs
    matrices[:, :, 3, 1] = -1.0
    path = tmp_path / "m.npy"
    np.save(path, matrices)
    response = client.post("/api/session/matrix", json={"path": str(path)})
    return response.json()["id"]


def test_views_endpoint_lists_coef_tiles() -> None:
    store.clear()
    client = TestClient(create_app())
    body = client.get("/api/views").json()
    keys = {view["key"] for view in body["views"]}
    assert "coef_tiles" in keys
    spec = next(v for v in body["views"] if v["key"] == "coef_tiles")
    assert spec["kind"] == "scalar"
    assert spec["channels"] == 12


def test_coef_tiles_returns_float32_binary(tmp_path) -> None:
    store.clear()
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, height=6, width=8)

    response = client.get(f"/api/matrix/{matrix_id}/view/coef_tiles")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["X-Height"] == "6"
    assert response.headers["X-Width"] == "8"
    assert response.headers["X-Channels"] == "12"
    assert response.headers["X-Kind"] == "scalar"

    array = np.frombuffer(response.content, dtype="<f4").reshape(6, 8, 12)
    # チャンネル順は (i, j) の行優先: 0=in_R->out_R, 1=in_R->out_G, ... 11=bias->out_B
    assert np.allclose(array[:, :, 0], np.arange(8, dtype=np.float32)[None, :])
    assert np.allclose(array[:, :, 10], -1.0)


def test_unknown_view_is_404(tmp_path) -> None:
    store.clear()
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, height=6, width=8)
    assert client.get(f"/api/matrix/{matrix_id}/view/nosuch").status_code == 404
