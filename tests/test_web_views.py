"""View registry and the coefficient-tile view."""

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


def test_bad_number_param_is_400_not_500(tmp_path) -> None:
    """A malformed ``number`` query param must 400, not crash into a 500.

    Registers a throwaway view with a ``number`` param rather than adding a
    real view — no shipped view uses ``type="number"`` today.
    """
    from yoshidacomp_web.views.base import REGISTRY, ParamSpec, ViewResult, ViewSpec, register

    def _render(context, params) -> ViewResult:
        height, width = context.array.shape[0], context.array.shape[1]
        return ViewResult(
            data=np.zeros((height, width, 1), dtype=np.float32),
            vmin=0.0,
            vmax=1.0,
            labels=["x"],
        )

    spec = ViewSpec(
        key="_test_number_param",
        label="test-only",
        kind="scalar",
        channels=1,
        params=[ParamSpec(name="scale", label="scale", type="number", default=1.0)],
        render=_render,
    )
    register(spec)
    try:
        store.clear()
        client = TestClient(create_app())
        matrix_id = _load(client, tmp_path, height=6, width=8)

        response = client.get(
            f"/api/matrix/{matrix_id}/view/_test_number_param",
            params={"scale": "abc"},
        )
        assert response.status_code == 400
        assert "scale" in response.json()["detail"]
        assert "abc" in response.json()["detail"]
    finally:
        REGISTRY.pop(spec.key, None)
