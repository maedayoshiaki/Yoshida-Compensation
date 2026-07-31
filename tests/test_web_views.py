"""View registry and the coefficient-tile view."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp.config import reload_config  # noqa: E402
from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store  # noqa: E402

# The real config.toml (1920x1080 projector) is restored before/after every
# test by the autouse fixture in tests/conftest.py. This file still imports
# ``reload_config`` itself: the active-region mask test below
# (test_coef_tiles_masks_outside_active_region) loads its own tiny stand-in
# projector config mid-test, which conftest's fixture then cleans up
# afterwards.
# 実 config.toml（1920x1080プロジェクタ）は tests/conftest.py の autouse
# フィクスチャが毎テストの前後で復元する。本ファイルが ``reload_config``
# を引き続き import しているのは、下記の有効領域マスクのテストが
# テスト内で小さな代用プロジェクタconfigを自前で読み込むため
# （後始末は conftest 側のフィクスチャが行う）。


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


def test_coef_tiles_masks_outside_active_region(tmp_path) -> None:
    """The NaN mask must honour ``(x, y, w, h)`` as-is, not a transposed read.

    Every other test matrix in this file is 6x8, which ``classify_space``
    puts in "camera" space, so ``active_region`` is ``None`` and the masking
    branch in ``coef_tiles._render`` never runs -- a transposed ``x``/``y``
    (or ``w``/``h``) in ``mask[y : y + h, x : x + w]``, or a dropped
    ``.copy()`` before ``data[~mask] = np.nan``, would go undetected.

    This test loads a small stand-in projector config (rather than
    allocating a real 1920x1080x12 float array) so the matrix classifies as
    "projector" and ``active_region`` is set. The frame is non-square
    (height != width) and the active rectangle sits off-centre near a
    corner with unequal width/height, so a row/column transposition lands
    on a disjoint set of pixels: the full finite/NaN mask is compared
    against the expected rectangle with ``np.array_equal``, and two single
    points are checked explicitly -- one inside the true rectangle, one
    inside where a swapped reading would have placed it instead.

    既存の行列はすべて6x8で "camera" 空間に分類され、``active_region`` は
    ``None`` のままマスク分岐が一度も実行されない。``x``/``y``（あるいは
    ``w``/``h``）の取り違えや ``.copy()`` の欠落は検出されない。実寸の
    1920x1080x12 floatを確保する代わりに、小さな代用プロジェクタconfigを
    読み込んで "projector" 空間に分類させる。縦横比が非対称かつ矩形も
    非対称・非中央に置くことで、行列転置バグが別の画素集合に着地し、
    このテストが確実に失敗するようにしている。
    """
    height, width = 6, 10
    x, y, w, h = 3, 1, 4, 2  # active rect occupies rows 1-2, cols 3-6.

    config_path = tmp_path / "stand_in_config.toml"
    config_path.write_text(
        f"""
[projector]
width = {width}
height = {height}
active_x = {x}
active_y = {y}
active_width = {w}
active_height = {h}
""",
        encoding="utf-8",
    )
    reload_config(config_path)

    store.clear()
    client = TestClient(create_app())
    matrix_id = _load(client, tmp_path, height=height, width=width)

    response = client.get(f"/api/matrix/{matrix_id}/view/coef_tiles")
    assert response.status_code == 200

    data = np.frombuffer(response.content, dtype="<f4").reshape(height, width, 12)
    finite_mask = np.all(np.isfinite(data), axis=2)

    expected_mask = np.zeros((height, width), dtype=bool)
    expected_mask[y : y + h, x : x + w] = True
    assert np.array_equal(finite_mask, expected_mask), (
        "active-region mask does not match (x, y, w, h) as configured -- "
        "check for a transposed x/y or w/h in the mask slice"
    )

    # A point inside the true rectangle (row=y, col=x) must be finite; the
    # point a transposed reading (row=x, col=y) would have kept instead must
    # be NaN, since x != y and w != h here.
    assert np.all(np.isfinite(data[y, x]))
    assert np.all(np.isnan(data[x, y]))


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
