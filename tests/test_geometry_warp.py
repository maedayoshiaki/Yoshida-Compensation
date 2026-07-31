"""Geometry warp extraction tests (no hardware, synthetic P2C map)."""

from __future__ import annotations

import numpy as np
import pytest


def test_center_rect_centers_and_clamps() -> None:
    from yoshidacomp.geometry.warp import center_rect

    assert center_rect(100, 50, 200, 100) == (50, 25, 100, 50)
    # 画像がプロジェクタより大きい場合はクランプされる
    x, y, w, h = center_rect(400, 300, 200, 100)
    assert (x, y) == (0, 0)
    assert w <= 200 and h <= 100


def test_identity_p2c_preserves_matrix(tmp_path) -> None:
    """恒等P2C（proj座標=cam座標）なら行列がほぼそのまま返ること。"""
    from yoshidacomp.geometry.warp import warp_color_mixing_matrices_to_projector

    width, height = 8, 6
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    rows = np.stack(
        [xs.ravel(), ys.ravel(), xs.ravel(), ys.ravel()], axis=1
    ).astype(np.float32)
    map_path = tmp_path / "identity_p2c.npy"
    np.save(map_path, rows)

    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    matrices[:, :, 0, 0] = 2.0
    matrices[:, :, 3, 1] = -0.5

    warped = warp_color_mixing_matrices_to_projector(
        matrices, str(map_path), width, height, width, height, warp_method="p2c"
    )
    assert warped.shape == (height, width, 4, 3)
    assert warped[height // 2, width // 2, 0, 0] == pytest.approx(2.0, abs=1e-4)
    assert warped[height // 2, width // 2, 3, 1] == pytest.approx(-0.5, abs=1e-4)


def test_rejects_wrong_matrix_shape(tmp_path) -> None:
    from yoshidacomp.geometry.warp import warp_color_mixing_matrices_to_projector

    map_path = tmp_path / "p2c.npy"
    np.save(map_path, np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        warp_color_mixing_matrices_to_projector(
            np.zeros((4, 4, 3), dtype=np.float32), str(map_path), 4, 4, 4, 4
        )
