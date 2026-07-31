"""Hardware-free generation pipeline tests."""

from __future__ import annotations

import numpy as np
import pytest


def test_srgb_roundtrip() -> None:
    from yoshidacomp.compensation.pipeline import linear_to_srgb, srgb_to_linear

    values = np.array([0.0, 0.02, 0.5, 1.0], dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(values)), values, atol=1e-5)


def test_decode_target_modes() -> None:
    from yoshidacomp.compensation.pipeline import decode_target

    codes = np.full((1, 1, 3), 128, dtype=np.uint8)
    assert np.allclose(decode_target(codes, mode="none"), 128 / 255.0, atol=1e-6)
    assert decode_target(codes, mode="srgb")[0, 0, 0] < 128 / 255.0
    assert np.allclose(
        decode_target(codes, mode="gamma", gamma=2.0), (128 / 255.0) ** 2.0, atol=1e-6
    )
    with pytest.raises(ValueError):
        decode_target(codes, mode="gamma", gamma=0.0)


def test_linearizers_registered() -> None:
    from yoshidacomp.compensation.pipeline import LINEARIZERS

    assert set(LINEARIZERS) == {"none", "gamma", "inverse_lut"}


def test_generate_projection_image_identity_matrix() -> None:
    """恒等行列 + linearizer=none なら、投影画像は線形化したターゲットに一致する。"""
    from dataclasses import dataclass

    from yoshidacomp.compensation.pipeline import generate_projection_image

    @dataclass(frozen=True)
    class Projector:
        width: int = 4
        height: int = 4
        gamma: float = 2.2

    matrices = np.zeros((4, 4, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 1.0

    target = np.full((4, 4, 3), 255, dtype=np.uint8)
    linear, projected = generate_projection_image(
        target,
        matrices,
        projector=Projector(),
        active_region=(0, 0, 4, 4),
        linearizer="none",
        target_decode="srgb",
    )
    assert linear.shape == (4, 4, 3)
    assert projected.shape == (4, 4, 3)
    # 白のsRGBは線形1.0 → 恒等行列 → 255
    assert int(linear[0, 0, 0]) == 255
    assert np.array_equal(linear, projected)  # linearizer="none" は素通し
