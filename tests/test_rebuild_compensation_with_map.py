"""Tests for target transfer decoding in the offline compensation rebuild."""

from __future__ import annotations

import numpy as np
import pytest

from examples.python import rebuild_compensation_with_map as rebuild
from examples.python import sample


def test_decode_target_supports_direct_exact_srgb_and_simple_gamma() -> None:
    target = np.array([[[0, 128, 255]]], dtype=np.uint8)
    normalized = target.astype(np.float32) / 255.0

    direct = rebuild._decode_target(target, mode="none", gamma=2.2)
    exact = rebuild._decode_target(target, mode="srgb", gamma=2.2)
    gamma = rebuild._decode_target(target, mode="gamma", gamma=2.2)

    assert np.allclose(direct, normalized, atol=1e-7)
    assert np.allclose(exact, sample.srgb_to_linear(normalized), atol=1e-7)
    assert np.allclose(gamma, np.power(normalized, 2.2), atol=1e-7)
    assert not np.isclose(exact[0, 0, 1], gamma[0, 0, 1])
    assert gamma[0, 0, 0] == 0.0
    assert gamma[0, 0, 2] == 1.0


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_decode_target_rejects_invalid_gamma(value: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        rebuild._decode_target(
            np.zeros((1, 1, 3), dtype=np.uint8),
            mode="gamma",
            gamma=value,
        )
