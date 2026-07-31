"""Tests for target transfer decoding used by the offline compensation rebuild.

``_decode_target`` moved out of ``examples/python/rebuild_compensation_with_map.py``
into ``yoshidacomp.compensation.pipeline.decode_target``; the CLI now calls it
through :func:`~yoshidacomp.compensation.pipeline.generate_projection_image`.
"""

from __future__ import annotations

import numpy as np
import pytest

# Import smoke for the CLI itself. Once ``_decode_target`` moved into
# yoshidacomp.compensation.pipeline (see module docstring above), this file
# stopped referencing rebuild_compensation_with_map.py entirely, so its
# module-level code -- including the sys.path inserts for the repo root and
# external/GrayCode/src, and its yoshidacomp/GrayCode imports -- had zero
# test coverage. Importing it here, even unused, re-exercises that code at
# collection time so a broken import in the CLI fails the suite. Do not
# remove as "unused".
from examples.python import rebuild_compensation_with_map as rebuild  # noqa: F401
from yoshidacomp.compensation.pipeline import decode_target, srgb_to_linear


def test_decode_target_supports_direct_exact_srgb_and_simple_gamma() -> None:
    target = np.array([[[0, 128, 255]]], dtype=np.uint8)
    normalized = target.astype(np.float32) / 255.0

    direct = decode_target(target, mode="none", gamma=2.2)
    exact = decode_target(target, mode="srgb", gamma=2.2)
    gamma = decode_target(target, mode="gamma", gamma=2.2)

    assert np.allclose(direct, normalized, atol=1e-7)
    assert np.allclose(exact, srgb_to_linear(normalized), atol=1e-7)
    assert np.allclose(gamma, np.power(normalized, 2.2), atol=1e-7)
    assert not np.isclose(exact[0, 0, 1], gamma[0, 0, 1])
    assert gamma[0, 0, 0] == 0.0
    assert gamma[0, 0, 2] == 1.0


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_decode_target_rejects_invalid_gamma(value: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        decode_target(
            np.zeros((1, 1, 3), dtype=np.uint8),
            mode="gamma",
            gamma=value,
        )
