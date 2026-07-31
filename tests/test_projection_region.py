from types import SimpleNamespace

import numpy as np
import pytest

from yoshidacomp.compensation.color_mixing_matrix import (
    calc_color_mixing_matrices,
    generate_projection_patterns,
)
from yoshidacomp.compensation.projection_region import (
    embed_in_active_region,
    mask_to_active_region,
    resolve_active_region,
)


def _projector(**overrides):
    values = {
        "width": 8,
        "height": 6,
        "active_x": 0,
        "active_y": 0,
        "active_width": 0,
        "active_height": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_resolve_active_region_uses_full_frame_when_disabled() -> None:
    assert resolve_active_region(_projector()) == (0, 0, 8, 6)


def test_resolve_active_region_returns_configured_rectangle() -> None:
    assert resolve_active_region(
        _projector(active_x=2, active_y=1, active_width=4, active_height=3)
    ) == (2, 1, 4, 3)


def test_resolve_active_region_rejects_out_of_bounds_rectangle() -> None:
    with pytest.raises(ValueError, match="must fit"):
        resolve_active_region(
            _projector(active_x=5, active_y=1, active_width=4, active_height=3)
        )


def test_mask_to_active_region_blacks_everything_outside_rectangle() -> None:
    image = np.full((6, 8, 3), 17, dtype=np.uint8)

    masked = mask_to_active_region(image, (2, 1, 4, 3))

    expected = np.zeros_like(image)
    expected[1:4, 2:6] = 17
    assert np.array_equal(masked, expected)
    assert np.array_equal(image, np.full_like(image, 17))


def test_embed_in_active_region_resizes_and_places_target() -> None:
    image = np.full((2, 2, 3), 23, dtype=np.float32)

    embedded = embed_in_active_region(image, (8, 6), (2, 1, 4, 3))

    assert embedded.shape == (6, 8, 3)
    assert np.all(embedded[1:4, 2:6] == 23)
    assert np.count_nonzero(embedded) == 4 * 3 * 3


def test_color_matrix_fit_handles_unilluminated_pixels() -> None:
    patterns = generate_projection_patterns(
        2,
        1,
        num_divisions=2,
        dtype=np.uint8,
    )
    captures = []
    for pattern in patterns:
        frame = pattern.astype(np.float32) / 255.0
        frame[:, 0] = 0.0
        captures.append(frame)

    matrices = calc_color_mixing_matrices(
        patterns,
        captures,
        safety_margin=0.1,
        min_batch_size=1,
        use_gpu=False,
    )

    assert matrices.shape == (1, 2, 4, 3)
    assert np.isfinite(matrices).all()
    assert np.allclose(matrices[0, 1, :3], np.eye(3), atol=2e-5)
    assert np.allclose(matrices[0, 1, 3], 0.0, atol=2e-5)
