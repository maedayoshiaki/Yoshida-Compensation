"""Numpy vs. torch agreement for the inverse-gamma linearizer.

Task 1 of the target-and-lut plan (see
``.superpowers/sdd/2026-08-02-target-and-lut/task-1-brief.md``, and
``.superpowers/sdd/2026-08-02-target-and-lut/task-1-report.md`` for the
full history including a reversal): whether a pure-numpy reimplementation
of ``color_mixing_matrix.apply_inverse_gamma_correction`` can replace it in
``pipeline.py``'s ``_linearize_gamma``.

**It now does.** An earlier version of this module's docstring said it
could not, because float32 output isn't bit-for-bit identical to torch's.
That measurement was correct but the conclusion drawn from it was wrong:
uint8 -- the only dtype any real caller in this codebase passes through
``LINEARIZERS["gamma"]`` -- matches byte-for-byte, exhaustively, against
*both* of torch's own backends; and the float32 gap (1 ULP) is smaller than
the gap between torch's CPU and CUDA backends disagreeing with *each
other* on the same grid. There was never a bit-exact float32 "torch answer"
to preserve -- ``_linearize_gamma`` calls torch with the CUDA default, so
which value it produced already depended on which device happened to run
it. Separately, and decisively: the *first call* to the torch version (not
just importing it) cost ~7.9s and initialized CUDA -- verbatim the
interactive-GUI stall the brief warns a trivial elementwise power doesn't
justify. See :func:`yoshidacomp.compensation.pipeline._apply_inverse_gamma_numpy`'s
docstring for the full, corrected measurement table.

This file does three different things, deliberately kept apart:

1. Measure agreement between the two implementations (skipped if torch
   isn't installed -- these tests need torch to compare against).
2. Lock in the exponent-direction convention using the numpy version
   alone (never skipped -- this is the trap the brief calls out: passing
   ``projector.gamma`` directly, instead of its reciprocal, darkens the
   image instead of brightening it).
3. Confirm the actual wiring (``LINEARIZERS["gamma"]`` /
   ``generate_projection_image``), not just the raw function, produces
   output bit-identical to what the torch version used to produce -- so
   switching did not silently change any CLI-visible output.

The subprocess-based "does importing/calling this module pull torch"
boundary tests live in ``tests/test_package_boundaries.py`` alongside the
repo's other layering guards, not here.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Agreement measurement (needs torch to compare against; skip without it)
# ---------------------------------------------------------------------------


def test_gamma_numpy_matches_torch_byte_for_byte_on_uint8() -> None:
    """uint8 の全256値・両方向で、torch版とバイト単位で一致すること（CPU・CUDA双方）。

    This is an exhaustive sweep, not a sample -- it walks every possible
    quantization boundary a uint8 input can land on. This is the decisive
    measurement for the switch: every real caller in this codebase passes
    uint8 through ``LINEARIZERS["gamma"]``, and it matches torch exactly,
    on both backends, so the switch changes no delivered output.
    """
    torch = pytest.importorskip("torch")
    from yoshidacomp.compensation.color_mixing_matrix import (
        apply_inverse_gamma_correction,
    )
    from yoshidacomp.compensation.pipeline import _apply_inverse_gamma_numpy

    image = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    assert image.dtype == np.uint8
    assert sorted(np.unique(image).tolist()) == list(range(256))

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        for gamma in (1.0 / 2.2, 2.2):
            numpy_out = _apply_inverse_gamma_numpy(image, gamma=gamma)
            torch_out = apply_inverse_gamma_correction(image, gamma=gamma, device=device)

            assert numpy_out.dtype == np.uint8
            assert numpy_out.shape == image.shape
            np.testing.assert_array_equal(
                numpy_out,
                np.asarray(torch_out),
                err_msg=f"uint8 mismatch at gamma={gamma}, device={device}",
            )


def test_gamma_numpy_matches_torch_within_one_ulp_on_float32_in_range() -> None:
    """float32 [0, 1] の細かいグリッドで torch 版とほぼ一致すること（CPU比較）。

    Not byte-for-byte: ``np.power`` and ``torch.pow`` disagree by up to 1
    ULP (~5.96e-8 in this magnitude range) on a real, measured fraction of
    a fine grid -- 222/12003 against torch-CPU on this exact grid (the
    number this test's tolerance is set against). An ``assert_array_equal``
    here would fail permanently and for a reason that has nothing to do
    with a real bug -- see the module docstring and
    ``pipeline._apply_inverse_gamma_numpy``'s docstring for why this
    doesn't block wiring the numpy version in: torch's CUDA backend
    disagrees with torch's own CPU backend on 762/12003 of this same grid,
    a *larger* gap than this function's. The tolerance below (2e-7, roughly
    3-4x the observed max diff) is deliberately tight enough that a clamp
    bug or a wrong normalization scale (both of which move results by
    orders of magnitude more than 1 ULP) still fails this test.
    """
    pytest.importorskip("torch")
    from yoshidacomp.compensation.color_mixing_matrix import (
        apply_inverse_gamma_correction,
    )
    from yoshidacomp.compensation.pipeline import _apply_inverse_gamma_numpy

    grid = np.linspace(0.0, 1.0, 4001, dtype=np.float32)
    image = np.stack([grid, grid, grid], axis=-1).reshape(-1, 1, 3).astype(np.float32)

    numpy_out = _apply_inverse_gamma_numpy(image, gamma=1.0 / 2.2)
    torch_out = np.asarray(
        apply_inverse_gamma_correction(image, gamma=1.0 / 2.2, device="cpu")
    )

    assert numpy_out.dtype == np.float32
    np.testing.assert_allclose(numpy_out, torch_out, atol=2e-7, rtol=0)

    # Pin the fact that they are *not* identical, so this test would fail
    # loudly (rather than silently pass a stricter bound) if a future numpy
    # or torch release happens to close the gap -- worth knowing about, not
    # something a green suite should hide.
    max_diff = float(np.max(np.abs(numpy_out.astype(np.float64) - torch_out.astype(np.float64))))
    assert 0.0 < max_diff <= 2e-7, max_diff


def test_gamma_numpy_matches_torch_on_out_of_range_float32() -> None:
    """範囲外（負値・1超）でも、clampとNaNの扱いがtorch版と一致すること。

    This is exactly the case the brief warns a [0, 1]-only test would
    hide: a missing ``np.clip`` would leave values above 1 or (after a
    fractional-exponent NaN) unclamped, and this is the only place that
    would show up.
    """
    pytest.importorskip("torch")
    from yoshidacomp.compensation.color_mixing_matrix import (
        apply_inverse_gamma_correction,
    )
    from yoshidacomp.compensation.pipeline import _apply_inverse_gamma_numpy

    grid = np.linspace(-0.5, 1.5, 4001, dtype=np.float32)
    image = np.stack([grid, grid, grid], axis=-1).reshape(-1, 1, 3).astype(np.float32)
    assert np.any(grid < 0.0) and np.any(grid > 1.0)

    numpy_out = _apply_inverse_gamma_numpy(image, gamma=1.0 / 2.2)
    torch_out = np.asarray(
        apply_inverse_gamma_correction(image, gamma=1.0 / 2.2, device="cpu")
    )

    numpy_nan = np.isnan(numpy_out)
    torch_nan = np.isnan(torch_out)
    np.testing.assert_array_equal(numpy_nan, torch_nan)
    assert numpy_nan.any(), "expected negative bases to produce NaN in both"

    finite = ~numpy_nan
    np.testing.assert_allclose(
        numpy_out[finite], torch_out[finite], atol=2e-7, rtol=0
    )

    # Every value fed through gamma=1/2.2 (a fractional, positive exponent)
    # that is >= 0 and raised above 1 must have been clamped back to 1.0.
    above_one_inputs = grid >= 1.0
    assert np.all(numpy_out.reshape(-1, 3)[above_one_inputs] <= 1.0)
    assert np.all(numpy_out.reshape(-1, 3)[above_one_inputs] == 1.0)


# ---------------------------------------------------------------------------
# 2. Direction convention -- numpy-only, must run even without torch
# ---------------------------------------------------------------------------


def test_gamma_direction_is_not_inverted_by_the_numpy_version() -> None:
    """gamma=1/2.2 は明るく、gamma=2.2 はそのまま渡すと暗くなること。

    This fixes the documented trap: ``apply_inverse_gamma_correction`` (and
    its numpy counterpart) use *gamma* directly as the exponent, so calling
    it with ``projector.gamma`` instead of ``1 / projector.gamma`` silently
    darkens the image instead of brightening it. This does not need torch
    at all -- it is a property of ``_apply_inverse_gamma_numpy`` alone --
    so it is not skipped when torch is unavailable.
    """
    from yoshidacomp.compensation.pipeline import _apply_inverse_gamma_numpy

    mid = np.full((1, 1, 3), 64, dtype=np.uint8)

    brightened = _apply_inverse_gamma_numpy(mid, gamma=1.0 / 2.2)
    darkened = _apply_inverse_gamma_numpy(mid, gamma=2.2)

    assert int(brightened[0, 0, 0]) > 64, "1/gamma must brighten a midtone"
    assert int(darkened[0, 0, 0]) < 64, "gamma (uninverted) must darken a midtone"
    assert int(brightened[0, 0, 0]) != int(darkened[0, 0, 0])


# ---------------------------------------------------------------------------
# 3. Wiring: LINEARIZERS["gamma"] / generate_projection_image, bit-equality
# ---------------------------------------------------------------------------


def test_linearizers_gamma_matches_torch_byte_for_byte() -> None:
    """``LINEARIZERS["gamma"]``（実際の呼び出し経路）が torch 版とバイト一致すること。

    The earlier tests in this file compare ``_apply_inverse_gamma_numpy``
    directly; this one goes through ``_linearize_gamma`` /
    ``LINEARIZERS["gamma"]`` -- the actual call path every CLI script and
    ``generate_projection_image`` use -- to prove the *wiring*, not just
    the underlying function, is correct: a plausible bug here would be
    forgetting to pass ``gamma=1.0 / projector.gamma`` (using
    ``projector.gamma`` directly) at the call site, which
    ``test_gamma_direction_is_not_inverted_by_the_numpy_version`` above
    only checks against the raw function, not this wiring.
    """
    torch = pytest.importorskip("torch")
    from dataclasses import dataclass

    from yoshidacomp.compensation.color_mixing_matrix import (
        apply_inverse_gamma_correction,
    )
    from yoshidacomp.compensation.pipeline import LINEARIZERS

    @dataclass(frozen=True)
    class Projector:
        gamma: float = 2.2

    image = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    wired_out = LINEARIZERS["gamma"](image, projector=Projector(), lut=None)
    reference = apply_inverse_gamma_correction(
        image, gamma=1.0 / Projector().gamma, device="cpu"
    )

    assert wired_out.dtype == np.uint8
    np.testing.assert_array_equal(wired_out, np.asarray(reference))


def test_generate_projection_image_gamma_output_is_byte_identical_to_torch() -> None:
    """CLI が実際に使う経路 (``generate_projection_image``, linearizer="gamma",
    dtype=uint8) の出力が、torch を直接呼んだ場合とバイト単位で一致すること。

    This is the end-to-end check the CLI scripts' output depends on:
    ``examples/python/sample.py`` and
    ``examples/python/rebuild_compensation_with_map.py`` both go through
    exactly this function with ``linearizer="gamma"`` and the default
    ``dtype=np.uint8``. A diverse (not flat/symmetric) target and matrix
    are used so the compensation values feeding into the gamma step cover
    many different code points, not just one.
    """
    torch = pytest.importorskip("torch")
    from dataclasses import dataclass

    from yoshidacomp.compensation.color_mixing_matrix import (
        apply_inverse_gamma_correction,
    )
    from yoshidacomp.compensation.pipeline import generate_projection_image
    from yoshidacomp.compensation.photometric_compensation import (
        calc_compensation_image,
    )

    @dataclass(frozen=True)
    class Projector:
        width: int = 16
        height: int = 16
        gamma: float = 2.2

    rng = np.random.default_rng(20260802)
    target = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    # A non-trivial, asymmetric matrix: each output channel is a distinct mix
    # of R/G/B plus a per-channel bias, so target diversity survives into
    # compensation diversity instead of collapsing onto one value.
    matrices = np.zeros((16, 16, 4, 3), dtype=np.float32)
    matrices[:, :, 0, 0] = 0.9
    matrices[:, :, 1, 0] = 0.05
    matrices[:, :, 1, 1] = 0.85
    matrices[:, :, 2, 1] = 0.1
    matrices[:, :, 2, 2] = 0.95
    matrices[:, :, 0, 2] = 0.05
    matrices[:, :, 3, 0] = 0.02
    matrices[:, :, 3, 1] = -0.01
    matrices[:, :, 3, 2] = 0.01

    _, wired_codes = generate_projection_image(
        target,
        matrices,
        projector=Projector(),
        active_region=(0, 0, 16, 16),
        linearizer="gamma",
        target_decode="none",
    )

    # Independently reconstruct what the pre-switch torch path would have
    # produced: same compensation step, then the torch gamma correction
    # directly (device="cpu" -- uint8 already measured identical on CUDA
    # too, see test_gamma_numpy_matches_torch_byte_for_byte_on_uint8).
    compensation = calc_compensation_image(
        target_image=target.astype(np.float32) / 255.0,
        color_mixing_matrices=matrices,
        dtype=np.uint8,
        clip=True,
    )
    from yoshidacomp.compensation.projection_region import mask_to_active_region

    compensation = mask_to_active_region(compensation, (0, 0, 16, 16))
    reference_codes = np.asarray(
        apply_inverse_gamma_correction(
            compensation, gamma=1.0 / Projector().gamma, device="cpu"
        )
    )

    assert wired_codes.dtype == np.uint8
    # Sanity: the target/matrix combination above must actually exercise a
    # range of compensation values, not collapse to one -- otherwise this
    # test would pass even with a badly broken implementation.
    assert len(np.unique(compensation)) > 10
    np.testing.assert_array_equal(wired_codes, reference_codes)
