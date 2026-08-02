"""Numpy vs. torch agreement for the inverse-gamma linearizer.

Task 1 of the target-and-lut plan (see
``.superpowers/sdd/2026-08-02-target-and-lut/task-1-brief.md``): decide,
by measurement, whether a pure-numpy reimplementation of
``color_mixing_matrix.apply_inverse_gamma_correction`` can replace it in
``pipeline.py``'s ``_linearize_gamma``. It cannot -- float32 output does
not agree bit-for-bit (see the docstring on
``pipeline._apply_inverse_gamma_numpy`` and
``.superpowers/sdd/2026-08-02-target-and-lut/task-1-report.md`` for the
measured numbers) -- so ``_linearize_gamma`` keeps calling the torch
version (now lazily imported) and the numpy version is exposed only for
the web layer's own direct use.

This file therefore does two different things, deliberately kept apart:

1. Measure agreement between the two implementations (skipped if torch
   isn't installed -- these tests need torch to compare against).
2. Lock in the exponent-direction convention using the numpy version
   alone (never skipped -- this is the trap the brief calls out: passing
   ``projector.gamma`` directly, instead of its reciprocal, darkens the
   image instead of brightening it).
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Agreement measurement (needs torch to compare against; skip without it)
# ---------------------------------------------------------------------------


def test_gamma_numpy_matches_torch_byte_for_byte_on_uint8() -> None:
    """uint8 の全256値・両方向で、torch版とバイト単位で一致すること。

    This is an exhaustive sweep, not a sample -- it walks every possible
    quantization boundary a uint8 input can land on. Verified against the
    CPU torch backend for reproducibility outside this machine's GPU; a
    one-off exploratory check (not asserted here) additionally confirmed
    0/768 mismatches against the CUDA backend too.
    """
    pytest.importorskip("torch")
    from yoshidacomp.compensation.color_mixing_matrix import (
        apply_inverse_gamma_correction,
    )
    from yoshidacomp.compensation.pipeline import _apply_inverse_gamma_numpy

    image = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    assert image.dtype == np.uint8
    assert sorted(np.unique(image).tolist()) == list(range(256))

    for gamma in (1.0 / 2.2, 2.2):
        numpy_out = _apply_inverse_gamma_numpy(image, gamma=gamma)
        torch_out = apply_inverse_gamma_correction(image, gamma=gamma, device="cpu")

        assert numpy_out.dtype == np.uint8
        assert numpy_out.shape == image.shape
        np.testing.assert_array_equal(
            numpy_out,
            np.asarray(torch_out),
            err_msg=f"uint8 mismatch at gamma={gamma}",
        )


def test_gamma_numpy_matches_torch_within_one_ulp_on_float32_in_range() -> None:
    """float32 [0, 1] の細かいグリッドで torch 版とほぼ一致すること。

    Not byte-for-byte: ``np.power`` and ``torch.pow`` disagree by up to 1
    ULP (~5.96e-8 in this magnitude range) on a real, measured fraction of
    a fine grid (774/12003 samples in the exploratory sweep). An
    ``assert_array_equal`` here would fail permanently and for a reason
    that has nothing to do with a real bug -- see the module docstring.
    The tolerance below (2e-7, roughly 3-4x the observed max diff) is deliberately
    tight enough that a clamp bug or a wrong normalization scale (both of
    which move results by orders of magnitude more than 1 ULP) still fails
    this test.
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
    # or torch release happens to close the gap -- that would be worth
    # revisiting the "do not switch" decision over, not something a green
    # suite should hide.
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
# 3. Import boundary
# ---------------------------------------------------------------------------


def test_importing_pipeline_does_not_pull_torch() -> None:
    """``pipeline`` の import だけでは torch / psutil / color_mixing_matrix /
    scipy を引かないこと。

    This started out weaker -- only asserting ``color_mixing_matrix`` and
    ``psutil`` were absent -- because ``yoshidacomp.geometry.warp`` (also
    imported by this module, for ``prepare_target_image_for_compensation()``)
    used to do an unconditional ``import torch`` at its own module scope,
    which made the blanket "no torch" claim genuinely false regardless of
    what this task did to ``pipeline.py``. Measured before that was fixed:
    importing ``pipeline`` took ~9.4-13.0s wall time (three trials) and put
    ``torch`` in ``sys.modules``.

    That was fixed *in ``warp.py`` itself*: its two ``import torch`` use
    sites (inside ``warp_camera_array_to_projector`` and ``invwarp_image``)
    are local imports, following the exact pattern that module's own
    docstring already used for the ``graycode`` git submodule. Because the
    fix lives inside those function bodies, it changes no module-level
    attribute of ``warp`` or of ``pipeline`` -- ``tests/test_sample_mocked.py``'s
    ``_patch_warp_helper``, which monkeypatches
    ``warp_image``/``invwarp_image`` as attributes on the ``pipeline``
    module (~10 call sites), keeps working untouched, since ``pipeline.py``
    still imports those two names at its own module scope from ``warp``
    (only ``warp``'s *internal* torch import moved). That got the wall time
    to ~2.0s (three trials, versus ~9.4-13.0s before).

    A profile of that remaining ~2.0s showed 73% of it (~1.5s) was
    ``scipy.interpolate``, pulled in by this module's own
    ``from scipy.interpolate import PchipInterpolator`` at module scope.
    Checked before moving it: ``PchipInterpolator`` has exactly one call
    site in the whole file, inside ``_build_channel_lut`` (LUT
    *construction*, called only from ``build_inverse_lut``, which
    ``pipeline.py`` does not even import). ``apply_inverse_lut`` -- the
    function every ``LINEARIZERS["inverse_lut"]`` call actually goes
    through -- uses only ``np.interp``. So the same treatment applied:
    the import moved into ``_build_channel_lut``. Measured after: ~0.55s
    (three trials), with ``scipy.interpolate`` absent from ``sys.modules``.

    ``yoshidacomp.geometry.warp`` and ``cv2`` are both still imported (the
    former because ``pipeline.py`` still imports
    ``WarpMethod``/``invwarp_image``/``warp_image`` from it at module
    scope; the latter is an existing, accepted dependency of the view
    layer), but neither drags torch, psutil, or scipy along with it
    anymore.

    Run in a subprocess: in-process, some other test collected earlier in
    the same session may have already imported these modules, which would
    make the assertion pass regardless of whether ``pipeline.py`` (or
    ``warp.py``/``inverse_lut.py``) is actually responsible.
    """
    import subprocess
    import sys

    code = (
        "import sys; import yoshidacomp.compensation.pipeline; "
        "mods = sys.modules; "
        "print('color_mixing_matrix=%s' % any('color_mixing_matrix' in m for m in mods)); "
        "print('psutil=%s' % ('psutil' in mods)); "
        "print('geometry.warp=%s' % ('yoshidacomp.geometry.warp' in mods)); "
        "print('torch=%s' % ('torch' in mods)); "
        "print('scipy=%s' % any('scipy' in m for m in mods)); "
        "print('cv2=%s' % ('cv2' in mods))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    reported = dict(line.split("=", 1) for line in out.stdout.strip().splitlines())

    assert reported["color_mixing_matrix"] == "False", out.stdout
    assert reported["psutil"] == "False", out.stdout
    assert reported["torch"] == "False", out.stdout
    assert reported["scipy"] == "False", out.stdout
    # geometry.warp and cv2 are legitimately still imported -- pipeline.py
    # needs WarpMethod/invwarp_image/warp_image from warp, and cv2 is an
    # existing, accepted dependency -- but none of them may bring torch,
    # psutil, or scipy along for the ride.
    assert reported["geometry.warp"] == "True", out.stdout
    assert reported["cv2"] == "True", out.stdout
