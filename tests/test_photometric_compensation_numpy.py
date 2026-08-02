"""Numpy vs. torch agreement for calc_compensation_image (Task 1, Addendum 4).

``photometric_compensation.calc_compensation_image`` used to run its
per-pixel ``[R, G, B, 1] @ M`` as a batched ``torch.bmm`` (GPU-accelerated
when CUDA is available). Measured (see
``.superpowers/sdd/2026-08-02-target-and-lut/task-1-report.md``, Addendum
4): this is 12 multiply-adds per pixel, ~24 MFLOP for a 1920x1080 frame --
not a workload GPU batching helps with -- and importing torch cost ~7.1s
of a ~7.6-7.7s first-call stall that was previously unconditional on
which linearizer ``generate_projection_image`` used. The numpy
reimplementation matches torch-CPU byte-for-byte on every uint8 input
tested, including all 6,220,800 pixel-channels of the real 1920x1080
matrix in ``data/linear_pipeline_1080/`` (that file is gitignored, so it
is not a dependency of any test here -- see the comment on
``_ASYMMETRY_RATIO`` below for how its measured properties inform the
synthetic fixtures used instead). Where it differs from torch-CUDA, the
difference is confined to pixels where torch-CUDA already differs from
torch-CPU by the same amount -- torch's own cross-device noise floor, not
something a numpy reimplementation could either introduce or avoid.
"""

from __future__ import annotations

import numpy as np
import pytest

# The real per-pixel matrices this function is actually applied to are
# strongly non-symmetric. Measured on data/linear_pipeline_1080's real
# 1920x1080 gamma matrix, within its valid (illuminated) region:
# max|offdiag| / max|diag| = 0.262. tests/test_web_surface_view.py cites the
# same real-data family at 0.251 (a different capture,
# data/comp_test_20260724/matrices_gamma.npy). Neither file is committed to
# git (*.npy is gitignored), so this ratio -- not the file -- is what the
# synthetic fixtures below reproduce.
_ASYMMETRY_RATIO = 0.25


def _realistic_matrices(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Per-pixel (H, W, 4, 3) matrices with a realistic diag/offdiag ratio.

    Every pixel gets its *own* random diagonal and off-diagonal terms (not
    one matrix broadcast over the frame) so a mismatch anywhere in a large
    grid has many independent chances to show up, the same reason
    ``test_web_surface_view.py``'s non-symmetric fixture exists.
    """
    diag = rng.uniform(0.5, 1.5, size=(height, width, 3)).astype(np.float32)
    matrices = np.zeros((height, width, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = diag[:, :, c]
    off = (rng.uniform(-1.0, 1.0, size=(height, width, 3, 3)) * _ASYMMETRY_RATIO).astype(
        np.float32
    )
    for row in range(3):
        for col in range(3):
            if row != col:
                matrices[:, :, row, col] = off[:, :, row, col] * diag[:, :, col]
    matrices[:, :, 3, :] = rng.uniform(-0.05, 0.05, size=(height, width, 3)).astype(np.float32)
    return matrices


# The torch reference implementation lives in tests/_torch_reference.py, not
# here, so it has exactly one copy to keep in sync -- see that module's
# docstring for why (a second, independently-drifting copy is exactly the
# "I1" failure mode from the task report).
from _torch_reference import torch_calc_compensation_image as _torch_calc_compensation_image


def test_calc_compensation_image_matches_torch_byte_for_byte_uint8() -> None:
    """uint8 出力が、非対称な実測相当の行列・多様なターゲット・非正方フレームで
    torch 版とバイト単位で一致すること（CPU・CUDA双方）。

    A non-square (48x64) frame with a distinct random matrix *and* a
    distinct random target color at every pixel -- 3072 pixels x 3
    channels = 9216 elements, no two evaluated on the same inputs.
    """
    torch = pytest.importorskip("torch")
    from yoshidacomp.compensation.photometric_compensation import calc_compensation_image

    rng = np.random.default_rng(20260802)
    height, width = 48, 64
    target = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    matrices = _realistic_matrices(rng, height, width)
    assert len(np.unique(target)) > 10, "fixture must not be degenerate"

    numpy_out = calc_compensation_image(
        target_image=target, color_mixing_matrices=matrices, dtype=np.uint8, clip=True
    )
    assert numpy_out.dtype == np.uint8
    assert numpy_out.shape == (height, width, 3)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        torch_out = _torch_calc_compensation_image(
            target, matrices, device, dtype=np.uint8, clip=True
        )
        np.testing.assert_array_equal(
            numpy_out, torch_out, err_msg=f"uint8 mismatch vs. torch-{device}"
        )


def test_calc_compensation_image_uint8_never_diverges_beyond_cudas_own_noise() -> None:
    """1920x1080相当の規模で、numpyとtorch-CUDAの相違はtorch-CUDAとtorch-CPUの
    自己不一致と完全に一致する画素に限られること（=それを超えないこと）。

    This is the decisive judgment-power test: it does not merely assert a
    tolerance, it asserts that every single element where numpy disagrees
    with torch-CUDA is *also* an element where torch-CUDA disagrees with
    torch-CPU, by the same amount -- i.e. numpy introduces zero divergence
    beyond torch's own cross-device inconsistency. A numpy implementation
    with a real bug (wrong element order, wrong dtype cast, an off-by-one)
    would fail this by producing mismatches at *different* positions than
    torch's own self-disagreement, or with different magnitudes.

    Frame size matches ``data/linear_pipeline_1080``'s real matrix
    (1920x1080) because torch's CPU/CUDA divergence is rare enough on real
    data (measured, on that real matrix, as 9 mismatched elements out of
    6,220,800 for one target and 2-3 for two others -- the count is a
    property of the target, not the matrix; the *set-equality* property
    this test asserts is what reproduces across targets, see
    ``photometric_compensation.calc_compensation_image``'s docstring) that a
    small grid usually finds zero disagreements at all -- which would make
    this test unable to fail in the one way that matters. Runtime is small
    (a few hundred ms of numpy plus a torch call on each backend).

    **Coverage gap on a GPU-less machine: this test skips entirely**, so
    the "identical positions" property -- the decisive one for the switch
    decision -- goes unasserted there. The exposure is narrow:
    ``test_calc_compensation_image_matches_torch_float32_for_the_record``
    still exercises the exact-vs-torch-CPU element order and cast on every
    machine (proven by mutation: reversing the summation order fails it),
    so a structural bug (wrong element order, wrong axis) would still be
    caught without CUDA -- only the CUDA-specific noise-floor property
    would not be.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available to compare against")
    from yoshidacomp.compensation.photometric_compensation import calc_compensation_image

    rng = np.random.default_rng(1080)
    height, width = 1080, 1920
    target = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    matrices = _realistic_matrices(rng, height, width)

    numpy_out = calc_compensation_image(
        target_image=target, color_mixing_matrices=matrices, dtype=np.uint8, clip=True
    )
    torch_cpu = _torch_calc_compensation_image(target, matrices, "cpu", dtype=np.uint8, clip=True)
    torch_cuda = _torch_calc_compensation_image(
        target, matrices, "cuda", dtype=np.uint8, clip=True
    )

    # numpy must be exactly identical to the deterministic CPU backend.
    np.testing.assert_array_equal(numpy_out, torch_cpu)

    diff_numpy_cuda = np.abs(numpy_out.astype(np.int64) - torch_cuda.astype(np.int64))
    diff_cuda_cpu = np.abs(torch_cuda.astype(np.int64) - torch_cpu.astype(np.int64))

    # Every element where numpy and CUDA disagree must be an element where
    # CUDA already disagreed with CPU -- same positions, same magnitude.
    # This is strictly stronger than a tolerance check: a bug that shifted
    # results at *different* pixels by the same small amount would still
    # pass a naive "max diff <= 1" test but fail this one.
    np.testing.assert_array_equal(
        diff_numpy_cuda,
        diff_cuda_cpu,
        err_msg=(
            "numpy vs. torch-CUDA divergence does not exactly match "
            "torch-CUDA vs. torch-CPU's own self-disagreement -- numpy is "
            "introducing a divergence beyond torch's cross-device noise floor"
        ),
    )


def test_calc_compensation_image_matches_torch_float32_for_the_record() -> None:
    """float32 出力の一致度を記録する（許容誤差検証、CPU厳密一致・CUDAは
    torch自己不一致の幅で判定）。

    Not the decisive dtype (no caller passes float32 through this path
    today) but recorded for completeness, the same way the gamma
    linearizer's float32 gap was recorded even though uint8 decided that
    switch.
    """
    torch = pytest.importorskip("torch")
    from yoshidacomp.compensation.photometric_compensation import calc_compensation_image

    rng = np.random.default_rng(20260802)
    height, width = 48, 64
    target = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    matrices = _realistic_matrices(rng, height, width)

    numpy_out = calc_compensation_image(
        target_image=target, color_mixing_matrices=matrices, dtype=np.float32, clip=True
    )
    torch_cpu = _torch_calc_compensation_image(
        target, matrices, "cpu", dtype=np.float32, clip=True
    )
    np.testing.assert_array_equal(numpy_out, torch_cpu)

    if torch.cuda.is_available():
        torch_cuda = _torch_calc_compensation_image(
            target, matrices, "cuda", dtype=np.float32, clip=True
        )
        # Not exact -- torch's own CUDA/CPU backends disagree with each
        # other on this same computation (measured up to ~1e-5 on the real
        # 1080p matrix; this smaller fixture's bound is looser only because
        # it is a much smaller sample of the same noise, not a different
        # phenomenon).
        np.testing.assert_allclose(numpy_out, torch_cuda, atol=5e-5, rtol=0)
