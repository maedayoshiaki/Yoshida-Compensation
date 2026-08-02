"""Shared torch reference implementation for Task 1's numpy-vs-torch tests.

``torch_calc_compensation_image`` below is the pre-switch
``calc_compensation_image`` (deleted from production in commit
``de1ad61``), pinned to an explicit *device* instead of auto-selecting
CUDA, so a test can compare numpy's output against a specific, reproducible
torch backend.

This is deliberately the **only** place this algorithm survives outside git
history. Any test that needs an independent torch reference for the
compensation matmul imports it from here rather than keeping its own copy,
so there is exactly one copy to keep in sync if ``calc_compensation_image``'s
semantics ever change, and no risk of two tests silently drifting apart
(or of a "reference" that quietly starts calling the same numpy
implementation it is supposed to be checking -- see
``.superpowers/sdd/2026-08-02-target-and-lut/task-1-report.md``, "I1", for
exactly that failure mode and why this module exists to prevent a repeat).

Not named ``test_*.py`` on purpose: pytest would otherwise try to collect it
as a test module. Importable as a plain top-level module (``import
_torch_reference``) because ``tests/`` has no ``__init__.py`` and pytest's
conftest-driven bootstrap puts the directory on ``sys.path``, the same
mechanism every other test module in this directory already relies on.

このファイルは唯一の生き残ったtorch参照実装である。他のテストファイルは
それぞれ独自のコピーを持たずここからimportする。
"""

from __future__ import annotations

import numpy as np


def torch_calc_compensation_image(
    target_image: np.ndarray,
    color_mixing_matrices: np.ndarray,
    device: str,
    dtype: np.typing.DTypeLike = np.float32,
    clip: bool = True,
) -> np.ndarray:
    """Same computation as the pre-switch ``calc_compensation_image``, pinned
    to *device* (bypassing its old auto-select) so CPU and CUDA can be
    compared side by side against the same inputs, and so a test's
    "reference" side is provably independent of whatever
    ``calc_compensation_image`` currently does.
    """
    import torch

    height, width, _ = target_image.shape
    target_dtype = target_image.dtype
    if target_dtype == np.uint8:
        norm_target_image = target_image.astype(np.float32) / 255.0
    elif target_dtype == np.uint16:
        norm_target_image = target_image.astype(np.float32) / 65535.0
    else:
        norm_target_image = target_image.astype(np.float32)

    dev = torch.device(device)
    norm_target_tensor = torch.from_numpy(norm_target_image).float().to(dev)
    color_mixing_matrices_tensor = (
        torch.from_numpy(np.ascontiguousarray(color_mixing_matrices))
        .float()
        .reshape(-1, color_mixing_matrices.shape[2], color_mixing_matrices.shape[3])
        .to(dev)
    )
    num_pixels = height * width
    norm_target_tensor = norm_target_tensor.reshape(num_pixels, 3)
    bias = torch.ones((num_pixels, 1), device=dev, dtype=norm_target_tensor.dtype)
    norm_target_tensor = torch.cat([norm_target_tensor, bias], dim=1)

    compensation_tensor = torch.bmm(
        norm_target_tensor.unsqueeze(1), color_mixing_matrices_tensor
    ).squeeze(1)

    if clip:
        compensation_tensor = torch.clamp(compensation_tensor, 0.0, 1.0)
    compensation_image = compensation_tensor.reshape(height, width, 3).cpu().numpy()

    if dtype == np.uint8:
        compensation_image = compensation_image * 255.0
        compensation_image = np.floor(compensation_image + 0.5).astype(dtype)
    elif dtype == np.uint16:
        compensation_image = compensation_image * 65535.0
        compensation_image = np.floor(compensation_image + 0.5).astype(dtype)
    else:
        compensation_image = compensation_image.astype(dtype)

    return compensation_image
