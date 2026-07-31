"""Apply saved compensation matrices to new target images.

Usage:
    uv run python data/comp_test_20260724/apply_compensation.py \
        <target.(png|jpg)> [condition] [fit_space]

    condition: none | gamma | lut_ch | lut_gray   (default lut_ch)
    fit_space: srgb | linear                      (default linear)

Produces `<target>_comp_<condition>_<fit_space>.png` -- the projector
input image (1280x720). Project it fullscreen at (3440, 0).

Notes:
  - Matrices were measured 2026-07-24 for the canvas/projector/camera
    geometry of that day; re-run the graycode + fitting stages if the
    physical setup changes.
  - `srgb` matrices expect the sRGB target directly; `linear` matrices
    expect the target decoded to linear (handled here automatically).
  - The condition's own linearization is applied to the compensation
    image before saving (lut_* require data/inverse_lut*.npz).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.python.color_mixing_matrix import apply_inverse_gamma_correction
from src.python.inverse_lut import apply_inverse_lut, load_inverse_lut
from src.python.photometric_compensation import calc_compensation_image

HERE = Path(__file__).resolve().parent
SIZE = (1280, 720)
GAMMA = 2.2


def srgb_decode(srgb: np.ndarray) -> np.ndarray:
    x = np.clip(srgb, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def linearize(img: np.ndarray, condition: str) -> np.ndarray:
    if condition == "none":
        return img
    if condition == "gamma":
        return np.asarray(apply_inverse_gamma_correction(img, gamma=1.0 / GAMMA))
    lut_file = "inverse_lut.npz" if condition == "lut_ch" else "inverse_lut_gray.npz"
    lut = load_inverse_lut(REPO / "data" / lut_file)
    return apply_inverse_lut(img, lut, dtype=np.uint8)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    target_path = Path(sys.argv[1])
    condition = sys.argv[2] if len(sys.argv) > 2 else "lut_ch"
    fit_space = sys.argv[3] if len(sys.argv) > 3 else "linear"
    assert condition in ("none", "gamma", "lut_ch", "lut_gray")
    assert fit_space in ("srgb", "linear")

    prefix = "matrices_lin_" if fit_space == "linear" else "matrices_"
    matrices = np.load(HERE / f"{prefix}{condition}.npy")

    rgb = cv2.cvtColor(cv2.imread(str(target_path)), cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (SIZE[1], SIZE[0]):
        rgb = cv2.resize(rgb, SIZE, interpolation=cv2.INTER_AREA)

    if fit_space == "linear":
        target = srgb_decode(rgb.astype(np.float32) / 255.0).astype(np.float32)
    else:
        target = rgb

    comp = calc_compensation_image(
        target_image=target, color_mixing_matrices=matrices, dtype=np.uint8
    )
    proj_img = np.asarray(linearize(comp, condition), dtype=np.uint8)

    out = target_path.with_name(
        f"{target_path.stem}_comp_{condition}_{fit_space}.png"
    )
    cv2.imwrite(str(out), cv2.cvtColor(proj_img, cv2.COLOR_RGB2BGR))
    print(f"saved projector input image: {out}")


if __name__ == "__main__":
    main()
