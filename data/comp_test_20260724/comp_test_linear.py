"""Linear-space variant of the 4-condition compensation comparison.

Hypothesis test: the standard pipeline fits affine matrices against
sRGB-ENCODED captures, where the projector's native gamma (~2.2) and the
sRGB encode (~1/2.4) nearly cancel -- favoring the 'none' condition. Here
the matrices are re-fitted in LINEAR camera space (decoding the saved
warped fit captures, no recapture needed); with a linearized projector
the chain is then truly affine, so 'gamma'/'lut' should win.

Evaluation stays identical to the sRGB run: captured results are scored
in sRGB space against the sRGB target inside the same mask.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(r"C:\py_scripts\Yoshida-Compensation")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from external.GrayCode.src.python.interpolate_p2c import load_p2c_numpy_array
from external.GrayCode.src.python.warp_image import InpaintMethod, PixelMapWarperTorch
from src.python.camera import create_camera_backend
from src.python.color_mixing_matrix import (
    apply_inverse_gamma_correction,
    generate_projection_patterns,
)
from src.python.config import get_config, reload_config
from src.python.inverse_lut import apply_inverse_lut, load_inverse_lut
from src.python.photometric_compensation import calc_compensation_image

SCRATCH = Path(__file__).resolve().parent
CONFIG = SCRATCH / "config_display5.toml"
OUT = REPO / "data" / "comp_test_20260724"
TARGET_DIR = REPO / "data" / "comp_test_targets"

POS = (3440, 0)
SIZE = (1280, 720)
GAMMA = 2.2


def srgb_encode(linear):
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_decode(srgb):
    x = np.clip(srgb, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def make_linearizers():
    lut_ch = load_inverse_lut(REPO / "data" / "inverse_lut.npz")
    lut_gray = load_inverse_lut(REPO / "data" / "inverse_lut_gray.npz")
    return {
        "none": lambda img: img,
        "gamma": lambda img: np.asarray(
            apply_inverse_gamma_correction(img, gamma=1.0 / GAMMA)
        ),
        "lut_ch": lambda img: apply_inverse_lut(img, lut_ch, dtype=np.uint8),
        "lut_gray": lambda img: apply_inverse_lut(img, lut_gray, dtype=np.uint8),
    }


def fit_matrices_linear(patterns, cond: str) -> np.ndarray:
    n = len(patterns)
    h, w = SIZE[1], SIZE[0]
    caps = []
    for i in range(n):
        bgr = cv2.imread(str(OUT / f"fit_captures_{cond}" / f"warped_{i:02d}.png"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        caps.append(srgb_decode(rgb).astype(np.float32))
    cap = torch.from_numpy(np.stack(caps)).reshape(n, h * w, 3).permute(1, 0, 2)
    bias = torch.ones((h * w, n, 1), dtype=cap.dtype)
    cap = torch.cat([cap, bias], dim=2)
    proj_ref = torch.from_numpy(
        np.stack([p[0, 0, :].astype(np.float32) / 255.0 for p in patterns])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sol = torch.linalg.lstsq(
        cap.to(device), proj_ref.unsqueeze(0).expand(h * w, -1, -1).to(device)
    ).solution
    return sol.cpu().numpy().reshape(h, w, 4, 3).astype(np.float32)


def masked_metrics(result01, target01, mask):
    diff = (result01 - target01)[mask]
    mse = float(np.mean(diff**2))
    psnr = float(10 * np.log10(1.0 / mse)) if mse > 0 else float("inf")

    def ssim_map(a, b):
        c1, c2 = 0.01**2, 0.03**2
        mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
        va = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a**2
        vb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b**2
        cov = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
        return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
            (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
        )

    gray_r = cv2.cvtColor(result01.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.float64)
    gray_t = cv2.cvtColor(target01.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.float64)
    ssim = float(np.mean(ssim_map(gray_r, gray_t)[mask]))
    lab_r = cv2.cvtColor(result01.astype(np.float32), cv2.COLOR_RGB2Lab)
    lab_t = cv2.cvtColor(target01.astype(np.float32), cv2.COLOR_RGB2Lab)
    de = float(np.mean(np.linalg.norm((lab_r - lab_t)[mask], axis=-1)))
    return {"psnr": psnr, "ssim": ssim, "deltaE76": de}


def main() -> None:
    reload_config(CONFIG)
    cfg = get_config()

    linearizers = make_linearizers()
    mask = np.load(OUT / "valid_mask.npy")
    warper = PixelMapWarperTorch(
        load_p2c_numpy_array(str(REPO / "result_p2c_compensated_delaunay.npy"))
    )
    patterns = generate_projection_patterns(SIZE[0], SIZE[1], dtype=np.uint8)

    targets = []
    for p in sorted(TARGET_DIR.glob("*.jpg")) + sorted(TARGET_DIR.glob("*.png")):
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (SIZE[1], SIZE[0]):
            rgb = cv2.resize(rgb, SIZE, interpolation=cv2.INTER_AREA)
        targets.append((p.stem, rgb))

    conditions = ["none", "gamma", "lut_ch", "lut_gray"]
    matrices = {}
    for cond in conditions:
        t0 = time.monotonic()
        m = fit_matrices_linear(patterns, cond)
        matrices[cond] = m
        np.save(OUT / f"matrices_lin_{cond}.npy", m)
        print(f"[{cond}] linear-space matrices fitted offline "
              f"in {time.monotonic() - t0:.0f}s")

    results = {}
    window = "comp_test_linear"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, POS[0], POS[1])
    cv2.resizeWindow(window, SIZE[0], SIZE[1])
    cv2.waitKey(50)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(100)

    backend = create_camera_backend(cfg.camera)
    try:
        with backend:
            for cond in conditions:
                lin = linearizers[cond]
                for tname, target in targets:
                    target_linear = srgb_decode(
                        target.astype(np.float32) / 255.0
                    ).astype(np.float32)
                    comp = calc_compensation_image(
                        target_image=target_linear,
                        color_mixing_matrices=matrices[cond],
                        dtype=np.uint8,
                    )
                    proj_img = np.asarray(lin(comp), dtype=np.uint8)
                    cv2.imshow(window, cv2.cvtColor(proj_img, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(cfg.camera.wait_key_ms)
                    time.sleep(0.3)
                    frame = backend.capture_linear_rgb()
                    srgb8 = (srgb_encode(frame) * 255).round().astype(np.uint8)
                    src = torch.from_numpy(
                        srgb8.astype(np.float32) / 255.0
                    ).permute(2, 0, 1)
                    warped = (
                        warper.backward_warp(
                            src, dst_size=SIZE, inpaint=InpaintMethod.NONE
                        )
                        .permute(1, 2, 0)
                        .numpy()
                    )
                    res_dir = OUT / "results_linear" / tname
                    res_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(res_dir / f"captured_{cond}.png"),
                        cv2.cvtColor(
                            (np.clip(warped, 0, 1) * 255).astype(np.uint8),
                            cv2.COLOR_RGB2BGR,
                        ),
                    )
                    met = masked_metrics(
                        warped, target.astype(np.float32) / 255.0, mask
                    )
                    results[f"{cond}/{tname}"] = met
                    print(
                        f"[{cond}] {tname}: PSNR={met['psnr']:.2f}dB "
                        f"SSIM={met['ssim']:.4f} dE76={met['deltaE76']:.2f}"
                    )
                    del frame, srgb8, warped
    finally:
        cv2.destroyWindow(window)

    (OUT / "metrics_linear.json").write_text(json.dumps(results, indent=2))
    print(f"saved metrics to {OUT / 'metrics_linear.json'}")


if __name__ == "__main__":
    main()
