"""Uncompensated baseline: project each target AS-IS (no matrices, no
linearization) into the projection region and evaluate identically to the
compensation conditions. Gives the reference score without compensation.
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
from src.python.config import get_config, reload_config

SCRATCH = Path(__file__).resolve().parent
OUT = REPO / "data" / "linear_pipeline_1080"
TARGET_DIR = REPO / "data" / "comp_test_targets_square"
POS = (3440, 0)
SIZE = (1920, 1080)
TARGET_SIZE = 512


def srgb_encode(linear):
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_decode(srgb):
    x = np.clip(srgb, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def metrics_pair(result01, target01):
    diff = result01 - target01
    mse = float(np.mean(diff**2))
    psnr = float(10 * np.log10(1.0 / mse)) if mse > 0 else float("inf")

    def ssim_mean(a, b):
        c1, c2 = 0.01**2, 0.03**2
        mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
        va = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a**2
        vb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b**2
        cov = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
        s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
            (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
        )
        return float(np.mean(s))

    gray_r = cv2.cvtColor(result01.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.float64)
    gray_t = cv2.cvtColor(target01.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.float64)
    return psnr, ssim_mean(gray_r, gray_t)


def main() -> None:
    reload_config(SCRATCH / "config_display5.toml")
    cfg = get_config()
    rx, ry, rs = json.loads((OUT / "region.json").read_text())["region"]

    warper = PixelMapWarperTorch(
        load_p2c_numpy_array(str(REPO / "result_p2c_compensated_delaunay.npy"))
    )

    win = "uncompensated"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win, POS[0], POS[1])
    cv2.resizeWindow(win, SIZE[0], SIZE[1])
    cv2.waitKey(50)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    results = {}
    backend = create_camera_backend(cfg.camera)
    try:
        with backend:
            for p in sorted(TARGET_DIR.glob("*.png")):
                target = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
                frame[ry : ry + rs, rx : rx + rs] = cv2.resize(
                    target, (rs, rs), interpolation=cv2.INTER_AREA
                )
                cv2.imshow(win, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(cfg.camera.wait_key_ms)
                time.sleep(0.3)
                cam = backend.capture_linear_rgb()
                src = torch.from_numpy(
                    np.ascontiguousarray(cam, dtype=np.float32)
                ).permute(2, 0, 1)
                warped = warper.backward_warp(
                    src, dst_size=SIZE, inpaint=InpaintMethod.NONE
                ).permute(1, 2, 0).numpy()
                square_lin = cv2.resize(
                    np.clip(warped[ry : ry + rs, rx : rx + rs], 0, 1),
                    (TARGET_SIZE, TARGET_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
                res_srgb = srgb_encode(square_lin).astype(np.float32)
                target01 = target.astype(np.float32) / 255.0
                psnr_s, ssim_s = metrics_pair(res_srgb, target01)
                target_lin = srgb_decode(target01)
                psnr_l, _ = metrics_pair(square_lin.astype(np.float32),
                                         target_lin.astype(np.float32))
                lab_r = cv2.cvtColor(res_srgb, cv2.COLOR_RGB2Lab)
                lab_t = cv2.cvtColor(target01.astype(np.float32), cv2.COLOR_RGB2Lab)
                de = float(np.mean(np.linalg.norm(lab_r - lab_t, axis=-1)))
                res_dir = OUT / "final_results" / p.stem
                res_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(res_dir / "square_uncompensated.png"),
                    cv2.cvtColor((res_srgb * 255).round().astype(np.uint8),
                                 cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(res_dir / "proj_input_uncompensated.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                )
                results[p.stem] = {
                    "psnr_srgb": psnr_s, "ssim_srgb": ssim_s,
                    "deltaE76": de, "psnr_linear": psnr_l,
                }
                print(f"[uncompensated] {p.stem}: PSNR_srgb={psnr_s:.2f}dB "
                      f"SSIM_srgb={ssim_s:.4f} dE76={de:.2f}")
                del cam, warped
    finally:
        cv2.destroyWindow(win)

    (OUT / "final_results" / "metrics_uncompensated.json").write_text(
        json.dumps(results, indent=2)
    )
    print("done")


if __name__ == "__main__":
    main()
