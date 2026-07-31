"""Fully linear compensation pipeline at 1920x1080 (user directive).

Design (everything linear, no sRGB anywhere in processing):
  - projector is ALWAYS linearized with the measured channel-ramp
    inverse LUT (data/inverse_lut.npz) -> behaves as a linear projector
  - camera frames stay LINEAR (raw-developed, no sRGB encode)
  - per-pixel 4x3 matrices are fitted between linear pattern values and
    linear camera values (warped to projector space)
  - targets (sRGB files) are decoded to linear before compensation
  - evaluation happens on the 512x512 square in linear space (sRGB-space
    metrics are also reported for comparability with earlier tables)

Stages:
  1. capture 27 fitting patterns (LUT-linearized), warp, fit, save
  2. derive the square projection region from the fresh decode mask
  3. evaluate the 3 square targets (lut_ch primary, gamma reference)

Run from the repository root AFTER graycode_run.py at 1920x1080.
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
OUT = REPO / "data" / "linear_pipeline_1080"
TARGET_DIR = REPO / "data" / "comp_test_targets_square"

POS = (3440, 0)
SIZE = (1920, 1080)
TARGET_SIZE = 512
GAMMA = 2.2

COVERS = [
    (-5120, 0, 1920, 1080),
    (-3200, 0, 1280, 800),
    (-1920, 0, 1920, 1200),
    (0, 0, 3440, 1440),
]


def srgb_encode(linear):
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_decode(srgb):
    x = np.clip(srgb, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def make_linearizers():
    lut_ch = load_inverse_lut(REPO / "data" / "inverse_lut.npz")
    return {
        "lut_ch": lambda img: apply_inverse_lut(img, lut_ch, dtype=np.uint8),
        "gamma": lambda img: np.asarray(
            apply_inverse_gamma_correction(img, gamma=1.0 / GAMMA)
        ),
    }


def derive_square_region(inset: int = 8):
    raw = np.load(REPO / "result_p2c.npy", allow_pickle=True).item()
    valid = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    for (px, py) in raw.keys():
        x, y = int(round(px)), int(round(py))
        if 0 <= x < SIZE[0] and 0 <= y < SIZE[1]:
            valid[y, x] = 1
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    dist = cv2.distanceTransform(valid, cv2.DIST_C, 5)
    cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
    half = int(dist[cy, cx]) - 2 - inset
    return (int(cx - half), int(cy - half), int(2 * half)), valid


class Rig:
    def __init__(self, cam_cfg, wait_ms: int):
        self.wait_ms = wait_ms
        self.names = []
        for i, (x, y, w, h) in enumerate(COVERS):
            name = f"cover_{i}"
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(name, x, y)
            cv2.resizeWindow(name, w, h)
            cv2.waitKey(30)
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(name, np.zeros((h, w, 3), np.uint8))
            self.names.append(name)
        self.window = "linear_pipeline"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window, POS[0], POS[1])
        cv2.resizeWindow(self.window, SIZE[0], SIZE[1])
        cv2.waitKey(50)
        cv2.setWindowProperty(
            self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.waitKey(100)
        self.backend = create_camera_backend(cam_cfg)

    def __enter__(self):
        self.backend.__enter__()
        return self

    def __exit__(self, *exc):
        self.backend.__exit__(*exc)
        cv2.destroyWindow(self.window)
        for name in self.names:
            cv2.destroyWindow(name)

    def project_capture_linear(self, rgb_uint8: np.ndarray) -> np.ndarray:
        cv2.imshow(self.window, cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
        cv2.waitKey(self.wait_ms)
        time.sleep(0.3)
        return self.backend.capture_linear_rgb()  # linear float, no encode


def warp_linear(warper, linear: np.ndarray) -> np.ndarray:
    src = torch.from_numpy(np.ascontiguousarray(linear, dtype=np.float32)).permute(2, 0, 1)
    out = warper.backward_warp(src, dst_size=SIZE, inpaint=InpaintMethod.NONE)
    return out.permute(1, 2, 0).numpy()


def fit_matrices(patterns, warped_caps) -> np.ndarray:
    n = len(patterns)
    h, w = SIZE[1], SIZE[0]
    cap = torch.from_numpy(np.stack(warped_caps)).reshape(n, h * w, 3).permute(1, 0, 2)
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


def evaluate(square_lin, target_srgb01):
    """Metrics in linear space (primary) and sRGB space (reference)."""
    target_lin = srgb_decode(target_srgb01)
    psnr_l, ssim_l = metrics_pair(square_lin.astype(np.float32), target_lin.astype(np.float32))
    res_srgb = srgb_encode(square_lin).astype(np.float32)
    psnr_s, ssim_s = metrics_pair(res_srgb, target_srgb01.astype(np.float32))
    lab_r = cv2.cvtColor(res_srgb, cv2.COLOR_RGB2Lab)
    lab_t = cv2.cvtColor(target_srgb01.astype(np.float32), cv2.COLOR_RGB2Lab)
    de = float(np.mean(np.linalg.norm(lab_r - lab_t, axis=-1)))
    return {
        "psnr_linear": psnr_l, "ssim_linear": ssim_l,
        "psnr_srgb": psnr_s, "ssim_srgb": ssim_s, "deltaE76": de,
    }


def main() -> None:
    reload_config(CONFIG)
    cfg = get_config()
    OUT.mkdir(parents=True, exist_ok=True)

    linearizers = make_linearizers()
    warper = PixelMapWarperTorch(
        load_p2c_numpy_array(str(REPO / "result_p2c_compensated_delaunay.npy"))
    )

    region, valid = derive_square_region()
    rx, ry, rs = region
    print(f"square projection region (x, y, side) = {region}")
    np.save(OUT / "valid_mask.npy", valid.astype(bool))
    (OUT / "region.json").write_text(json.dumps({"region": list(region)}))

    patterns = generate_projection_patterns(SIZE[0], SIZE[1], dtype=np.uint8)
    targets = []
    for p in sorted(TARGET_DIR.glob("*.png")):
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        targets.append((p.stem, rgb))
    print(f"{len(patterns)} patterns, {len(targets)} targets")

    results = {}
    with Rig(cfg.camera, cfg.camera.wait_key_ms) as rig:
        # Stage 1: fit matrices per condition (linear capture, linear fit).
        mats = {}
        for cond in ("lut_ch", "gamma"):
            lin = linearizers[cond]
            warped_caps = []
            t0 = time.monotonic()
            for pat in patterns:
                frame = rig.project_capture_linear(lin(pat))
                warped_caps.append(warp_linear(warper, frame).astype(np.float32))
                del frame
            m = fit_matrices(patterns, warped_caps)
            mats[cond] = m
            np.save(OUT / f"matrices_linear_{cond}.npy", m)
            np.save(
                OUT / f"fit_captures_linear_{cond}.npy",
                np.stack(warped_caps).astype(np.float16),
            )
            print(f"[{cond}] matrices fitted in {time.monotonic() - t0:.0f}s")
            del warped_caps

        # Stage 2: square-target evaluation.
        for cond in ("lut_ch", "gamma"):
            lin = linearizers[cond]
            for tname, target in targets:
                tgt_region01 = srgb_decode(
                    cv2.resize(target, (rs, rs), interpolation=cv2.INTER_AREA)
                    .astype(np.float32) / 255.0
                ).astype(np.float32)
                comp = calc_compensation_image(
                    target_image=tgt_region01,
                    color_mixing_matrices=mats[cond][ry : ry + rs, rx : rx + rs],
                    dtype=np.uint8,
                )
                frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
                frame[ry : ry + rs, rx : rx + rs] = comp
                proj_img = np.asarray(lin(frame), dtype=np.uint8)
                cam = rig.project_capture_linear(proj_img)
                warped = warp_linear(warper, cam)
                square_lin = cv2.resize(
                    np.clip(warped[ry : ry + rs, rx : rx + rs], 0, 1),
                    (TARGET_SIZE, TARGET_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
                res_dir = OUT / "results" / tname
                res_dir.mkdir(parents=True, exist_ok=True)
                np.save(res_dir / f"square_linear_{cond}.npy",
                        square_lin.astype(np.float16))
                cv2.imwrite(
                    str(res_dir / f"square_{cond}.png"),
                    cv2.cvtColor(
                        (srgb_encode(square_lin) * 255).round().astype(np.uint8),
                        cv2.COLOR_RGB2BGR,
                    ),
                )
                met = evaluate(square_lin, target.astype(np.float32) / 255.0)
                results[f"{cond}/{tname}"] = met
                print(
                    f"[{cond}] {tname}: PSNR_lin={met['psnr_linear']:.2f}dB "
                    f"PSNR_srgb={met['psnr_srgb']:.2f}dB "
                    f"SSIM_srgb={met['ssim_srgb']:.4f} dE76={met['deltaE76']:.2f}"
                )
                del cam, warped

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
