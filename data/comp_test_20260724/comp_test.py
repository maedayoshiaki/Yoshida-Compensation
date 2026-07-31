"""4-condition photometric compensation comparison.

Conditions (projector linearization applied to BOTH the matrix-fitting
patterns and the final compensation image):
    none     - no correction
    gamma    - analytic inverse gamma 2.2 (current pipeline behavior)
    lut_ch   - measured inverse LUT, channel_ramp calibration
    lut_gray - measured inverse LUT, gray calibration

Per condition:
  1. Project the 27 linear color-grid patterns through the condition's
     linearization; capture (sRGB uint8), backward-warp into projector
     space (p2c map), and fit per-pixel 4x3 matrices at 1280x720.
     Matrices are saved to data/comp_test_20260724/matrices_<cond>.npy
     and are reusable for any future 1280x720 target.
  2. For each target image: compensation = [target,1] @ M, linearize,
     project, capture, warp to projector space, and score against the
     target (masked PSNR / SSIM / CIEDE76) inside the valid-map mask.

Run from the repository root.
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
from external.GrayCode.src.python.warp_image import (
    InpaintMethod,
    PixelMapWarperTorch,
)
from external.GrayCode.src.python.config import reload_config as gc_reload_config
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
P2C_MAP = REPO / "result_p2c_compensated_delaunay.npy"

POS = (3440, 0)
SIZE = (1280, 720)  # (width, height)
GAMMA = 2.2


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


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


class ProjectorCapture:
    """One fullscreen projector window + one persistent camera session."""

    def __init__(self, cam_cfg, wait_ms: int):
        self.wait_ms = wait_ms
        self.window = "comp_test"
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

    def project_capture_srgb(self, rgb_uint8: np.ndarray) -> np.ndarray:
        cv2.imshow(self.window, cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
        cv2.waitKey(self.wait_ms)
        time.sleep(0.3)
        frame = self.backend.capture_linear_rgb()
        return (srgb_encode(frame) * 255.0).round().astype(np.uint8)


def build_warper():
    pixel_map = load_p2c_numpy_array(str(P2C_MAP))
    warper = PixelMapWarperTorch(pixel_map)
    # Valid-evaluation mask: only projector pixels that were DIRECTLY
    # decoded by graycode (the interpolated map nearest-neighbor-fills
    # ~half the frame outside the canvas, which must not be scored).
    raw = np.load(REPO / "result_p2c.npy", allow_pickle=True).item()
    valid = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    for (px, py) in raw.keys():
        x, y = int(round(px)), int(round(py))
        if 0 <= x < SIZE[0] and 0 <= y < SIZE[1]:
            valid[y, x] = 1
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.erode(valid, np.ones((11, 11), np.uint8))
    return warper, mask.astype(bool)


def warp_cam_to_proj(warper, srgb_uint8: np.ndarray) -> np.ndarray:
    src = torch.from_numpy(srgb_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
    out = warper.backward_warp(src, dst_size=SIZE, inpaint=InpaintMethod.NONE)
    return out.permute(1, 2, 0).cpu().numpy()  # float [0,1], (720, 1280, 3)


def fit_matrices(patterns, warped_captures) -> np.ndarray:
    """Per-pixel lstsq: [captured, 1] @ M = linear pattern color."""
    n = len(patterns)
    h, w = SIZE[1], SIZE[0]
    cap = torch.from_numpy(np.stack(warped_captures)).reshape(n, h * w, 3)
    cap = cap.permute(1, 0, 2)  # (HW, N, 3)
    bias = torch.ones((h * w, n, 1), dtype=cap.dtype)
    cap = torch.cat([cap, bias], dim=2)  # (HW, N, 4)
    proj_ref = torch.from_numpy(
        np.stack([p[0, 0, :].astype(np.float32) / 255.0 for p in patterns])
    )  # (N, 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sol = torch.linalg.lstsq(
        cap.to(device), proj_ref.unsqueeze(0).expand(h * w, -1, -1).to(device)
    ).solution  # (HW, 4, 3)
    return sol.cpu().numpy().reshape(h, w, 4, 3).astype(np.float32)


def masked_metrics(result01: np.ndarray, target01: np.ndarray, mask: np.ndarray):
    diff = (result01 - target01)[mask]
    mse = float(np.mean(diff**2))
    psnr = float(10 * np.log10(1.0 / mse)) if mse > 0 else float("inf")

    def ssim_map(a, b):
        c1, c2 = 0.01**2, 0.03**2
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
        va = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a**2
        vb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b**2
        cov = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
        return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
            (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
        )

    gray_r = cv2.cvtColor(result01.astype(np.float32), cv2.COLOR_RGB2GRAY)
    gray_t = cv2.cvtColor(target01.astype(np.float32), cv2.COLOR_RGB2GRAY)
    ssim = float(np.mean(ssim_map(gray_r, gray_t)[mask]))

    lab_r = cv2.cvtColor(result01.astype(np.float32), cv2.COLOR_RGB2Lab)
    lab_t = cv2.cvtColor(target01.astype(np.float32), cv2.COLOR_RGB2Lab)
    de = float(np.mean(np.linalg.norm((lab_r - lab_t)[mask], axis=-1)))
    return {"psnr": psnr, "ssim": ssim, "deltaE76": de}


def main() -> None:
    reload_config(CONFIG)
    gc_reload_config(CONFIG)
    cfg = get_config()
    OUT.mkdir(parents=True, exist_ok=True)

    linearizers = make_linearizers()
    warper, mask = build_warper()
    np.save(OUT / "valid_mask.npy", mask)
    print(f"valid mask: {mask.mean() * 100:.1f}% of projector pixels")

    patterns = generate_projection_patterns(SIZE[0], SIZE[1], dtype=np.uint8)
    print(f"{len(patterns)} fitting patterns")

    targets = []
    for p in sorted(TARGET_DIR.glob("*.jpg")) + sorted(TARGET_DIR.glob("*.png")):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (SIZE[1], SIZE[0]):
            rgb = cv2.resize(rgb, SIZE, interpolation=cv2.INTER_AREA)
        targets.append((p.stem, rgb))
    print(f"{len(targets)} target images: {[t[0] for t in targets]}")

    conditions = ["none", "gamma", "lut_ch", "lut_gray"]
    matrices = {}
    results = {}

    with ProjectorCapture(cfg.camera, cfg.camera.wait_key_ms) as pc:
        # Stage 1: fit matrices per condition.
        for cond in conditions:
            lin = linearizers[cond]
            warped_caps = []
            t0 = time.monotonic()
            for i, pat in enumerate(patterns):
                cap = pc.project_capture_srgb(lin(pat))
                warped_caps.append(warp_cam_to_proj(warper, cap))
                del cap
            m = fit_matrices(patterns, warped_caps)
            matrices[cond] = m
            np.save(OUT / f"matrices_{cond}.npy", m)
            cap_dir = OUT / f"fit_captures_{cond}"
            cap_dir.mkdir(exist_ok=True)
            for i, wc in enumerate(warped_caps):
                cv2.imwrite(
                    str(cap_dir / f"warped_{i:02d}.png"),
                    cv2.cvtColor(
                        (np.clip(wc, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR,
                    ),
                )
            print(
                f"[{cond}] fitted matrices in {time.monotonic() - t0:.0f}s "
                f"-> matrices_{cond}.npy"
            )
            del warped_caps

        # Stage 2: compensate, project, capture, evaluate.
        for cond in conditions:
            lin = linearizers[cond]
            for tname, target in targets:
                comp = calc_compensation_image(
                    target_image=target,
                    color_mixing_matrices=matrices[cond],
                    dtype=np.uint8,
                )
                proj_img = np.asarray(lin(comp), dtype=np.uint8)
                cap = pc.project_capture_srgb(proj_img)
                warped = warp_cam_to_proj(warper, cap)
                res_dir = OUT / "results" / tname
                res_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(res_dir / f"comp_input_{cond}.png"),
                    cv2.cvtColor(comp, cv2.COLOR_RGB2BGR),
                )
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
                del cap, warped

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"saved metrics to {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
