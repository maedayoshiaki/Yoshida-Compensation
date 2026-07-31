"""Square-target compensation workflow evaluation.

Workflow requested by the user:
  - targets are square images (512x512)
  - the projection region is an explicit square rect in projector space
    (default: largest inscribed square in the decoded canvas, with an
    8 px guard inset -> (416, 142, 468))
  - the captured camera frame is warped to projector space, the region
    is cropped and resized back to 512x512, and image
    processing/metrics run on that square

Evaluates the 4 linearization conditions x 2 matrix fit spaces using
the matrices saved by comp_test.py / comp_test_linear.py (no refit).

Usage:
    uv run python square_comp_test.py [x y side]
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
from src.python.color_mixing_matrix import apply_inverse_gamma_correction
from src.python.config import get_config, reload_config
from src.python.inverse_lut import apply_inverse_lut, load_inverse_lut
from src.python.photometric_compensation import calc_compensation_image

SCRATCH = Path(__file__).resolve().parent
CONFIG = SCRATCH / "config_display5.toml"
OUT = REPO / "data" / "comp_test_20260724"
SQ_OUT = OUT / "square_results"
TARGET_DIR = REPO / "data" / "comp_test_targets_square"

POS = (3440, 0)
SIZE = (1280, 720)
TARGET_SIZE = 512
GAMMA = 2.2
DEFAULT_REGION = (416, 142, 468)  # (x, y, side) in projector px


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


def metrics_square(result01, target01):
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
    ssim = ssim_mean(gray_r, gray_t)
    lab_r = cv2.cvtColor(result01.astype(np.float32), cv2.COLOR_RGB2Lab)
    lab_t = cv2.cvtColor(target01.astype(np.float32), cv2.COLOR_RGB2Lab)
    de = float(np.mean(np.linalg.norm(lab_r - lab_t, axis=-1)))
    return {"psnr": psnr, "ssim": ssim, "deltaE76": de}


def main() -> None:
    region = DEFAULT_REGION
    if len(sys.argv) == 4:
        region = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    rx, ry, rs = region
    print(f"projection region (x, y, side) = {region}")

    reload_config(CONFIG)
    cfg = get_config()
    SQ_OUT.mkdir(parents=True, exist_ok=True)

    linearizers = make_linearizers()
    warper = PixelMapWarperTorch(
        load_p2c_numpy_array(str(REPO / "result_p2c_compensated_delaunay.npy"))
    )

    targets = []
    for p in sorted(TARGET_DIR.glob("*.png")):
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        assert rgb.shape[:2] == (TARGET_SIZE, TARGET_SIZE), p
        targets.append((p.stem, rgb))
    print(f"{len(targets)} square targets")

    conditions = ["none", "gamma", "lut_ch", "lut_gray"]
    fit_spaces = ["srgb", "linear"]
    matrices = {}
    for fs in fit_spaces:
        prefix = "matrices_lin_" if fs == "linear" else "matrices_"
        for cond in conditions:
            m = np.load(OUT / f"{prefix}{cond}.npy")
            matrices[(fs, cond)] = m[ry : ry + rs, rx : rx + rs]

    results = {}
    window = "square_comp"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, POS[0], POS[1])
    cv2.resizeWindow(window, SIZE[0], SIZE[1])
    cv2.waitKey(50)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(100)

    backend = create_camera_backend(cfg.camera)
    try:
        with backend:
            for fs in fit_spaces:
                for cond in conditions:
                    lin = linearizers[cond]
                    for tname, target in targets:
                        tgt_region = cv2.resize(
                            target, (rs, rs), interpolation=cv2.INTER_AREA
                        )
                        if fs == "linear":
                            tgt_in = srgb_decode(
                                tgt_region.astype(np.float32) / 255.0
                            ).astype(np.float32)
                        else:
                            tgt_in = tgt_region
                        comp = calc_compensation_image(
                            target_image=tgt_in,
                            color_mixing_matrices=matrices[(fs, cond)],
                            dtype=np.uint8,
                        )
                        frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
                        frame[ry : ry + rs, rx : rx + rs] = comp
                        proj_img = np.asarray(lin(frame), dtype=np.uint8)
                        cv2.imshow(
                            window, cv2.cvtColor(proj_img, cv2.COLOR_RGB2BGR)
                        )
                        cv2.waitKey(cfg.camera.wait_key_ms)
                        time.sleep(0.3)
                        cam = backend.capture_linear_rgb()
                        srgb8 = (srgb_encode(cam) * 255).round().astype(np.uint8)
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
                        crop = warped[ry : ry + rs, rx : rx + rs]
                        square = cv2.resize(
                            np.clip(crop, 0, 1),
                            (TARGET_SIZE, TARGET_SIZE),
                            interpolation=cv2.INTER_AREA,
                        )
                        res_dir = SQ_OUT / tname
                        res_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(
                            str(res_dir / f"square_{fs}_{cond}.png"),
                            cv2.cvtColor(
                                (square * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                            ),
                        )
                        met = metrics_square(
                            square.astype(np.float32),
                            target.astype(np.float32) / 255.0,
                        )
                        results[f"{fs}/{cond}/{tname}"] = met
                        print(
                            f"[{fs}/{cond}] {tname}: PSNR={met['psnr']:.2f}dB "
                            f"SSIM={met['ssim']:.4f} dE76={met['deltaE76']:.2f}"
                        )
                        del cam, srgb8, warped
    finally:
        cv2.destroyWindow(window)

    payload = {"region": list(region), "target_size": TARGET_SIZE, "metrics": results}
    (SQ_OUT / "metrics_square.json").write_text(json.dumps(payload, indent=2))
    print(f"saved metrics to {SQ_OUT / 'metrics_square.json'}")


if __name__ == "__main__":
    main()
