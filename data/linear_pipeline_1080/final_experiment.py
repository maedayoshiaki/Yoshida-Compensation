"""Final all-conditions compensation experiment (fully linear pipeline).

Conditions: lut_ch / lut_gray / gamma / none — the projector
linearization applied to both the fitting patterns and the compensation
image. Everything else is the fully linear pipeline at 1920x1080:
linear capture, linear-space matrix fit, sRGB targets decoded to linear,
square workflow (region from data/linear_pipeline_1080/region.json).

Matrices for lut_ch and gamma are reused from the earlier run (physical
setup unchanged); none and lut_gray are fitted fresh here.
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


class Rig:
    def __init__(self, cam_cfg, wait_ms: int):
        self.wait_ms = wait_ms
        self.window = "final_experiment"
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

    def project_capture_linear(self, rgb_uint8: np.ndarray) -> np.ndarray:
        cv2.imshow(self.window, cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
        cv2.waitKey(self.wait_ms)
        time.sleep(0.3)
        return self.backend.capture_linear_rgb()


def warp_linear(warper, linear: np.ndarray) -> np.ndarray:
    src = torch.from_numpy(
        np.ascontiguousarray(linear, dtype=np.float32)
    ).permute(2, 0, 1)
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
    # Ridge-regularized normal equations instead of lstsq: pixels outside
    # the decoded canvas are exactly zero across all patterns (rank
    # deficient), which makes the CUDA gels driver abort. The tiny ridge
    # keeps every pixel solvable with negligible bias on valid pixels.
    a = cap.to(device)  # (HW, N, 4)
    b = proj_ref.unsqueeze(0).expand(h * w, -1, -1).to(device)  # (HW, N, 3)
    ata = a.transpose(1, 2) @ a  # (HW, 4, 4)
    atb = a.transpose(1, 2) @ b  # (HW, 4, 3)
    eye = torch.eye(4, device=device).unsqueeze(0) * 1e-6
    sol = torch.linalg.solve(ata + eye, atb)
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
    target_lin = srgb_decode(target_srgb01)
    psnr_l, ssim_l = metrics_pair(
        square_lin.astype(np.float32), target_lin.astype(np.float32)
    )
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

    region = tuple(json.loads((OUT / "region.json").read_text())["region"])
    rx, ry, rs = region
    print(f"region = {region}")

    linearizers = make_linearizers()
    warper = PixelMapWarperTorch(
        load_p2c_numpy_array(str(REPO / "result_p2c_compensated_delaunay.npy"))
    )
    patterns = generate_projection_patterns(SIZE[0], SIZE[1], dtype=np.uint8)
    targets = []
    for p in sorted(TARGET_DIR.glob("*.png")):
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        targets.append((p.stem, rgb))

    conditions = ["none", "gamma", "lut_ch", "lut_gray"]
    mats = {}
    to_fit = []
    for cond in conditions:
        path = OUT / f"matrices_linear_{cond}.npy"
        if path.exists():
            mats[cond] = np.load(path)
            print(f"[{cond}] reusing existing matrices")
        else:
            to_fit.append(cond)

    results = {}
    with Rig(cfg.camera, cfg.camera.wait_key_ms) as rig:
        # Debug: capture solid white at three brightness levels first, so
        # every run archives the surface/exposure/projector state for
        # cross-run comparison.
        dbg = OUT / "debug_whites"
        dbg.mkdir(parents=True, exist_ok=True)
        white_stats = {}
        for lvl in (255, 170, 85):
            frame = rig.project_capture_linear(
                np.full((SIZE[1], SIZE[0], 3), lvl, np.uint8)
            )
            cv2.imwrite(
                str(dbg / f"white_{lvl:03d}.png"),
                cv2.cvtColor(
                    (srgb_encode(frame) * 255).round().astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                ),
            )
            mean8 = frame.reshape(-1, 3).mean(axis=0) * 255
            white_stats[str(lvl)] = [float(x) for x in mean8]
            print(f"[debug] white {lvl}: frame mean8 = "
                  f"{np.array2string(mean8, precision=2)}")
            del frame
        (dbg / "stats.json").write_text(json.dumps(white_stats, indent=2))

        for cond in to_fit:
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
            print(f"[{cond}] matrices fitted in {time.monotonic() - t0:.0f}s")
            del warped_caps

        # Uncompensated baseline: project each target AS-IS (no matrices,
        # no linearization) so compensation scores have a reference point.
        for tname, target in targets:
            frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
            frame[ry : ry + rs, rx : rx + rs] = cv2.resize(
                target, (rs, rs), interpolation=cv2.INTER_AREA
            )
            cam = rig.project_capture_linear(frame)
            warped = warp_linear(warper, cam)
            square_lin = cv2.resize(
                np.clip(warped[ry : ry + rs, rx : rx + rs], 0, 1),
                (TARGET_SIZE, TARGET_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            res_dir = OUT / "final_results" / tname
            res_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(res_dir / "proj_input_uncompensated.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(res_dir / "square_uncompensated.png"),
                cv2.cvtColor(
                    (srgb_encode(square_lin) * 255).round().astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                ),
            )
            met = evaluate(square_lin, target.astype(np.float32) / 255.0)
            results[f"uncompensated/{tname}"] = met
            print(
                f"[uncompensated] {tname}: PSNR_srgb={met['psnr_srgb']:.2f}dB "
                f"SSIM_srgb={met['ssim_srgb']:.4f} dE76={met['deltaE76']:.2f}"
            )
            del cam, warped

        for cond in conditions:
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
                res_dir = OUT / "final_results" / tname
                res_dir.mkdir(parents=True, exist_ok=True)
                # Save the projector-input images alongside the captured
                # result: the region-sized compensation image and the full
                # linearized 1920x1080 frame actually sent to the projector.
                cv2.imwrite(
                    str(res_dir / f"comp_region_{cond}.png"),
                    cv2.cvtColor(comp, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(res_dir / f"proj_input_{cond}.png"),
                    cv2.cvtColor(proj_img, cv2.COLOR_RGB2BGR),
                )
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
                    f"[{cond}] {tname}: PSNR_srgb={met['psnr_srgb']:.2f}dB "
                    f"SSIM_srgb={met['ssim_srgb']:.4f} dE76={met['deltaE76']:.2f} "
                    f"PSNR_lin={met['psnr_linear']:.2f}dB"
                )
                del cam, warped

    (OUT / "final_results" / "metrics_final.json").write_text(
        json.dumps({"region": list(region), "metrics": results}, indent=2)
    )
    print("done")


if __name__ == "__main__":
    main()
