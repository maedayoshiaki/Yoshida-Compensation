"""Generate per-pixel DeltaE76, local PSNR, and local SSIM maps.

The input images are the 512x512 ``warped_*`` images emitted by
``evaluate_compensation_conditions.py``.  Each target/condition gets raw
``.npy`` maps and one horizontal PNG row in the order DeltaE76, PSNR, SSIM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_CONDITIONS = (
    "no_comp",
    "comp_no_gamma",
    "comp_gamma",
    "comp_inverse_lut",
    "comp_inverse_lut_gray",
)


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _delta_e_map(reference: np.ndarray, actual: np.ndarray) -> np.ndarray:
    ref_lab = cv2.cvtColor(reference.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    act_lab = cv2.cvtColor(actual.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    return np.linalg.norm(ref_lab - act_lab, axis=2).astype(np.float32)


def _psnr_map(reference: np.ndarray, actual: np.ndarray) -> np.ndarray:
    ref = reference.astype(np.float32) / 255.0
    act = actual.astype(np.float32) / 255.0
    mse = np.mean(np.square(ref - act), axis=2)
    local_mse = cv2.GaussianBlur(mse, (11, 11), 1.5)
    return (10.0 * np.log10(1.0 / np.maximum(local_mse, 1e-8))).astype(np.float32)


def _ssim_map(reference: np.ndarray, actual: np.ndarray) -> np.ndarray:
    ref = reference.astype(np.float32) / 255.0
    act = actual.astype(np.float32) / 255.0
    c1, c2 = 0.01**2, 0.03**2
    mu_ref = cv2.GaussianBlur(ref, (11, 11), 1.5)
    mu_act = cv2.GaussianBlur(act, (11, 11), 1.5)
    sigma_ref = cv2.GaussianBlur(ref * ref, (11, 11), 1.5) - mu_ref * mu_ref
    sigma_act = cv2.GaussianBlur(act * act, (11, 11), 1.5) - mu_act * mu_act
    sigma_cross = cv2.GaussianBlur(ref * act, (11, 11), 1.5) - mu_ref * mu_act
    numerator = (2.0 * mu_ref * mu_act + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_ref * mu_ref + mu_act * mu_act + c1) * (
        sigma_ref + sigma_act + c2
    )
    return np.mean(numerator / np.maximum(denominator, 1e-8), axis=2).astype(np.float32)


def _color_map(values: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
    return cv2.applyColorMap(
        np.round(scaled * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )


def _panel(image_bgr: np.ndarray, title: str, values: np.ndarray) -> np.ndarray:
    panel = image_bgr.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 44), (0, 0, 0), -1)
    summary = f"{title}  mean={np.mean(values):.3f}"
    cv2.putText(
        panel,
        summary,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for target_path in sorted(args.target_dir.glob("*_sq.png"), key=lambda p: p.name.casefold()):
        target_name = target_path.stem
        reference = _load_rgb(target_path)
        for condition in args.conditions:
            actual_path = args.evaluation_dir / f"warped_{target_name}_{condition}.png"
            actual = _load_rgb(actual_path)
            if actual.shape[:2] != reference.shape[:2]:
                actual = cv2.resize(
                    actual,
                    (reference.shape[1], reference.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            delta_e = _delta_e_map(reference, actual)
            psnr = _psnr_map(reference, actual)
            ssim = _ssim_map(reference, actual)
            map_dir = args.output_dir / target_name / condition
            map_dir.mkdir(parents=True, exist_ok=True)
            np.save(map_dir / "deltaE76.npy", delta_e)
            np.save(map_dir / "psnr_local_db.npy", psnr)
            np.save(map_dir / "ssim_local.npy", ssim)
            delta_vis = _color_map(delta_e, 0.0, 40.0)
            psnr_vis = _color_map(psnr, 0.0, 40.0)
            ssim_vis = _color_map(ssim, 0.0, 1.0)
            cv2.imwrite(str(map_dir / "deltaE76.png"), delta_vis)
            cv2.imwrite(str(map_dir / "psnr_local_db.png"), psnr_vis)
            cv2.imwrite(str(map_dir / "ssim_local.png"), ssim_vis)
            row = np.hstack(
                [
                    _panel(delta_vis, "DeltaE76", delta_e),
                    _panel(psnr_vis, "local PSNR [dB]", psnr),
                    _panel(ssim_vis, "local SSIM", ssim),
                ]
            )
            row_path = args.output_dir / f"metric_maps_{target_name}_{condition}.png"
            cv2.imwrite(str(row_path), row)
            summary.setdefault(target_name, {})[condition] = {
                "delta_e76_mean": float(np.mean(delta_e)),
                "delta_e76_max": float(np.max(delta_e)),
                "psnr_local_db_mean": float(np.mean(psnr)),
                "psnr_local_db_min": float(np.min(psnr)),
                "ssim_local_mean": float(np.mean(ssim)),
                "ssim_local_min": float(np.min(ssim)),
            }
    (args.output_dir / "metric_maps_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
