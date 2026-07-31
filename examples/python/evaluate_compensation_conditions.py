"""Capture and compare four or five projector-compensation conditions.

The four conditions are:

1. no compensation
2. compensation without projector gamma correction
3. compensation with analytic projector gamma correction
4. compensation with a measured inverse LUT

All patterns are captured in one triggered batch so camera development and
projection setup costs are shared across the comparison.

Pass ``--gray-lut-comp-dir`` to append a fifth condition using an inverse LUT
fitted from gray-ramp projections.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.python import sample
from src.python.config import reload_config


BASE_CONDITIONS = (
    "no_comp",
    "comp_no_gamma",
    "comp_gamma",
    "comp_inverse_lut",
)


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_first_rgb(directory: Path, *names: str) -> np.ndarray:
    for name in names:
        path = directory / name
        if path.exists():
            return _load_rgb(path)
    raise FileNotFoundError(
        f"None of the expected images exists in {directory}: {names}"
    )


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _psnr(reference: np.ndarray, actual: np.ndarray) -> float:
    difference = (
        reference.astype(np.float32) / 255.0
        - actual.astype(np.float32) / 255.0
    )
    mse = float(np.mean(np.square(difference)))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))


def _ssim(reference: np.ndarray, actual: np.ndarray) -> float:
    reference_f = reference.astype(np.float32) / 255.0
    actual_f = actual.astype(np.float32) / 255.0
    c1 = 0.01**2
    c2 = 0.03**2
    mu_ref = cv2.GaussianBlur(reference_f, (11, 11), 1.5)
    mu_actual = cv2.GaussianBlur(actual_f, (11, 11), 1.5)
    mu_ref_sq = mu_ref * mu_ref
    mu_actual_sq = mu_actual * mu_actual
    mu_cross = mu_ref * mu_actual
    sigma_ref_sq = cv2.GaussianBlur(reference_f * reference_f, (11, 11), 1.5)
    sigma_ref_sq -= mu_ref_sq
    sigma_actual_sq = cv2.GaussianBlur(actual_f * actual_f, (11, 11), 1.5)
    sigma_actual_sq -= mu_actual_sq
    sigma_cross = cv2.GaussianBlur(reference_f * actual_f, (11, 11), 1.5)
    sigma_cross -= mu_cross
    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_ref_sq + mu_actual_sq + c1) * (
        sigma_ref_sq + sigma_actual_sq + c2
    )
    return float(np.mean(numerator / denominator))


def _delta_e76(reference: np.ndarray, actual: np.ndarray) -> float:
    reference_lab = cv2.cvtColor(
        reference.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB
    )
    actual_lab = cv2.cvtColor(
        actual.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB
    )
    return float(np.mean(np.linalg.norm(reference_lab - actual_lab, axis=2)))


def _metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    return {
        "psnr_db": _psnr(reference, actual),
        "ssim": _ssim(reference, actual),
        "delta_e76": _delta_e76(reference, actual),
    }


def _label_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        panel,
        label,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--target-dir",
        type=Path,
        help="Override paths.target_image_dir from the configuration.",
    )
    parser.add_argument("--raw-comp-dir", type=Path, required=True)
    parser.add_argument("--gamma-comp-dir", type=Path, required=True)
    parser.add_argument("--lut-comp-dir", type=Path, required=True)
    parser.add_argument(
        "--gray-lut-comp-dir",
        type=Path,
        help="Optional fifth condition: gray-ramp-fitted inverse LUT.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--color-levels", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = reload_config(args.config)
    if cfg.camera.half_size:
        raise ValueError(
            "Half-resolution RAW is prohibited; camera.half_size must be false."
        )
    proj = cfg.projector
    active_region = (
        proj.active_x,
        proj.active_y,
        proj.active_width,
        proj.active_height,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = BASE_CONDITIONS + (
        ("comp_inverse_lut_gray",) if args.gray_lut_comp_dir is not None else ()
    )

    target_dir = args.target_dir or Path(cfg.paths.target_image_dir)
    target_paths = sorted(
        (
            path
            for path in target_dir.iterdir()
            if path.suffix.lower() in sample.SUPPORTED_TARGET_IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if not target_paths:
        raise FileNotFoundError(
            f"No target images found in {target_dir}"
        )

    patterns: list[np.ndarray] = []
    sequence: list[tuple[str, str, np.ndarray]] = []
    for index, target_path in enumerate(target_paths):
        target = _load_rgb(target_path)
        condition_patterns = {
            "no_comp": sample.embed_in_active_region(
                target, (proj.width, proj.height), active_region
            ),
            "comp_no_gamma": _load_first_rgb(
                args.raw_comp_dir,
                f"no_gamma_comp_image_{index:02d}.png",
                f"inv_lut_comp_image_{index:02d}.png",
            ),
            "comp_gamma": _load_rgb(
                args.gamma_comp_dir / f"inv_gamma_comp_image_{index:02d}.png"
            ),
            "comp_inverse_lut": _load_rgb(
                args.lut_comp_dir / f"inv_lut_comp_image_{index:02d}.png"
            ),
        }
        if args.gray_lut_comp_dir is not None:
            condition_patterns["comp_inverse_lut_gray"] = _load_rgb(
                args.gray_lut_comp_dir / f"inv_lut_comp_image_{index:02d}.png"
            )
        for condition in conditions:
            patterns.append(condition_patterns[condition])
            sequence.append((target_path.stem, condition, target))

    window_name = "projection_window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, proj.pos_x, proj.pos_y)
    cv2.resizeWindow(window_name, proj.width, proj.height)
    cv2.waitKey(50)
    cv2.setWindowProperty(
        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )
    if hasattr(cv2, "WND_PROP_TOPMOST"):
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    try:
        capture_started = time.perf_counter()
        captures = sample.capture_projection_patterns(
            patterns, cfg=cfg, window_name=window_name
        )
        capture_seconds = time.perf_counter() - capture_started
    finally:
        cv2.destroyWindow(window_name)
    if captures is None:
        raise RuntimeError("Triggered comparison capture failed.")

    image_results: dict[str, dict[str, dict[str, float]]] = {}
    warped_by_name: dict[str, dict[str, np.ndarray]] = {}
    warp_seconds: list[float] = []
    x, y, width, height = active_region
    for index, (capture, (target_name, condition, target)) in enumerate(
        zip(captures, sequence, strict=True)
    ):
        capture_uint8 = sample.capture_to_srgb_uint8(capture)
        _save_rgb(
            output_dir / f"capture_{index:02d}_{target_name}_{condition}.png",
            capture_uint8,
        )

        warp_started = time.perf_counter()
        warped_linear = sample.warp_image(
            capture,
            cfg.paths.p2c_map,
            proj.width,
            proj.height,
            proj.width,
            proj.height,
            warp_method="p2c",
        )
        warp_seconds.append(time.perf_counter() - warp_started)
        warped_uint8 = sample.capture_to_srgb_uint8(warped_linear)
        active_capture = warped_uint8[y : y + height, x : x + width]
        evaluated = cv2.resize(
            active_capture,
            (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        _save_rgb(
            output_dir / f"warped_{target_name}_{condition}.png",
            evaluated,
        )
        image_results.setdefault(target_name, {})[condition] = _metrics(
            target, evaluated
        )
        warped_by_name.setdefault(target_name, {})[condition] = evaluated

    for target_name, condition_images in warped_by_name.items():
        target = next(
            target
            for name, condition, target in sequence
            if name == target_name and condition == "no_comp"
        )
        comparison = np.hstack(
            [_label_panel(target, "target")]
            + [
                _label_panel(condition_images[condition], condition)
                for condition in conditions
            ]
        )
        _save_rgb(
            output_dir / f"comparison_{target_name}_requested{len(conditions)}.png",
            comparison,
        )

    averages = {
        condition: {
            metric: float(
                np.mean(
                    [
                        image_results[name][condition][metric]
                        for name in image_results
                    ]
                )
            )
            for metric in ("psnr_db", "ssim", "delta_e76")
        }
        for condition in conditions
    }
    payload = {
        "setup_date": "2026-07-30",
        "camera": {
            "iso": cfg.camera.iso,
            "av": cfg.camera.av,
            "tv": cfg.camera.tv,
            "image_quality": cfg.camera.image_quality,
            "initial_wait_ms": cfg.camera.wait_key_ms,
            "trigger_switch_wait_ms": cfg.camera.trigger_wait_key_ms,
            "half_size": cfg.camera.half_size,
            "hdr_enabled": cfg.camera.hdr_enabled,
            "hdr_max_stops": cfg.inverse_lut.hdr_max_stops,
            "hdr_trust_low": cfg.inverse_lut.trust_low,
            "hdr_trust_high": cfg.inverse_lut.trust_high,
        },
        "p2c_map": cfg.paths.p2c_map,
        "target_dir": str(target_dir),
        "requested_conditions": list(conditions),
        "color_levels_per_channel": args.color_levels,
        "calibration_patterns_per_compensation_matrix": args.color_levels**3,
        "capture_and_development_seconds": capture_seconds,
        "warp_seconds": warp_seconds,
        "images": [
            {"name": name, "conditions": image_results[name]}
            for name in image_results
        ],
        "average": averages,
    }
    metrics_path = output_dir / f"metrics_requested{len(conditions)}.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved comparison results to: {output_dir}")


if __name__ == "__main__":
    main()
