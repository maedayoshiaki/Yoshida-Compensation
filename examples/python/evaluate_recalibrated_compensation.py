"""Capture and compare no compensation with one newly calibrated gamma result."""

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

from examples.python import evaluate_compensation_conditions as evaluation
from examples.python import sample
from yoshidacomp.config import reload_config


_CONDITIONS = ("no_comp", "comp_gamma")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gamma-comp-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = reload_config(args.config)
    if cfg.camera.half_size:
        raise ValueError("Half-resolution RAW is prohibited.")
    proj = cfg.projector
    active_region = (
        proj.active_x,
        proj.active_y,
        proj.active_width,
        proj.active_height,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target_paths = sorted(
        (
            path
            for path in Path(cfg.paths.target_image_dir).iterdir()
            if path.suffix.lower() in sample.SUPPORTED_TARGET_IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if not target_paths:
        raise FileNotFoundError(
            f"No target images found in {cfg.paths.target_image_dir}"
        )

    patterns: list[np.ndarray] = []
    sequence: list[tuple[str, str, np.ndarray]] = []
    for index, target_path in enumerate(target_paths):
        target = evaluation._load_rgb(target_path)
        condition_patterns = {
            "no_comp": sample.embed_in_active_region(
                target, (proj.width, proj.height), active_region
            ),
            "comp_gamma": evaluation._load_rgb(
                args.gamma_comp_dir / f"inv_gamma_comp_image_{index:02d}.png"
            ),
        }
        for condition in _CONDITIONS:
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

    x, y, width, height = active_region
    image_results: dict[str, dict[str, dict[str, float]]] = {}
    evaluated_images: dict[str, dict[str, np.ndarray]] = {}
    warp_seconds: list[float] = []
    for index, (capture, (target_name, condition, target)) in enumerate(
        zip(captures, sequence, strict=True)
    ):
        evaluation._save_rgb(
            output_dir / f"capture_{index:02d}_{target_name}_{condition}.png",
            sample.capture_to_srgb_uint8(capture),
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
        warped = sample.capture_to_srgb_uint8(warped_linear)
        active = warped[y : y + height, x : x + width]
        evaluated = cv2.resize(
            active,
            (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        evaluation._save_rgb(
            output_dir / f"warped_{target_name}_{condition}.png", evaluated
        )
        image_results.setdefault(target_name, {})[condition] = evaluation._metrics(
            target, evaluated
        )
        evaluated_images.setdefault(target_name, {})[condition] = evaluated

    for target_name, conditions in evaluated_images.items():
        target = next(
            target
            for name, condition, target in sequence
            if name == target_name and condition == "no_comp"
        )
        comparison = np.hstack(
            [
                evaluation._label_panel(target, "target"),
                evaluation._label_panel(conditions["no_comp"], "no_comp"),
                evaluation._label_panel(
                    conditions["comp_gamma"], "new_comp_gamma"
                ),
            ]
        )
        evaluation._save_rgb(
            output_dir / f"comparison_{target_name}.png", comparison
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
        for condition in _CONDITIONS
    }
    payload = {
        "setup_date": "2026-07-30",
        "display_scaling_percent": 100,
        "camera": {
            "iso": cfg.camera.iso,
            "av": cfg.camera.av,
            "tv": cfg.camera.tv,
            "image_quality": cfg.camera.image_quality,
            "initial_wait_ms": cfg.camera.wait_key_ms,
            "trigger_switch_wait_ms": cfg.camera.trigger_wait_key_ms,
            "half_size": cfg.camera.half_size,
        },
        "p2c_map": cfg.paths.p2c_map,
        "conditions": list(_CONDITIONS),
        "capture_and_development_seconds": capture_seconds,
        "warp_seconds": warp_seconds,
        "images": [
            {"name": name, "conditions": image_results[name]}
            for name in image_results
        ],
        "average": averages,
    }
    metrics_path = output_dir / "metrics_recalibrated.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved comparison results to: {output_dir}")


if __name__ == "__main__":
    main()
