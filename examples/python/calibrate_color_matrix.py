"""Capture an HDR photometric calibration and fit a per-pixel colour matrix."""

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
from yoshidacomp.compensation import inverse_lut
from yoshidacomp.config import reload_config
from yoshidacomp.compensation.projection_region import mask_to_active_region, resolve_active_region


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--linearizer",
        choices=("none", "gamma", "inverse_lut"),
        required=True,
    )
    parser.add_argument("--lut", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        image = sample.capture_to_srgb_uint8(image)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to save image: {path}")


def main() -> None:
    args = _parse_args()
    cfg = reload_config(args.config)
    if cfg.camera.half_size:
        raise ValueError("Full-resolution RAW is required; half_size must be false.")
    if not cfg.camera.hdr_enabled:
        raise ValueError("camera.hdr_enabled must be true for HDR calibration.")
    if args.linearizer == "inverse_lut" and args.lut is None:
        raise ValueError("--lut is required for --linearizer inverse_lut.")

    proj = cfg.projector
    active_region = resolve_active_region(proj)
    model_patterns = sample.generate_projection_patterns(
        proj.width,
        proj.height,
        dtype=np.dtype(np.uint8),
    )
    linear_patterns = [
        mask_to_active_region(pattern, active_region)
        for pattern in model_patterns
    ]

    lut = None
    if args.linearizer == "inverse_lut":
        lut = inverse_lut.load_inverse_lut(args.lut)

    projected_patterns: list[np.ndarray] = []
    for pattern in linear_patterns:
        if args.linearizer == "none":
            projected = pattern
        elif args.linearizer == "gamma":
            projected = sample.apply_inverse_gamma_correction(
                pattern,
                gamma=1 / proj.gamma,
            )
            if hasattr(projected, "cpu"):
                projected = projected.cpu().numpy()
        else:
            projected = inverse_lut.apply_inverse_lut(
                pattern,
                lut,
                dtype=np.uint8,
            )
        projected_patterns.append(np.asarray(projected, dtype=np.uint8))

    output_dir = args.output_dir
    linear_dir = output_dir / "linear_patterns"
    projected_dir = output_dir / "projected_patterns"
    captures_dir = output_dir / "captures"
    for directory in (linear_dir, projected_dir, captures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for index, (linear, projected) in enumerate(
        zip(linear_patterns, projected_patterns, strict=True)
    ):
        _save_rgb(linear_dir / f"linear_pattern_{index:02d}.png", linear)
        _save_rgb(projected_dir / f"projected_pattern_{index:02d}.png", projected)

    window_name = "projection_window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, proj.pos_x, proj.pos_y)
    cv2.resizeWindow(window_name, proj.width, proj.height)
    cv2.waitKey(50)
    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )
    if hasattr(cv2, "WND_PROP_TOPMOST"):
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    try:
        capture_started = time.perf_counter()
        captures = sample.capture_projection_patterns(
            projected_patterns,
            cfg=cfg,
            window_name=window_name,
        )
        capture_seconds = time.perf_counter() - capture_started
    finally:
        cv2.destroyWindow(window_name)
    if captures is None:
        raise RuntimeError("HDR colour-matrix capture failed.")

    for index, capture in enumerate(captures):
        _save_rgb(captures_dir / f"hdr_capture_{index:02d}.png", capture)

    fit_started = time.perf_counter()
    matrix = sample.calc_color_mixing_matrices(
        proj_images=model_patterns,
        captured_images=captures,
    )
    fit_seconds = time.perf_counter() - fit_started
    matrix_path = output_dir / "color_mixing_matrices.npy"
    np.save(matrix_path, matrix)

    metadata = {
        "config": str(args.config),
        "linearizer": args.linearizer,
        "inverse_lut": str(args.lut) if args.lut else None,
        "patterns": len(model_patterns),
        "color_levels_per_channel": cfg.compensation.num_divisions,
        "camera": {
            "iso": cfg.camera.iso,
            "av": cfg.camera.av,
            "base_tv": cfg.camera.tv,
            "half_size": cfg.camera.half_size,
            "hdr_enabled": cfg.camera.hdr_enabled,
            "hdr_max_stops": cfg.inverse_lut.hdr_max_stops,
            "hdr_trust_low": cfg.inverse_lut.trust_low,
            "hdr_trust_high": cfg.inverse_lut.trust_high,
            "hdr_gate_roi": [
                cfg.inverse_lut.roi_x,
                cfg.inverse_lut.roi_y,
                cfg.inverse_lut.roi_width,
                cfg.inverse_lut.roi_height,
            ],
        },
        "projector": {
            "frame": [proj.width, proj.height],
            "active_region": list(active_region),
            "refresh_rate": proj.refresh_rate,
        },
        "capture_and_development_seconds": capture_seconds,
        "matrix_fit_seconds": fit_seconds,
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": str(matrix.dtype),
        "matrix_path": str(matrix_path),
    }
    (output_dir / "calibration_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
