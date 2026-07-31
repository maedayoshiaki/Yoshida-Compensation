"""Capture black/white projections and measure linear RAW exposure."""

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
from yoshidacomp.config import reload_config
from yoshidacomp.compensation.projection_region import mask_to_active_region, resolve_active_region


def _statistics(image: np.ndarray) -> dict[str, object]:
    return {
        "mean_rgb": np.mean(image, axis=(0, 1)).tolist(),
        "max_rgb": np.max(image, axis=(0, 1)).tolist(),
        "fraction_any_ge_0_98": float(np.mean(np.any(image >= 0.98, axis=2))),
        "fraction_any_ge_0_95": float(np.mean(np.any(image >= 0.95, axis=2))),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = reload_config(args.config)
    proj = cfg.projector
    active_region = resolve_active_region(proj)
    black = np.zeros((proj.height, proj.width, 3), dtype=np.uint8)
    white = mask_to_active_region(
        np.full((proj.height, proj.width, 3), 255, dtype=np.uint8),
        active_region,
    )

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
        started = time.perf_counter()
        captures = sample.capture_projection_patterns(
            [black, white], cfg=cfg, window_name=window_name
        )
        elapsed = time.perf_counter() - started
    finally:
        cv2.destroyWindow(window_name)
    if captures is None:
        raise RuntimeError("Exposure-check capture failed.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ("black", "white")
    for name, capture in zip(names, captures, strict=True):
        viewable = sample.capture_to_srgb_uint8(capture)
        sample._save_rgb_debug_image(
            args.output_dir / f"capture_{name}.png", viewable
        )

    scale = 2 if cfg.camera.half_size else 1
    x = cfg.inverse_lut.roi_x // scale
    y = cfg.inverse_lut.roi_y // scale
    width = cfg.inverse_lut.roi_width // scale
    height = cfg.inverse_lut.roi_height // scale
    roi = (x, y, width, height)
    payload = {
        "iso": cfg.camera.iso,
        "av": cfg.camera.av,
        "tv": cfg.camera.tv,
        "half_size": cfg.camera.half_size,
        "hdr_enabled": cfg.camera.hdr_enabled,
        "hdr_max_stops": cfg.inverse_lut.hdr_max_stops,
        "hdr_trust_low": cfg.inverse_lut.hdr_trust_low,
        "hdr_trust_high": cfg.inverse_lut.hdr_trust_high,
        "image_shape": list(captures[0].shape),
        "clean_roi": list(roi),
        "capture_and_development_seconds": elapsed,
        "black": _statistics(captures[0][y : y + height, x : x + width]),
        "white": _statistics(captures[1][y : y + height, x : x + width]),
    }
    output_path = args.output_dir / "exposure_check.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved exposure check to: {args.output_dir}")


if __name__ == "__main__":
    main()
