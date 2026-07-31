"""Project four gray code values and capture them as full-resolution RAW."""

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
from src.python.projection_region import mask_to_active_region, resolve_active_region


_LEVELS = (0, 85, 170, 255)


def _statistics(image: np.ndarray) -> dict[str, object]:
    return {
        "mean_rgb_linear": np.mean(image, axis=(0, 1)).tolist(),
        "mean_linear": float(np.mean(image)),
        "max_rgb_linear": np.max(image, axis=(0, 1)).tolist(),
        "fraction_any_ge_0_98": float(np.mean(np.any(image >= 0.98, axis=2))),
        "fraction_any_ge_0_95": float(np.mean(np.any(image >= 0.95, axis=2))),
    }


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        image = sample.capture_to_srgb_uint8(image)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to save image: {path}")


def _contact_sheet(crops: list[np.ndarray], levels: tuple[int, ...]) -> np.ndarray:
    tile_size = 480
    label_height = 52
    tiles: list[np.ndarray] = []
    for crop, level in zip(crops, levels, strict=True):
        viewable = sample.capture_to_srgb_uint8(crop)
        tile = cv2.resize(viewable, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        labeled = np.zeros((tile_size + label_height, tile_size, 3), dtype=np.uint8)
        labeled[label_height:] = tile
        cv2.putText(
            labeled,
            f"projector gray code {level}",
            (14, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(labeled)
    return np.concatenate(tiles, axis=1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = reload_config(args.config)
    if cfg.camera.half_size:
        raise ValueError("Full-resolution RAW is required; half_size must be false.")

    proj = cfg.projector
    active_region = resolve_active_region(proj)
    patterns = [
        mask_to_active_region(
            np.full((proj.height, proj.width, 3), level, dtype=np.uint8),
            active_region,
        )
        for level in _LEVELS
    ]

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
            patterns, cfg=cfg, window_name=window_name
        )
        capture_seconds = time.perf_counter() - started
    finally:
        cv2.destroyWindow(window_name)
    if captures is None:
        raise RuntimeError("Grayscale capture failed.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y, width, height = active_region
    roi_x = cfg.inverse_lut.roi_x
    roi_y = cfg.inverse_lut.roi_y
    roi_width = cfg.inverse_lut.roi_width
    roi_height = cfg.inverse_lut.roi_height
    clean_roi = (roi_x, roi_y, roi_width, roi_height)

    warped_crops: list[np.ndarray] = []
    measurements: list[dict[str, object]] = []
    for index, (level, capture) in enumerate(
        zip(_LEVELS, captures, strict=True)
    ):
        _save_rgb(output_dir / f"capture_full_{index:02d}_gray_{level:03d}.png", capture)
        warped = sample.warp_image(
            capture,
            cfg.paths.p2c_map,
            proj.width,
            proj.height,
            proj.width,
            proj.height,
            warp_method="p2c",
        )
        crop = warped[y : y + height, x : x + width]
        warped_crops.append(crop)
        _save_rgb(output_dir / f"capture_warped_active_{index:02d}_gray_{level:03d}.png", crop)
        camera_roi = capture[
            roi_y : roi_y + roi_height,
            roi_x : roi_x + roi_width,
        ]
        measurements.append(
            {
                "index": index,
                "projector_code": level,
                "projector_fraction": level / 255.0,
                "camera_clean_roi": _statistics(camera_roi),
                "warped_active": _statistics(crop),
            }
        )

    sheet = _contact_sheet(warped_crops, _LEVELS)
    _save_rgb(output_dir / "grayscale_4level_contact_sheet.png", sheet)

    responses = [
        float(item["camera_clean_roi"]["mean_linear"])  # type: ignore[index]
        for item in measurements
    ]
    payload = {
        "camera": {
            "iso": cfg.camera.iso,
            "av": cfg.camera.av,
            "tv": cfg.camera.tv,
            "half_size": cfg.camera.half_size,
            "hdr_enabled": cfg.camera.hdr_enabled,
            "hdr_max_stops": cfg.inverse_lut.hdr_max_stops,
            "hdr_trust_low": cfg.inverse_lut.hdr_trust_low,
            "hdr_trust_high": cfg.inverse_lut.hdr_trust_high,
            "capture_shape": list(captures[0].shape),
        },
        "projector": {
            "display_position": [proj.pos_x, proj.pos_y],
            "frame_size": [proj.width, proj.height],
            "active_region": list(active_region),
            "levels": list(_LEVELS),
        },
        "clean_roi_camera": list(clean_roi),
        "capture_and_development_seconds": capture_seconds,
        "strictly_increasing_clean_roi_mean": bool(
            all(b > a for a, b in zip(responses, responses[1:]))
        ),
        "measurements": measurements,
    }
    metrics_path = output_dir / "grayscale_4level_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved grayscale capture to: {output_dir}")


if __name__ == "__main__":
    main()
