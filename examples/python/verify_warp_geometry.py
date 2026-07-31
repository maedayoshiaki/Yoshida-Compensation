"""Project a checker, capture it, and verify P2C/C2P warp alignment."""

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


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    image_u8 = sample.capture_to_srgb_uint8(image)
    cv2.imwrite(str(path), cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR))


def _alignment_metrics(
    expected: np.ndarray,
    actual: np.ndarray,
    active_region: tuple[int, int, int, int],
) -> dict[str, object]:
    x, y, width, height = active_region
    expected_gray = cv2.cvtColor(
        expected[y : y + height, x : x + width], cv2.COLOR_RGB2GRAY
    ).astype(np.float32)
    actual_gray = cv2.cvtColor(
        actual[y : y + height, x : x + width], cv2.COLOR_RGB2GRAY
    ).astype(np.float32)
    margin = max(8, min(width, height) // 8)
    expected_inner = expected_gray[margin:-margin, margin:-margin]
    actual_inner = actual_gray[margin:-margin, margin:-margin]

    expected_edges = cv2.Laplacian(
        cv2.GaussianBlur(expected_inner, (0, 0), 1.0), cv2.CV_32F
    )
    actual_edges = cv2.Laplacian(
        cv2.GaussianBlur(actual_inner, (0, 0), 1.0), cv2.CV_32F
    )
    shift, response = cv2.phaseCorrelate(expected_edges, actual_edges)
    correlation = float(
        np.corrcoef(expected_inner.ravel(), actual_inner.ravel())[0, 1]
    )
    rmse = float(
        np.sqrt(
            np.mean(
                np.square((expected_inner - actual_inner) / 255.0)
            )
        )
    )
    return {
        "phase_shift_xy_px": [float(shift[0]), float(shift[1])],
        "phase_response": float(response),
        "inner_grayscale_correlation": correlation,
        "inner_grayscale_rmse": rmse,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--p2c-map", type=Path, required=True)
    parser.add_argument("--c2p-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = reload_config(args.config)
    if cfg.camera.half_size:
        raise ValueError(
            "Geometry verification requires full-resolution RAW; "
            "camera.half_size must be false."
        )
    proj = cfg.projector
    active_region = resolve_active_region(proj)
    checker = mask_to_active_region(_load_rgb(args.checker), active_region)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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
            [checker], cfg=cfg, window_name=window_name
        )
        capture_seconds = time.perf_counter() - capture_started
    finally:
        cv2.destroyWindow(window_name)
    if captures is None:
        raise RuntimeError("Checker capture failed.")
    capture = captures[0]
    _save_rgb(output_dir / "checker_capture_full.png", capture)
    _save_rgb(output_dir / "checker_expected.png", checker)

    results: dict[str, object] = {
        "camera": {
            "iso": cfg.camera.iso,
            "av": cfg.camera.av,
            "tv": cfg.camera.tv,
            "half_size": cfg.camera.half_size,
            "capture_shape": list(capture.shape),
        },
        "active_region": list(active_region),
        "capture_and_development_seconds": capture_seconds,
        "warps": {},
    }
    for method, map_path in (
        ("p2c", args.p2c_map),
        ("c2p", args.c2p_map),
    ):
        started = time.perf_counter()
        warped = sample.warp_image(
            capture,
            str(map_path),
            proj.width,
            proj.height,
            proj.width,
            proj.height,
            warp_method=method,
        )
        elapsed = time.perf_counter() - started
        warped_u8 = sample.capture_to_srgb_uint8(warped)
        warped_u8 = mask_to_active_region(warped_u8, active_region)
        _save_rgb(output_dir / f"checker_warped_{method}.png", warped_u8)
        results["warps"][method] = {
            "map": str(map_path),
            "warp_seconds": elapsed,
            **_alignment_metrics(checker, warped_u8, active_region),
        }

    (output_dir / "geometry_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
