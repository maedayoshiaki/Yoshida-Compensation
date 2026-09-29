"""Evaluate arbitrary compensation conditions with the immediate capture path.

Projects, for every target image, the uncompensated target plus one
compensation image per named condition directory, captures each with a
single base exposure on the saturation-guarded immediate path, warps to
projector space, and scores PSNR / SSIM / dE76 against the target under the
linear-unified contract (uint8 quantization only, no tone encoding).

任意の補償条件ディレクトリ群を、即時撮影経路（単一基準露光）で実写評価する。
リニア統一の契約（uint8量子化のみ・ガンマ禁止）で PSNR / SSIM / ΔE76 を
採点する。条件ディレクトリには ``comp_image_{i:02d}.png`` を置く
（``generate_tps_and_matrix_compensation.py`` の出力形式）。

Usage:
    uv run --no-sync python examples/python/evaluate_conditions_immediate.py \
        --config config.color_grid_iso200_hdr_d65_20260812.toml \
        --condition comp_matrix=data/tps_compensation_20260812/comp_matrix \
        --condition comp_tps=data/tps_compensation_20260812/comp_tps \
        --output-dir data/tps_compensation_20260812/evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.python import sample
from examples.python.evaluate_compensation_conditions import (
    _capture_to_linear_uint8,
    _label_panel,
    _load_rgb,
    _metrics,
    _save_rgb,
)
from yoshidacomp.compensation.pipeline import load_target_images
from yoshidacomp.compensation.projection_region import (
    embed_in_active_region,
    resolve_active_region,
)
from yoshidacomp.config import reload_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="named condition directory containing comp_image_{i:02d}.png",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    from yoshidacomp.capture import create_camera_backend

    args = _parse_args()
    cfg = reload_config(args.config)
    if cfg.camera.half_size:
        raise ValueError("half_size must be false.")
    proj = cfg.projector
    active_region = resolve_active_region(proj)
    x, y, width, height = active_region
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions: List[Tuple[str, Path]] = []
    for spec in args.condition:
        name, _, directory = spec.partition("=")
        if not directory:
            raise ValueError(f"--condition expects NAME=DIR, got {spec!r}")
        conditions.append((name, Path(directory)))

    target_dir = args.target_dir or Path(cfg.paths.target_image_dir)
    targets = load_target_images(str(target_dir))

    # Build the projection sequence: per target, no_comp first then each
    # condition (all conditions of one target adjacent, like the classic
    # evaluator, so slow drift cancels within a target's comparison).
    sequence: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
    for index, (target_name, target_uint8) in enumerate(targets):
        name = Path(target_name).stem  # load_target_images returns filenames
        no_comp = embed_in_active_region(
            target_uint8, (proj.width, proj.height), active_region
        )
        sequence.append((name, "no_comp", target_uint8, no_comp))
        for condition, directory in conditions:
            pattern = _load_rgb(directory / f"comp_image_{index:02d}.png")
            sequence.append((name, condition, target_uint8, pattern))

    camera_backend = create_camera_backend(cfg.camera)
    window_name = "conditions_evaluation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, proj.pos_x, proj.pos_y)
    cv2.resizeWindow(window_name, proj.width, proj.height)
    cv2.waitKey(50)
    cv2.setWindowProperty(
        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )
    if hasattr(cv2, "WND_PROP_TOPMOST"):
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    image_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    warped_by_name: Dict[str, Dict[str, np.ndarray]] = {}
    started = time.perf_counter()
    with camera_backend:
        try:
            for index, (name, condition, target_uint8, pattern) in enumerate(
                sequence
            ):
                cv2.imshow(
                    window_name, cv2.cvtColor(pattern, cv2.COLOR_RGB2BGR)
                )
                cv2.waitKey(cfg.camera.wait_key_ms)
                time.sleep(args.settle_seconds)
                capture = np.asarray(
                    camera_backend.capture_linear_rgb(), dtype=np.float32
                )
                _save_rgb(
                    output_dir / f"capture_{index:02d}_{name}_{condition}.png",
                    _capture_to_linear_uint8(capture),
                )
                warped = sample.warp_image(
                    capture,
                    cfg.paths.p2c_map,
                    proj.width,
                    proj.height,
                    proj.width,
                    proj.height,
                    warp_method="p2c",
                )
                del capture
                warped_uint8 = _capture_to_linear_uint8(warped)
                del warped
                evaluated = cv2.resize(
                    warped_uint8[y : y + height, x : x + width],
                    (target_uint8.shape[1], target_uint8.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
                _save_rgb(output_dir / f"warped_{name}_{condition}.png", evaluated)
                image_results.setdefault(name, {})[condition] = _metrics(
                    target_uint8, evaluated
                )
                warped_by_name.setdefault(name, {})[condition] = evaluated
                per = (time.perf_counter() - started) / (index + 1)
                print(
                    f"[{index + 1}/{len(sequence)}] {name} {condition} "
                    f"({per:.1f}s/shot)",
                    flush=True,
                )
        finally:
            cv2.destroyWindow(window_name)

    condition_names = ["no_comp"] + [c for c, _ in conditions]
    for name, images in warped_by_name.items():
        target_uint8 = next(t for n, c, t, _ in sequence if n == name)
        panels = [_label_panel(target_uint8, "target")] + [
            _label_panel(images[c], c) for c in condition_names if c in images
        ]
        _save_rgb(output_dir / f"comparison_{name}.png", np.hstack(panels))

    averages = {
        condition: {
            metric: float(
                np.mean(
                    [image_results[n][condition][metric] for n in image_results]
                )
            )
            for metric in ("psnr_db", "ssim", "delta_e76")
        }
        for condition in condition_names
    }
    payload = {
        "config": str(args.config),
        "capture": "immediate_single_exposure",
        "capture_encoding": "linear_uint8",
        "settle_seconds": args.settle_seconds,
        "conditions": condition_names,
        "targets": list(image_results.keys()),
        "per_image": image_results,
        "averages": averages,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(averages, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
