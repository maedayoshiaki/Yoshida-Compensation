"""Rebuild compensation images from a saved camera-space colour matrix.

This avoids recapturing photometric calibration patterns when only the
camera/projector geometric map has changed.
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
from src.python import inverse_lut
from yoshidacomp.config import reload_config
from yoshidacomp.compensation.projection_region import mask_to_active_region, resolve_active_region


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--target-dir",
        type=Path,
        help="Override paths.target_image_dir from the configuration.",
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--linearizer",
        choices=("none", "gamma", "inverse_lut"),
        required=True,
    )
    parser.add_argument("--lut", type=Path)
    parser.add_argument(
        "--target-decode",
        choices=("none", "srgb", "gamma"),
        default="srgb",
        help=(
            "Use normalized target code values directly, decode with the exact "
            "sRGB EOTF, or decode with a simple power-law gamma."
        ),
    )
    parser.add_argument(
        "--target-gamma",
        type=float,
        default=2.2,
        help="Power used when --target-decode=gamma (default: 2.2).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _decode_target(
    target_uint8: np.ndarray,
    *,
    mode: str,
    gamma: float,
) -> np.ndarray:
    """Convert uint8 target values into the compensation target domain."""
    normalized = target_uint8.astype(np.float32) / 255.0
    if mode == "none":
        return normalized
    if mode == "srgb":
        return sample.srgb_to_linear(normalized).astype(np.float32)
    if mode == "gamma":
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("--target-gamma must be a finite positive number.")
        return np.power(normalized, gamma).astype(np.float32)
    raise ValueError(f"Unsupported target decode mode: {mode!r}")


def main() -> None:
    args = _parse_args()
    cfg = reload_config(args.config)
    if cfg.camera.half_size:
        raise ValueError(
            "Half-resolution RAW is prohibited; camera.half_size must be false."
        )
    if cfg.paths.warp_method != "p2c":
        raise ValueError("Offline rebuild currently requires paths.warp_method='p2c'.")
    if args.linearizer == "inverse_lut" and args.lut is None:
        raise ValueError("--lut is required for --linearizer inverse_lut.")

    proj = cfg.projector
    active_region = resolve_active_region(proj)
    target_dir = args.target_dir or Path(cfg.paths.target_image_dir)
    targets = sample._load_target_images(str(target_dir))
    if not targets:
        raise FileNotFoundError(
            f"No target images found in {target_dir}"
        )

    started = time.perf_counter()
    matrix = np.load(args.matrix, mmap_mode="r")
    if matrix.ndim != 4 or matrix.shape[2:] != (4, 3):
        raise ValueError(
            f"Expected matrix shape (H, W, 4, 3), got {matrix.shape}."
        )
    load_seconds = time.perf_counter() - started

    warp_started = time.perf_counter()
    warped_matrix = sample.warp_color_mixing_matrices_to_projector(
        matrix,
        cfg.paths.p2c_map,
        proj.width,
        proj.height,
        proj.width,
        proj.height,
        warp_method="p2c",
    )
    warp_seconds = time.perf_counter() - warp_started

    lut = None
    if args.linearizer == "inverse_lut":
        lut = inverse_lut.load_inverse_lut(args.lut)

    linear_dir = args.output_dir / "compensation_linear"
    projector_dir = args.output_dir / "compensation_projector"
    per_target: list[dict[str, object]] = []
    for index, (target_name, target_uint8) in enumerate(targets):
        target_started = time.perf_counter()
        target_linear = _decode_target(
            target_uint8,
            mode=args.target_decode,
            gamma=args.target_gamma,
        )
        target_linear = sample.embed_in_active_region(
            target_linear,
            (proj.width, proj.height),
            active_region,
        )
        compensation = sample.calc_compensation_image(
            target_image=target_linear,
            color_mixing_matrices=warped_matrix,
            dtype=np.uint8,
        )
        compensation = mask_to_active_region(compensation, active_region)

        if args.linearizer == "none":
            projector_image = compensation
        elif args.linearizer == "gamma":
            projector_image = sample.apply_inverse_gamma_correction(
                compensation,
                gamma=1 / proj.gamma,
            )
            if hasattr(projector_image, "cpu"):
                projector_image = projector_image.cpu().numpy()
        else:
            projector_image = inverse_lut.apply_inverse_lut(
                compensation,
                lut,
                dtype=np.uint8,
            )

        _save_rgb(
            linear_dir / f"compensation_image_{index:02d}.png",
            compensation,
        )
        suffix = {
            "none": "no_gamma",
            "gamma": "inv_gamma",
            "inverse_lut": "inv_lut",
        }[args.linearizer]
        _save_rgb(
            projector_dir / f"{suffix}_comp_image_{index:02d}.png",
            np.asarray(projector_image, dtype=np.uint8),
        )
        per_target.append(
            {
                "name": target_name,
                "seconds": time.perf_counter() - target_started,
            }
        )

    metadata = {
        "config": str(args.config),
        "camera_half_size": cfg.camera.half_size,
        "matrix": str(args.matrix),
        "matrix_shape": list(matrix.shape),
        "p2c_map": cfg.paths.p2c_map,
        "linearizer": args.linearizer,
        "inverse_lut": str(args.lut) if args.lut else None,
        "target_decode": args.target_decode,
        "target_gamma": args.target_gamma if args.target_decode == "gamma" else None,
        "target_dir": str(target_dir),
        "active_region": list(active_region),
        "matrix_load_seconds": load_seconds,
        "matrix_warp_seconds": warp_seconds,
        "targets": per_target,
        "total_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rebuild_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
