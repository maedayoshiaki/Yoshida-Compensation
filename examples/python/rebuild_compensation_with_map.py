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

from yoshidacomp.geometry.graycode_path import ensure_on_syspath

# warp_color_mixing_matrices_to_projector() imports graycode lazily, so the
# submodule still has to be on sys.path before that call happens.
# warp_color_mixing_matrices_to_projector() は graycode を遅延importするため、
# 呼び出し前に submodule を sys.path に載せておく必要がある。
ensure_on_syspath()

from yoshidacomp.compensation import inverse_lut
from yoshidacomp.compensation.pipeline import (
    LINEARIZER_FILENAME_PREFIXES,
    generate_projection_image,
    load_target_images,
)
from yoshidacomp.config import reload_config
from yoshidacomp.compensation.projection_region import resolve_active_region
from yoshidacomp.geometry.warp import warp_color_mixing_matrices_to_projector


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
    """Write an ``(H, W, 3)`` uint8 RGB array as a PNG.

    Raises:
        OSError: ``cv2.imwrite`` refused the path. It signals failure by
            returning ``False`` rather than raising, and the return value used
            to be discarded here -- so a run that could not write (a locked
            file, a full disk, a read-only directory) still reached the end of
            ``main()`` and printed a complete metadata JSON to stdout,
            describing files that are not there. This writes into the same
            99 GB tree the web endpoint does, and prints success more loudly.
            ``packages/web/yoshidacomp_web/generate.py::_save_rgb`` checks the
            same value for the same reason.
            ``cv2.imwrite`` は失敗しても例外を出さず ``False`` を返す。戻り値を
            捨てていたため、書けなかった実行でもメタデータJSONを「成功」として
            表示していた。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {path} (cv2.imwrite refused the path)")


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
    targets = load_target_images(str(target_dir))
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
    warped_matrix = warp_color_mixing_matrices_to_projector(
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
        compensation, projector_image = generate_projection_image(
            target_uint8,
            warped_matrix,
            projector=proj,
            active_region=active_region,
            linearizer=args.linearizer,
            lut=lut,
            target_decode=args.target_decode,
            target_gamma=args.target_gamma,
        )

        _save_rgb(
            linear_dir / f"compensation_image_{index:02d}.png",
            compensation,
        )
        # Read from the core rather than written out here: ``POST /api/generate``
        # (``packages/web/yoshidacomp_web/generate.py``) writes the same files
        # and a second copy of this mapping would be two dicts that must agree
        # with nothing making them. See ``LINEARIZER_FILENAME_PREFIXES``.
        suffix = LINEARIZER_FILENAME_PREFIXES[args.linearizer]
        _save_rgb(
            projector_dir / f"{suffix}_comp_image_{index:02d}.png",
            projector_image,
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
