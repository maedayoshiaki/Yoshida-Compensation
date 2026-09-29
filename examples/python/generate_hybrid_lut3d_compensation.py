"""Hybrid compensation: per-pixel colour matrix + TPS-baked 3-D LUT.

Fits, from the SAME calibration frames as the matrix/TPS conditions
(``capture_tps_calibration.py`` output), a pipeline-shaped hybrid:

    target T --[per-pixel 4x3 matrix]--> linear intent c
             --[baked 3-D LUT (TPS inverse, trilinear)]--> projector code

The matrix is fitted against the *normalized reference response* ``c_k``
(spatial mean of each calibration frame, black-subtracted and
white-normalized), and the 3-D LUT is the reference-point inverse map
``c -> p``. This puts a measured 3-D map in the exact pipeline seat the
per-channel 1-D inverse LUT occupies today, while the matrix keeps the
per-pixel spatial correction.

ハイブリッド補償: 「目標 → 画素別4×3行列 → TPS焼き込み3D LUT（三線形）→
投影コード」。行列の目標は各校正色の基準応答（空間平均を黒引き・白正規化した
c_k）、3D LUT は基準点の逆写像 c→p。現行パイプラインの 1D 逆LUT の席に
実測3D写像を差し替えつつ、画素別の空間補正は行列が担う。

Usage:
    uv run --no-sync python examples/python/generate_hybrid_lut3d_compensation.py \
        --config config.color_grid_iso200_hdr_d65_20260812.toml \
        --calibration-dir data/tps_calibration_20260812 \
        --output-dir data/tps_compensation_20260812
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

from examples.python.generate_tps_and_matrix_compensation import (
    _load_calibration,
    fit_pixel_matrices,
    upsample_matrices,
)
from yoshidacomp.compensation import color_lut3d as cl3
from yoshidacomp.compensation.photometric_compensation import (
    calc_compensation_image,
    quantize_unit_image,
)
from yoshidacomp.compensation.pipeline import load_target_images
from yoshidacomp.compensation.projection_region import (
    embed_in_active_region,
    mask_to_active_region,
    resolve_active_region,
)
from yoshidacomp.config import reload_config

CONDITION = "comp_matrix_lut3d"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument("--lut-size", type=int, default=33)
    parser.add_argument("--smoothing", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = reload_config(args.config)
    proj = cfg.projector
    active_region = resolve_active_region(proj)
    x0, y0, aw, ah = active_region

    meta = json.loads(
        (args.calibration_dir / "metadata.json").read_text(encoding="utf-8")
    )
    ds = int(meta["downsample"])
    frames, codes = _load_calibration(args.calibration_dir)
    print(f"Loaded {len(frames)} calibration frames (ds={ds}).")

    # Reference response = spatial mean over the active region per colour.
    region = np.s_[y0 // ds : (y0 + ah) // ds, x0 // ds : (x0 + aw) // ds]
    mean_q = frames[:, region[0], region[1], :].reshape(len(frames), -1, 3).mean(
        axis=1
    )
    c = cl3.normalize_reference_response(mean_q, codes)
    lut = cl3.bake_inverse_lut3d(
        c, codes, size=args.lut_size, smoothing=args.smoothing
    )
    print(
        f"Baked {args.lut_size}^3 LUT (smoothing {args.smoothing:g}, "
        f"input box {np.round(lut.meta['input_lo'], 3)}..)"
    )

    # Per-pixel matrix mapping measured colour -> linear intent c.
    started = time.perf_counter()
    matrices_c = fit_pixel_matrices(frames, c)
    matrices_c_full = upsample_matrices(matrices_c, (proj.width, proj.height))
    print(f"Matrix fit (target=c): {time.perf_counter() - started:.1f}s")

    output_dir = args.output_dir
    cond_dir = output_dir / CONDITION
    cond_dir.mkdir(parents=True, exist_ok=True)
    cl3.save_lut3d(output_dir / "inverse_lut3d.npz", lut)
    np.save(output_dir / "matrices_c_ds.npy", matrices_c)

    target_dir = args.target_dir or Path(cfg.paths.target_image_dir)
    targets = load_target_images(str(target_dir))
    for index, (name, target_uint8) in enumerate(targets):
        target_full = embed_in_active_region(
            target_uint8.astype(np.float32) / 255.0,
            (proj.width, proj.height),
            active_region,
        )
        intent = calc_compensation_image(
            target_full, matrices_c_full, dtype=np.float32, clip=True
        )
        comp = cl3.apply_lut3d(intent, lut, dtype=np.float32)
        comp = mask_to_active_region(comp, active_region)
        cv2.imwrite(
            str(cond_dir / f"comp_image_{index:02d}.png"),
            cv2.cvtColor(quantize_unit_image(comp, np.uint8), cv2.COLOR_RGB2BGR),
        )
        print(f"[{index + 1}/{len(targets)}] {name}")

    (output_dir / "hybrid_metadata.json").write_text(
        json.dumps(
            {
                "condition": CONDITION,
                "calibration_dir": str(args.calibration_dir),
                "lut_size": args.lut_size,
                "smoothing": args.smoothing,
                "target_decode": "none",
                "model": "target -> per-pixel 4x3 matrix (to normalized "
                "reference response) -> baked TPS inverse 3D LUT",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Hybrid compensation images written to {cond_dir}")


if __name__ == "__main__":
    main()
