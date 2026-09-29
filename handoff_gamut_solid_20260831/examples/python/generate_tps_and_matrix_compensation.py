"""Fit matrix and node-TPS compensation from stored calibration frames.

Consumes the output of ``capture_tps_calibration.py`` (projector-space,
downsampled linear frames of the 6^3 grid) and produces compensation
images for three conditions fitted from IDENTICAL data:

- ``comp_matrix``       per-pixel affine inverse map ``p = [q, 1] @ M``
                        (the classic ``comp_no_gamma`` model class)
- ``comp_gamma_matrix`` per-pixel affine map to linear intent followed by
                        the analytic 1/2.2 linearizer (``comp_gamma`` class)
- ``comp_tps``          node-grid thin-plate-spline inverse map
                        (Grundhöfer & Iwai style, block_tps_map)

``capture_tps_calibration.py`` の保存フレーム（プロジェクタ座標・縮小・
リニア）から、同一データで3条件の補償画像を生成する。

Usage:
    uv run --no-sync python examples/python/generate_tps_and_matrix_compensation.py \
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
from typing import Dict, List, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yoshidacomp.compensation import block_tps_map as btm
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

GAMMA = 2.2


def fit_pixel_matrices(
    frames: np.ndarray, codes: np.ndarray, ridge: float = 1e-8
) -> np.ndarray:
    """Per-pixel least-squares inverse map ``codes ~= [q, 1] @ M``.

    画素ごとの最小二乗逆写像。``frames`` は (N, H, W, 3)、``codes`` は
    (N, 3)。返り値は (H, W, 4, 3)。有効領域の外（q が退化する画素）でも
    解けるよう、正規方程式に微小リッジを足す。
    """
    frames = np.asarray(frames, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.float64)
    n, height, width, _ = frames.shape
    design = np.concatenate(
        [frames, np.ones((n, height, width, 1))], axis=-1
    )  # (N, H, W, 4)
    xtx = np.einsum("nhwi,nhwj->hwij", design, design)
    xtx += ridge * np.eye(4)
    xty = np.einsum("nhwi,nj->hwij", design, codes)
    return np.linalg.solve(xtx, xty).astype(np.float32)  # (H, W, 4, 3)


def upsample_matrices(
    matrices: np.ndarray, frame_size: Tuple[int, int]
) -> np.ndarray:
    """Bilinearly upsample a (h, w, 4, 3) matrix field to ``frame_size``.

    行列場 (h, w, 4, 3) を (width, height) へ双線形アップサンプルする。
    """
    width, height = frame_size
    h, w = matrices.shape[:2]
    flat = matrices.reshape(h, w, 12)
    up = cv2.resize(flat, (width, height), interpolation=cv2.INTER_LINEAR)
    return up.reshape(height, width, 4, 3)


def _load_calibration(calibration_dir: Path):
    """Load stored frames and codes (non-sentinel, status ok)."""
    manifest = [
        json.loads(line)
        for line in (calibration_dir / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    rows: Dict[Tuple[int, int, int], dict] = {}
    for row in manifest:
        if row.get("status") != "ok" or row.get("drive") == "sentinel":
            continue
        rows[tuple(row["requested_rgb"])] = row  # last capture wins
    codes = []
    frames = []
    for rgb, row in sorted(rows.items()):
        data = np.load(calibration_dir / row["frame"])
        frames.append(np.asarray(data["frame"], dtype=np.float32))
        codes.append(np.asarray(rgb, dtype=np.float64) / 255.0)
    return np.stack(frames), np.asarray(codes)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument("--node-grid", type=int, default=33)
    parser.add_argument(
        "--node-window",
        type=int,
        default=5,
        help="node averaging window in DOWNSAMPLED pixels",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=-1.0,
        help="TPS regularisation; negative = automatic LOOCV selection",
    )
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
    print(f"Loaded {len(frames)} calibration frames {frames.shape[1:]} (ds={ds}).")

    # ── matrix fits (both model classes) on downsampled frames ─────────
    started = time.perf_counter()
    matrices_p = fit_pixel_matrices(frames, codes)
    matrices_t = fit_pixel_matrices(frames, codes**GAMMA)
    print(f"Matrix fits: {time.perf_counter() - started:.1f}s")
    matrices_p_full = upsample_matrices(matrices_p, (proj.width, proj.height))
    matrices_t_full = upsample_matrices(matrices_t, (proj.width, proj.height))

    # ── node TPS fit ────────────────────────────────────────────────────
    started = time.perf_counter()
    node_y, node_x = btm.node_grid(active_region, (args.node_grid, args.node_grid))
    samples = btm.extract_node_samples(
        list(frames), node_y / ds, node_x / ds, window=args.node_window
    )
    smoothing = None if args.smoothing < 0 else float(args.smoothing)
    tps_map = btm.fit_block_tps_map(
        samples, codes, node_y, node_x, active_region, smoothing=smoothing
    )
    print(
        f"TPS fit ({args.node_grid}x{args.node_grid} nodes, smoothing="
        f"{tps_map.smoothing:g}): {time.perf_counter() - started:.1f}s"
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    btm.save_block_tps_map(output_dir / "tps_map.npz", tps_map)
    np.save(output_dir / "matrices_p_ds.npy", matrices_p)
    np.save(output_dir / "matrices_t_ds.npy", matrices_t)

    # ── compensation images ─────────────────────────────────────────────
    target_dir = args.target_dir or Path(cfg.paths.target_image_dir)
    targets = load_target_images(str(target_dir))
    names = []
    for index, (name, target_uint8) in enumerate(targets):
        names.append(name)
        # Linear unification: target code values ARE the linear intent
        # (--target-decode none of the r3 chain).
        target_linear = target_uint8.astype(np.float32) / 255.0
        target_full = embed_in_active_region(
            target_linear, (proj.width, proj.height), active_region
        )

        outputs = {}
        outputs["comp_matrix"] = calc_compensation_image(
            target_full, matrices_p_full, dtype=np.float32, clip=True
        )
        linear_intent = calc_compensation_image(
            target_full, matrices_t_full, dtype=np.float32, clip=True
        )
        outputs["comp_gamma_matrix"] = np.power(
            linear_intent, 1.0 / GAMMA, dtype=np.float32
        )
        started = time.perf_counter()
        outputs["comp_tps"] = btm.apply_block_tps_map(tps_map, target_full)
        tps_seconds = time.perf_counter() - started

        for condition, image in outputs.items():
            image = mask_to_active_region(image, active_region)
            cond_dir = output_dir / condition
            cond_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(cond_dir / f"comp_image_{index:02d}.png"),
                cv2.cvtColor(
                    quantize_unit_image(image, np.uint8), cv2.COLOR_RGB2BGR
                ),
            )
        print(
            f"[{index + 1}/{len(targets)}] {name} done (TPS {tps_seconds:.1f}s)"
        )

    (output_dir / "generation_metadata.json").write_text(
        json.dumps(
            {
                "calibration_dir": str(args.calibration_dir),
                "config": str(args.config),
                "targets": names,
                "target_decode": "none",
                "conditions": ["comp_matrix", "comp_gamma_matrix", "comp_tps"],
                "node_grid": args.node_grid,
                "node_window_ds_px": args.node_window,
                "tps_smoothing": tps_map.smoothing,
                "gamma": GAMMA,
                "downsample": ds,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Compensation images written to {output_dir}")


if __name__ == "__main__":
    main()
