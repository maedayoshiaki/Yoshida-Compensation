"""Capture a 6^3 colour calibration and store projector-space frames.

Projects the 216-colour GRID6 set (plus drift sentinels) with the
saturation-guarded immediate HDR capture path, warps every frame to
projector space, downsamples it, and stores one ``.npz`` per colour plus a
JSONL manifest (crash-safe, resumable). The stored frames feed BOTH the
per-pixel matrix fit and the node-grid TPS fit, so the two compensation
models can be compared from identical calibration data.

6³=216色の校正セット（+ドリフト番兵）を飽和ガード付き即時HDR撮影で投影・
撮影し、各フレームをプロジェクタ座標へワープ→ダウンサンプルして色ごとの
``.npz`` と JSONL マニフェストに逐次保存する（再開可能）。保存フレームは
画素別行列 fit とノードTPS fit の両方に使い、同一データでモデル比較する。

Usage:
    uv run --no-sync python examples/python/capture_tps_calibration.py \
        --config config.color_grid_iso200_hdr_d65_20260812.toml \
        --output-dir data/tps_calibration_20260812 [--downsample 4]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.python import sample
from examples.python.measure_color_grid_response import (
    DRIVE_SENTINEL,
    GRID_LEVELS,
    SENTINEL_RGB,
    PatchJob,
    filter_pending,
    load_completed_job_ids,
    plan_jobs,
)
from yoshidacomp.compensation.inverse_lut import (
    _PROJECTION_SETTLE_SECONDS,
    capture_hdr_linear,
    scale_shutter_tv,
    snap_to_flicker_free_tv,
    tv_to_seconds,
)
from yoshidacomp.compensation.projection_region import (
    mask_to_active_region,
    resolve_active_region,
)
from yoshidacomp.config import reload_config


def build_grid_patch_set() -> List[Tuple[int, int, int]]:
    """The 6^3 grid alone (the ramps/diagonals stay in the response study).

    6³ グリッドのみ（ランプ・対角線は応答調査側の設計）。
    """
    return [
        (r, g, b)
        for r in GRID_LEVELS
        for g in GRID_LEVELS
        for b in GRID_LEVELS
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sentinel-every", type=int, default=30)
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="spatial downsample factor for the stored projector-space frames",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="projector settle after a pattern switch (adaptive-brightness "
        "transient is ~1%% at 2s; see docs/color_grid_response_20260812.md)",
    )
    parser.add_argument("--max-jobs", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    from yoshidacomp.capture import create_camera_backend

    args = _parse_args()
    cfg = reload_config(args.config)
    if cfg.camera.half_size:
        raise ValueError("Full-resolution RAW is required; half_size must be false.")
    proj = cfg.projector
    ilut_cfg = cfg.inverse_lut
    active_region = resolve_active_region(proj)
    roi = (ilut_cfg.roi_x, ilut_cfg.roi_y, ilut_cfg.roi_width, ilut_cfg.roi_height)
    hdr_enabled = bool(ilut_cfg.hdr_enabled)
    ds = max(1, int(args.downsample))

    jobs = plan_jobs(
        build_grid_patch_set(), [], seed=args.seed,
        sentinel_every=args.sentinel_every,
    )
    output_dir = args.output_dir
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    completed = load_completed_job_ids(manifest_path)
    pending = filter_pending(jobs, completed)
    print(
        f"Plan: {len(jobs)} jobs, {len(completed)} done, {len(pending)} pending."
    )

    refresh_hz = float(getattr(proj, "refresh_rate", 0.0) or 0.0)
    base_tv = cfg.camera.tv
    if refresh_hz > 0:
        base_tv = snap_to_flicker_free_tv(base_tv, refresh_hz=refresh_hz)
    base_seconds = tv_to_seconds(base_tv)

    metadata = {
        "config": str(args.config),
        "purpose": "6^3 calibration frames for matrix-vs-TPS compensation",
        "downsample": ds,
        "settle_seconds": args.settle_seconds,
        "seed": args.seed,
        "camera": {
            "iso": cfg.camera.iso,
            "av": cfg.camera.av,
            "base_tv": base_tv,
            "hdr_enabled": hdr_enabled,
            "trust_low": ilut_cfg.trust_low,
            "trust_high": ilut_cfg.trust_high,
            "hdr_max_stops": ilut_cfg.hdr_max_stops,
        },
        "projector": {
            "frame": [proj.width, proj.height],
            "active_region": list(active_region),
        },
        "p2c_map": cfg.paths.p2c_map,
        "roi_gate": list(roi),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    camera_backend = create_camera_backend(cfg.camera)
    window_name = "tps_calibration"

    def make_solid(rgb: Tuple[int, int, int]) -> np.ndarray:
        image = np.empty((proj.height, proj.width, 3), dtype=np.uint8)
        image[:, :] = np.asarray(rgb, dtype=np.uint8)
        return mask_to_active_region(image, active_region)

    def roi_mean(frame: np.ndarray) -> np.ndarray:
        x, y, w, h = roi
        return frame[y : y + h, x : x + w, :3].reshape(-1, 3).mean(axis=0)

    with camera_backend:
        supported_tv: Optional[List[str]] = None
        if hdr_enabled:
            try:
                supported_tv = camera_backend.supported_shutter_speeds()
            except Exception as exc:  # noqa: BLE001
                print(f"supported Tv unavailable ({exc!r}).", file=sys.stderr)
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
            done = 0
            started = time.perf_counter()
            for job in pending:
                if args.max_jobs and done >= args.max_jobs:
                    break
                image = make_solid(job.requested_rgb)
                exposures_taken: List[Tuple[str, float]] = []
                # capture_hdr_linear needs the FULL frame here (the merge and
                # the guard both act on it); reduce only after the merge.
                last_frame: dict = {}

                def capture_at(scale: float):
                    cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(cfg.camera.wait_key_ms)
                    if scale == 1.0:
                        time.sleep(args.settle_seconds)
                    if scale == 1.0 or not hdr_enabled:
                        tv_label, ratio = base_tv, 1.0
                    else:
                        tv_label = scale_shutter_tv(
                            base_tv, scale,
                            supported=supported_tv, refresh_hz=refresh_hz,
                        )
                        ratio = tv_to_seconds(tv_label) / base_seconds
                    if hdr_enabled:
                        frame = camera_backend.capture_linear_rgb(tv=tv_label)
                    else:
                        frame = camera_backend.capture_linear_rgb()
                    exposures_taken.append((tv_label, float(ratio)))
                    frame = np.asarray(frame, dtype=np.float32)
                    return frame if scale == 1.0 else (frame, ratio)

                job_started = time.perf_counter()
                record = {
                    "job_id": job.job_id,
                    "drive": job.drive,
                    "requested_rgb": list(job.requested_rgb),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                try:
                    merged = capture_hdr_linear(
                        capture_at,
                        enabled=hdr_enabled,
                        trust_low=ilut_cfg.trust_low,
                        trust_high=ilut_cfg.trust_high,
                        max_stops=ilut_cfg.hdr_max_stops,
                    )
                    mean = roi_mean(merged)
                    warped = sample.warp_image(
                        merged,
                        cfg.paths.p2c_map,
                        proj.width,
                        proj.height,
                        proj.width,
                        proj.height,
                        warp_method="p2c",
                    )
                    del merged
                    gc.collect()
                    warped = np.asarray(warped, dtype=np.float32)
                    small = cv2.resize(
                        warped,
                        (proj.width // ds, proj.height // ds),
                        interpolation=cv2.INTER_AREA,
                    )
                    del warped
                    gc.collect()
                    frame_name = f"frame_{job.job_id:03d}.npz"
                    np.savez_compressed(
                        frames_dir / frame_name,
                        frame=small.astype(np.float32),
                        requested_rgb=np.asarray(job.requested_rgb, np.uint8),
                        drive=job.drive,
                    )
                    record.update(
                        {
                            "status": "ok",
                            "frame": f"frames/{frame_name}",
                            "roi_mean": [float(v) for v in mean],
                            "exposures": [[t, r] for t, r in exposures_taken],
                            "elapsed_seconds": time.perf_counter() - job_started,
                        }
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 - logged, run continues
                    record.update({"status": "error", "error": repr(exc)})
                    print(f"job {job.job_id} failed: {exc!r}", file=sys.stderr)
                with manifest_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                done += 1
                per = (time.perf_counter() - started) / done
                print(
                    f"[{done}/{len(pending)}] job {job.job_id} "
                    f"{job.requested_rgb} ({per:.1f}s/patch, "
                    f"ETA {per * (len(pending) - done) / 60:.0f}min)",
                    flush=True,
                )
        finally:
            cv2.destroyWindow(window_name)
    print("Calibration capture finished.")


if __name__ == "__main__":
    main()
