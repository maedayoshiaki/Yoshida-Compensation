"""Measure the projector->camera colour response over a 3-D RGB patch set.

Project solid-colour patches spanning the RGB input cube -- a 6x6x6 grid,
17-level single-channel / gray ramps, and 17-level two-channel diagonals
(321 unique colours) -- capture each with the calibrated HDR pipeline, and
record the linear-RGB mean of the measurement ROI to a JSONL file, one line
per capture (crash-safe, resumable). A second, smaller set is driven
through the measured gray-fit inverse LUT so the linearity of the
*linearised* response can be verified directly.

入力RGB立方体を面的にカバーする単色パッチ群（6³グリッド、17段の単ch/グレー
ランプ、17段の2ch対角線、計321色）を投影し、HDR撮影で計測ROIのリニアRGB
平均を JSONL に1行ずつ逐次記録する（クラッシュ耐性・再開可能）。加えて
gray-fit 逆LUTを通した小セットを駆動し、「線形化後に本当に線形か」を直接
検証できるようにする。

設計書: docs/superpowers/specs/2026-08-12-color-grid-response-design.md

Usage:
    uv run --no-sync python examples/python/measure_color_grid_response.py \
        --config config.color_grid_iso200_hdr_d65_20260812.toml \
        --lut data/iso200_hdr_isoscan_d65_20260807/inverse_lut_gray/inverse_lut.npz \
        --output-dir data/color_grid_response_20260812 [--smoke-only] [--max-jobs N]
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

from yoshidacomp.compensation.inverse_lut import (
    _PROJECTION_SETTLE_SECONDS,
    apply_inverse_lut,
    capture_hdr_linear,
    load_inverse_lut,
    scale_shutter_tv,
    snap_to_flicker_free_tv,
    tv_to_seconds,
)
from yoshidacomp.compensation.projection_region import (
    mask_to_active_region,
    resolve_active_region,
)
from yoshidacomp.config import reload_config

# Drive kinds recorded in the JSONL. 記録される駆動系の種別。
DRIVE_RAW = "raw"
DRIVE_LUT_GRAY = "lut_gray"
DRIVE_SENTINEL = "sentinel"

SENTINEL_RGB: Tuple[int, int, int] = (128, 128, 128)

# 6^3 grid levels (exact multiples of 51) and 17-level ramp values.
# 6³ グリッドのレベル（51の倍数）と17段ランプの8bit値。
GRID_LEVELS: Tuple[int, ...] = (0, 51, 102, 153, 204, 255)
RAMP_STEPS = 17


@dataclass(frozen=True)
class PatchJob:
    """One planned projection+capture. 計画された投影・撮影1回分。"""

    job_id: int
    drive: str  # DRIVE_RAW / DRIVE_LUT_GRAY / DRIVE_SENTINEL
    requested_rgb: Tuple[int, int, int]


def ramp_values_8bit(steps: int = RAMP_STEPS) -> List[int]:
    """17 equally spaced 8-bit levels: round(i*255/16), i=0..16."""
    return [int(round(i * 255.0 / (steps - 1))) for i in range(steps)]


def build_raw_patch_set() -> List[Tuple[int, int, int]]:
    """All raw-drive patches: grid + ramps + diagonals, deduped (321 total).

    生駆動パッチ全部: グリッド+ランプ+対角線、重複排除で計321色。
    順序はグリッド→R/G/Bランプ→グレー→対角線（初出順を保持）。
    """
    patches: List[Tuple[int, int, int]] = []
    seen: set = set()

    def add(rgb: Tuple[int, int, int]) -> None:
        if rgb not in seen:
            seen.add(rgb)
            patches.append(rgb)

    for r in GRID_LEVELS:
        for g in GRID_LEVELS:
            for b in GRID_LEVELS:
                add((r, g, b))
    ramp = ramp_values_8bit()
    for channel in range(3):
        for v in ramp:
            rgb = [0, 0, 0]
            rgb[channel] = v
            add(tuple(rgb))
    for v in ramp:
        add((v, v, v))
    for pair in ((0, 1), (0, 2), (1, 2)):
        for v in ramp:
            rgb = [0, 0, 0]
            rgb[pair[0]] = v
            rgb[pair[1]] = v
            add(tuple(rgb))
    return patches


def build_lut_check_set() -> List[Tuple[int, int, int]]:
    """Through-LUT verification patches (35 unique requested colours).

    LUT通し確認パッチ（要求色35色）: グレー17段 + 単ch4段 + 混合6色。
    """
    patches: List[Tuple[int, int, int]] = []
    seen: set = set()

    def add(rgb: Tuple[int, int, int]) -> None:
        if rgb not in seen:
            seen.add(rgb)
            patches.append(rgb)

    for v in ramp_values_8bit():
        add((v, v, v))
    for channel in range(3):
        for v in (51, 102, 153, 204):
            rgb = [0, 0, 0]
            rgb[channel] = v
            add(tuple(rgb))
    for rgb in (
        (128, 128, 0),
        (128, 0, 128),
        (0, 128, 128),
        (192, 64, 64),
        (64, 192, 64),
        (64, 64, 192),
    ):
        add(rgb)
    return patches


def plan_jobs(
    raw_patches: Sequence[Tuple[int, int, int]],
    lut_patches: Sequence[Tuple[int, int, int]],
    seed: int = 0,
    sentinel_every: int = 30,
) -> List[PatchJob]:
    """Build the full deterministic measurement plan.

    決定的な計測計画を組む: 生セット・LUTセットをそれぞれ固定シードで
    シャッフルして連結し、先頭・30枚ごと・末尾にドリフト番兵を挿入する。
    job_id は計画順の連番（再開時のスキップ判定キー）。
    """
    raw_order = list(raw_patches)
    random.Random(seed).shuffle(raw_order)
    lut_order = list(lut_patches)
    random.Random(seed + 1).shuffle(lut_order)

    measurements: List[Tuple[str, Tuple[int, int, int]]] = [
        (DRIVE_RAW, rgb) for rgb in raw_order
    ] + [(DRIVE_LUT_GRAY, rgb) for rgb in lut_order]

    jobs: List[PatchJob] = []

    def add(drive: str, rgb: Tuple[int, int, int]) -> None:
        jobs.append(PatchJob(job_id=len(jobs), drive=drive, requested_rgb=rgb))

    add(DRIVE_SENTINEL, SENTINEL_RGB)
    for index, (drive, rgb) in enumerate(measurements):
        if index > 0 and index % sentinel_every == 0:
            add(DRIVE_SENTINEL, SENTINEL_RGB)
        add(drive, rgb)
    add(DRIVE_SENTINEL, SENTINEL_RGB)
    return jobs


def load_completed_job_ids(jsonl_path: Path) -> set:
    """Job ids already measured successfully (status == "ok").

    JSONL から計測済み（status=="ok"）の job_id 集合を読む。壊れた行・
    error 行は未計測扱いにする（再実行で埋まる）。
    """
    completed: set = set()
    if not jsonl_path.exists():
        return completed
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "ok" and isinstance(record.get("job_id"), int):
            completed.add(record["job_id"])
    return completed


def filter_pending(jobs: Sequence[PatchJob], completed: set) -> List[PatchJob]:
    """Jobs still to run, preserving plan order. 未計測ジョブ（計画順）。"""
    return [job for job in jobs if job.job_id not in completed]


def projected_rgb_for_job(
    job: PatchJob, lut
) -> Tuple[int, int, int]:
    """8-bit colour actually sent to the projector for this job.

    このジョブで実際に投影する8bit色。raw/sentinel は要求色そのまま、
    lut_gray は逆LUT適用後の色。
    """
    if job.drive == DRIVE_LUT_GRAY:
        if lut is None:
            raise ValueError("lut_gray drive requires a loaded inverse LUT.")
        pixel = np.array([[list(job.requested_rgb)]], dtype=np.uint8)
        out = apply_inverse_lut(pixel, lut, dtype=np.uint8)
        return tuple(int(v) for v in np.asarray(out).reshape(3))
    return job.requested_rgb


def _roi_slice(
    frame: np.ndarray, roi: Tuple[int, int, int, int]
) -> np.ndarray:
    """ROI rectangle of a frame, clipped to bounds (copy-free view)."""
    height, width = frame.shape[:2]
    x, y, w, h = (int(v) for v in roi)
    x0 = max(0, min(x, width))
    y0 = max(0, min(y, height))
    x1 = max(x0, min(x + w, width))
    y1 = max(y0, min(y + h, height))
    patch = frame[y0:y1, x0:x1, :3]
    return patch if patch.size else frame[:, :, :3]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_camera_properties(camera_backend) -> Dict:
    """Best-effort readback of body properties (WB etc.) for the metadata.

    カメラ本体プロパティ（WB等）のベストエフォート読み戻し。失敗しても
    中断しない（metadata に記録するだけ）。
    """
    try:
        controller = getattr(camera_backend, "_controller", None)
        if controller is None or not hasattr(controller, "get_properties"):
            return {"status": "unavailable", "reason": "no live controller"}
        props = dict(controller.get_properties())
        try:
            import edsdk as _edsdk
            from edsdk.constants.properties import PropID

            props["ColorTemperature"] = int(
                _edsdk.GetPropertyData(
                    controller._cam, PropID.ColorTemperature, 0
                )
            )
        except Exception:  # noqa: BLE001 - metadata only, never fatal
            pass
        return {"status": "ok", "properties": props}
    except Exception as exc:  # noqa: BLE001 - metadata only, never fatal
        return {"status": "error", "error": repr(exc)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--lut",
        type=Path,
        required=True,
        help="gray-fit inverse LUT (.npz) for the through-LUT check set",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sentinel-every", type=int, default=30)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="run only the black/white smoke test and exit",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        help="stop after this many captures (0 = no limit; for testing)",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="save the base-exposure ROI crop PNG per patch",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=_PROJECTION_SETTLE_SECONDS,
        help=(
            "projector settle time after switching patterns. The projector's "
            "adaptive brightness (iris) needs ~2s to stabilize after a "
            "content change (measured 2026-08-12: +-2-4%% at 0.6s, ~1%% at "
            "1-2s, stable by 7s)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    import cv2

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
    if hdr_enabled and cfg.camera.backend.strip().lower() != "canon_edsdk":
        print(
            "measure_color_grid_response: HDR requires the canon_edsdk backend; "
            "falling back to single exposure.",
            file=sys.stderr,
        )
        hdr_enabled = False

    lut = load_inverse_lut(args.lut)

    jobs = plan_jobs(
        build_raw_patch_set(),
        build_lut_check_set(),
        seed=args.seed,
        sentinel_every=args.sentinel_every,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "measurements.jsonl"
    completed = load_completed_job_ids(jsonl_path)
    pending = filter_pending(jobs, completed)
    print(
        f"Plan: {len(jobs)} jobs total, {len(completed)} already measured, "
        f"{len(pending)} pending."
    )

    # Flicker-free shutter (integer multiple of the projector refresh period).
    refresh_hz = float(getattr(proj, "refresh_rate", 0.0) or 0.0)
    base_tv = cfg.camera.tv
    if refresh_hz > 0:
        snapped = snap_to_flicker_free_tv(base_tv, refresh_hz=refresh_hz)
        if snapped != base_tv:
            print(
                f"shutter {base_tv!r} is not flicker-free at {refresh_hz:g}Hz; "
                f"using {snapped!r}.",
                file=sys.stderr,
            )
        base_tv = snapped
    base_seconds = tv_to_seconds(base_tv)

    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    camera_backend = create_camera_backend(cfg.camera)
    window_name = "color_grid_response"

    def make_solid(projected_rgb: Tuple[int, int, int]) -> np.ndarray:
        image = np.empty((proj.height, proj.width, 3), dtype=np.uint8)
        image[:, :] = np.asarray(projected_rgb, dtype=np.uint8)
        return mask_to_active_region(image, active_region)

    def project(image: np.ndarray, settle: bool) -> None:
        cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(cfg.camera.wait_key_ms)
        if settle:
            time.sleep(args.settle_seconds)

    metadata = {
        "config": str(args.config),
        "spec": "docs/superpowers/specs/2026-08-12-color-grid-response-design.md",
        "lut": str(args.lut),
        "lut_source": lut.meta.get("source") if hasattr(lut, "meta") else None,
        "seed": args.seed,
        "sentinel_every": args.sentinel_every,
        "jobs_total": len(jobs),
        "jobs_precompleted": len(completed),
        "camera": {
            "backend": cfg.camera.backend,
            "iso": cfg.camera.iso,
            "av": cfg.camera.av,
            "base_tv": base_tv,
            "hdr_enabled": hdr_enabled,
            "hdr_max_stops": ilut_cfg.hdr_max_stops,
            "trust_low": ilut_cfg.trust_low,
            "trust_high": ilut_cfg.trust_high,
        },
        "measurement_roi": list(roi),
        "settle_seconds": args.settle_seconds,
        "projector": {
            "frame": [proj.width, proj.height],
            "active_region": list(active_region),
            "refresh_rate": proj.refresh_rate,
        },
        "started_at": _utc_now(),
    }

    with camera_backend:
        metadata["camera_properties_readback"] = _read_camera_properties(
            camera_backend
        )
        supported_tv: Optional[List[str]] = None
        if hdr_enabled:
            try:
                supported_tv = camera_backend.supported_shutter_speeds()
            except Exception as exc:  # noqa: BLE001 - snap falls back to table
                print(
                    f"supported_shutter_speeds unavailable ({exc!r}); "
                    "using the standard Tv table.",
                    file=sys.stderr,
                )

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
            # ── Smoke test: black vs white, single base exposure ────────
            smoke = {}
            for name, rgb in (("black", (0, 0, 0)), ("white", (255, 255, 255))):
                project(make_solid(rgb), settle=True)
                frame = camera_backend.capture_linear_rgb()
                patch = _roi_slice(np.asarray(frame, dtype=np.float32), roi)
                smoke[name] = {
                    "mean": [float(v) for v in patch.reshape(-1, 3).mean(axis=0)],
                    "std": [float(v) for v in patch.reshape(-1, 3).std(axis=0)],
                }
                del frame, patch
                gc.collect()
            black_g = smoke["black"]["mean"][1]
            white_g = smoke["white"]["mean"][1]
            metadata["smoke_test"] = smoke
            contrast_ok = white_g > max(10.0 * black_g, 0.02)
            level_ok = 0.05 <= white_g <= 1.2
            metadata["smoke_test"]["passed"] = bool(contrast_ok and level_ok)
            (output_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"Smoke test: black G={black_g:.5f}, white G={white_g:.5f} "
                f"-> {'OK' if metadata['smoke_test']['passed'] else 'FAILED'}"
            )
            if not metadata["smoke_test"]["passed"]:
                raise RuntimeError(
                    "Smoke test failed: projector response not detected at the "
                    f"measurement ROI (black G={black_g:.5f}, white G={white_g:.5f}). "
                    "Check projector window position, power, and camera setup."
                )
            if args.smoke_only:
                print("--smoke-only: exiting after smoke test.")
                return

            # ── Main measurement loop ───────────────────────────────────
            done_this_run = 0
            loop_started = time.perf_counter()
            for job in pending:
                if args.max_jobs and done_this_run >= args.max_jobs:
                    print(f"--max-jobs {args.max_jobs} reached; stopping.")
                    break
                projected = projected_rgb_for_job(job, lut)
                image = make_solid(projected)
                exposures_taken: List[Tuple[str, float]] = []
                base_std: List[float] = []
                job_started = time.perf_counter()

                def capture_at(scale: float):
                    # Brackets keep the same pattern; only the first exposure
                    # needs the projector settle delay.
                    project(image, settle=scale == 1.0)
                    if scale == 1.0 or not hdr_enabled:
                        tv_label, ratio = base_tv, 1.0
                    else:
                        tv_label = scale_shutter_tv(
                            base_tv,
                            scale,
                            supported=supported_tv,
                            refresh_hz=refresh_hz,
                        )
                        ratio = tv_to_seconds(tv_label) / base_seconds
                    if hdr_enabled:
                        frame = camera_backend.capture_linear_rgb(tv=tv_label)
                    else:
                        # Non-Canon fallbacks may not accept a tv override.
                        frame = camera_backend.capture_linear_rgb()
                    patch = _roi_slice(np.asarray(frame, dtype=np.float32), roi)
                    mean = patch.reshape(-1, 3).mean(axis=0).astype(np.float32)
                    if scale == 1.0:
                        base_std[:] = [
                            float(v) for v in patch.reshape(-1, 3).std(axis=0)
                        ]
                        if args.debug_dir is not None:
                            bgr = cv2.cvtColor(
                                (np.clip(patch, 0.0, 1.0) * 255.0)
                                .round()
                                .astype(np.uint8),
                                cv2.COLOR_RGB2BGR,
                            )
                            cv2.imwrite(
                                str(
                                    args.debug_dir
                                    / (
                                        f"job{job.job_id:03d}_{job.drive}_"
                                        f"r{projected[0]:03d}g{projected[1]:03d}"
                                        f"b{projected[2]:03d}.png"
                                    )
                                ),
                                bgr,
                            )
                    exposures_taken.append((tv_label, float(ratio)))
                    # Full-resolution frames are ~390 MB; free before the next
                    # RAW is developed (two live frames crash the decoder).
                    del frame, patch
                    gc.collect()
                    reduced = mean.reshape(1, 1, 3)
                    return reduced if scale == 1.0 else (reduced, ratio)

                record = {
                    "job_id": job.job_id,
                    "drive": job.drive,
                    "requested_rgb": list(job.requested_rgb),
                    "projected_rgb": list(projected),
                    "timestamp": _utc_now(),
                }
                try:
                    merged = capture_hdr_linear(
                        capture_at,
                        enabled=hdr_enabled,
                        trust_low=ilut_cfg.trust_low,
                        trust_high=ilut_cfg.trust_high,
                        max_stops=ilut_cfg.hdr_max_stops,
                    )
                    record.update(
                        {
                            "status": "ok",
                            "measured_mean": [
                                float(v)
                                for v in np.asarray(merged).reshape(3)
                            ],
                            "base_roi_std": base_std,
                            "exposures": [
                                [tv, ratio] for tv, ratio in exposures_taken
                            ],
                            "elapsed_seconds": time.perf_counter() - job_started,
                        }
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 - logged, run continues
                    record.update({"status": "error", "error": repr(exc)})
                    print(
                        f"job {job.job_id} failed: {exc!r} (continuing)",
                        file=sys.stderr,
                    )
                with jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                done_this_run += 1
                elapsed = time.perf_counter() - loop_started
                per_job = elapsed / done_this_run
                remaining = len(pending) - done_this_run
                print(
                    f"[{done_this_run}/{len(pending)}] job {job.job_id} "
                    f"{job.drive} {job.requested_rgb} -> "
                    f"{record.get('measured_mean', 'ERROR')} "
                    f"({per_job:.1f}s/patch, ETA {per_job * remaining / 60:.0f}min)"
                )
        finally:
            cv2.destroyWindow(window_name)

    metadata["finished_at"] = _utc_now()
    metadata["jobs_measured_this_run"] = done_this_run
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    failures = [
        j.job_id
        for j in jobs
        if j.job_id not in load_completed_job_ids(jsonl_path)
    ]
    if failures:
        print(
            f"Unmeasured/failed job ids ({len(failures)}): {failures[:20]}"
            f"{'...' if len(failures) > 20 else ''}\n"
            "Re-run the same command to fill them in."
        )
    else:
        print("All jobs measured.")


if __name__ == "__main__":
    main()
