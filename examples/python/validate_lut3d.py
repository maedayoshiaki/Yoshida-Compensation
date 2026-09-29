"""Validate a baked 3-D inverse LUT on real hardware.

Requests a set of normalized target colours ``c`` (gray ramp,
single-channel ramps, random in-gamut colours, held-out calibration
colours), maps each through the baked 3-D LUT to projector codes,
projects, captures, and measures the achieved normalized colour at the
LUT's own reference (active-region mean of the projector-space frame).
If the LUT is accurate, achieved ``c`` equals requested ``c`` -- the 3-D
analogue of ``validate_inverse_lut.py``, including for mixed colours the
1-D LUT provably cannot linearize.

焼き込み済み3D LUTの実機検証。要求色 c（グレー/単ch/ランダム/held-out）を
3D LUT で投影コードへ写して投影・撮影し、LUT自身の基準（プロジェクタ座標の
有効領域平均）で達成色を測る。正確なら達成 c ≒ 要求 c になる。
1D LUT では原理的に線形化できない混合色まで含めた検証。

Usage:
    uv run --no-sync python examples/python/validate_lut3d.py \
        --config config.color_grid_iso200_hdr_d65_20260812.toml \
        --lut data/tps_compensation_20260812/inverse_lut3d.npz \
        --calibration-dir data/tps_calibration_20260812 \
        --output-dir data/lut3d_validation_20260813
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
from yoshidacomp.compensation import color_lut3d as cl3
from yoshidacomp.compensation.inverse_lut import (
    capture_hdr_linear,
    scale_shutter_tv,
    snap_to_flicker_free_tv,
    tv_to_seconds,
)
from yoshidacomp.compensation.photometric_compensation import quantize_unit_image
from yoshidacomp.compensation.projection_region import (
    mask_to_active_region,
    resolve_active_region,
)
from yoshidacomp.config import reload_config

GRAY_STEPS = 17
SINGLE_LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)
N_RANDOM = 24
N_HELD_OUT = 12


def build_validation_requests(
    calibration_c: np.ndarray, seed: int = 0
) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Deterministic list of (set_name, requested c) validation colours.

    決定的な検証色リスト。ランダム色は校正実測色の軸平行ボックス内から
    固定シードで抽選し、held-out は校正実測色そのものから選ぶ
    （理想的な LUT なら達成色が要求色に一致するはずの点）。
    """
    rng = np.random.default_rng(seed)
    requests: List[Tuple[str, Tuple[float, float, float]]] = []
    for i in range(GRAY_STEPS):
        t = i / (GRAY_STEPS - 1)
        requests.append(("gray", (t, t, t)))
    for channel in range(3):
        for level in SINGLE_LEVELS:
            c = [0.0, 0.0, 0.0]
            c[channel] = level
            requests.append(("single", tuple(c)))
    lo = calibration_c.min(axis=0)
    hi = calibration_c.max(axis=0)
    for point in rng.uniform(lo, hi, size=(N_RANDOM, 3)):
        requests.append(("random_box", tuple(float(v) for v in point)))
    picks = rng.choice(len(calibration_c), size=N_HELD_OUT, replace=False)
    for index in picks:
        requests.append(
            ("held_out", tuple(float(v) for v in calibration_c[index]))
        )
    return requests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--downsample", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    from yoshidacomp.capture import create_camera_backend
    from examples.python.generate_tps_and_matrix_compensation import (
        _load_calibration,
    )

    args = _parse_args()
    cfg = reload_config(args.config)
    proj = cfg.projector
    ilut_cfg = cfg.inverse_lut
    active_region = resolve_active_region(proj)
    x0, y0, aw, ah = active_region
    ds = max(1, int(args.downsample))
    hdr_enabled = bool(ilut_cfg.hdr_enabled)

    lut = cl3.load_lut3d(args.lut)
    frames, codes = _load_calibration(args.calibration_dir)
    region = np.s_[y0 // ds : (y0 + ah) // ds, x0 // ds : (x0 + aw) // ds]
    mean_q = frames[:, region[0], region[1], :].reshape(len(frames), -1, 3).mean(
        axis=1
    )
    calibration_c = cl3.normalize_reference_response(mean_q, codes)
    del frames
    gc.collect()

    requests = build_validation_requests(calibration_c, seed=args.seed)
    # Raw black / white anchor the normalization of THIS run's measurements.
    jobs: List[Tuple[str, Tuple[float, float, float], Tuple[int, int, int]]] = [
        ("anchor_black", (0.0, 0.0, 0.0), (0, 0, 0)),
        ("anchor_white", (1.0, 1.0, 1.0), (255, 255, 255)),
    ]
    for set_name, c in requests:
        p = cl3.apply_lut3d(
            np.asarray(c, dtype=np.float32).reshape(1, 1, 3), lut
        ).reshape(3)
        p8 = tuple(int(v) for v in quantize_unit_image(p, np.uint8))
        jobs.append((set_name, c, p8))
    # Drift sentinels (raw mid gray) sprinkled through the run.
    for position in (2, len(jobs) // 2, len(jobs)):
        jobs.insert(position, ("sentinel", (-1.0, -1.0, -1.0), (128, 128, 128)))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "measurements.jsonl"
    done = 0
    if jsonl_path.exists():
        done = sum(
            1
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    print(f"{len(jobs)} jobs, resuming after {done}.")

    refresh_hz = float(getattr(proj, "refresh_rate", 0.0) or 0.0)
    base_tv = snap_to_flicker_free_tv(cfg.camera.tv, refresh_hz=refresh_hz)
    base_seconds = tv_to_seconds(base_tv)
    roi = (ilut_cfg.roi_x, ilut_cfg.roi_y, ilut_cfg.roi_width, ilut_cfg.roi_height)

    camera_backend = create_camera_backend(cfg.camera)
    window_name = "lut3d_validation"

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
            started = time.perf_counter()
            for index, (set_name, c_req, p8) in enumerate(jobs):
                if index < done:
                    continue
                image = np.empty((proj.height, proj.width, 3), dtype=np.uint8)
                image[:, :] = np.asarray(p8, dtype=np.uint8)
                image = mask_to_active_region(image, active_region)

                def capture_at(scale: float):
                    cv2.imshow(
                        window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    )
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
                    frame = np.asarray(frame, dtype=np.float32)
                    return frame if scale == 1.0 else (frame, ratio)

                merged = capture_hdr_linear(
                    capture_at,
                    enabled=hdr_enabled,
                    trust_low=ilut_cfg.trust_low,
                    trust_high=ilut_cfg.trust_high,
                    max_stops=ilut_cfg.hdr_max_stops,
                )
                warped = sample.warp_image(
                    merged, cfg.paths.p2c_map,
                    proj.width, proj.height, proj.width, proj.height,
                    warp_method="p2c",
                )
                del merged
                gc.collect()
                small = cv2.resize(
                    np.asarray(warped, dtype=np.float32),
                    (proj.width // ds, proj.height // ds),
                    interpolation=cv2.INTER_AREA,
                )
                del warped
                gc.collect()
                mean = small[region].reshape(-1, 3).mean(axis=0)
                record = {
                    "index": index,
                    "set": set_name,
                    "requested_c": list(c_req),
                    "projected_rgb8": list(p8),
                    "measured_mean_q": [float(v) for v in mean],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                with jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                per = (time.perf_counter() - started) / (index - done + 1)
                print(
                    f"[{index + 1}/{len(jobs)}] {set_name} c={np.round(c_req, 3)}"
                    f" -> p={p8} ({per:.1f}s/patch)",
                    flush=True,
                )
        finally:
            cv2.destroyWindow(window_name)

    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "lut": str(args.lut),
                "calibration_dir": str(args.calibration_dir),
                "seed": args.seed,
                "settle_seconds": args.settle_seconds,
                "sets": {
                    "gray": GRAY_STEPS,
                    "single": len(SINGLE_LEVELS) * 3,
                    "random_box": N_RANDOM,
                    "held_out": N_HELD_OUT,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Validation capture finished:", jsonl_path)


if __name__ == "__main__":
    main()
