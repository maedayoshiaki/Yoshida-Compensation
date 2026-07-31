"""Final inverse-LUT build with jitter/drift mitigations.

Mitigations over the plain build_inverse_lut.py run:
  1. Base exposure 1/15 (av13, iso100): integrates 4 refresh cycles,
     cutting the partial-cycle capture jitter from ~3.3% to ~1.1% std.
  2. Three capture passes, each with an independently shuffled projection
     order; build_inverse_lut aggregates duplicate levels by mean, so
     residual per-shot jitter and slow drift average out instead of
     imprinting on the curve at a fixed level.
Writes per-pass measurement dumps (JSON) next to this script and saves
the merged LUT to data/inverse_lut.npz.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(r"C:\py_scripts\Yoshida-Compensation")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.python import inverse_lut as ilut
from src.python.config import get_config, reload_config

SCRATCH = Path(__file__).resolve().parent
CONFIG = SCRATCH / "config_display5.toml"
PASSES = 3


def main() -> None:
    reload_config(CONFIG)
    cfg = get_config()
    ilut_cfg = cfg.inverse_lut

    samples = ilut.generate_calibration_patterns(
        cfg.projector.width,
        cfg.projector.height,
        num_levels=ilut_cfg.num_levels,
        source=ilut_cfg.source,
        dtype=np.uint8,
    )
    print(f"{len(samples)} patterns per pass, {PASSES} passes (shuffled orders)")

    all_measurements = []
    meta = {}
    try:
        for p in range(PASSES):
            order = list(samples)
            random.Random(1000 + p).shuffle(order)
            t0 = time.monotonic()
            data = ilut.run_calibration_capture(
                order, window_name=f"ilut_pass{p}"
            )
            dt = time.monotonic() - t0
            print(f"pass {p}: {len(data.measurements)} measurements in {dt:.0f}s")
            meta = data.meta
            all_measurements.extend(data.measurements)
            dump = [
                {
                    "input_rgb": m.input_rgb,
                    "measured_rgb": m.measured_rgb,
                    "recorded_channels": list(m.recorded_channels),
                    "kind": m.kind,
                }
                for m in data.measurements
            ]
            (SCRATCH / f"final_pass{p}.json").write_text(json.dumps(dump))
    finally:
        cv2.destroyAllWindows()

    merged = ilut.CalibrationData(measurements=all_measurements, meta=meta)
    lut = ilut.build_inverse_lut(merged, source=ilut_cfg.source)
    out = ilut.save_inverse_lut(ilut_cfg.inverse_lut_path, lut)
    print(f"Saved merged inverse LUT to: {out}")


if __name__ == "__main__":
    main()
