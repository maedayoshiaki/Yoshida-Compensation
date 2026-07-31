"""Capture 5 shuffled passes of the GRAY ramp and build a median LUT.

Gray-source calibration matches gray-drive usage on this non-additive
projector (channel_ramp calibrates single-channel drive, which this
projector renders very differently from neutrals).
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
OUT_LUT = REPO / "data" / "inverse_lut_gray.npz"


def main() -> None:
    reload_config(CONFIG)
    cfg = get_config()
    ilut_cfg = cfg.inverse_lut

    samples = ilut.generate_calibration_patterns(
        cfg.projector.width,
        cfg.projector.height,
        num_levels=ilut_cfg.num_levels,
        source="gray",
        dtype=np.uint8,
    )
    print(f"{len(samples)} gray patterns per pass, {PASSES} passes")

    try:
        for p in range(PASSES):
            order = list(samples)
            random.Random(2000 + p).shuffle(order)
            t0 = time.monotonic()
            data = ilut.run_calibration_capture(order, window_name=f"gray_pass{p}")
            print(f"pass {p}: {len(data.measurements)} measurements "
                  f"in {time.monotonic() - t0:.0f}s")
            dump = [
                {
                    "input_rgb": m.input_rgb,
                    "measured_rgb": m.measured_rgb,
                    "recorded_channels": list(m.recorded_channels),
                    "kind": m.kind,
                }
                for m in data.measurements
            ]
            (SCRATCH / f"gray_pass{p}.json").write_text(json.dumps(dump))
    finally:
        cv2.destroyAllWindows()

    # Median-of-passes per (channel, level), then build with source="gray".
    per_channel: list[dict] = [{}, {}, {}]
    for p in range(PASSES):
        dump = json.loads((SCRATCH / f"gray_pass{p}.json").read_text())
        for m in dump:
            if m["kind"] != "gray":
                continue
            for c in m["recorded_channels"]:
                lvl = round(m["input_rgb"][c], 6)
                per_channel[c].setdefault(lvl, []).append(m["measured_rgb"][c])

    measurements = []
    outliers = 0
    levels = sorted(per_channel[0])
    for lvl in levels:
        measured = []
        for c in range(3):
            vals = np.array(per_channel[c][lvl], dtype=np.float64)
            med = float(np.median(vals))
            if med > 1e-4:
                outliers += int(np.sum(np.abs(vals / med - 1) > 0.05))
            measured.append(med)
        measurements.append(
            ilut.CalibrationMeasurement(
                input_rgb=(lvl, lvl, lvl),
                measured_rgb=tuple(measured),
                recorded_channels=(0, 1, 2),
                kind="gray",
            )
        )
    print(f"gray levels: {len(levels)}, >5% deviants rejected by median: {outliers}")

    data = ilut.CalibrationData(
        measurements=measurements,
        meta={"source": "gray", "aggregation": f"median-of-{PASSES}"},
    )
    lut = ilut.build_inverse_lut(data, source="gray")
    lut.meta["aggregation"] = f"median-of-{PASSES}"
    out = ilut.save_inverse_lut(OUT_LUT, lut)
    print(f"Saved gray-source inverse LUT to: {out}")
    for c, name in enumerate("RGB"):
        row = lut.lut[c]
        print(f"{name}: monotonic={bool(np.all(np.diff(row) >= -1e-6))} "
              f"L(0.25)={row[64]:.3f} L(0.5)={row[128]:.3f} "
              f"L(0.75)={row[192]:.3f} L(1.0)={row[255]:.3f}")


if __name__ == "__main__":
    main()
