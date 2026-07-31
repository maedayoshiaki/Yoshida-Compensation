"""Rebuild the inverse LUT using per-level MEDIAN across the 5 passes.

The library's _aggregate_by_level averages duplicate levels; a mean is
vulnerable to the observed single-shot 10-23% dither-phase outliers, so
this script collapses each (channel, level) to its median across passes
first and feeds one synthetic measurement per level to
build_inverse_lut. Also reports the outlier count that the median
rejected (value deviating >5% from the median).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"C:\py_scripts\Yoshida-Compensation")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.python import inverse_lut as ilut
from src.python.config import get_config, reload_config

SCRATCH = Path(__file__).resolve().parent
CONFIG = SCRATCH / "config_display5.toml"
N_PASSES = 3

reload_config(CONFIG)
cfg = get_config()

per_channel: list[dict] = [{}, {}, {}]
for p in range(N_PASSES):
    dump = json.loads((SCRATCH / f"final_pass{p}.json").read_text())
    for m in dump:
        if m["kind"] != "channel":
            continue
        for c in m["recorded_channels"]:
            lvl = round(m["input_rgb"][c], 6)
            per_channel[c].setdefault(lvl, []).append(m["measured_rgb"][c])

measurements = []
outliers = 0
for c in range(3):
    for lvl, vals in sorted(per_channel[c].items()):
        vals = np.array([v for v in vals if v == v], dtype=np.float64)
        med = float(np.median(vals))
        if med > 1e-4:
            outliers += int(np.sum(np.abs(vals / med - 1) > 0.05))
        input_rgb = [0.0, 0.0, 0.0]
        input_rgb[c] = float(lvl)
        measured = [float("nan")] * 3
        measured[c] = med
        measurements.append(
            ilut.CalibrationMeasurement(
                input_rgb=tuple(input_rgb),
                measured_rgb=tuple(measured),
                recorded_channels=(c,),
                kind="channel",
            )
        )
        n = len(vals)
        if n != N_PASSES:
            print(f"warn: channel {c} level {lvl}: {n} samples (expected {N_PASSES})")

print(f"levels: {sum(len(d) for d in per_channel)}, "
      f">5% deviants rejected by median: {outliers}")

data = ilut.CalibrationData(
    measurements=measurements,
    meta={"source": "channel_ramp", "aggregation": "median-of-5"},
)
lut = ilut.build_inverse_lut(data, source="channel_ramp")
lut.meta["aggregation"] = "median-of-5"
out = ilut.save_inverse_lut(cfg.inverse_lut.inverse_lut_path, lut)
print(f"Saved median-aggregated inverse LUT to: {out}")

for c, name in enumerate("RGB"):
    row = lut.lut[c]
    mono = bool(np.all(np.diff(row) >= -1e-6))
    print(f"{name}: monotonic={mono} L(0.25)={row[64]:.3f} L(0.5)={row[128]:.3f} "
          f"L(0.75)={row[192]:.3f} L(1.0)={row[255]:.3f}")
