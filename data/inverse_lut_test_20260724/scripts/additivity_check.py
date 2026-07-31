"""Check projector additivity: gray-drive response vs single-channel drive.

Uses existing data only (no capture):
  - gray drive: the raw-mode curve of the gray validation npz
    (33 levels, absolute linear ROI means, same av/iso/tv as calibration)
  - single-channel drive: median-of-5 channel-ramp calibration values

If the projector were additive (per-channel independence), the gray
response in channel c would equal the channel-c ramp value at the same
level plus the small leak floor. Reports the ratio gray/single per level.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRATCH = Path(__file__).resolve().parent

val = np.load(SCRATCH / "validation" / "inverse_lut_validation.npz")
levels = val["levels"]  # (33,) in [0,1]
gray_raw = val["measured_raw"]  # (33, 3) absolute linear ROI means

per_channel: list[dict] = [{}, {}, {}]
for p in range(5):
    dump = json.loads((SCRATCH / f"final_pass{p}.json").read_text())
    for m in dump:
        if m["kind"] != "channel":
            continue
        for c in m["recorded_channels"]:
            lvl = round(m["input_rgb"][c], 6)
            per_channel[c].setdefault(lvl, []).append(m["measured_rgb"][c])

print("level | gray measured (R,G,B) | single-ramp median (R,G,B) | ratio gray/single")
print("-" * 100)
for i, lvl in enumerate(levels):
    key = round(float(lvl), 6)
    singles = []
    for c in range(3):
        if key in per_channel[c]:
            singles.append(float(np.median(per_channel[c][key])))
        else:
            singles.append(np.nan)
    singles = np.array(singles)
    g = gray_raw[i]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(singles > 1e-4, g / singles, np.nan)
    print(
        f"{lvl:5.3f} | {g[0]:7.4f} {g[1]:7.4f} {g[2]:7.4f} | "
        f"{singles[0]:7.4f} {singles[1]:7.4f} {singles[2]:7.4f} | "
        f"{ratio[0]:5.2f} {ratio[1]:5.2f} {ratio[2]:5.2f}"
    )
