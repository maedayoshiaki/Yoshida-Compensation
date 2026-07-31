"""Analyze calibration capture stability: step ratios along each ramp.

Prints, for each channel ramp, the measured value and the step ratio
vs the previous level (expected smooth ~ (p2/p1)^gamma), flagging
non-monotonic or erratic steps. Helps distinguish level-dependent
projector adaptation from one-off transients.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRATCH = Path(sys.argv[1])

rows = []
with open(SCRATCH / "build_debug" / "captures.csv", newline="") as f:
    for rec in csv.DictReader(f):
        if rec["stage"] != "post_hdr":
            continue
        tag = rec["pattern"]
        idx = int(tag.split("_")[0])
        rgb = tag.split("_")[-1]
        r, g, b = int(rgb[1:4]), int(rgb[5:8]), int(rgb[9:12])
        rows.append((idx, r, g, b, float(rec["roi_R"]), float(rec["roi_G"]), float(rec["roi_B"])))

rows.sort()
GAMMA = 2.2

for c, name in enumerate("RGB"):
    ramp = [(rec[1 + c], rec[4 + c], rec[0]) for rec in rows
            if rec[1 + c] > 0 and rec[1] + rec[2] + rec[3] == rec[1 + c]]
    ramp.sort()
    print(f"\n--- {name} ramp ---")
    print(f"{'lvl':>4} {'idx':>4} {'measured':>10} {'ratio':>7} {'expect':>7} flag")
    prev = None
    for lvl, m, idx in ramp:
        if prev is not None and prev[1] > 0:
            ratio = m / prev[1]
            expect = (lvl / prev[0]) ** GAMMA
            flag = ""
            if ratio < 1.0:
                flag = "NON-MONOTONIC"
            elif abs(ratio - expect) / expect > 0.15:
                flag = "erratic"
            print(f"{lvl:>4} {idx:>4} {m:>10.6f} {ratio:>7.3f} {expect:>7.3f} {flag}")
        else:
            print(f"{lvl:>4} {idx:>4} {m:>10.6f} {'':>7} {'':>7}")
        prev = (lvl, m)
