"""Forensics on the ~23% cross-pass outliers.

For each channel's worst level, print the three pass values, their
ratios to the median, and the projection-order neighbors in the
offending pass (the JSON dumps are in projection order).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRATCH = Path(__file__).resolve().parent

passes = [json.loads((SCRATCH / f"final_pass{p}.json").read_text()) for p in range(3)]

TARGETS = [(0, 0.34375), (1, 0.28125), (2, 0.25)]

for c, lvl in TARGETS:
    name = "RGB"[c]
    vals = []
    for pi, dump in enumerate(passes):
        for pos, m in enumerate(dump):
            if m["kind"] != "channel" or c not in m["recorded_channels"]:
                continue
            if abs(m["input_rgb"][c] - lvl) < 1e-6:
                vals.append((pi, pos, m["measured_rgb"][c]))
    med = float(np.median([v[2] for v in vals]))
    print(f"\n=== {name} level {lvl:.5f} (median {med:.6f}) ===")
    for pi, pos, v in vals:
        marker = "  <-- OUTLIER" if abs(v / med - 1) > 0.05 else ""
        print(f"  pass {pi} @ position {pos:2d}: {v:.6f} ratio={v / med:.3f}{marker}")
        if abs(v / med - 1) > 0.05:
            dump = passes[pi]
            for q in range(max(0, pos - 2), min(len(dump), pos + 3)):
                mm = dump[q]
                rgb = tuple(round(x * 255) for x in mm["input_rgb"])
                meas = tuple(
                    f"{x:.5f}" if x == x else "nan" for x in mm["measured_rgb"]
                )
                tagpos = "*" if q == pos else " "
                print(f"     {tagpos} pos {q:2d}: input={rgb} measured={meas}")
