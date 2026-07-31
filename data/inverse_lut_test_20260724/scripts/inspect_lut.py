"""Sanity-check the built inverse LUT: shape, monotonicity, and curve plot.

Also plots the measured forward response knots from the calibration
captures.csv (post-HDR ROI means) for comparison against pure gamma.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(r"C:\py_scripts\Yoshida-Compensation")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.python.inverse_lut import load_inverse_lut

SCRATCH = Path(sys.argv[1])
lut = load_inverse_lut(REPO / "data" / "inverse_lut.npz")
print(f"meta: {lut.meta}")
print(f"lut shape: {lut.lut.shape}, t_grid: {lut.t_grid.shape}")
for c, name in enumerate("RGB"):
    row = lut.lut[c]
    mono = np.all(np.diff(row) >= -1e-6)
    idx = [0, 64, 128, 192, 255]
    vals = ", ".join(f"L({lut.t_grid[i]:.2f})={row[i]:.3f}" for i in idx)
    ident = np.allclose(row, lut.t_grid, atol=1e-6)
    print(f"{name}: monotonic={mono} identity_fallback={ident} {vals}")

# Forward response knots from captures.csv (post_hdr rows of channel ramps).
rows = []
with open(SCRATCH / "build_debug" / "captures.csv", newline="") as f:
    for rec in csv.DictReader(f):
        if rec["stage"] != "post_hdr":
            continue
        tag = rec["pattern"]  # e.g. 12_channel_r000g167b000
        rgb = tag.split("_")[-1]
        r, g, b = int(rgb[1:4]), int(rgb[5:8]), int(rgb[9:12])
        rows.append((r, g, b, float(rec["roi_R"]), float(rec["roi_G"]), float(rec["roi_B"])))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ("tab:red", "tab:green", "tab:blue")
for c, (ax, name, col) in enumerate(zip(axes, "RGB", colors)):
    pts = [(rec[c] / 255.0, rec[3 + c]) for rec in rows
           if rec[c] > 0 or (rec[0] == rec[1] == rec[2] == 0)]
    pts.sort()
    p = np.array([q[0] for q in pts])
    m = np.array([q[1] for q in pts])
    m0, mw = m[0], m.max()
    n = (m - m0) / (mw - m0)
    ax.plot(p, n, "o-", ms=3, color=col, label="measured (normalized)")
    ax.plot(p, p ** 2.2, "k--", lw=1, label="gamma 2.2")
    ax.plot(lut.t_grid, lut.lut[c], "-", color="gray", label="inverse LUT L(t)")
    ax.set_title(f"{name}")
    ax.set_xlabel("projector input p / target t")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("Forward response (knots) and inverse LUT")
fig.tight_layout()
out = SCRATCH / "lut_inspection.png"
fig.savefig(out, dpi=110)
print(f"saved: {out}")
