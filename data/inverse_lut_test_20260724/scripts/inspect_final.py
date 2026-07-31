"""Inspect the 3-pass build: per-level repeatability and merged curves."""

from __future__ import annotations

import json
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

SCRATCH = Path(__file__).resolve().parent

passes = []
for p in range(3):
    passes.append(json.loads((SCRATCH / f"final_pass{p}.json").read_text()))

# Collect per-channel {level: [pass values]}.
per_channel: list[dict] = [{}, {}, {}]
for pi, dump in enumerate(passes):
    for m in dump:
        if m["kind"] != "channel":
            continue
        for c in m["recorded_channels"]:
            lvl = m["input_rgb"][c]
            v = m["measured_rgb"][c]
            per_channel[c].setdefault(round(lvl, 6), [None, None, None])[pi] = v

print("Cross-pass repeatability of recorded values (per channel):")
worst = []
for c, name in enumerate("RGB"):
    spreads = []
    for lvl, vals in sorted(per_channel[c].items()):
        vals = [v for v in vals if v is not None]
        mean = np.mean(vals)
        if mean > 1e-4 and len(vals) == 3:
            spread = (max(vals) - min(vals)) / mean
            spreads.append((spread, lvl, mean))
    spreads.sort(reverse=True)
    med = np.median([s[0] for s in spreads])
    print(f"  {name}: median spread={med * 100:.2f}%  worst 3: "
          + ", ".join(f"lvl {l:.3f}: {s * 100:.1f}%" for s, l, m in spreads[:3]))
    worst.append(spreads[0][0])

lut = load_inverse_lut(REPO / "data" / "inverse_lut.npz")
print(f"\nmeta: {lut.meta}")
for c, name in enumerate("RGB"):
    row = lut.lut[c]
    mono = np.all(np.diff(row) >= -1e-6)
    print(f"{name}: monotonic={mono} L(0.25)={row[64]:.3f} L(0.5)={row[128]:.3f} "
          f"L(0.75)={row[192]:.3f} L(1.0)={row[255]:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ("tab:red", "tab:green", "tab:blue")
for c, (ax, name, col) in enumerate(zip(axes, "RGB", colors)):
    lvls = np.array(sorted(per_channel[c]))
    means = np.array([np.mean([v for v in per_channel[c][l] if v is not None])
                      for l in lvls])
    m0, mw = means[0], means.max()
    n = (means - m0) / (mw - m0)
    for pi, marker in enumerate("x+."):
        vals = np.array([per_channel[c][l][pi] for l in lvls], dtype=np.float64)
        ax.plot(lvls, (vals - m0) / (mw - m0), marker, ms=4, alpha=0.5,
                color=col, label=f"pass {pi}")
    ax.plot(lvls, n, "-", lw=1.2, color=col, label="mean")
    ax.plot(lvls, lvls ** 2.2, "k--", lw=1, label="gamma 2.2")
    ax.plot(lut.t_grid, lut.lut[c], "-", color="gray", label="inverse LUT")
    ax.set_title(name)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.suptitle("3-pass merged forward response and inverse LUT")
fig.tight_layout()
out = SCRATCH / "final_inspection.png"
fig.savefig(out, dpi=110)
print(f"saved: {out}")
