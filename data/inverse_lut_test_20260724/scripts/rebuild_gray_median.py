"""Rebuild the gray-source LUT from the existing gray_pass*.json dumps
(median-of-5 per level), using the trust bounds in the current config.
No capture involved."""

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
PASSES = 5
OUT_LUT = REPO / "data" / "inverse_lut_gray.npz"

reload_config(CONFIG)
cfg = get_config()
print(f"trust_low={cfg.inverse_lut.trust_low} trust_high={cfg.inverse_lut.trust_high}")

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
for lvl in sorted(per_channel[0]):
    measured = tuple(
        float(np.median(np.array(per_channel[c][lvl], dtype=np.float64)))
        for c in range(3)
    )
    measurements.append(
        ilut.CalibrationMeasurement(
            input_rgb=(lvl, lvl, lvl),
            measured_rgb=measured,
            recorded_channels=(0, 1, 2),
            kind="gray",
        )
    )

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
