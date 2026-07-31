"""Capture passes 3 and 4 (shuffled) to enable 5-sample median aggregation."""

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

COVERS = [
    (-5120, 0, 1920, 1080),
    (-3200, 0, 1280, 800),
    (-1920, 0, 1920, 1200),
    (0, 0, 3440, 1440),
]


def make_covers() -> list[str]:
    names = []
    for i, (x, y, w, h) in enumerate(COVERS):
        name = f"cover_{i}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(name, x, y)
        cv2.resizeWindow(name, w, h)
        cv2.waitKey(30)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(name, np.zeros((h, w, 3), np.uint8))
        names.append(name)
    cv2.waitKey(200)
    return names


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

    covers = make_covers()
    try:
        for p in (3, 4):
            order = list(samples)
            random.Random(1000 + p).shuffle(order)
            t0 = time.monotonic()
            data = ilut.run_calibration_capture(order, window_name=f"ilut_pass{p}")
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
            (SCRATCH / f"final_pass{p}.json").write_text(json.dumps(dump))
    finally:
        for name in covers:
            cv2.destroyWindow(name)
    print("extra passes done")


if __name__ == "__main__":
    main()
