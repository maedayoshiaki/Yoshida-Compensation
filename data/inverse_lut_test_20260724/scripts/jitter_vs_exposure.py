"""Isolate shot-to-shot jitter vs exposure length at matched signal level.

Captures solid r128 8 times at four settings whose (tv x av x iso)
product keeps the sensor signal roughly constant, so only the number of
integrated refresh cycles changes:

    1/60 av8  iso200  (~1 cycle)
    1/30 av8  iso100  (~2 cycles)
    1/15 av11 iso100  (~4 cycles)
    1/8  av16 iso100  (~7.5 cycles)

If the jitter comes from partial-refresh / color-wheel phase sampling
(or shutter timing), the relative std should fall roughly in proportion
to the exposure time.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

REPO = Path(r"C:\py_scripts\Yoshida-Compensation")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.python.camera import create_camera_backend
from src.python.config import get_config, reload_config

POS = (3440, 0)
SIZE = (1280, 720)
ROI = (2778, 826, 1837, 2324)

SETTINGS = [
    ("1/60", "8", "200"),
    ("1/30", "8", "100"),
    ("1/15", "11", "100"),
    ("1/8", "16", "100"),
]
N = 8


def roi_mean(frame: np.ndarray) -> np.ndarray:
    x, y, w, h = ROI
    return frame[y : y + h, x : x + w, :3].reshape(-1, 3).mean(axis=0)


def main() -> None:
    reload_config(REPO / "config.toml")
    cfg = get_config()

    window = "jitter"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, POS[0], POS[1])
    cv2.resizeWindow(window, SIZE[0], SIZE[1])
    cv2.waitKey(50)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    img[:, :, 2] = 128  # BGR: red 128
    cv2.imshow(window, img)
    cv2.waitKey(200)
    time.sleep(1.0)

    try:
        for tv, av, iso in SETTINGS:
            cam_cfg = replace(cfg.camera, av=av, iso=iso, tv=tv)
            backend = create_camera_backend(cam_cfg)
            vals = []
            with backend:
                for _ in range(N):
                    frame = backend.capture_linear_rgb()
                    vals.append(roi_mean(frame)[0])
                    del frame
            arr = np.array(vals)
            rel = arr / arr.mean()
            print(
                f"tv={tv:>5} av={av:>3} iso={iso:>3}: meanR={arr.mean():.6f} "
                f"rel_std={rel.std() * 100:.3f}% pk-pk={(rel.max() - rel.min()) * 100:.3f}%"
            )
    finally:
        cv2.destroyWindow(window)


if __name__ == "__main__":
    main()
