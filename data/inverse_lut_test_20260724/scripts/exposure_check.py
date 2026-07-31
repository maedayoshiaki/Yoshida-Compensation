"""Pick av/iso so projected white stays below saturation at tv=1/60.

1/60 is the shortest flicker-free shutter at 60 Hz, so the HDR bracket
cannot recover highlights beyond it; white must not clip at the base
exposure. Tests candidate (av, iso) pairs and reports ROI stats.
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
from src.python.inverse_lut import _PROJECTION_SETTLE_SECONDS

POS = (3440, 0)
SIZE = (1280, 720)
ROI = (2778, 826, 1837, 2324)

CANDIDATES = [("8", "200"), ("8", "100"), ("11", "100")]


def main() -> None:
    reload_config(REPO / "config.toml")
    cfg = get_config()

    window = "exposure_check"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, POS[0], POS[1])
    cv2.resizeWindow(window, SIZE[0], SIZE[1])
    cv2.waitKey(50)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    img = np.full((SIZE[1], SIZE[0], 3), 255, np.uint8)
    cv2.imshow(window, img)
    cv2.waitKey(cfg.camera.wait_key_ms)
    time.sleep(_PROJECTION_SETTLE_SECONDS)

    x, y, w, h = ROI
    try:
        for av, iso in CANDIDATES:
            cam_cfg = replace(cfg.camera, av=av, iso=iso, tv="1/60")
            backend = create_camera_backend(cam_cfg)
            with backend:
                frame = backend.capture_linear_rgb()
            crop = frame[y : y + h, x : x + w, :3].reshape(-1, 3)
            mean8 = crop.mean(axis=0) * 255
            p999 = np.percentile(crop, 99.9, axis=0) * 255
            sat = (crop >= 245 / 255.0).mean(axis=0)
            print(
                f"av={av} iso={iso} tv=1/60: "
                f"mean8={np.array2string(mean8, precision=2)} "
                f"p99.9={np.array2string(p999, precision=2)} "
                f"sat245={np.array2string(sat, precision=5)}"
            )
            del frame, crop
    finally:
        cv2.destroyWindow(window)


if __name__ == "__main__":
    main()
