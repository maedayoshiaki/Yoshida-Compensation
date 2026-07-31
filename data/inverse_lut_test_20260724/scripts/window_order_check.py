"""Check whether the repo's window sequence (fullscreen -> move) actually
places the projection window on the target display at (3440, 0).

Captures with white shown via the repo order and reports the ROI mean;
compares against the known-good order (move -> fullscreen).
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


def roi_mean8(frame: np.ndarray) -> np.ndarray:
    x, y, w, h = ROI
    return frame[y : y + h, x : x + w, :3].reshape(-1, 3).mean(axis=0) * 255


def main() -> None:
    reload_config(REPO / "config.toml")
    cfg = get_config()
    cam_cfg = replace(cfg.camera, av="8", iso="200", tv="1/60")

    img = np.full((SIZE[1], SIZE[0], 3), 255, np.uint8)
    backend = create_camera_backend(cam_cfg)
    with backend:
        # Repo order: namedWindow -> fullscreen -> moveWindow (as in
        # run_calibration_capture / validate_inverse_lut).
        w1 = "order_repo"
        cv2.namedWindow(w1, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(w1, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(w1, POS[0], POS[1])
        cv2.imshow(w1, img)
        cv2.waitKey(300)
        time.sleep(_PROJECTION_SETTLE_SECONDS)
        try:
            rect1 = cv2.getWindowImageRect(w1)
        except Exception as exc:  # noqa: BLE001
            rect1 = f"unavailable ({exc})"
        frame = backend.capture_linear_rgb()
        cv2.destroyWindow(w1)
        cv2.waitKey(50)
        print(f"repo order   : rect={rect1} roi_mean8={np.array2string(roi_mean8(frame), precision=2)}")
        del frame

        # Known-good order: normal -> move -> resize -> fullscreen.
        w2 = "order_good"
        cv2.namedWindow(w2, cv2.WINDOW_NORMAL)
        cv2.moveWindow(w2, POS[0], POS[1])
        cv2.resizeWindow(w2, SIZE[0], SIZE[1])
        cv2.waitKey(50)
        cv2.setWindowProperty(w2, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(w2, img)
        cv2.waitKey(300)
        time.sleep(_PROJECTION_SETTLE_SECONDS)
        try:
            rect2 = cv2.getWindowImageRect(w2)
        except Exception as exc:  # noqa: BLE001
            rect2 = f"unavailable ({exc})"
        frame = backend.capture_linear_rgb()
        cv2.destroyWindow(w2)
        print(f"good order   : rect={rect2} roi_mean8={np.array2string(roi_mean8(frame), precision=2)}")


if __name__ == "__main__":
    main()
