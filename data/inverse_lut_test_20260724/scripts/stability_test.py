"""Diagnose temporal instability of the projector/scene during capture.

Test 1 (drift): project solid red r128 and capture it 12 times in a row;
report the time series of the ROI mean. Photon/read noise on a 4M-px ROI
mean is negligible, so any variation is real (projector drift, ambient).

Test 2 (APL adaptation): capture r128 right after showing BLACK for 5 s
vs right after showing WHITE for 5 s, at delays 0.3/2/5 s after the
switch. A content-adaptive projector (dynamic iris / eco mode) shows a
prior-content dependence and a settle-time dependence.
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


def roi_mean(frame: np.ndarray) -> np.ndarray:
    x, y, w, h = ROI
    return frame[y : y + h, x : x + w, :3].reshape(-1, 3).mean(axis=0)


def main() -> None:
    reload_config(REPO / "config.toml")
    cfg = get_config()
    cam_cfg = replace(cfg.camera, av="8", iso="200", tv="1/60")

    window = "stability"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, POS[0], POS[1])
    cv2.resizeWindow(window, SIZE[0], SIZE[1])
    cv2.waitKey(50)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(rgb):
        img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
        img[:, :] = rgb[::-1]  # BGR for imshow
        cv2.imshow(window, img)
        cv2.waitKey(100)

    red128 = (128, 0, 0)
    backend = create_camera_backend(cam_cfg)
    t0 = time.monotonic()
    try:
        with backend:
            print("=== Test 1: 12 repeats of solid r128 ===")
            show(red128)
            time.sleep(0.5)
            series = []
            for i in range(12):
                frame = backend.capture_linear_rgb()
                m = roi_mean(frame)
                series.append(m)
                print(
                    f"t={time.monotonic() - t0:6.1f}s  #{i:02d} "
                    f"R={m[0]:.6f} G={m[1]:.6f} B={m[2]:.6f}"
                )
                del frame
            arr = np.array(series)
            rel = arr[:, 0] / arr[:, 0].mean()
            print(
                f"R rel spread: min={rel.min():.4f} max={rel.max():.4f} "
                f"pk-pk={(rel.max() - rel.min()) * 100:.2f}%  "
                f"std={rel.std() * 100:.3f}%"
            )

            print("\n=== Test 2: r128 after 5s BLACK vs 5s WHITE ===")
            for prior_name, prior in [("black", (0, 0, 0)), ("white", (255, 255, 255))] * 2:
                show(prior)
                time.sleep(5.0)
                show(red128)
                deadline = time.monotonic()
                for delay in (0.3, 2.0, 5.0):
                    while time.monotonic() - deadline < delay:
                        time.sleep(0.02)
                    frame = backend.capture_linear_rgb()
                    m = roi_mean(frame)
                    print(
                        f"prior={prior_name:<5} delay={delay:>3.1f}s "
                        f"R={m[0]:.6f} G={m[1]:.6f} B={m[2]:.6f}"
                    )
                    del frame
    finally:
        cv2.destroyWindow(window)


if __name__ == "__main__":
    main()
