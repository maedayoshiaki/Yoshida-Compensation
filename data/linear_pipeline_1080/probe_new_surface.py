"""Probe the NEW projection surface: exposure + auto ROI for LUT calibration.

Projects white and black at 1920x1080, reports ROI-free stats, derives a
measurement ROI from the white-black difference (largest modulated
region, eroded, central 70% box), and saves previews.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(r"C:\py_scripts\Yoshida-Compensation")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.python.camera import create_camera_backend
from src.python.config import get_config, reload_config

SCRATCH = Path(__file__).resolve().parent
POS = (3440, 0)
SIZE = (1920, 1080)


def srgb_encode(linear):
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def save_preview(frame, roi, out_path, max_width=1500):
    h, w = frame.shape[:2]
    scale = max_width / float(w)
    small = cv2.resize(srgb_encode(frame), (max_width, int(round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    img8 = (np.clip(small, 0, 1) * 255).round().astype(np.uint8)
    bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    if roi is not None:
        x, y, rw, rh = roi
        cv2.rectangle(bgr, (int(x * scale), int(y * scale)),
                      (int((x + rw) * scale), int((y + rh) * scale)), (0, 255, 0), 3)
    cv2.imwrite(str(out_path), bgr)


def main() -> None:
    out_dir = SCRATCH / "probe_new"
    out_dir.mkdir(exist_ok=True)
    reload_config(SCRATCH / "config_display5.toml")
    cfg = get_config()

    win = "probe_new"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win, POS[0], POS[1])
    cv2.resizeWindow(win, SIZE[0], SIZE[1])
    cv2.waitKey(50)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frames = {}
    backend = create_camera_backend(cfg.camera)
    try:
        with backend:
            for name, val in (("white", 255), ("black", 0)):
                cv2.imshow(win, np.full((SIZE[1], SIZE[0], 3), val, np.uint8))
                cv2.waitKey(cfg.camera.wait_key_ms)
                time.sleep(0.4)
                frames[name] = backend.capture_linear_rgb()
    finally:
        cv2.destroyWindow(win)

    white, black = frames["white"], frames["black"]
    diff = np.clip(white.astype(np.float32) - black.astype(np.float32), 0, None)
    lum = diff @ np.array([0.30, 0.55, 0.15], np.float32)
    small = cv2.resize(lum, (lum.shape[1] // 8, lum.shape[0] // 8),
                       interpolation=cv2.INTER_AREA)
    thresh = 0.5 * np.percentile(small, 99)
    mask = (small >= thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    num, labels, stats_cc, _ = cv2.connectedComponentsWithStats(mask, 8)
    largest = 1 + int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA]))
    comp = (labels == largest).astype(np.uint8)
    bx, by, bw, bh = cv2.boundingRect(comp)
    radius = max(3, int(round(0.06 * bw)))
    comp = cv2.erode(comp, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    ex, ey, ew, eh = cv2.boundingRect(comp)
    cx, cy = ex + ew / 2.0, ey + eh / 2.0
    rw_, rh_ = 0.7 * ew, 0.7 * eh
    roi = (int(round((cx - rw_ / 2) * 8)), int(round((cy - rh_ / 2) * 8)),
           int(round(rw_ * 8)), int(round(rh_ * 8)))
    print(f"proposed LUT ROI (x, y, w, h) = {roi}")

    x, y, w, h = roi
    for name, frame in frames.items():
        crop = frame[y : y + h, x : x + w, :3].reshape(-1, 3)
        mean8 = crop.mean(axis=0) * 255
        p999 = np.percentile(crop, 99.9, axis=0) * 255
        sat = (crop >= 245 / 255.0).mean(axis=0)
        print(f"[{name}] mean8={np.array2string(mean8, precision=2)} "
              f"p99.9={np.array2string(p999, precision=2)} "
              f"sat245={np.array2string(sat, precision=6)}")
        save_preview(frame, roi, out_dir / f"probe_{name}.png")
    np.save(out_dir / "roi.npy", np.array(roi))


if __name__ == "__main__":
    main()
