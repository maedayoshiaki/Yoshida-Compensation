"""Derive a measurement ROI for the inverse-LUT run from white/black captures.

Projects white and black on the projector display, computes the
luminance difference, finds the largest strongly-modulated region (the
projection surface), erodes it away from edges/ruler, and proposes a
centered ROI. Saves previews with the ROI drawn and prints exposure
statistics inside the ROI for white/black at 1/30 and 1/60.
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
from src.python.inverse_lut import _PROJECTION_SETTLE_SECONDS

POS = (3440, 0)
SIZE = (1280, 720)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def save_preview(frame: np.ndarray, roi, out_path: Path, max_width: int = 1400):
    h, w = frame.shape[:2]
    scale = max_width / float(w)
    small = cv2.resize(
        srgb_encode(frame), (max_width, int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    img8 = (np.clip(small, 0, 1) * 255).round().astype(np.uint8)
    bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    if roi is not None:
        x, y, rw, rh = roi
        p0 = (int(round(x * scale)), int(round(y * scale)))
        p1 = (int(round((x + rw) * scale)), int(round((y + rh) * scale)))
        cv2.rectangle(bgr, p0, p1, (0, 255, 0), 3)
    cv2.imwrite(str(out_path), bgr)


def stats(frame: np.ndarray, roi, trust_low=10.0, trust_high=245.0):
    x, y, rw, rh = roi
    crop = frame[y : y + rh, x : x + rw, :3].reshape(-1, 3)
    return {
        "mean8": crop.mean(axis=0) * 255,
        "p99_8": np.percentile(crop, 99, axis=0) * 255,
        "sat": (crop >= trust_high / 255.0).mean(axis=0),
        "dark": (crop <= trust_low / 255.0).mean(axis=0),
    }


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    reload_config(REPO / "config.toml")
    cfg = get_config()

    window = "derive_roi"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, POS[0], POS[1])
    cv2.resizeWindow(window, SIZE[0], SIZE[1])
    cv2.waitKey(50)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(value: int):
        img = np.full((SIZE[1], SIZE[0], 3), value, np.uint8)
        cv2.imshow(window, img)
        cv2.waitKey(cfg.camera.wait_key_ms)
        time.sleep(_PROJECTION_SETTLE_SECONDS)

    frames = {}
    backend = create_camera_backend(cfg.camera)
    try:
        with backend:
            for value, tv in [(255, "1/60"), (255, "1/30"), (0, "1/60"), (0, "1/30")]:
                show(value)
                frames[(value, tv)] = backend.capture_linear_rgb(tv=tv)
    finally:
        cv2.destroyWindow(window)

    white = frames[(255, "1/60")]
    black = frames[(0, "1/60")]
    diff = np.clip(white.astype(np.float32) - black.astype(np.float32), 0, None)
    lum = diff @ np.array([0.30, 0.55, 0.15], dtype=np.float32)

    # Downscale for morphology speed (factor 8), threshold at half the P99.
    small = cv2.resize(lum, (lum.shape[1] // 8, lum.shape[0] // 8),
                       interpolation=cv2.INTER_AREA)
    thresh = 0.5 * np.percentile(small, 99)
    mask = (small >= thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    num, labels, stats_cc, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num < 2:
        raise SystemExit("No modulated region found -- projector not visible?")
    largest = 1 + int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA]))
    comp = (labels == largest).astype(np.uint8)
    # Erode hard so the ROI stays well inside the surface (away from the
    # ruler tape / edges): erosion radius ~6% of the component width.
    bx, by, bw, bh = cv2.boundingRect(comp)
    radius = max(3, int(round(0.06 * bw)))
    comp = cv2.erode(comp, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    if comp.sum() == 0:
        raise SystemExit("Erosion removed everything -- lower the radius.")
    ex, ey, ew, eh = cv2.boundingRect(comp)

    # Centered box: central 70% of the eroded bounding box, back to full-res.
    cx, cy = ex + ew / 2.0, ey + eh / 2.0
    rw, rh = 0.7 * ew, 0.7 * eh
    roi = (
        int(round((cx - rw / 2) * 8)),
        int(round((cy - rh / 2) * 8)),
        int(round(rw * 8)),
        int(round(rh * 8)),
    )
    print(f"proposed roi (x, y, w, h) = {roi}")

    for (value, tv), frame in frames.items():
        s = stats(frame, roi)
        name = f"{'white' if value == 255 else 'black'}_{tv.replace('/', '-')}"
        print(
            f"[{name}] mean8={np.array2string(s['mean8'], precision=2)} "
            f"p99_8={np.array2string(s['p99_8'], precision=2)} "
            f"sat={np.array2string(s['sat'], precision=4)} "
            f"dark={np.array2string(s['dark'], precision=4)}"
        )
        save_preview(frame, roi, out_dir / f"roi_{name}.png")

    np.save(out_dir / "proposed_roi.npy", np.array(roi))


if __name__ == "__main__":
    main()
