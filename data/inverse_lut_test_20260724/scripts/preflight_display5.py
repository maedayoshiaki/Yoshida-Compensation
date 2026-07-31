"""Pre-flight check for the inverse-LUT run on DISPLAY5.

Projects white / mid-gray / black on the projector window (DISPLAY5),
captures each with the Canon backend, and reports ROI statistics plus
saves viewable previews (sRGB-encoded, ROI rectangle drawn) so the
projection placement and exposure can be verified before the long
calibration run.
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
from src.python.inverse_lut import (
    _PROJECTION_SETTLE_SECONDS,
    reduce_frame,
    snap_to_flicker_free_tv,
)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def save_preview(frame: np.ndarray, roi, out_path: Path, max_width: int = 1600):
    h, w = frame.shape[:2]
    scale = max_width / float(w)
    small = cv2.resize(
        srgb_encode(frame), (max_width, int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    img8 = (np.clip(small, 0, 1) * 255).round().astype(np.uint8)
    bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    x, y, rw, rh = roi
    p0 = (int(round(x * scale)), int(round(y * scale)))
    p1 = (int(round((x + rw) * scale)), int(round((y + rh) * scale)))
    cv2.rectangle(bgr, p0, p1, (0, 255, 0), 2)
    cv2.imwrite(str(out_path), bgr)


def roi_stats(frame: np.ndarray, roi, trust_low: float, trust_high: float):
    x, y, rw, rh = roi
    crop = frame[y : y + rh, x : x + rw, :3]
    mean = crop.reshape(-1, 3).mean(axis=0)
    p1 = np.percentile(crop.reshape(-1, 3), 1, axis=0)
    p99 = np.percentile(crop.reshape(-1, 3), 99, axis=0)
    frac_sat = (crop >= trust_high / 255.0).reshape(-1, 3).mean(axis=0)
    frac_dark = (crop <= trust_low / 255.0).reshape(-1, 3).mean(axis=0)
    return mean, p1, p99, frac_sat, frac_dark


def main() -> None:
    config_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    reload_config(config_path)
    cfg = get_config()
    proj = cfg.projector
    ilut = cfg.inverse_lut
    roi = (ilut.roi_x, ilut.roi_y, ilut.roi_width, ilut.roi_height)

    base_tv = snap_to_flicker_free_tv(
        cfg.camera.tv, refresh_hz=float(proj.refresh_rate)
    )
    short_tv = "1/60"

    print(f"projector window: pos=({proj.pos_x},{proj.pos_y}) "
          f"size={proj.width}x{proj.height}")
    print(f"roi={roi}  base_tv={base_tv}  short_tv={short_tv}")

    window = "preflight_display5"
    cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(window, proj.pos_x, proj.pos_y)

    shots = [
        ("white_base", 255, base_tv),
        ("white_short", 255, short_tv),
        ("gray128_base", 128, base_tv),
        ("black_base", 0, base_tv),
    ]

    backend = create_camera_backend(cfg.camera)
    try:
        with backend:
            for name, value, tv in shots:
                img = np.full((proj.height, proj.width, 3), value, np.uint8)
                cv2.imshow(window, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(cfg.camera.wait_key_ms)
                time.sleep(_PROJECTION_SETTLE_SECONDS)
                frame = backend.capture_linear_rgb(tv=tv)
                mean, p1, p99, frac_sat, frac_dark = roi_stats(
                    frame, roi, ilut.trust_low, ilut.trust_high
                )
                print(f"\n[{name}] tv={tv} frame={frame.shape}")
                print(f"  roi mean (8bit) : {np.array2string(mean * 255, precision=2)}")
                print(f"  roi p1   (8bit) : {np.array2string(p1 * 255, precision=2)}")
                print(f"  roi p99  (8bit) : {np.array2string(p99 * 255, precision=2)}")
                print(f"  frac >= trust_high: {np.array2string(frac_sat, precision=4)}")
                print(f"  frac <= trust_low : {np.array2string(frac_dark, precision=4)}")
                save_preview(frame, roi, out_dir / f"preflight_{name}.png")
                del frame
    finally:
        cv2.destroyWindow(window)

    print(f"\npreviews saved to: {out_dir}")


if __name__ == "__main__":
    main()
