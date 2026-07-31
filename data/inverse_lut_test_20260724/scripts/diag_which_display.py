"""Find which physical display the camera is looking at.

Shows a fullscreen WHITE window on each candidate display in turn,
captures a frame, and reports the whole-frame mean; the display whose
white raises the mean is the projector the camera sees. Also prints
cv2.getWindowImageRect to confirm the window really landed on the
requested display.
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

# (label, x, y, w, h) from Windows Forms Screen enumeration
DISPLAYS = [
    ("DISPLAY5", -5120, 0, 1920, 1080),
    ("DISPLAY3", -3200, 0, 1280, 800),
    ("DISPLAY2", -1920, 0, 1920, 1200),
    ("DISPLAY4", 3440, 0, 1280, 720),
]

TV = "1/60"


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def save_preview(frame: np.ndarray, out_path: Path, max_width: int = 1200):
    h, w = frame.shape[:2]
    scale = max_width / float(w)
    small = cv2.resize(
        srgb_encode(frame), (max_width, int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    img8 = (np.clip(small, 0, 1) * 255).round().astype(np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))


def show_fullscreen_white(window: str, x: int, y: int, w: int, h: int) -> None:
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, x, y)
    cv2.resizeWindow(window, w, h)
    cv2.waitKey(50)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.imshow(window, img)
    cv2.waitKey(300)


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    reload_config(REPO / "config.toml")
    cfg = get_config()

    backend = create_camera_backend(cfg.camera)
    results = []
    with backend:
        # Baseline: no window anywhere.
        time.sleep(_PROJECTION_SETTLE_SECONDS)
        frame = backend.capture_linear_rgb(tv=TV)
        base_mean = frame.reshape(-1, 3).mean(axis=0)
        print(f"baseline (no window): mean8={np.array2string(base_mean * 255, precision=2)}")
        save_preview(frame, out_dir / "diag_baseline.png")
        del frame

        for label, x, y, w, h in DISPLAYS:
            window = f"diag_{label}"
            show_fullscreen_white(window, x, y, w, h)
            try:
                rect = cv2.getWindowImageRect(window)
            except Exception as exc:  # noqa: BLE001
                rect = f"unavailable ({exc})"
            time.sleep(_PROJECTION_SETTLE_SECONDS)
            frame = backend.capture_linear_rgb(tv=TV)
            cv2.destroyWindow(window)
            cv2.waitKey(50)
            mean = frame.reshape(-1, 3).mean(axis=0)
            delta = (mean - base_mean) * 255
            results.append((label, mean))
            print(
                f"{label}: requested=({x},{y},{w},{h}) actual_rect={rect} "
                f"mean8={np.array2string(mean * 255, precision=2)} "
                f"delta8={np.array2string(delta, precision=2)}"
            )
            save_preview(frame, out_dir / f"diag_{label}.png")
            del frame

    best = max(results, key=lambda r: float(np.sum(r[1])))
    print(f"\nbrightest with white: {best[0]}")


if __name__ == "__main__":
    main()
