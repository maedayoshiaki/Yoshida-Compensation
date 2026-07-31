"""GrayCode geometric calibration with a corrected projection window.

Replaces cap_graycode's capture stage (its fullscreen-then-move window
order leaves a 3440x1440 window on this machine, which would stretch the
patterns and corrupt the decoded geometry) with a move-then-fullscreen
window and ONE persistent Canon session, then runs decode and both
interpolation stages from the GrayCode package.

Run from the repository root (outputs result_*.npy into CWD).
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

from external.GrayCode.src.python import decode, interpolate_c2p, interpolate_p2c
from external.GrayCode.src.python.config import (
    get_config as gc_get_config,
    reload_config as gc_reload_config,
)
from src.python.camera import create_camera_backend
from src.python.config import get_config, reload_config

SCRATCH = Path(__file__).resolve().parent
CONFIG = SCRATCH / "config_display5.toml"

POS = (3440, 0)
SIZE = (1920, 1080)  # (width, height)

COVERS = [
    (-5120, 0, 1920, 1080),
    (-3200, 0, 1280, 800),
    (-1920, 0, 1920, 1200),
    (0, 0, 3440, 1440),
]


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def main() -> None:
    reload_config(CONFIG)
    gc_reload_config(CONFIG)
    cfg = get_config()
    gc_cfg = gc_get_config()

    pattern_dir = Path(gc_cfg.paths.pattern_dir)
    capture_dir = Path(gc_cfg.paths.captured_dir)
    pattern_dir.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate patterns (step=1: native size, no expansion loop needed).
    graycode = cv2.structured_light.GrayCodePattern.create(SIZE[0], SIZE[1])
    _, patterns = graycode.generate()
    patterns = list(patterns)
    patterns.append(255 * np.ones((SIZE[1], SIZE[0]), np.uint8))
    patterns.append(np.zeros((SIZE[1], SIZE[0]), np.uint8))
    for i, pat in enumerate(patterns):
        cv2.imwrite(str(pattern_dir / f"pattern_{i:02d}.png"), pat)
    print(f"generated {len(patterns)} graycode patterns")

    # 2. Covers + projection window (move BEFORE fullscreen).
    cover_names = []
    for i, (x, y, w, h) in enumerate(COVERS):
        name = f"cover_{i}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(name, x, y)
        cv2.resizeWindow(name, w, h)
        cv2.waitKey(30)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(name, np.zeros((h, w, 3), np.uint8))
        cover_names.append(name)
    window = "graycode_pattern"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, POS[0], POS[1])
    cv2.resizeWindow(window, SIZE[0], SIZE[1])
    cv2.waitKey(50)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(100)

    # 3. Capture all patterns on one persistent camera session.
    backend = create_camera_backend(cfg.camera)
    t0 = time.monotonic()
    try:
        with backend:
            for i, pat in enumerate(patterns):
                cv2.imshow(window, pat)
                cv2.waitKey(cfg.camera.wait_key_ms)
                time.sleep(0.3)
                frame = backend.capture_linear_rgb()
                srgb8 = (srgb_encode(frame) * 255.0).round().astype(np.uint8)
                gray = cv2.cvtColor(srgb8, cv2.COLOR_RGB2GRAY)
                cv2.imwrite(str(capture_dir / f"capture_{i:02d}.png"), gray)
                print(f"captured {i + 1}/{len(patterns)}")
                del frame, srgb8, gray
    finally:
        cv2.destroyWindow(window)
        for name in cover_names:
            cv2.destroyWindow(name)
    print(f"capture stage done in {time.monotonic() - t0:.0f}s")

    # 4. Decode -> result_c2p.npy / result_p2c.npy
    cam_size = decode.main(["decode.py", str(SIZE[1]), str(SIZE[0]), "1", "1"])
    if cam_size is None:
        raise SystemExit("decode failed")
    cam_height, cam_width = cam_size
    print(f"decoded, camera size = {cam_width}x{cam_height}")

    # 5. Interpolations -> result_c2p_compensated_delaunay.npy /
    #    result_p2c_compensated_delaunay.npy
    interpolate_c2p.main(
        ["interpolate_c2p.py", "result_c2p.npy", str(cam_height), str(cam_width),
         "delaunay"]
    )
    interpolate_p2c.main(
        ["interpolate_p2c.py", "result_p2c.npy", str(SIZE[1]), str(SIZE[0])]
    )
    print("graycode pipeline finished")


if __name__ == "__main__":
    main()
