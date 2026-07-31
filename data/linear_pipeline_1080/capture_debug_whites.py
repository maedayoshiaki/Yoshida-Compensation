"""Capture solid-white debug frames at three brightness levels.

Projects (v, v, v) for v in {255, 170, 85} without any correction,
captures linear, and saves viewable sRGB PNGs plus per-level stats.
Use across runs to compare surface / exposure / projector state.

Output: data/linear_pipeline_1080/debug_whites/
    white_255.png / white_170.png / white_085.png  (full camera frame, sRGB)
    stats.json  (frame + region means in 8bit units, camera settings, timestamp)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO = Path(r"C:\py_scripts\Yoshida-Compensation")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.python.camera import create_camera_backend
from src.python.config import get_config, reload_config

SCRATCH = Path(__file__).resolve().parent
OUT = REPO / "data" / "linear_pipeline_1080" / "debug_whites"
POS = (3440, 0)
SIZE = (1920, 1080)
LEVELS = (255, 170, 85)


def srgb_encode(linear):
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def main() -> None:
    reload_config(SCRATCH / "config_display5.toml")
    cfg = get_config()
    OUT.mkdir(parents=True, exist_ok=True)

    region = None
    region_path = OUT.parent / "region.json"
    if region_path.exists():
        region = json.loads(region_path.read_text())["region"]

    win = "debug_whites"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win, POS[0], POS[1])
    cv2.resizeWindow(win, SIZE[0], SIZE[1])
    cv2.waitKey(50)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    stats = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "camera": {"av": cfg.camera.av, "tv": cfg.camera.tv, "iso": cfg.camera.iso},
        "levels": {},
    }
    backend = create_camera_backend(cfg.camera)
    try:
        with backend:
            for lvl in LEVELS:
                cv2.imshow(win, np.full((SIZE[1], SIZE[0], 3), lvl, np.uint8))
                cv2.waitKey(cfg.camera.wait_key_ms)
                time.sleep(0.3)
                frame = backend.capture_linear_rgb()
                cv2.imwrite(
                    str(OUT / f"white_{lvl:03d}.png"),
                    cv2.cvtColor(
                        (srgb_encode(frame) * 255).round().astype(np.uint8),
                        cv2.COLOR_RGB2BGR,
                    ),
                )
                entry = {
                    "frame_mean8": [
                        float(x) for x in frame.reshape(-1, 3).mean(axis=0) * 255
                    ],
                }
                if region is not None:
                    # region is in projector space; report the full-frame mean
                    # plus a central-crop mean as a stable scalar reference.
                    h, w = frame.shape[:2]
                    crop = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
                    entry["center_crop_mean8"] = [
                        float(x) for x in crop.reshape(-1, 3).mean(axis=0) * 255
                    ]
                stats["levels"][str(lvl)] = entry
                print(f"white {lvl}: frame mean8 = "
                      f"{np.array2string(np.array(entry['frame_mean8']), precision=2)}")
                del frame
    finally:
        cv2.destroyWindow(win)

    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
