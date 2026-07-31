"""One-shot geometric sanity check of the fresh p2c map.

Projects nature_1 (raw), captures, warps the capture back into projector
space, and saves target / warped-capture / blend images for visual
alignment inspection.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(r"C:\py_scripts\Yoshida-Compensation")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from external.GrayCode.src.python.interpolate_p2c import load_p2c_numpy_array
from external.GrayCode.src.python.warp_image import InpaintMethod, PixelMapWarperTorch
from src.python.camera import create_camera_backend
from src.python.config import get_config, reload_config

SCRATCH = Path(__file__).resolve().parent
POS = (3440, 0)
SIZE = (1280, 720)


def srgb_encode(linear):
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def main() -> None:
    reload_config(SCRATCH / "config_display5.toml")
    cfg = get_config()

    target = cv2.cvtColor(
        cv2.imread(str(REPO / "data" / "comp_test_targets" / "nature_1.jpg")),
        cv2.COLOR_BGR2RGB,
    )
    target = cv2.resize(target, SIZE, interpolation=cv2.INTER_AREA)

    window = "sanity"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, POS[0], POS[1])
    cv2.resizeWindow(window, SIZE[0], SIZE[1])
    cv2.waitKey(50)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window, cv2.cvtColor(target, cv2.COLOR_RGB2BGR))
    cv2.waitKey(300)
    time.sleep(0.4)

    backend = create_camera_backend(cfg.camera)
    try:
        with backend:
            frame = backend.capture_linear_rgb()
    finally:
        cv2.destroyWindow(window)
    srgb8 = (srgb_encode(frame) * 255).round().astype(np.uint8)

    warper = PixelMapWarperTorch(load_p2c_numpy_array(str(REPO / "result_p2c_compensated_delaunay.npy")))
    src = torch.from_numpy(srgb8.astype(np.float32) / 255.0).permute(2, 0, 1)
    warped = warper.backward_warp(src, dst_size=SIZE, inpaint=InpaintMethod.NONE)
    warped8 = (np.clip(warped.permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)

    out = SCRATCH / "warp_sanity"
    out.mkdir(exist_ok=True)
    cv2.imwrite(str(out / "target.png"), cv2.cvtColor(target, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "warped_capture.png"), cv2.cvtColor(warped8, cv2.COLOR_RGB2BGR))
    blend = cv2.addWeighted(target, 0.5, warped8, 0.5, 0)
    cv2.imwrite(str(out / "blend.png"), cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
    edge_t = cv2.Canny(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), 60, 150)
    edge_w = cv2.Canny(cv2.cvtColor(warped8, cv2.COLOR_RGB2GRAY), 60, 150)
    overlay = np.dstack([edge_w, edge_t, np.zeros_like(edge_t)])
    cv2.imwrite(str(out / "edges_overlay.png"), overlay)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
