"""Probe: does a sequential ascending R-ramp reproduce the Aug-7 response?

Sequence A: project (v,0,0) ascending v=0..255 (17 levels), single exposure.
Sequence B: hold gray 128 ~10 s, then jump to (255,0,0) and capture at
0 / ~7 / ~20 / ~45 s dwell to observe adaptation relaxation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path("C:/py_scripts/Yoshida-Compensation")
sys.path.insert(0, str(REPO))

from yoshidacomp.capture import create_camera_backend
from yoshidacomp.compensation.inverse_lut import _PROJECTION_SETTLE_SECONDS, reduce_frame
from yoshidacomp.compensation.projection_region import (
    mask_to_active_region,
    resolve_active_region,
)
from yoshidacomp.config import reload_config

cfg = reload_config(REPO / "config.color_grid_iso200_hdr_d65_20260812.toml")
proj = cfg.projector
il = cfg.inverse_lut
roi = (il.roi_x, il.roi_y, il.roi_width, il.roi_height)
active = resolve_active_region(proj)
out_path = Path(sys.argv[1])

window = "seq_ramp_probe"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.moveWindow(window, proj.pos_x, proj.pos_y)
cv2.resizeWindow(window, proj.width, proj.height)
cv2.waitKey(50)
cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
if hasattr(cv2, "WND_PROP_TOPMOST"):
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)


def show(rgb):
    image = np.empty((proj.height, proj.width, 3), dtype=np.uint8)
    image[:, :] = np.asarray(rgb, dtype=np.uint8)
    image = mask_to_active_region(image, active)
    cv2.imshow(window, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(cfg.camera.wait_key_ms)


results = {"sequence_a": [], "sequence_b": []}
backend = create_camera_backend(cfg.camera)
with backend:
    # Sequence A: ascending R ramp, same order as the Aug-7 calibration.
    levels = [int(round(i * 255 / 16)) for i in range(17)]
    for v in levels:
        show((v, 0, 0))
        time.sleep(_PROJECTION_SETTLE_SECONDS)
        frame = backend.capture_linear_rgb()
        mean = reduce_frame(frame, roi)
        del frame
        results["sequence_a"].append({"input": v, "mean": [float(x) for x in mean]})
        print(f"A r={v:3d} -> {['%.4f' % x for x in mean]}", flush=True)

    # Sequence B: gray hold, then jump to full red; observe relaxation.
    show((128, 128, 128))
    time.sleep(10.0)
    show((255, 0, 0))
    time.sleep(_PROJECTION_SETTLE_SECONDS)
    t_jump = time.perf_counter()
    for label in ("t0", "t7", "t20", "t45"):
        frame = backend.capture_linear_rgb()
        mean = reduce_frame(frame, roi)
        del frame
        dwell = time.perf_counter() - t_jump
        results["sequence_b"].append(
            {"dwell_s": round(dwell, 1), "mean": [float(x) for x in mean]}
        )
        print(f"B dwell={dwell:5.1f}s -> {['%.4f' % x for x in mean]}", flush=True)
        target = {"t0": 7, "t7": 20, "t20": 45, "t45": None}[label]
        if target is not None:
            while time.perf_counter() - t_jump < target:
                cv2.waitKey(100)

cv2.destroyWindow(window)
out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
print("saved:", out_path)
