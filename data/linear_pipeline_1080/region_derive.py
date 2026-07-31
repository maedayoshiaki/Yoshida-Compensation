"""Derive the square projection region from the fresh graycode decode and
save region.json + a preview into data/linear_pipeline_1080."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(r"C:\py_scripts\Yoshida-Compensation")
OUT = REPO / "data" / "linear_pipeline_1080"
SIZE = (1920, 1080)
INSET = 8

OUT.mkdir(parents=True, exist_ok=True)
raw = np.load(REPO / "result_p2c.npy", allow_pickle=True).item()
valid = np.zeros((SIZE[1], SIZE[0]), np.uint8)
for (px, py) in raw.keys():
    x, y = int(round(px)), int(round(py))
    if 0 <= x < SIZE[0] and 0 <= y < SIZE[1]:
        valid[y, x] = 1
valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
dist = cv2.distanceTransform(valid, cv2.DIST_C, 5)
cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
half = int(dist[cy, cx]) - 2 - INSET
region = (int(cx - half), int(cy - half), int(2 * half))
print(f"square projection region (x, y, side) = {region}")

np.save(OUT / "valid_mask.npy", valid.astype(bool))
(OUT / "region.json").write_text(json.dumps({"region": list(region)}))

vis = np.dstack([valid * 130] * 3)
x, y, s = region
cv2.rectangle(vis, (x, y), (x + s, y + s), (0, 255, 0), 4)
cv2.putText(vis, f"region ({x}, {y}, side {s})", (x + 10, y + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
cv2.imwrite(str(OUT / "square_region_preview.png"), vis)
print("region preview saved")
