"""Simple manual capture and warp example script.

Captures an image from the camera, applies geometric warping to
projector coordinates, and saves both the raw and warped results.

手動キャプチャおよびワーピングのサンプルスクリプト。

カメラから画像をキャプチャし、プロジェクタ座標系への幾何学的ワーピングを
適用して、元画像とワーピング済み画像の両方を保存する。
"""

from examples.python.sample import capture_image, warp_image
from src.python.config import get_config
import cv2
import os

cfg = get_config()

captured_image = capture_image()
if captured_image is None:
    print("Failed to capture image")
    exit(1)
os.makedirs("data/manual_capture/wo_warp", exist_ok=True)
cv2.imwrite(
    "data/manual_capture/wo_warp/captured_image.png",
    cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR),
)

warped_image = warp_image(
    captured_image,
    cfg.paths.c2p_map,
    cfg.projector.width,
    cfg.projector.height,
    512,
    512,
)

os.makedirs("data/manual_capture/with_warp", exist_ok=True)
cv2.imwrite(
    "data/manual_capture/with_warp/warped_image.png",
    cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR),
)
print("Captured and warped images saved.")
