"""OpenCV camera backend for generic USB/UVC devices.

Uses ``cv2.VideoCapture`` to capture from commodity cameras and converts
captured sRGB frames to linear RGB.

汎用 USB/UVC カメラ向けの OpenCV バックエンド。

``cv2.VideoCapture`` で取得した sRGB フレームをリニア RGB に変換する。
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import CameraBackend, CameraCaptureError


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB image values to linear RGB.

    Args:
        rgb: sRGB image array in range [0.0, 1.0].
            範囲 [0.0, 1.0] の sRGB 画像配列。

    Returns:
        Linear RGB image array in range [0.0, 1.0].
            範囲 [0.0, 1.0] のリニア RGB 画像配列。
    """
    threshold = 0.04045
    return np.where(
        rgb <= threshold,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


class OpenCVCamera(CameraBackend):
    """Capture backend using cv2.VideoCapture.

    ``device_index`` で指定したデバイスから 1 フレーム取得する。
    """

    def capture_linear_rgb(self) -> np.ndarray:
        """Capture one frame from OpenCV device as linear RGB.

        Returns:
            Linear RGB image in shape ``(H, W, 3)`` and dtype ``float32``.
                形状 ``(H, W, 3)``、dtype ``float32`` のリニア RGB 画像。

        Raises:
            CameraCaptureError: If camera opening or frame capture fails.
                カメラのオープンまたはフレーム取得に失敗した場合。
        """
        cap = cv2.VideoCapture(int(self.config.device_index))
        if not cap.isOpened():
            raise CameraCaptureError(
                f"Failed to open camera device_index={self.config.device_index}."
            )
        try:
            ok, frame_bgr = cap.read()
        finally:
            cap.release()

        if not ok or frame_bgr is None:
            raise CameraCaptureError("Failed to read frame from OpenCV camera.")

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        linear_rgb = _srgb_to_linear(rgb).astype(np.float32)
        return np.clip(linear_rgb, 0.0, 1.0)
