"""OpenCV camera backend for generic USB/UVC devices."""

from __future__ import annotations

import cv2
import numpy as np

from .base import CameraBackend, CameraCaptureError


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    threshold = 0.04045
    return np.where(
        rgb <= threshold,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


class OpenCVCamera(CameraBackend):
    """Capture backend using cv2.VideoCapture."""

    def capture_linear_rgb(self) -> np.ndarray:
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
