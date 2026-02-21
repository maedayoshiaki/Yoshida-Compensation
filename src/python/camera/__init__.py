"""Camera backend abstraction and factory helpers."""

from .base import CameraBackend, CameraCaptureError
from .factory import available_camera_backends, create_camera_backend

__all__ = [
    "CameraBackend",
    "CameraCaptureError",
    "available_camera_backends",
    "create_camera_backend",
]
