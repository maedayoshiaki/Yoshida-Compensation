"""Base interface for camera capture backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.python.config import CameraConfig


class CameraCaptureError(RuntimeError):
    """Raised when capture fails at the camera backend layer."""


class CameraBackend(ABC):
    """Abstract camera backend.

    Backends return linear RGB images in range [0.0, 1.0].
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config

    @abstractmethod
    def capture_linear_rgb(self) -> np.ndarray:
        """Capture one frame as float32 linear RGB image."""
