"""Camera backend factory."""

from __future__ import annotations

from src.python.config import CameraConfig

from .base import CameraBackend, CameraCaptureError
from .canon_edsdk import CanonEdsdkCamera
from .opencv import OpenCVCamera

_BACKENDS: dict[str, type[CameraBackend]] = {
    "canon_edsdk": CanonEdsdkCamera,
    "opencv": OpenCVCamera,
}


def available_camera_backends() -> tuple[str, ...]:
    """Return registered camera backend names."""
    return tuple(sorted(_BACKENDS.keys()))


def create_camera_backend(config: CameraConfig) -> CameraBackend:
    """Instantiate camera backend from config."""
    backend_name = config.backend.strip().lower()
    backend_cls = _BACKENDS.get(backend_name)
    if backend_cls is None:
        names = ", ".join(available_camera_backends())
        raise CameraCaptureError(
            f"Unknown camera backend: '{config.backend}'. Available backends: {names}."
        )
    return backend_cls(config)
