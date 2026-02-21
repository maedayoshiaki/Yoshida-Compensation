"""Camera backend factory.

Maps backend names from configuration to concrete camera backend
implementations.

カメラバックエンドのファクトリ。

設定で指定されたバックエンド名を具体的な実装クラスに解決する。
"""

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
    """Return registered camera backend names.

    Returns:
        Sorted backend names.
            利用可能なバックエンド名のソート済みタプル。
    """
    return tuple(sorted(_BACKENDS.keys()))


def create_camera_backend(config: CameraConfig) -> CameraBackend:
    """Instantiate camera backend from config.

    Args:
        config: Camera configuration object.
            カメラ設定オブジェクト。

    Returns:
        Concrete camera backend instance.
            具体的なカメラバックエンドインスタンス。

    Raises:
        CameraCaptureError: If backend name is not registered.
            設定されたバックエンド名が未登録の場合。
    """
    backend_name = config.backend.strip().lower()
    backend_cls = _BACKENDS.get(backend_name)
    if backend_cls is None:
        names = ", ".join(available_camera_backends())
        raise CameraCaptureError(
            f"Unknown camera backend: '{config.backend}'. Available backends: {names}."
        )
    return backend_cls(config)
