"""Camera backend abstraction and factory helpers.

Provides a unified entry point for camera capture backends used by the
project.

カメラキャプチャバックエンドの抽象化とファクトリをまとめた公開モジュール。

プロジェクト内で利用するカメラ実装への統一的な入口を提供する。
"""

from .base import CameraBackend, CameraCaptureError
from .factory import available_camera_backends, create_camera_backend

__all__ = [
    "CameraBackend",
    "CameraCaptureError",
    "available_camera_backends",
    "create_camera_backend",
]
