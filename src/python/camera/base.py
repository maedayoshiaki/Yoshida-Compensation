"""Base interface for camera capture backends.

Defines a backend contract so camera-specific code stays isolated from
pipeline logic.

カメラキャプチャバックエンドの基底インターフェース。

カメラ固有実装をパイプライン本体から分離するための共通契約を定義する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.python.config import CameraConfig


class CameraCaptureError(RuntimeError):
    """Raised when capture fails at the camera backend layer.

    カメラバックエンド層でキャプチャに失敗したときに送出される例外。
    """


class CameraBackend(ABC):
    """Abstract camera backend.

    Backends return linear RGB images in range [0.0, 1.0].

    抽象カメラバックエンド。

    すべての実装は、範囲 [0.0, 1.0] のリニア RGB 画像を返す。
    """

    def __init__(self, config: CameraConfig) -> None:
        """Initialize backend with camera configuration.

        Args:
            config: Camera section of the project configuration.
                プロジェクト設定の camera セクション。
        """
        self.config = config

    @abstractmethod
    def capture_linear_rgb(self) -> np.ndarray:
        """Capture one frame as float32 linear RGB image.

        Returns:
            Linear RGB image in shape ``(H, W, 3)`` and dtype ``float32``.
                形状 ``(H, W, 3)``、dtype ``float32`` のリニア RGB 画像。

        Raises:
            CameraCaptureError: When capture cannot be completed.
                キャプチャを完了できない場合。
        """
