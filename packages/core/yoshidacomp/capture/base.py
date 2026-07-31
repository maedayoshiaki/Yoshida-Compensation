"""Base interface for camera capture backends.

Defines a backend contract so camera-specific code stays isolated from
pipeline logic.

カメラキャプチャバックエンドの基底インターフェース。

カメラ固有実装をパイプライン本体から分離するための共通契約を定義する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from yoshidacomp.config import CameraConfig


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

    def __enter__(self) -> "CameraBackend":
        """Enter a capture session. Default is a no-op.

        撮影セッションに入る。既定は何もしない。

        Backends that hold expensive per-session state (e.g. a Canon EDSDK
        session) override this to open it once and reuse it across many
        captures. Stateless backends (e.g. OpenCV) keep the no-op.

        セッション単位の高コストな状態（例: Canon EDSDK セッション）を持つ
        バックエンドは、これを上書きして 1 回開き多数の撮影で使い回す。
        状態を持たないバックエンド（例: OpenCV）は no-op のまま。
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit a capture session. Default is a no-op.

        撮影セッションを抜ける。既定は何もしない。
        """
        return None

    @abstractmethod
    def capture_linear_rgb(self, tv: "str | None" = None) -> np.ndarray:
        """Capture one frame as float32 linear RGB image.

        Args:
            tv: Optional shutter-speed override for this shot. Backends that
                support exposure control honor it; others ignore it.
                この 1 枚のシャッター速度の任意上書き。露光制御に対応する
                バックエンドのみ反映し、他は無視する。

        Returns:
            Linear RGB image in shape ``(H, W, 3)`` and dtype ``float32``.
                形状 ``(H, W, 3)``、dtype ``float32`` のリニア RGB 画像。

        Raises:
            CameraCaptureError: When capture cannot be completed.
                キャプチャを完了できない場合。
        """
