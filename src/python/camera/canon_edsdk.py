"""Canon EDSDK camera backend.

Implements capture for Canon cameras via EDSDK and converts RAW frames
to linear RGB for the photometric pipeline.

Canon EDSDK を利用したカメラバックエンド実装。

Canon カメラから RAW を取得し、測光パイプライン向けのリニア RGB に変換する。
"""

from __future__ import annotations

import numpy as np

from .base import CameraBackend, CameraCaptureError


def _raw_to_linear_rgb(raw_bytes: bytes) -> np.ndarray:
    """Convert RAW bytes to normalized linear RGB.

    Args:
        raw_bytes: RAW image payload provided by EDSDK.
            EDSDK から渡される RAW 画像データ。

    Returns:
        Linear RGB image in range [0.0, 1.0], dtype ``float32``.
            範囲 [0.0, 1.0] のリニア RGB 画像（dtype ``float32``）。

    Raises:
        CameraCaptureError: If required dependencies are missing or RAW
            conversion fails.
            必要な依存関係が不足している場合、または RAW 変換に失敗した場合。
    """
    try:
        import rawpy
    except ImportError as exc:
        raise CameraCaptureError(
            "rawpy is required for canon_edsdk backend. Install extras with `uv sync --extra sample`."
        ) from exc

    with rawpy.imread(raw_bytes) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            bright=1,
            output_bps=16,
            output_color=rawpy.ColorSpace.raw,
            use_camera_wb=True,
            use_auto_wb=False,
            no_auto_scale=False,
            user_sat=None,
        )
    if rgb is None:
        raise CameraCaptureError("RAW conversion failed for canon_edsdk backend.")
    return rgb.astype(np.float32) / 65535.0


class CanonEdsdkCamera(CameraBackend):
    """Capture backend backed by Canon EDSDK.

    Canon EDSDK を用いて 1 枚撮影し、リニア RGB 画像を返す。
    """

    def capture_linear_rgb(self) -> np.ndarray:
        """Capture one frame from Canon camera as linear RGB.

        Returns:
            Linear RGB image in shape ``(H, W, 3)`` and dtype ``float32``.
                形状 ``(H, W, 3)``、dtype ``float32`` のリニア RGB 画像。

        Raises:
            CameraCaptureError: If EDSDK is unavailable or capture fails.
                EDSDK が利用できない場合、または撮影に失敗した場合。
        """
        try:
            from edsdk.camera_controller import CameraController
        except ImportError as exc:
            raise CameraCaptureError(
                "edsdk is required for canon_edsdk backend."
            ) from exc

        with CameraController() as camera:
            camera.set_properties(
                av=self.config.av,
                tv=self.config.tv,
                iso=self.config.iso,
                image_quality=self.config.image_quality,
            )
            images = camera.capture_numpy(raw_processor=_raw_to_linear_rgb)

        if not images or images[0] is None:
            raise CameraCaptureError("No image returned from canon_edsdk backend.")

        image = images[0]
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Keep the return contract stable even if raw_processor behavior changes.
        # raw_processor の挙動が変わっても戻り値契約を維持する。
        if image.max() > 1.0:
            image /= 65535.0
        return np.clip(image, 0.0, 1.0)
