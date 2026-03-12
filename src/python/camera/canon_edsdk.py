"""Canon EDSDK camera backend.

Implements capture for Canon cameras via EDSDK and converts RAW frames
to linear RGB for the photometric pipeline.

Canon EDSDK を利用したカメラバックエンド実装。

Canon カメラから RAW を取得し、測光パイプライン向けのリニア RGB に変換する。
"""

from __future__ import annotations

import io

import numpy as np

from .base import CameraBackend, CameraCaptureError


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB image values to linear RGB."""
    threshold = 0.04045
    return np.where(
        rgb <= threshold,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def _image_quality_looks_raw(image_quality: str) -> bool:
    """Best-effort check whether the configured quality includes RAW data."""
    return "R" in image_quality.strip().upper()


def _normalize_capture_output(image: np.ndarray, image_quality: str) -> np.ndarray:
    """Normalize controller output to float32 linear RGB in [0, 1]."""
    if image is None:
        raise CameraCaptureError("No image returned from canon_edsdk backend.")

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise CameraCaptureError(
            f"canon_edsdk backend returned unexpected image shape {image.shape}."
        )

    normalized = image[..., :3].astype(np.float32, copy=False)
    if np.issubdtype(image.dtype, np.integer):
        max_value = float(np.iinfo(image.dtype).max)
        if max_value <= 0.0:
            raise CameraCaptureError(
                f"canon_edsdk backend returned unsupported dtype {image.dtype}."
            )
        normalized /= max_value
    elif np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(normalized)) if normalized.size else 0.0
        if max_value > 1.0:
            scale = 255.0 if max_value <= 255.0 + 1e-6 else 65535.0
            normalized /= scale
    else:
        raise CameraCaptureError(
            f"canon_edsdk backend returned unsupported dtype {image.dtype}."
        )

    if not _image_quality_looks_raw(image_quality):
        normalized = _srgb_to_linear(normalized)

    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


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

    try:
        payload = bytes(raw_bytes)
    except Exception as exc:
        raise CameraCaptureError(
            "RAW payload from canon_edsdk is not bytes-like."
        ) from exc

    try:
        with rawpy.imread(io.BytesIO(payload)) as raw:
            rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                bright=1,
                output_bps=16,
                output_color=rawpy.ColorSpace.sRGB,
                use_camera_wb=True,
                use_auto_wb=False,
                no_auto_scale=False,
                user_sat=None,
            )
    except Exception as exc:
        raise CameraCaptureError(
            f"RAW conversion failed for canon_edsdk backend: {exc}"
        ) from exc

    if rgb is None:
        raise CameraCaptureError("RAW conversion failed for canon_edsdk backend.")
    return np.clip(rgb.astype(np.float32) / 65535.0, 0.0, 1.0)


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

        return _normalize_capture_output(images[0], self.config.image_quality)
