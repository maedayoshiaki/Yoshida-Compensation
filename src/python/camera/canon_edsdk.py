"""Canon EDSDK camera backend."""

from __future__ import annotations

import numpy as np

from .base import CameraBackend, CameraCaptureError


def _raw_to_linear_rgb(raw_bytes: bytes) -> np.ndarray:
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
    """Capture backend backed by Canon EDSDK."""

    def capture_linear_rgb(self) -> np.ndarray:
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

        # Keep contract stable even if raw_processor behavior changes.
        if image.max() > 1.0:
            image /= 65535.0
        return np.clip(image, 0.0, 1.0)
