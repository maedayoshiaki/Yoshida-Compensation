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


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB image values to sRGB."""
    threshold = 0.0031308
    return np.where(
        rgb <= threshold,
        rgb * 12.92,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
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

    normalized = np.ascontiguousarray(normalized, dtype=np.float32)
    np.clip(normalized, 0.0, 1.0, out=normalized)  # in-place: avoid a full-frame copy
    return normalized


def _raw_to_linear_rgb(raw_bytes: bytes, half_size: bool = False) -> np.ndarray:
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
                half_size=half_size,
            )
    except Exception as exc:
        raise CameraCaptureError(
            f"RAW conversion failed for canon_edsdk backend: {exc}"
        ) from exc

    if rgb is None:
        raise CameraCaptureError("RAW conversion failed for canon_edsdk backend.")
    # In-place scale + clip to keep peak memory low: a 32 MP frame is ~390 MB
    # as float32, and separate ``/`` and ``np.clip`` calls would each allocate
    # another ~390 MB (~1 GB peak), which OOMs on low-RAM machines.
    out = rgb.astype(np.float32)
    del rgb
    out /= 65535.0
    np.clip(out, 0.0, 1.0, out=out)
    return out


def capture_srgb_uint8(camera_controller, image_quality: str) -> np.ndarray:
    """Capture one still image and return it as RGB uint8 regardless of RAW/JPEG mode."""
    raw_processor = _raw_to_linear_rgb if _image_quality_looks_raw(image_quality) else None
    images = camera_controller.capture_numpy(raw_processor=raw_processor)

    if not images or images[0] is None:
        raise CameraCaptureError("No image returned from canon_edsdk backend.")

    linear_rgb = _normalize_capture_output(images[0], image_quality)
    srgb = _linear_to_srgb(linear_rgb)
    return np.clip(srgb * 255.0, 0.0, 255.0).astype(np.uint8)


class CanonEdsdkCamera(CameraBackend):
    """Capture backend backed by Canon EDSDK.

    Canon EDSDK を用いて撮影し、リニア RGB 画像を返す。

    Can be used as a context manager to hold ONE EDSDK session open across
    many captures::

        with CanonEdsdkCamera(config) as cam:
            for _ in range(n):
                frame = cam.capture_linear_rgb()

    EDSDK must be initialized once per process; opening a fresh session per
    shot (``InitializeSDK``/``TerminateSDK`` each time) makes the SDK
    unstable and raises ``EDS_ERR_DEVICE_BUSY`` / ``EDS_ERR_INTERNAL_ERROR``
    during batch capture. Outside a ``with`` block each call still opens a
    one-shot session (fine for single captures).

    context manager として使うと、多数の撮影で 1 つの EDSDK セッションを
    開いたまま使い回せる。EDSDK はプロセスで 1 回だけ初期化する設計であり、
    撮影ごとに ``InitializeSDK``/``TerminateSDK`` を繰り返すと SDK が不安定に
    なり、バッチ撮影中に ``EDS_ERR_DEVICE_BUSY`` / ``EDS_ERR_INTERNAL_ERROR``
    が発生する。``with`` の外では従来どおり 1 回ごとにセッションを開く
    （単発撮影には十分）。
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        # Persistent CameraController while used as a context manager.
        # context manager として使う間だけ保持する永続 CameraController。
        self._controller = None

    def __enter__(self) -> "CanonEdsdkCamera":
        """Open one EDSDK session and apply capture properties once.

        1 つの EDSDK セッションを開き、撮影プロパティを 1 回だけ適用する。
        """
        try:
            from edsdk.camera_controller import CameraController
        except ImportError as exc:
            raise CameraCaptureError(
                "edsdk is required for canon_edsdk backend."
            ) from exc

        controller = CameraController(register_property_events=False)
        controller.__enter__()
        try:
            controller.set_properties(
                av=self.config.av,
                tv=self.config.tv,
                iso=self.config.iso,
                image_quality=self.config.image_quality,
                validate=True,
                tolerate_not_supported=True,
            )
        except Exception:
            controller.__exit__(None, None, None)
            raise
        self._controller = controller
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        controller, self._controller = self._controller, None
        if controller is not None:
            controller.__exit__(exc_type, exc, tb)

    def _capture_via(self, controller, tv) -> np.ndarray:
        """Set the shutter (if given) and capture one linear RGB frame."""
        if tv is not None:
            controller.set_properties(
                tv=tv, validate=True, tolerate_not_supported=True
            )
        # ``half_size`` develops the RAW at half resolution (~1/4 the memory),
        # essential on low-RAM machines. The measurement (ROI mean of solid
        # patterns) is unaffected; the ROI must be set in capture-resolution px.
        half = bool(getattr(self.config, "half_size", False))
        images = controller.capture_numpy(
            raw_processor=lambda rb: _raw_to_linear_rgb(rb, half_size=half)
        )
        if not images or images[0] is None:
            raise CameraCaptureError("No image returned from canon_edsdk backend.")
        return _normalize_capture_output(images[0], self.config.image_quality)

    def capture_linear_rgb(self, tv: "str | None" = None) -> np.ndarray:
        """Capture one frame from the Canon camera as linear RGB.

        Canon カメラから 1 枚撮影し、リニア RGB 画像を返す。

        Reuses the open session when inside a ``with`` block; otherwise opens
        a one-shot session. ``tv`` overrides the shutter speed for this shot
        (used by the simple-HDR bracket).

        ``with`` ブロック内なら開いているセッションを再利用し、そうでなければ
        単発セッションを開く。``tv`` はこの 1 枚のシャッター速度を上書きする
        （簡易 HDR ブラケットで使用）。

        Returns:
            Linear RGB image of shape ``(H, W, 3)``, dtype ``float32``.
                形状 ``(H, W, 3)``、dtype ``float32`` のリニア RGB 画像。

        Raises:
            CameraCaptureError: If EDSDK is unavailable or capture fails.
                EDSDK が利用できない場合、または撮影に失敗した場合。
        """
        if self._controller is not None:
            return self._capture_via(self._controller, tv)

        try:
            from edsdk.camera_controller import CameraController
        except ImportError as exc:
            raise CameraCaptureError(
                "edsdk is required for canon_edsdk backend."
            ) from exc

        with CameraController(register_property_events=False) as camera:
            camera.set_properties(
                av=self.config.av,
                tv=tv if tv is not None else self.config.tv,
                iso=self.config.iso,
                image_quality=self.config.image_quality,
                validate=True,
                tolerate_not_supported=True,
            )
            return self._capture_via(camera, None)

    def supported_shutter_speeds(self) -> list[str]:
        """List the Tv (shutter) values this camera body accepts.

        この機種が受理する Tv（シャッター）値の一覧を返す。

        Queries EDSDK property descriptors only; it does not release the
        shutter. Used by the simple-HDR bracket to snap scaled exposures
        onto values the body will accept.

        EDSDK のプロパティ記述子を照会するのみで、シャッターは切らない。
        簡易 HDR ブラケットが、スケール後の露光を機種が受理する値へ
        スナップするために使う。

        Only values the body's own parser round-trips are returned:
        ``list_supported()`` can report display forms that ``set_properties``
        later rejects (e.g. seconds-notation third-stops like ``0"3 (1/3)``),
        which would abort a bracket. Such values are filtered out here.

        戻すのは機種自身のパーサが往復できる値のみ。``list_supported()`` は
        後で ``set_properties`` が弾く表示形式（例 ``0"3 (1/3)`` の秒表記
        1/3段）を返すことがあり、ブラケットを中断させるため、ここで除外する。

        Returns:
            Accepted Tv display strings (e.g. ``"1/30"``, ``0"5``, ``"2"``).
                受理される Tv 表示文字列（例 ``"1/30"``, ``0"5``, ``"2"``）。

        Raises:
            CameraCaptureError: If EDSDK is unavailable.
                EDSDK が利用できない場合。
        """
        if self._controller is not None:
            return self._accepted_shutter_speeds(self._controller)

        try:
            from edsdk.camera_controller import CameraController
        except ImportError as exc:
            raise CameraCaptureError(
                "edsdk is required for canon_edsdk backend."
            ) from exc

        with CameraController(register_property_events=False) as camera:
            return self._accepted_shutter_speeds(camera)

    @staticmethod
    def _accepted_shutter_speeds(controller) -> list[str]:
        """Return supported Tv values the body's parser also accepts.

        機種のパーサが受理する（往復できる）サポート Tv 値だけを返す。
        """
        try:
            from edsdk.camera_controller import _parse_tv
        except ImportError:
            _parse_tv = None

        values = list(controller.list_supported().get("Tv", []))
        if _parse_tv is None:
            return [str(tv) for tv in values]

        accepted: list[str] = []
        for tv in values:
            try:
                _parse_tv(tv)
            except Exception:
                continue  # display form set_properties would reject
            accepted.append(str(tv))
        return accepted
