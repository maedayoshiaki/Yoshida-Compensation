"""Canon EDSDK camera backend.

Implements capture for Canon cameras via EDSDK and converts RAW frames
to linear RGB for the photometric pipeline.

Canon EDSDK を利用したカメラバックエンド実装。

Canon カメラから RAW を取得し、測光パイプライン向けのリニア RGB に変換する。
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import tempfile
import traceback
from pathlib import Path
from typing import Iterator

import cv2
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


def _create_camera_controller(controller_type):
    """Create a controller whose EDSDK session lives in a protected worker.

    edsdk-python 0.2.0 supports ``arm_triggered_capture()`` through the
    protected worker, so the fast in-memory trigger path remains available.
    Keeping EDSDK out of the projection process also lets the worker close the
    camera session when that parent process exits unexpectedly.
    """
    return controller_type(
        register_property_events=False,
        protected=True,
    )


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
    if half_size:
        raise CameraCaptureError(
            "Half-resolution RAW development is disabled. "
            "Use full-resolution RAW (camera.half_size=false)."
        )

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


def _captured_bytes_to_linear_rgb(
    payload: bytes,
    filename: str,
    image_quality: str,
    *,
    half_size: bool = False,
) -> np.ndarray:
    """Develop one triggered-capture asset into linear RGB."""
    suffix = Path(filename).suffix.lower()
    if suffix in {".cr2", ".cr3", ".raw"}:
        return _raw_to_linear_rgb(payload, half_size=half_size)

    encoded = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise CameraCaptureError(
            f"Failed to decode triggered capture asset {filename!r}."
        )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _normalize_capture_output(rgb, image_quality)


def _develop_raw_to_npy_worker(
    raw_path: str,
    linear_path: str,
    half_size: bool,
    error_path: str,
) -> None:
    """Develop one RAW in a short-lived process and persist linear RGB."""
    try:
        payload = Path(raw_path).read_bytes()
        linear_rgb = _raw_to_linear_rgb(payload, half_size=half_size)
        np.save(linear_path, linear_rgb, allow_pickle=False)
    except BaseException:
        Path(error_path).write_text(traceback.format_exc(), encoding="utf-8")
        raise


class CanonEdsdkTriggeredBatch:
    """Spool fast triggered captures, then develop them sequentially.

    The camera transfers each asset to memory, but RAW development is deferred
    until every projection in the batch has been captured. Assets are spooled
    to a temporary directory to avoid retaining an entire calibration pass in
    RAM.
    """

    def __init__(self, camera: "CanonEdsdkCamera", tv: str) -> None:
        controller = camera._controller
        if controller is None:
            raise CameraCaptureError(
                "Triggered batch capture requires an open camera session."
            )
        if not hasattr(controller, "arm_triggered_capture"):
            raise CameraCaptureError(
                "edsdk-python >= 0.2.0 is required for triggered batch capture."
            )

        self._camera = camera
        # The default TemporaryDirectory path is retained for normal use.
        # Some managed Windows environments deny writes to directories created
        # below the process temp root, even when the root itself is writable.
        # In that case callers can provide an already-writable spool directory
        # and we write batch files directly into it with a unique prefix.
        spool_override = os.environ.get("YOSHIDA_EDSDK_TRIGGER_SPOOL_DIR")
        if spool_override:
            self._temporary = None
            self._spool_dir = Path(spool_override)
            self._spool_dir.mkdir(parents=True, exist_ok=True)
            self._spool_prefix = f"{os.getpid()}-{id(self):x}-"
        else:
            self._temporary = tempfile.TemporaryDirectory(
                prefix="yoshida-edsdk-trigger-"
            )
            self._spool_dir = Path(self._temporary.name)
            self._spool_prefix = ""
        self._paths: list[tuple[Path, str]] = []
        self._mode = controller.arm_triggered_capture(
            transfer="memory",
            defaults={"tv": tv},
        )
        self._armed = True

    @staticmethod
    def _select_asset(frame, wants_raw: bool):
        assets = list(frame.assets)
        if not assets:
            raise CameraCaptureError(
                "Triggered capture completed without an image asset."
            )
        if wants_raw:
            for asset in assets:
                if Path(asset.filename).suffix.lower() in {".cr2", ".cr3", ".raw"}:
                    return asset
        return assets[0]

    def trigger(self) -> None:
        """Capture exactly one frame and spool its encoded bytes."""
        if not self._armed:
            raise CameraCaptureError("Triggered capture batch is already closed.")
        frame = self._mode.trigger(wait=True)
        asset = self._select_asset(
            frame,
            _image_quality_looks_raw(self._camera.config.image_quality),
        )
        if asset.data is None:
            raise CameraCaptureError(
                f"Triggered capture asset {asset.filename!r} has no memory data."
            )
        index = len(self._paths)
        suffix = Path(asset.filename).suffix or ".bin"
        path = self._spool_dir / f"{self._spool_prefix}{index:06d}{suffix}"
        path.write_bytes(asset.data)
        self._paths.append((path, asset.filename))

    def _disarm(self) -> None:
        if self._armed:
            self._mode.disarm(leave_on_card=True)
            self._armed = False

    def iter_linear_rgb(self) -> Iterator[np.ndarray]:
        """Disarm capture and develop the spooled frames in trigger order."""
        self._disarm()
        half = bool(getattr(self._camera.config, "half_size", False))
        controller = getattr(self._camera, "_controller", None)
        isolated_raw_development = (
            controller is not None
            and type(controller).__module__.split(".", 1)[0] == "edsdk"
        )
        for path, filename in self._paths:
            suffix = Path(filename).suffix.lower()
            if isolated_raw_development and suffix in {".cr2", ".cr3", ".raw"}:
                linear_path = path.with_name(path.name + ".linear.npy")
                error_path = path.with_name(path.name + ".linear.error.txt")
                context = mp.get_context("spawn")
                worker = context.Process(
                    target=_develop_raw_to_npy_worker,
                    args=(str(path), str(linear_path), half, str(error_path)),
                )
                worker.start()
                worker.join()
                try:
                    if worker.exitcode != 0:
                        details = (
                            error_path.read_text(encoding="utf-8")
                            if error_path.exists()
                            else f"worker exit code {worker.exitcode}"
                        )
                        raise CameraCaptureError(
                            f"RAW development failed for {filename!r}:\n{details}"
                        )
                    developed = np.load(linear_path, allow_pickle=False)
                    try:
                        yield np.asarray(developed, dtype=np.float32)
                    finally:
                        del developed
                finally:
                    linear_path.unlink(missing_ok=True)
                    error_path.unlink(missing_ok=True)
                    path.unlink(missing_ok=True)
            else:
                try:
                    yield _captured_bytes_to_linear_rgb(
                        path.read_bytes(),
                        filename,
                        self._camera.config.image_quality,
                        half_size=half,
                    )
                finally:
                    path.unlink(missing_ok=True)

    def __enter__(self) -> "CanonEdsdkTriggeredBatch":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._disarm()
        finally:
            if self._temporary is not None:
                self._temporary.cleanup()
            else:
                for path in self._spool_dir.glob(f"{self._spool_prefix}*"):
                    path.unlink(missing_ok=True)


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
        if bool(getattr(config, "half_size", False)):
            raise CameraCaptureError(
                "Half-resolution RAW is prohibited; set camera.half_size=false."
            )
        # Persistent CameraController while used as a context manager.
        # context manager として使う間だけ保持する永続 CameraController。
        self._controller = None

    def _prepare_edsdk_import(self) -> None:
        dll_dir = str(getattr(self.config, "edsdk_dll_dir", "")).strip()
        if dll_dir:
            os.environ.setdefault("EDSDK_PYTHON_DLL_DIR", dll_dir)

    def __enter__(self) -> "CanonEdsdkCamera":
        """Open one EDSDK session and apply capture properties once.

        1 つの EDSDK セッションを開き、撮影プロパティを 1 回だけ適用する。
        """
        self._prepare_edsdk_import()
        try:
            from edsdk.camera_controller import CameraController
        except ImportError as exc:
            raise CameraCaptureError(
                "edsdk is required for canon_edsdk backend."
            ) from exc

        controller = _create_camera_controller(CameraController)
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

        self._prepare_edsdk_import()
        try:
            from edsdk.camera_controller import CameraController
        except ImportError as exc:
            raise CameraCaptureError(
                "edsdk is required for canon_edsdk backend."
            ) from exc

        with _create_camera_controller(CameraController) as camera:
            camera.set_properties(
                av=self.config.av,
                tv=tv if tv is not None else self.config.tv,
                iso=self.config.iso,
                image_quality=self.config.image_quality,
                validate=True,
                tolerate_not_supported=True,
            )
            return self._capture_via(camera, None)

    def triggered_batch(self, tv: str) -> CanonEdsdkTriggeredBatch:
        """Create a triggered batch that defers RAW development."""
        return CanonEdsdkTriggeredBatch(self, tv)

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

        self._prepare_edsdk_import()
        try:
            from edsdk.camera_controller import CameraController
        except ImportError as exc:
            raise CameraCaptureError(
                "edsdk is required for canon_edsdk backend."
            ) from exc

        with _create_camera_controller(CameraController) as camera:
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
