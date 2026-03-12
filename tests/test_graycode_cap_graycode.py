from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.python.config import CameraConfig as SharedCameraConfig

from external.GrayCode.src.python import cap_graycode


def _graycode_app_config(
    *,
    backend: str = "canon_edsdk",
    pattern_dir: str = "data/patterns",
    captured_dir: str = "data/captured",
):
    return SimpleNamespace(
        camera=SimpleNamespace(
            backend=backend,
            av="5",
            tv="1/15",
            iso="100",
            image_quality="LR",
            device_index=2,
            wait_key_ms=1,
        ),
        paths=SimpleNamespace(
            pattern_dir=pattern_dir,
            captured_dir=captured_dir,
        ),
    )


def _linear_to_srgb_uint8(value: float) -> int:
    if value <= 0.0031308:
        srgb = 12.92 * value
    else:
        srgb = 1.055 * (value ** (1.0 / 2.4)) - 0.055
    return int(np.clip(srgb * 255.0, 0.0, 255.0))


def _stub_window_calls(monkeypatch) -> None:
    for func_name in (
        "namedWindow",
        "setWindowProperty",
        "moveWindow",
        "imshow",
        "destroyAllWindows",
    ):
        monkeypatch.setattr(cap_graycode.cv2, func_name, lambda *args, **kwargs: None)
    monkeypatch.setattr(cap_graycode.cv2, "waitKey", lambda *args, **kwargs: 0)


def test_capture_uses_shared_camera_backend_and_maps_edsdk_alias(monkeypatch) -> None:
    seen: dict[str, object] = {}
    linear_rgb = np.array([[[0.0, 0.25, 1.0]]], dtype=np.float32)

    class FakeCamera:
        def capture_linear_rgb(self) -> np.ndarray:
            return linear_rgb

    monkeypatch.setattr(
        cap_graycode,
        "get_config",
        lambda: _graycode_app_config(backend="edsdk"),
    )

    def fake_create_camera_backend(config: SharedCameraConfig):
        seen["config"] = config
        return FakeCamera()

    monkeypatch.setattr(cap_graycode, "create_camera_backend", fake_create_camera_backend)

    captured = cap_graycode.capture()

    assert isinstance(seen["config"], SharedCameraConfig)
    assert seen["config"].backend == "canon_edsdk"
    assert seen["config"].device_index == 2
    assert captured.dtype == np.uint8
    assert captured.tolist() == [
        [[0, _linear_to_srgb_uint8(0.25), _linear_to_srgb_uint8(1.0)]]
    ]


def test_main_saves_grayscale_capture_via_shared_camera_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pattern_dir = tmp_path / "patterns"
    captured_dir = tmp_path / "captured"
    config_path = tmp_path / "graycode.toml"
    pattern_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(pattern_dir / "pattern_00.png"),
        np.full((1, 1), 255, dtype=np.uint8),
    )
    config_path.write_text(
        (
            "[paths]\n"
            f'pattern_dir = "{pattern_dir.as_posix()}"\n'
            f'captured_dir = "{captured_dir.as_posix()}"\n\n'
            "[camera]\n"
            'backend = "canon_edsdk"\n'
            'av = "5"\n'
            'tv = "1/15"\n'
            'iso = "100"\n'
            'image_quality = "LR"\n'
            "device_index = 0\n"
            "wait_key_ms = 1\n"
        ),
        encoding="utf-8",
    )

    linear_rgb = np.array([[[0.0, 0.25, 1.0]]], dtype=np.float32)

    class FakeCamera:
        def capture_linear_rgb(self) -> np.ndarray:
            return linear_rgb

    monkeypatch.setattr(cap_graycode, "create_camera_backend", lambda config: FakeCamera())
    _stub_window_calls(monkeypatch)

    cap_graycode.main(
        ["cap_graycode.py", "0", "0", "--config", str(config_path)]
    )

    saved = cv2.imread(str(captured_dir / "capture_00.png"), cv2.IMREAD_GRAYSCALE)

    assert saved is not None
    assert saved.shape == (1, 1)
    assert int(saved[0, 0]) == 109
