import io
import sys
import types

import numpy as np

from src.python.camera.canon_edsdk import CanonEdsdkCamera, _raw_to_linear_rgb
from src.python.config import CameraConfig


def _srgb_to_linear(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def test_raw_to_linear_rgb_reads_bytes_via_bytesio(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class FakeRaw:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def postprocess(self, **kwargs):
            seen["postprocess_kwargs"] = kwargs
            return np.full((2, 3, 3), 32768, dtype=np.uint16)

    def fake_imread(source):
        seen["source"] = source
        return FakeRaw()

    rawpy = types.ModuleType("rawpy")
    rawpy.imread = fake_imread
    rawpy.ColorSpace = types.SimpleNamespace(sRGB="sRGB")
    monkeypatch.setitem(sys.modules, "rawpy", rawpy)

    image = _raw_to_linear_rgb(b"raw-payload")

    assert isinstance(seen["source"], io.BytesIO)
    assert seen["source"].getvalue() == b"raw-payload"
    assert seen["postprocess_kwargs"]["output_color"] == "sRGB"
    assert image.dtype == np.float32
    assert np.allclose(image, 32768 / 65535.0)


def test_capture_linear_rgb_converts_jpeg_output_to_linear(monkeypatch) -> None:
    captured: dict[str, object] = {}
    jpeg_rgb = np.array([[[255, 128, 0]]], dtype=np.uint8)

    class FakeController:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def set_properties(self, **kwargs) -> None:
            captured["properties"] = kwargs

        def capture_numpy(self, raw_processor=None):
            captured["raw_processor"] = raw_processor
            return [jpeg_rgb]

    edsdk = types.ModuleType("edsdk")
    camera_controller = types.ModuleType("edsdk.camera_controller")
    camera_controller.CameraController = FakeController
    monkeypatch.setitem(sys.modules, "edsdk", edsdk)
    monkeypatch.setitem(sys.modules, "edsdk.camera_controller", camera_controller)

    camera = CanonEdsdkCamera(
        CameraConfig(av="5", tv="1/15", iso="400", image_quality="LJF")
    )
    linear_rgb = camera.capture_linear_rgb()

    assert captured["properties"] == {
        "av": "5",
        "tv": "1/15",
        "iso": "400",
        "image_quality": "LJF",
    }
    assert captured["raw_processor"] is _raw_to_linear_rgb
    assert linear_rgb.dtype == np.float32
    assert np.isclose(linear_rgb[0, 0, 0], 1.0)
    assert np.isclose(linear_rgb[0, 0, 1], _srgb_to_linear(128 / 255.0))
    assert np.isclose(linear_rgb[0, 0, 2], 0.0)
