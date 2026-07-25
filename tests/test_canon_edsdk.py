import io
import sys
import types
from types import SimpleNamespace

import numpy as np

from src.python.camera import canon_edsdk
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
        def __init__(self, *args, **kwargs) -> None:
            pass

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
        "validate": True,
        "tolerate_not_supported": True,
    }
    assert callable(captured["raw_processor"])
    assert linear_rgb.dtype == np.float32
    assert np.isclose(linear_rgb[0, 0, 0], 1.0)
    assert np.isclose(linear_rgb[0, 0, 1], _srgb_to_linear(128 / 255.0))
    assert np.isclose(linear_rgb[0, 0, 2], 0.0)


def test_triggered_batch_captures_before_developing(monkeypatch) -> None:
    events: list[str] = []

    class FakeMode:
        def __init__(self) -> None:
            self.index = 0

        def trigger(self, *, wait):
            assert wait is True
            events.append(f"trigger:{self.index}")
            frame = SimpleNamespace(
                assets=(
                    SimpleNamespace(
                        filename=f"frame-{self.index}.CR3",
                        data=f"raw-{self.index}".encode(),
                    ),
                )
            )
            self.index += 1
            return frame

        def disarm(self, *, leave_on_card):
            assert leave_on_card is True
            events.append("disarm")

    class FakeController:
        def arm_triggered_capture(self, **kwargs):
            assert kwargs == {
                "transfer": "memory",
                "defaults": {"tv": "1/30"},
            }
            return FakeMode()

    def fake_develop(payload, filename, image_quality, *, half_size):
        events.append(f"develop:{filename}")
        return np.full((1, 1, 3), int(payload[-1:]), dtype=np.float32)

    monkeypatch.setattr(
        canon_edsdk,
        "_captured_bytes_to_linear_rgb",
        fake_develop,
    )
    camera = CanonEdsdkCamera(CameraConfig(image_quality="LR"))
    camera._controller = FakeController()

    with camera.triggered_batch("1/30") as batch:
        batch.trigger()
        batch.trigger()
        assert events == ["trigger:0", "trigger:1"]
        frames = list(batch.iter_linear_rgb())

    assert events == [
        "trigger:0",
        "trigger:1",
        "disarm",
        "develop:frame-0.CR3",
        "develop:frame-1.CR3",
    ]
    assert [float(frame[0, 0, 0]) for frame in frames] == [0.0, 1.0]
