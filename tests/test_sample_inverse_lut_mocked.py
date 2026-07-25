"""Hardware-independent test for examples/python/sample_inverse_lut.py.

Mirrors tests/test_sample_mocked.py but exercises the LUT-based sample:
it feeds a saved (measured) inverse LUT and mocks the camera/projector,
then asserts the pipeline runs and linearizes via ``apply_inverse_lut``
(not the analytic inverse gamma).

examples/python/sample_inverse_lut.py のハードウェア非依存テスト。
tests/test_sample_mocked.py を踏襲しつつ、保存済み逆LUTを与えてカメラ/
プロジェクタをモックし、``apply_inverse_lut`` で線形化されることを検証する。
"""

from pathlib import Path
import importlib.util
import sys
import types

import cv2
import numpy as np


def _install_optional_dependency_stubs() -> None:
    if "colour" not in sys.modules:
        colour = types.ModuleType("colour")
        colour.characterisation = types.SimpleNamespace(
            polynomial_expansion_Finlayson2015=lambda *args, **kwargs: None
        )
        colour.algebra = types.SimpleNamespace(vecmul=lambda *args, **kwargs: None)
        colour.XYZ_to_sRGB = lambda image, apply_cctf_encoding=False: image
        sys.modules["colour"] = colour

    if "psutil" not in sys.modules:
        psutil = types.ModuleType("psutil")
        psutil.virtual_memory = lambda: types.SimpleNamespace(available=1 << 30)
        sys.modules["psutil"] = psutil


def _load_new_sample_module():
    _install_optional_dependency_stubs()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sample_path = repo_root / "examples" / "python" / "sample_inverse_lut.py"
    spec = importlib.util.spec_from_file_location("sample_inverse_lut_module", sample_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {sample_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


new_sample = _load_new_sample_module()

from src.python import inverse_lut as ilut


def _save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _save_identity_p2c_map(path: Path, width: int, height: int) -> None:
    rows = []
    for y in range(height):
        for x in range(width):
            rows.append([float(x), float(y), float(x), float(y)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(rows, dtype=np.float32))


def _save_identity_lut(path: Path, resolution: int = 256) -> None:
    """Save an identity inverse LUT (L_c(t) = t) for a deterministic pipeline."""
    t_grid = np.linspace(0.0, 1.0, resolution).astype(np.float32)
    lut_arr = np.stack([t_grid, t_grid, t_grid]).astype(np.float32)
    lut = ilut.InverseLut(
        lut=lut_arr, t_grid=t_grid, knots=[], meta={"source": "channel_ramp"}
    )
    ilut.save_inverse_lut(path, lut)


def _write_config(path: Path, *, width: int, height: int, lut_path: str) -> None:
    path.write_text(
        f"""
[projector]
gamma = 2.2
width = {width}
height = {height}
pos_x = 0
pos_y = 0

[camera]
backend = "opencv"
device_index = 0
wait_key_ms = 1

[paths]
c2p_map = ""
p2c_map = "maps/p2c.npy"
warp_method = "p2c"
target_image_space = "projector"
target_image_dir = "data/target_images"
linear_pattern_dir = "data/linear_proj_patterns"
inv_gamma_pattern_dir = "data/inv_gamma_proj_patterns"
captured_image_dir = "data/captured_images"
compensation_image_dir = "data/compensation_images"
inv_gamma_comp_dir = "data/inv_gamma_comp_images"

[compensation]
num_divisions = 2
use_gpu = false

[inverse_lut]
inverse_lut_path = "{lut_path}"
""".strip(),
        encoding="utf-8",
    )


def _stub_window_calls(monkeypatch) -> None:
    for func_name in (
        "namedWindow",
        "setWindowProperty",
        "moveWindow",
        "resizeWindow",
        "imshow",
        "destroyWindow",
    ):
        monkeypatch.setattr(new_sample.cv2, func_name, lambda *args, **kwargs: None)
    monkeypatch.setattr(new_sample.cv2, "waitKey", lambda *args, **kwargs: 0)


def test_sample_inverse_lut_runs_and_linearizes_with_lut(tmp_path, monkeypatch) -> None:
    width, height = 4, 3
    data_dir = tmp_path / "data"
    target_dir = data_dir / "target_images"
    map_path = tmp_path / "maps" / "p2c.npy"

    _save_identity_p2c_map(map_path, width, height)
    _save_identity_lut(tmp_path / "data" / "inverse_lut.npz")

    target_image = np.zeros((height, width, 3), dtype=np.uint8)
    target_image[..., 0] = 32
    target_image[..., 1] = 128
    target_image[..., 2] = 224
    _save_rgb_image(target_dir / "target.png", target_image)

    config_path = tmp_path / "config.toml"
    _write_config(
        config_path, width=width, height=height, lut_path="data/inverse_lut.npz"
    )

    monkeypatch.chdir(tmp_path)
    _stub_window_calls(monkeypatch)

    # Load sample.py once, mock its capture, and hand it back to the new sample.
    sample_mod = new_sample._load_sample_module()
    captured_images = iter(
        sample_mod.generate_projection_patterns(
            width, height, num_divisions=2, dtype=np.dtype(np.uint8)
        )
    )
    monkeypatch.setattr(
        sample_mod, "capture_image", lambda: next(captured_images, None)
    )
    monkeypatch.setattr(new_sample, "_load_sample_module", lambda: sample_mod)

    # Spy on apply_inverse_lut to prove the LUT path (not inverse gamma) is used.
    calls = {"count": 0}
    original_apply = ilut.apply_inverse_lut

    def _counting_apply(*args, **kwargs):
        calls["count"] += 1
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(ilut, "apply_inverse_lut", _counting_apply)

    new_sample.main(["sample_inverse_lut.py", "--config", str(config_path)])

    # The LUT was applied to all 8 patterns plus the single target compensation.
    assert calls["count"] == 8 + 1

    assert (data_dir / "color_mixing_matrices.npy").exists()
    assert (
        len(list((data_dir / "inv_lut_proj_patterns").glob("inv_lut_pattern_*.png")))
        == 8
    )
    assert (data_dir / "inv_lut_comp_images" / "inv_lut_comp_image_00.png").exists()
    assert (data_dir / "compensation_images" / "compensation_image_00.png").exists()


def test_sample_inverse_lut_exits_when_lut_missing(tmp_path, monkeypatch) -> None:
    width, height = 4, 3
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path, width=width, height=height, lut_path="data/does_not_exist.npz"
    )

    monkeypatch.chdir(tmp_path)

    try:
        new_sample.main(["sample_inverse_lut.py", "--config", str(config_path)])
    except SystemExit as exc:
        assert exc.code == 1
    else:  # pragma: no cover - failure path
        raise AssertionError("Expected SystemExit when the inverse LUT is missing.")
