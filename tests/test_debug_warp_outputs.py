from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

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


def _load_module(module_name: str):
    _install_optional_dependency_stubs()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


sample = _load_module("examples.python.sample")
debug_warp = _load_module("examples.python.debug_warp_outputs")


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


def _write_config(
    path: Path,
    *,
    width: int,
    height: int,
    warp_method: str,
    target_image_space: str,
    p2c_map: str = "",
    c2p_map: str = "",
) -> None:
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
c2p_map = "{c2p_map}"
p2c_map = "{p2c_map}"
warp_method = "{warp_method}"
target_image_space = "{target_image_space}"
target_image_dir = "data/target_images"
linear_pattern_dir = "data/linear_proj_patterns"
inv_gamma_pattern_dir = "data/inv_gamma_proj_patterns"
captured_image_dir = "data/captured_images"
compensation_image_dir = "data/compensation_images"
inv_gamma_comp_dir = "data/inv_gamma_comp_images"

[compensation]
num_divisions = 2
use_gpu = false
""".strip(),
        encoding="utf-8",
    )


def test_center_rect_clamps_to_projector_frame() -> None:
    assert sample.center_rect(2, 1, 8, 6) == (3, 2, 2, 1)
    assert sample.center_rect(12, 9, 8, 6) == (0, 0, 8, 6)


def test_debug_warp_outputs_saves_debug_images_and_report(
    tmp_path,
    monkeypatch,
) -> None:
    width, height = 4, 3
    data_dir = tmp_path / "data"
    target_dir = data_dir / "target_images"
    map_path = tmp_path / "maps" / "p2c.npy"

    _save_identity_p2c_map(map_path, width, height)

    target_image = np.zeros((height, width, 3), dtype=np.uint8)
    target_image[..., 0] = 32
    target_image[..., 1] = 128
    target_image[..., 2] = 224
    _save_rgb_image(target_dir / "target.png", target_image)

    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        width=width,
        height=height,
        warp_method="p2c",
        target_image_space="camera",
        p2c_map="maps/p2c.npy",
    )

    monkeypatch.chdir(tmp_path)

    exit_code = debug_warp.main(
        ["debug_warp_outputs.py", "--config", str(config_path)]
    )

    debug_dir = data_dir / "debug_warp_outputs_DEBUG_ONLY"
    report_path = debug_dir / "report.json"
    target_debug_dir = debug_dir / "00_target"

    assert exit_code == 0
    assert (debug_dir / "README.md").exists()
    assert report_path.exists()
    assert (target_debug_dir / "00_original.png").exists()
    assert (target_debug_dir / "01_projector_region.png").exists()
    assert (target_debug_dir / "02_prepared_for_compensation.png").exists()
    assert (target_debug_dir / "03_camera_to_projector_preview.png").exists()
    assert (target_debug_dir / "04_roundtrip_projector_to_camera.png").exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["warp_method"] == "p2c"
    assert report["target_image_space"] == "camera"
    assert report["map"]["inferred_camera_size"] == {"width": width, "height": height}
    assert report["targets"][0]["warnings"] == []
