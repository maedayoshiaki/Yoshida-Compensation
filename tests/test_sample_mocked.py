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


def _load_sample_module():
    _install_optional_dependency_stubs()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sample_path = repo_root / "examples" / "python" / "sample.py"
    spec = importlib.util.spec_from_file_location("sample_module", sample_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {sample_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sample = _load_sample_module()


def _save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _load_rgb_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


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
    target_image_space: str = "projector",
    c2p_map: str = "",
    p2c_map: str = "",
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


def _stub_window_calls(monkeypatch) -> None:
    for func_name in (
        "namedWindow",
        "setWindowProperty",
        "moveWindow",
        "imshow",
        "destroyWindow",
    ):
        monkeypatch.setattr(sample.cv2, func_name, lambda *args, **kwargs: None)
    monkeypatch.setattr(sample.cv2, "waitKey", lambda *args, **kwargs: 0)


def test_sample_runs_with_mocked_camera_and_projector(tmp_path, monkeypatch) -> None:
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
        p2c_map="maps/p2c.npy",
    )

    monkeypatch.chdir(tmp_path)
    _stub_window_calls(monkeypatch)

    captured_images = iter(
        sample.generate_projection_patterns(
            width,
            height,
            num_divisions=2,
            dtype=np.dtype(np.uint8),
        )
    )
    monkeypatch.setattr(
        sample,
        "capture_image",
        lambda: next(captured_images, None),
    )

    sample.main(["sample.py", "--config", str(config_path)])

    assert (data_dir / "color_mixing_matrices.npy").exists()
    assert len(list((data_dir / "linear_proj_patterns").glob("pattern_*.png"))) == 8
    assert (
        len(list((data_dir / "inv_gamma_proj_patterns").glob("inv_gamma_pattern_*.png")))
        == 8
    )
    assert len(list((data_dir / "captured_images").glob("captured_image_*.png"))) == 8
    assert (data_dir / "compensation_images" / "compensation_image_00.png").exists()
    assert (data_dir / "inv_gamma_comp_images" / "inv_gamma_comp_image_00.png").exists()


def test_capture_image_reports_camera_errors(monkeypatch, capsys) -> None:
    class FailingCamera:
        def capture_linear_rgb(self) -> np.ndarray:
            raise sample.CameraCaptureError("raw develop failed")

    monkeypatch.setattr(
        sample,
        "get_config",
        lambda: types.SimpleNamespace(camera=types.SimpleNamespace()),
    )
    monkeypatch.setattr(sample, "create_camera_backend", lambda cfg: FailingCamera())

    image = sample.capture_image()
    stderr = capsys.readouterr().err

    assert image is None
    assert "Camera capture failed: raw develop failed" in stderr


def test_sample_uses_camera_space_compensation_for_c2p(
    tmp_path,
    monkeypatch,
) -> None:
    width, height = 4, 3
    data_dir = tmp_path / "data"
    target_dir = data_dir / "target_images"
    target_image = np.full((height, width, 3), 17, dtype=np.uint8)
    _save_rgb_image(target_dir / "target.png", target_image)

    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        width=width,
        height=height,
        warp_method="c2p",
        target_image_space="camera",
        c2p_map="maps/c2p.npy",
    )

    monkeypatch.chdir(tmp_path)
    _stub_window_calls(monkeypatch)
    monkeypatch.setattr(
        sample,
        "generate_projection_patterns",
        lambda *args, **kwargs: [np.zeros((height, width, 3), dtype=np.uint8)],
    )
    monkeypatch.setattr(
        sample,
        "apply_inverse_gamma_correction",
        lambda image, gamma=None: image,
    )
    monkeypatch.setattr(
        sample,
        "capture_image",
        lambda: np.zeros((height, width, 3), dtype=np.uint8),
    )

    color_mixing_matrices = np.ones((height, width, 4, 3), dtype=np.float32)
    monkeypatch.setattr(
        sample,
        "calc_color_mixing_matrices",
        lambda *args, **kwargs: color_mixing_matrices,
    )

    calc_calls: list[tuple[np.ndarray, np.ndarray]] = []
    compensation_input = np.full((height, width, 3), 99, dtype=np.uint8)

    def fake_calc_compensation_image(
        target_image: np.ndarray,
        color_mixing_matrices: np.ndarray,
        dtype,
    ) -> np.ndarray:
        calc_calls.append((target_image.copy(), color_mixing_matrices))
        return compensation_input

    warp_calls: list[np.ndarray] = []

    def fake_warp_image(
        src_image: np.ndarray,
        pixel_map_path: str,
        proj_width: int,
        proj_height: int,
        image_width: int,
        image_height: int,
        warp_method: str = "c2p",
    ) -> np.ndarray:
        warp_calls.append(src_image.copy())
        return np.full((image_height, image_width, 3), 123, dtype=np.uint8)

    monkeypatch.setattr(sample, "calc_compensation_image", fake_calc_compensation_image)
    monkeypatch.setattr(sample, "warp_image", fake_warp_image)
    monkeypatch.setattr(
        sample,
        "invwarp_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("camera-space target should not be inverse warped for c2p")
        ),
    )
    monkeypatch.setattr(
        sample,
        "warp_color_mixing_matrices_to_projector",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("c2p should not warp color mixing matrices")
        ),
    )

    sample.main(["sample.py", "--config", str(config_path)])

    assert len(calc_calls) == 1
    calc_target_image, calc_cmm = calc_calls[0]
    assert np.array_equal(calc_target_image, target_image)
    assert calc_cmm is color_mixing_matrices
    assert len(warp_calls) == 1
    assert np.array_equal(warp_calls[0], compensation_input)


def test_sample_inverse_warps_projector_target_for_c2p(
    tmp_path,
    monkeypatch,
) -> None:
    width, height = 4, 3
    data_dir = tmp_path / "data"
    target_dir = data_dir / "target_images"
    target_image = np.full((height, width, 3), 23, dtype=np.uint8)
    camera_space_target = np.full((height, width, 3), 61, dtype=np.uint8)
    _save_rgb_image(target_dir / "target.png", target_image)

    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        width=width,
        height=height,
        warp_method="c2p",
        target_image_space="projector",
        c2p_map="maps/c2p.npy",
    )

    monkeypatch.chdir(tmp_path)
    _stub_window_calls(monkeypatch)
    monkeypatch.setattr(
        sample,
        "generate_projection_patterns",
        lambda *args, **kwargs: [np.zeros((height, width, 3), dtype=np.uint8)],
    )
    monkeypatch.setattr(
        sample,
        "apply_inverse_gamma_correction",
        lambda image, gamma=None: image,
    )
    monkeypatch.setattr(
        sample,
        "capture_image",
        lambda: np.zeros((height, width, 3), dtype=np.uint8),
    )

    color_mixing_matrices = np.ones((height, width, 4, 3), dtype=np.float32)
    monkeypatch.setattr(
        sample,
        "calc_color_mixing_matrices",
        lambda *args, **kwargs: color_mixing_matrices,
    )

    invwarp_calls: list[np.ndarray] = []

    def fake_invwarp_image(
        src_image: np.ndarray,
        pixel_map_path: str,
        proj_width: int,
        proj_height: int,
        cam_width: int,
        cam_height: int,
        warp_method: str = "c2p",
    ) -> np.ndarray:
        invwarp_calls.append(src_image.copy())
        return camera_space_target

    calc_calls: list[np.ndarray] = []

    def fake_calc_compensation_image(
        target_image: np.ndarray,
        color_mixing_matrices: np.ndarray,
        dtype,
    ) -> np.ndarray:
        calc_calls.append(target_image.copy())
        return np.full((height, width, 3), 88, dtype=np.uint8)

    warp_calls: list[np.ndarray] = []

    def fake_warp_image(
        src_image: np.ndarray,
        pixel_map_path: str,
        proj_width: int,
        proj_height: int,
        image_width: int,
        image_height: int,
        warp_method: str = "c2p",
    ) -> np.ndarray:
        warp_calls.append(src_image.copy())
        return np.full((image_height, image_width, 3), 123, dtype=np.uint8)

    monkeypatch.setattr(sample, "invwarp_image", fake_invwarp_image)
    monkeypatch.setattr(sample, "calc_compensation_image", fake_calc_compensation_image)
    monkeypatch.setattr(sample, "warp_image", fake_warp_image)
    monkeypatch.setattr(
        sample,
        "warp_color_mixing_matrices_to_projector",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("c2p should not warp color mixing matrices")
        ),
    )

    sample.main(["sample.py", "--config", str(config_path)])

    assert len(invwarp_calls) == 1
    assert np.array_equal(invwarp_calls[0], target_image)
    assert len(calc_calls) == 1
    assert np.array_equal(calc_calls[0], camera_space_target)
    assert len(warp_calls) == 1


def test_sample_saves_warp_debug_images_during_normal_run(
    tmp_path,
    monkeypatch,
) -> None:
    width, height = 4, 3
    data_dir = tmp_path / "data"
    target_dir = data_dir / "target_images"
    target_image = np.full((height, width, 3), 23, dtype=np.uint8)
    camera_space_target = np.full((height, width, 3), 61, dtype=np.uint8)
    before_warped_compensation = np.full((height, width, 3), 88, dtype=np.uint8)
    final_warped_compensation = np.full((height, width, 3), 123, dtype=np.uint8)
    _save_rgb_image(target_dir / "target.png", target_image)

    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        width=width,
        height=height,
        warp_method="c2p",
        target_image_space="projector",
        c2p_map="maps/c2p.npy",
    )

    monkeypatch.chdir(tmp_path)
    _stub_window_calls(monkeypatch)
    monkeypatch.setattr(
        sample,
        "generate_projection_patterns",
        lambda *args, **kwargs: [np.zeros((height, width, 3), dtype=np.uint8)],
    )
    monkeypatch.setattr(
        sample,
        "apply_inverse_gamma_correction",
        lambda image, gamma=None: image,
    )
    monkeypatch.setattr(
        sample,
        "capture_image",
        lambda: np.zeros((height, width, 3), dtype=np.uint8),
    )

    color_mixing_matrices = np.ones((height, width, 4, 3), dtype=np.float32)
    monkeypatch.setattr(
        sample,
        "calc_color_mixing_matrices",
        lambda *args, **kwargs: color_mixing_matrices,
    )
    monkeypatch.setattr(
        sample,
        "invwarp_image",
        lambda *args, **kwargs: camera_space_target,
    )
    monkeypatch.setattr(
        sample,
        "calc_compensation_image",
        lambda *args, **kwargs: before_warped_compensation,
    )
    monkeypatch.setattr(
        sample,
        "warp_image",
        lambda *args, **kwargs: final_warped_compensation,
    )
    monkeypatch.setattr(
        sample,
        "warp_color_mixing_matrices_to_projector",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("c2p should not warp color mixing matrices")
        ),
    )

    sample.main(["sample.py", "--config", str(config_path)])

    debug_dir = data_dir / "sample_warp_debug_DEBUG_ONLY"
    target_debug_dir = debug_dir / "00_target"

    assert (debug_dir / "README.md").exists()
    assert (debug_dir / "session.json").exists()
    assert (target_debug_dir / "metadata.json").exists()
    assert (target_debug_dir / "00_original_target.png").exists()
    assert (
        target_debug_dir / "01_prepare_target_projector_to_camera_output.png"
    ).exists()
    assert (
        target_debug_dir / "02_final_compensation_camera_to_projector_output.png"
    ).exists()

    saved_prepare = _load_rgb_image(
        target_debug_dir / "01_prepare_target_projector_to_camera_output.png"
    )
    saved_final = _load_rgb_image(
        target_debug_dir / "02_final_compensation_camera_to_projector_output.png"
    )
    assert np.array_equal(saved_prepare, camera_space_target)
    assert np.array_equal(saved_final, final_warped_compensation)


def test_sample_uses_projector_space_compensation_for_p2c(
    tmp_path,
    monkeypatch,
) -> None:
    width, height = 4, 3
    data_dir = tmp_path / "data"
    target_dir = data_dir / "target_images"
    target_image = np.full((height, width, 3), 21, dtype=np.uint8)
    _save_rgb_image(target_dir / "target.png", target_image)

    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        width=width,
        height=height,
        warp_method="p2c",
        target_image_space="projector",
        p2c_map="maps/p2c.npy",
    )

    monkeypatch.chdir(tmp_path)
    _stub_window_calls(monkeypatch)
    monkeypatch.setattr(
        sample,
        "generate_projection_patterns",
        lambda *args, **kwargs: [np.zeros((height, width, 3), dtype=np.uint8)],
    )
    monkeypatch.setattr(
        sample,
        "apply_inverse_gamma_correction",
        lambda image, gamma=None: image,
    )
    monkeypatch.setattr(
        sample,
        "capture_image",
        lambda: np.zeros((height, width, 3), dtype=np.uint8),
    )

    camera_space_cmm = np.ones((height, width, 4, 3), dtype=np.float32)
    projector_space_cmm = np.full((height, width, 4, 3), 2.0, dtype=np.float32)

    monkeypatch.setattr(
        sample,
        "calc_color_mixing_matrices",
        lambda *args, **kwargs: camera_space_cmm,
    )

    monkeypatch.setattr(
        sample,
        "warp_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("projector-space target should not be warped for p2c")
        ),
    )
    monkeypatch.setattr(
        sample,
        "invwarp_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("p2c should not inverse warp projector-space target")
        ),
    )

    matrix_warp_calls: list[np.ndarray] = []

    def fake_warp_color_mixing_matrices_to_projector(
        color_mixing_matrices: np.ndarray,
        pixel_map_path: str,
        proj_width: int,
        proj_height: int,
        image_width: int,
        image_height: int,
        warp_method: str = "p2c",
    ) -> np.ndarray:
        matrix_warp_calls.append(color_mixing_matrices.copy())
        return projector_space_cmm

    calc_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_calc_compensation_image(
        target_image: np.ndarray,
        color_mixing_matrices: np.ndarray,
        dtype,
    ) -> np.ndarray:
        calc_calls.append((target_image.copy(), color_mixing_matrices.copy()))
        return np.full((height, width, 3), 200, dtype=np.uint8)

    monkeypatch.setattr(
        sample,
        "warp_color_mixing_matrices_to_projector",
        fake_warp_color_mixing_matrices_to_projector,
    )
    monkeypatch.setattr(sample, "calc_compensation_image", fake_calc_compensation_image)

    sample.main(["sample.py", "--config", str(config_path)])

    assert len(matrix_warp_calls) == 1
    assert np.array_equal(matrix_warp_calls[0], camera_space_cmm)
    assert len(calc_calls) == 1
    calc_target_image, calc_cmm = calc_calls[0]
    assert np.array_equal(calc_target_image, target_image)
    assert np.array_equal(calc_cmm, projector_space_cmm)


def test_sample_warps_camera_target_for_p2c(
    tmp_path,
    monkeypatch,
) -> None:
    width, height = 4, 3
    data_dir = tmp_path / "data"
    target_dir = data_dir / "target_images"
    target_image = np.full((height, width, 3), 21, dtype=np.uint8)
    projector_space_target = np.full((height, width, 3), 44, dtype=np.uint8)
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
    _stub_window_calls(monkeypatch)
    monkeypatch.setattr(
        sample,
        "generate_projection_patterns",
        lambda *args, **kwargs: [np.zeros((height, width, 3), dtype=np.uint8)],
    )
    monkeypatch.setattr(
        sample,
        "apply_inverse_gamma_correction",
        lambda image, gamma=None: image,
    )
    monkeypatch.setattr(
        sample,
        "capture_image",
        lambda: np.zeros((height, width, 3), dtype=np.uint8),
    )

    camera_space_cmm = np.ones((height, width, 4, 3), dtype=np.float32)
    projector_space_cmm = np.full((height, width, 4, 3), 2.0, dtype=np.float32)

    monkeypatch.setattr(
        sample,
        "calc_color_mixing_matrices",
        lambda *args, **kwargs: camera_space_cmm,
    )

    warp_calls: list[np.ndarray] = []

    def fake_warp_image(
        src_image: np.ndarray,
        pixel_map_path: str,
        proj_width: int,
        proj_height: int,
        image_width: int,
        image_height: int,
        warp_method: str = "p2c",
    ) -> np.ndarray:
        warp_calls.append(src_image.copy())
        return projector_space_target

    monkeypatch.setattr(sample, "warp_image", fake_warp_image)
    monkeypatch.setattr(
        sample,
        "invwarp_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("camera-space target should not be inverse warped for p2c")
        ),
    )

    matrix_warp_calls: list[np.ndarray] = []

    def fake_warp_color_mixing_matrices_to_projector(
        color_mixing_matrices: np.ndarray,
        pixel_map_path: str,
        proj_width: int,
        proj_height: int,
        image_width: int,
        image_height: int,
        warp_method: str = "p2c",
    ) -> np.ndarray:
        matrix_warp_calls.append(color_mixing_matrices.copy())
        return projector_space_cmm

    calc_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_calc_compensation_image(
        target_image: np.ndarray,
        color_mixing_matrices: np.ndarray,
        dtype,
    ) -> np.ndarray:
        calc_calls.append((target_image.copy(), color_mixing_matrices.copy()))
        return np.full((height, width, 3), 200, dtype=np.uint8)

    monkeypatch.setattr(
        sample,
        "warp_color_mixing_matrices_to_projector",
        fake_warp_color_mixing_matrices_to_projector,
    )
    monkeypatch.setattr(sample, "calc_compensation_image", fake_calc_compensation_image)

    sample.main(["sample.py", "--config", str(config_path)])

    assert len(warp_calls) == 1
    assert np.array_equal(warp_calls[0], target_image)
    assert len(matrix_warp_calls) == 1
    assert np.array_equal(matrix_warp_calls[0], camera_space_cmm)
    assert len(calc_calls) == 1
    calc_target_image, calc_cmm = calc_calls[0]
    assert np.array_equal(calc_target_image, projector_space_target)
    assert np.array_equal(calc_cmm, projector_space_cmm)


def test_warp_camera_array_to_projector_uses_actual_output_shape_for_p2c(
    monkeypatch,
) -> None:
    src_array = np.zeros((6, 4, 3), dtype=np.uint8)

    class FakeWarper:
        def backward_warp(self, src_tensor, dst_size, inpaint):
            return sample.torch.zeros((src_tensor.shape[0], 2, 3), dtype=src_tensor.dtype)

    monkeypatch.setattr(sample, "_create_warper", lambda *args, **kwargs: FakeWarper())

    warped = sample._warp_camera_array_to_projector(
        src_array,
        "dummy.npy",
        proj_width=3,
        proj_height=2,
        image_width=src_array.shape[1],
        image_height=src_array.shape[0],
        warp_method="p2c",
    )

    assert warped.shape == (2, 3, 3)


def test_warp_camera_array_to_projector_uses_actual_output_shape_for_c2p(
    monkeypatch,
) -> None:
    src_array = np.zeros((6, 4, 3), dtype=np.uint8)

    class FakeWarper:
        def forward_warp(
            self,
            src_tensor,
            dst_size,
            splat_method,
            crop_rect,
            inpaint,
            aggregation,
        ):
            return sample.torch.zeros((src_tensor.shape[0], 2, 3), dtype=src_tensor.dtype)

    monkeypatch.setattr(sample, "_create_warper", lambda *args, **kwargs: FakeWarper())

    warped = sample._warp_camera_array_to_projector(
        src_array,
        "dummy.npy",
        proj_width=3,
        proj_height=2,
        image_width=src_array.shape[1],
        image_height=src_array.shape[0],
        warp_method="c2p",
    )

    assert warped.shape == (2, 3, 3)
