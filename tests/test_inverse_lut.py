from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from src.python.config import AppConfig, InverseLutConfig, reload_config
from src.python import inverse_lut as ilut


# ── Task 1: config ───────────────────────────────────────────────────


def test_inverse_lut_config_defaults():
    cfg = AppConfig()
    assert isinstance(cfg.inverse_lut, InverseLutConfig)
    assert cfg.inverse_lut.num_levels == 17
    assert cfg.inverse_lut.source == "both"
    assert cfg.inverse_lut.lut_resolution == 256
    assert abs(cfg.inverse_lut.reference_region - 0.2) < 1e-9
    assert abs(cfg.inverse_lut.trust_low - 10 / 255) < 1e-9
    assert abs(cfg.inverse_lut.trust_high - 245 / 255) < 1e-9
    assert cfg.inverse_lut.hdr_enabled is False
    assert cfg.inverse_lut.inverse_lut_path == "data/inverse_lut.npz"


def test_inverse_lut_config_loads_from_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[inverse_lut]
num_levels = 9
source = "gray"
lut_resolution = 128
reference_region = 0.5
trust_low = 0.05
trust_high = 0.9
hdr_enabled = true
inverse_lut_path = "data/custom_lut.npz"
""".strip(),
        encoding="utf-8",
    )
    cfg = reload_config(config_path)
    assert cfg.inverse_lut.num_levels == 9
    assert cfg.inverse_lut.source == "gray"
    assert cfg.inverse_lut.lut_resolution == 128
    assert cfg.inverse_lut.hdr_enabled is True
    assert cfg.inverse_lut.inverse_lut_path == "data/custom_lut.npz"
    reload_config(_REPO_ROOT / "config.toml")  # 後続テストのため既定に戻す


# ── Task 2: pattern generation ───────────────────────────────────────


def test_generate_calibration_patterns_counts_and_kinds():
    samples = ilut.generate_calibration_patterns(4, 3, num_levels=17, dtype=np.uint8)
    # グレー17 + 単独R/G/B 各17 = 68
    assert len(samples) == 68
    gray = [s for s in samples if s.kind == "gray"]
    channel = [s for s in samples if s.kind == "channel"]
    assert len(gray) == 17
    assert len(channel) == 51

    # グレーは r,g,b を記録、単独ランプは投影チャンネルのみ記録
    assert all(s.recorded_channels == (0, 1, 2) for s in gray)
    r_ramp = [s for s in channel if s.recorded_channels == (0,)]
    assert len(r_ramp) == 17
    for s in r_ramp:
        assert s.input_rgb[1] == 0.0 and s.input_rgb[2] == 0.0

    # 画像形状と dtype、最大階調
    assert gray[0].image.shape == (3, 4, 3)
    assert gray[0].image.dtype == np.uint8
    top_gray = max(gray, key=lambda s: s.input_rgb[0])
    assert int(top_gray.image[0, 0, 0]) == 255

    # 全黒の distinct 投影は 1 つ
    distinct_inputs = {tuple(round(v, 6) for v in s.input_rgb) for s in samples}
    assert len(distinct_inputs) == 1 + 4 * (17 - 1)  # 65


# ── Task 3: measurement / reduction ──────────────────────────────────


def _synth_frame(input_rgb, gammas, black, white=0.9, h=8, w=8):
    # white < trust_high (245/255) so the top level is not treated as saturated
    frame = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        frame[:, :, c] = black[c] + (white - black[c]) * (input_rgb[c] ** gammas[c])
    return frame


def test_reduce_frame_central_mean():
    frame = np.zeros((10, 10, 3), dtype=np.float32)
    frame[4:6, 4:6, :] = 1.0  # 中央のみ 1
    mean = ilut.reduce_frame(frame, reference_region=0.2)
    assert mean.shape == (3,)
    assert np.allclose(mean, 1.0)


def test_measure_records_only_recorded_channels_and_dedups():
    gammas = (2.2, 2.0, 1.8)
    black = (10 / 255, 8 / 255, 6 / 255)
    samples = ilut.generate_calibration_patterns(6, 6, num_levels=5, dtype=np.uint8)

    calls = {"n": 0}

    def capture_fn(sample):
        calls["n"] += 1
        return _synth_frame(sample.input_rgb, gammas, black)

    data = ilut.measure_calibration_samples(samples, capture_fn, reference_region=0.5)

    assert len(data.measurements) == len(samples)
    distinct = {tuple(round(v, 6) for v in s.input_rgb) for s in samples}
    assert calls["n"] == len(distinct)

    gray_m = next(m for m in data.measurements if m.kind == "gray" and m.input_rgb[0] > 0)
    assert all(not np.isnan(x) for x in gray_m.measured_rgb)
    r_m = next(
        m for m in data.measurements
        if m.kind == "channel" and m.recorded_channels == (0,) and m.input_rgb[0] > 0
    )
    assert not np.isnan(r_m.measured_rgb[0])
    assert np.isnan(r_m.measured_rgb[1]) and np.isnan(r_m.measured_rgb[2])

    v = r_m.input_rgb[0]
    assert abs(r_m.measured_rgb[0] - (black[0] + (0.9 - black[0]) * v ** gammas[0])) < 1e-5


# ── Task 4: capture orchestration ────────────────────────────────────


def test_run_calibration_capture_projects_and_measures(monkeypatch):
    for name in ("namedWindow", "setWindowProperty", "moveWindow", "imshow", "destroyWindow"):
        monkeypatch.setattr(ilut.cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(ilut.cv2, "waitKey", lambda *a, **k: 0)

    samples = ilut.generate_calibration_patterns(6, 6, num_levels=4, dtype=np.uint8)
    distinct = {tuple(round(v, 6) for v in s.input_rgb) for s in samples}

    frames = [np.full((6, 6, 3), 0.5, dtype=np.float32) for _ in range(len(distinct))]

    class FakeCamera:
        def __init__(self, frames):
            self._it = iter(frames)
            self.calls = 0

        def capture_linear_rgb(self):
            self.calls += 1
            return next(self._it)

    camera = FakeCamera(frames)
    data = ilut.run_calibration_capture(samples, camera_backend=camera, show_window=True)

    assert len(data.measurements) == len(samples)
    assert camera.calls == len(distinct)  # 入力で重複排除
    assert np.allclose(data.measurements[0].measured_rgb, 0.5)  # gray black は 3ch 記録


def test_capture_hdr_linear_stub_single_exposure():
    called = {"n": 0}

    def once():
        called["n"] += 1
        return np.full((4, 4, 3), 0.3, dtype=np.float32)

    frame = ilut.capture_hdr_linear(once, enabled=False)
    assert called["n"] == 1
    assert np.allclose(frame, 0.3)


# ── Task 5: build ────────────────────────────────────────────────────


def _build_from_synth(gammas, black, num_levels=17, white=0.9, **kwargs):
    # 設定に依存しないよう信頼境界と解像度を明示（既定と同値）
    kwargs.setdefault("trust_low", 10 / 255)
    kwargs.setdefault("trust_high", 245 / 255)
    kwargs.setdefault("lut_resolution", 256)
    samples = ilut.generate_calibration_patterns(6, 6, num_levels=num_levels, dtype=np.uint8)

    def capture_fn(sample):
        return _synth_frame(sample.input_rgb, gammas, black, white=white)

    data = ilut.measure_calibration_samples(samples, capture_fn, reference_region=0.5)
    return ilut.build_inverse_lut(data, **kwargs)


def test_build_inverse_lut_inverts_gamma_per_channel():
    gammas = (2.2, 2.0, 1.8)
    black = (10 / 255, 8 / 255, 6 / 255)
    lut = _build_from_synth(gammas, black, source="both", lut_resolution=256)

    assert lut.lut.shape == (3, 256)
    for c in range(3):
        assert abs(lut.lut[c, 0] - 0.0) < 1e-3
        assert abs(lut.lut[c, -1] - 1.0) < 1e-3
    for c in range(3):
        assert np.all(np.diff(lut.lut[c]) >= -1e-6)
    for c, g in enumerate(gammas):
        t = lut.t_grid
        expected = t ** (1.0 / g)
        assert np.max(np.abs(lut.lut[c] - expected)) < 0.05


def test_build_inverse_lut_black_subtraction_maps_zero_to_zero():
    lut = _build_from_synth((2.2, 2.2, 2.2), (10 / 255, 10 / 255, 10 / 255))
    assert np.all(lut.lut[:, 0] < 1e-3)


def test_build_inverse_lut_excludes_saturated_knots():
    measurements = []
    levels = np.linspace(0.0, 1.0, 6)
    for v in levels:
        m = 0.02 + 0.98 * v
        measured = 0.99 if m >= 0.95 else m
        measurements.append(
            ilut.CalibrationMeasurement((v, v, v), (measured, measured, measured), (0, 1, 2), "gray")
        )
    data = ilut.CalibrationData(measurements=measurements, meta={})
    lut = ilut.build_inverse_lut(
        data, source="gray", lut_resolution=64, trust_low=0.039, trust_high=0.95
    )
    # 飽和ノットを除外しても NaN なし・単調・[0,1]・黒引きが保たれる。
    # 白は飽和値(0.99)ではなく信頼できる最上位(p=0.8)で正規化されるため上限は ~0.8。
    assert not np.any(np.isnan(lut.lut))
    for c in range(3):
        assert np.all(np.diff(lut.lut[c]) >= -1e-6)
        assert lut.lut[c, 0] < 1e-3
        assert 0.6 < lut.lut[c, -1] < 0.95


def test_build_inverse_lut_source_gray_only_ignores_channel_ramps():
    gammas = (2.2, 2.2, 2.2)
    black = (0.02, 0.02, 0.02)
    lut = _build_from_synth(gammas, black, source="gray")
    assert lut.lut.shape[0] == 3
    assert np.max(np.abs(lut.lut[0] - lut.t_grid ** (1 / 2.2))) < 0.05


# ── Task 6: apply / save-load ────────────────────────────────────────


def test_apply_inverse_lut_linearizes_response():
    gammas = (2.2, 2.0, 1.8)
    black = (0.02, 0.02, 0.02)
    lut = _build_from_synth(gammas, black, source="both")

    t = np.linspace(0, 1, 16, dtype=np.float32)
    target = np.stack([t, t, t], axis=-1).reshape(1, 16, 3)
    projector_input = ilut.apply_inverse_lut(target, lut, dtype=np.float32)
    assert projector_input.shape == target.shape

    for c, g in enumerate(gammas):
        reproduced = projector_input[0, :, c] ** g
        assert np.max(np.abs(reproduced - t)) < 0.05


def test_apply_inverse_lut_uint8_roundtrip_dtype():
    lut = _build_from_synth((2.2, 2.2, 2.2), (0.02, 0.02, 0.02))
    image = np.full((2, 2, 3), 128, dtype=np.uint8)
    out = ilut.apply_inverse_lut(image, lut, dtype=np.uint8)
    assert out.dtype == np.uint8
    assert out.shape == (2, 2, 3)


def test_save_and_load_inverse_lut_roundtrip(tmp_path):
    lut = _build_from_synth((2.2, 2.0, 1.8), (0.02, 0.02, 0.02))
    path = tmp_path / "lut.npz"
    ilut.save_inverse_lut(path, lut)
    assert path.exists()
    loaded = ilut.load_inverse_lut(path)
    assert np.allclose(loaded.lut, lut.lut)
    assert np.allclose(loaded.t_grid, lut.t_grid)
    assert loaded.meta["source"] == lut.meta["source"]


# ── Task 7: CLI ──────────────────────────────────────────────────────


def test_cli_main_builds_and_saves_lut(tmp_path, monkeypatch):
    import importlib.util

    for name in ("namedWindow", "setWindowProperty", "moveWindow", "imshow", "destroyWindow"):
        monkeypatch.setattr(ilut.cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(ilut.cv2, "waitKey", lambda *a, **k: 0)

    cli_path = _REPO_ROOT / "examples" / "python" / "build_inverse_lut.py"
    spec = importlib.util.spec_from_file_location("build_inverse_lut_cli", cli_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config_path = tmp_path / "config.toml"
    out_lut = tmp_path / "out_lut.npz"
    config_path.write_text(
        f"""
[projector]
gamma = 2.2
width = 8
height = 6
pos_x = 0
pos_y = 0

[camera]
backend = "opencv"
device_index = 0
wait_key_ms = 1

[inverse_lut]
num_levels = 5
source = "both"
lut_resolution = 64
reference_region = 0.5
inverse_lut_path = "{out_lut.as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    gammas = (2.2, 2.2, 2.2)
    black = (0.02, 0.02, 0.02)

    def fake_run_capture(samples, **kwargs):
        def capture_fn(sample):
            return _synth_frame(sample.input_rgb, gammas, black)
        return ilut.measure_calibration_samples(samples, capture_fn, reference_region=0.5)

    monkeypatch.setattr(module.ilut, "run_calibration_capture", fake_run_capture)

    module.main(["build_inverse_lut.py", "--config", str(config_path)])

    assert out_lut.exists()
    loaded = ilut.load_inverse_lut(out_lut)
    assert loaded.lut.shape == (3, 64)
    reload_config(_REPO_ROOT / "config.toml")  # 既定へ戻す
