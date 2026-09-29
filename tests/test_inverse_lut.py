from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pytest

from yoshidacomp.config import (
    AppConfig,
    CameraConfig,
    InverseLutConfig,
    ProjectorConfig,
    reload_config,
)
from yoshidacomp.compensation import inverse_lut as ilut


# ── Task 1: config ───────────────────────────────────────────────────


def test_inverse_lut_config_defaults():
    cfg = AppConfig()
    assert isinstance(cfg.inverse_lut, InverseLutConfig)
    assert cfg.inverse_lut.num_levels == 17
    assert cfg.inverse_lut.source == "channel_ramp"
    assert cfg.inverse_lut.lut_resolution == 256
    assert cfg.inverse_lut.roi_x == 0
    assert cfg.inverse_lut.roi_y == 0
    assert cfg.inverse_lut.roi_width == 0
    assert cfg.inverse_lut.roi_height == 0
    assert cfg.inverse_lut.trust_low == 10.0
    assert cfg.inverse_lut.trust_high == 245.0
    assert cfg.inverse_lut.hdr_enabled is False
    assert cfg.inverse_lut.hdr_max_stops == 3
    assert cfg.inverse_lut.inverse_lut_path == "data/inverse_lut.npz"


def test_inverse_lut_config_loads_from_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[inverse_lut]
num_levels = 9
source = "gray"
lut_resolution = 128
roi_x = 100
roi_y = 80
roi_width = 320
roi_height = 240
trust_low = 12
trust_high = 250
hdr_enabled = true
hdr_max_stops = 5
inverse_lut_path = "data/custom_lut.npz"
""".strip(),
        encoding="utf-8",
    )
    cfg = reload_config(config_path)
    assert cfg.inverse_lut.num_levels == 9
    assert cfg.inverse_lut.source == "gray"
    assert cfg.inverse_lut.lut_resolution == 128
    assert cfg.inverse_lut.roi_x == 100
    assert cfg.inverse_lut.roi_y == 80
    assert cfg.inverse_lut.roi_width == 320
    assert cfg.inverse_lut.roi_height == 240
    assert cfg.inverse_lut.trust_low == 12
    assert cfg.inverse_lut.trust_high == 250
    assert cfg.inverse_lut.hdr_enabled is True
    assert cfg.inverse_lut.hdr_max_stops == 5
    assert cfg.inverse_lut.inverse_lut_path == "data/custom_lut.npz"
    reload_config(_REPO_ROOT / "config.toml")  # 後続テストのため既定に戻す


# ── Task 2: pattern generation ───────────────────────────────────────


def test_generate_calibration_patterns_channel_ramp_only():
    # source="channel_ramp": 単独R/G/B 各17 = 51、グレーは含まない
    samples = ilut.generate_calibration_patterns(
        4, 3, num_levels=17, source="channel_ramp", dtype=np.uint8
    )
    assert len(samples) == 51
    assert all(s.kind == "channel" for s in samples)

    # 各単独ランプは投影チャンネルのみ記録
    r_ramp = [s for s in samples if s.recorded_channels == (0,)]
    assert len(r_ramp) == 17
    for s in r_ramp:
        assert s.input_rgb[1] == 0.0 and s.input_rgb[2] == 0.0

    # 画像形状と dtype、最大階調
    assert r_ramp[0].image.shape == (3, 4, 3)
    assert r_ramp[0].image.dtype == np.uint8
    top_r = max(r_ramp, key=lambda s: s.input_rgb[0])
    assert int(top_r.image[0, 0, 0]) == 255

    # 全黒 (0,0,0) は 3 ランプで共有され distinct は 1 + 3*(17-1) = 49
    distinct_inputs = {tuple(round(v, 6) for v in s.input_rgb) for s in samples}
    assert len(distinct_inputs) == 1 + 3 * (17 - 1)  # 49


def test_generate_calibration_patterns_gray_only():
    # source="gray": グレー17 のみ、r,g,b を記録
    samples = ilut.generate_calibration_patterns(
        4, 3, num_levels=17, source="gray", dtype=np.uint8
    )
    assert len(samples) == 17
    assert all(s.kind == "gray" for s in samples)
    assert all(s.recorded_channels == (0, 1, 2) for s in samples)
    top_gray = max(samples, key=lambda s: s.input_rgb[0])
    assert int(top_gray.image[0, 0, 0]) == 255


def test_generate_calibration_patterns_rejects_invalid_source():
    import pytest

    with pytest.raises(ValueError):
        ilut.generate_calibration_patterns(4, 3, num_levels=5, source="both")


# ── Task 3: measurement / reduction ──────────────────────────────────


def _synth_frame(input_rgb, gammas, black, white=0.9, h=8, w=8):
    # white < trust_high (245/255) so the top level is not treated as saturated
    frame = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        frame[:, :, c] = black[c] + (white - black[c]) * (input_rgb[c] ** gammas[c])
    return frame


def test_reduce_frame_roi_and_full():
    frame = np.zeros((10, 10, 3), dtype=np.float32)
    frame[4:6, 4:6, :] = 1.0  # 中央 2x2 のみ 1

    # ROI で中央 2x2 を指定 → 平均 1.0
    roi_mean = ilut.reduce_frame(frame, roi=(4, 4, 2, 2))
    assert roi_mean.shape == (3,)
    assert np.allclose(roi_mean, 1.0)

    # roi=None → 全フレーム平均 = 4/100
    assert np.allclose(ilut.reduce_frame(frame, roi=None), 4.0 / 100.0)

    # width/height が 0 なら全フレーム扱い
    assert np.allclose(ilut.reduce_frame(frame, roi=(4, 4, 0, 0)), 4.0 / 100.0)

    # フレーム外へはみ出す ROI は境界にクリップ（frame[4:10, 4:10] = 6x6, 内 4 画素が 1）
    assert np.allclose(ilut.reduce_frame(frame, roi=(4, 4, 100, 100)), 4.0 / 36.0)


def test_measure_channel_ramp_records_only_projected_channel_and_dedups():
    gammas = (2.2, 2.0, 1.8)
    black = (10 / 255, 8 / 255, 6 / 255)
    samples = ilut.generate_calibration_patterns(
        6, 6, num_levels=5, source="channel_ramp", dtype=np.uint8
    )

    calls = {"n": 0}

    def capture_fn(sample):
        calls["n"] += 1
        return _synth_frame(sample.input_rgb, gammas, black)

    data = ilut.measure_calibration_samples(samples, capture_fn)

    assert len(data.measurements) == len(samples)
    distinct = {tuple(round(v, 6) for v in s.input_rgb) for s in samples}
    assert calls["n"] == len(distinct)  # 入力で重複排除（全黒は 1 回）

    # 単独ランプは投影チャンネルのみ記録、他は NaN
    r_m = next(
        m for m in data.measurements
        if m.recorded_channels == (0,) and m.input_rgb[0] > 0
    )
    assert not np.isnan(r_m.measured_rgb[0])
    assert np.isnan(r_m.measured_rgb[1]) and np.isnan(r_m.measured_rgb[2])

    v = r_m.input_rgb[0]
    assert abs(r_m.measured_rgb[0] - (black[0] + (0.9 - black[0]) * v ** gammas[0])) < 1e-5


def test_measure_gray_records_all_channels():
    gammas = (2.2, 2.0, 1.8)
    black = (10 / 255, 8 / 255, 6 / 255)
    samples = ilut.generate_calibration_patterns(
        6, 6, num_levels=5, source="gray", dtype=np.uint8
    )

    def capture_fn(sample):
        return _synth_frame(sample.input_rgb, gammas, black)

    data = ilut.measure_calibration_samples(samples, capture_fn)
    gray_m = next(m for m in data.measurements if m.input_rgb[0] > 0)
    assert gray_m.kind == "gray"
    assert all(not np.isnan(x) for x in gray_m.measured_rgb)


# ── Task 4: capture orchestration ────────────────────────────────────


def test_run_calibration_capture_projects_and_measures(monkeypatch):
    for name in (
        "namedWindow",
        "setWindowProperty",
        "moveWindow",
        "resizeWindow",
        "imshow",
        "destroyWindow",
    ):
        monkeypatch.setattr(ilut.cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(ilut.cv2, "waitKey", lambda *a, **k: 0)
    # Skip the real projection-settle sleep so the test stays fast.
    monkeypatch.setattr(ilut.time, "sleep", lambda *a, **k: None)

    samples = ilut.generate_calibration_patterns(
        6, 6, num_levels=4, source="channel_ramp", dtype=np.uint8
    )
    distinct = {tuple(round(v, 6) for v in s.input_rgb) for s in samples}

    frames = [np.full((6, 6, 3), 0.5, dtype=np.float32) for _ in range(len(distinct))]

    class FakeCamera:
        def __init__(self, frames):
            self._it = iter(frames)
            self.calls = 0

        def capture_linear_rgb(self, tv=None):
            self.calls += 1
            return next(self._it)

    camera = FakeCamera(frames)
    data = ilut.run_calibration_capture(samples, camera_backend=camera, show_window=True)

    assert len(data.measurements) == len(samples)
    assert camera.calls == len(distinct)  # 入力で重複排除
    # 単独ランプ先頭 (0,0,0) は投影チャンネル(0)のみ記録
    assert data.measurements[0].measured_rgb[0] == 0.5
    assert np.isnan(data.measurements[0].measured_rgb[1])


def test_run_calibration_capture_batches_triggered_hdr_by_need(monkeypatch):
    samples = ilut.generate_calibration_patterns(
        2, 2, num_levels=3, source="gray", dtype=np.uint8
    )

    class FakeBatch:
        def __init__(self, camera, tv):
            self.camera = camera
            self.tv = tv
            self.shots = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.camera.batches.append((self.tv, self.shots))

        def trigger(self):
            self.shots += 1

        def iter_linear_rgb(self):
            values = {
                "1/30": [0.02, 0.5, 1.0],
                "1/60": [0.6],
                "1/15": [0.04],
            }[self.tv]
            assert len(values) == self.shots
            for value in values:
                yield np.full((2, 2, 3), value, dtype=np.float32)

    class FakeTriggeredCamera:
        def __init__(self):
            self.batches = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def supported_shutter_speeds(self):
            return ["1/60", "1/30", "1/15"]

        def triggered_batch(self, tv):
            return FakeBatch(self, tv)

        def capture_linear_rgb(self, tv=None):
            raise AssertionError("immediate capture path must not be used")

    cfg = AppConfig(
        projector=ProjectorConfig(width=2, height=2, refresh_rate=0),
        camera=CameraConfig(
            backend="canon_edsdk",
            tv="1/30",
            wait_key_ms=300,
            trigger_wait_key_ms=1,
        ),
        inverse_lut=InverseLutConfig(
            num_levels=3,
            source="gray",
            hdr_enabled=True,
            hdr_max_stops=3,
        ),
    )
    camera = FakeTriggeredCamera()
    waits = []
    monkeypatch.setattr(ilut.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(
        ilut.cv2,
        "waitKey",
        lambda wait_ms: waits.append(wait_ms) or 0,
    )

    data = ilut.run_calibration_capture(
        samples,
        config=cfg,
        camera_backend=camera,
        show_window=True,
    )

    # Base: all patterns. Short/long: only the saturated/crushed pattern.
    assert camera.batches == [("1/30", 3), ("1/60", 1), ("1/15", 1)]
    assert waits == [50, 300, 1, 1, 300, 300]
    assert data.meta["capture_mode"] == "edsdk_triggered_batch"
    means = [measurement.measured_rgb[0] for measurement in data.measurements]
    assert means[0] == pytest.approx(0.02, abs=1e-6)
    assert means[1] == pytest.approx(0.5, abs=1e-6)
    assert means[2] == pytest.approx(1.2, abs=1e-6)


def test_capture_hdr_linear_single_exposure_when_disabled():
    called = {"n": 0}

    def capture_at(scale):
        called["n"] += 1
        return np.full((4, 4, 3), 0.3, dtype=np.float32)

    frame = ilut.capture_hdr_linear(capture_at, enabled=False)
    assert called["n"] == 1
    assert np.allclose(frame, 0.3)


def test_capture_hdr_linear_recovers_saturated_and_crushed():
    # true[0] は基準露光で飽和、true[2] は黒潰れ
    true = np.array([[[1.2, 0.5, 0.02]]], dtype=np.float32)

    def capture_at(scale):
        return np.clip(true * scale, 0.0, 1.0).astype(np.float32)

    out = ilut.capture_hdr_linear(
        capture_at, enabled=True, trust_low=10, trust_high=245, max_stops=3
    )
    assert abs(out[0, 0, 0] - 1.2) < 0.05    # 露光短縮で飽和を回復
    assert abs(out[0, 0, 1] - 0.5) < 0.02    # 中間はそのまま
    assert abs(out[0, 0, 2] - 0.02) < 0.005  # 露光延長で黒潰れを回復


def test_capture_hdr_linear_uses_actual_exposure_ratio():
    # capture_at が (frame, 実効露光比) を返す場合、公称 scale ではなく
    # 実効比で回復すること。実機は離散 Tv しか取れず、公称 2.0 倍を要求しても
    # 例えば 1.875 倍しか延ばせない、という状況を模す。
    true_val = 0.03  # trust_low(=10/255≈0.039) 以下＝黒潰れ
    actual_ratio = 1.875

    def capture_at(scale):
        if scale == 1.0:
            return np.full((1, 1, 3), true_val, dtype=np.float32)
        frame = np.full((1, 1, 3), true_val * actual_ratio, dtype=np.float32)
        return frame, actual_ratio  # (frame, 実効露光比)

    out = ilut.capture_hdr_linear(
        capture_at, enabled=True, trust_low=10, trust_high=245, max_stops=1
    )
    # 実効比 1.875 で割るので true_val に戻る（公称 2.0 で割ると誤差 6%）。
    assert abs(out[0, 0, 2] - true_val) < 1e-6


def test_capture_hdr_linear_channels_restricts_bracketing():
    # 単独チャンネル(青)相当: R,G は常時 ~0(黒潰れ), B は中間(問題なし)。
    # B=0.3 は ×2 でも 0.6 < trust_high なので、飽和予測ガードには
    # かからない（このテストは channels ゲートの効果だけを見る）。
    def _blue_only(scale):
        return np.array([[[0.0, 0.0, 0.3]]], dtype=np.float32)

    # channels=(2,) なら 暗い R/G は無視され、B が正常なのでブラケットしない。
    calls = {"n": 0}

    def capture_at(scale):
        calls["n"] += 1
        return _blue_only(scale)

    out = ilut.capture_hdr_linear(
        capture_at, enabled=True, trust_low=10, trust_high=245,
        max_stops=3, channels=(2,),
    )
    assert calls["n"] == 1  # 基準撮影のみ、無駄な長露光なし
    assert np.isclose(out[0, 0, 2], 0.3)

    # channels 未指定(全ch)だと、常時暗の R/G が黒潰れループを誘発する。
    calls_all = {"n": 0}

    def capture_at_all(scale):
        calls_all["n"] += 1
        return _blue_only(scale)

    ilut.capture_hdr_linear(
        capture_at_all, enabled=True, trust_low=10, trust_high=245, max_stops=3
    )
    assert calls_all["n"] > 1  # 基準 + R/G のための長露光ブラケット


def test_capture_hdr_linear_per_channel_best_exposure():
    # 露光に比例(1.0でクリップ)する合成カメラ。真の基準露光値:
    #   R=1.4(基準で飽和), G=0.5(中間), B=0.01(暗い)。
    true = np.array([[[1.4, 0.5, 0.01]]], dtype=np.float32)

    def capture_at(scale):
        return np.clip(true * scale, 0.0, 1.0).astype(np.float32)

    out = ilut.capture_hdr_linear(
        capture_at, enabled=True, trust_low=10, trust_high=245, max_stops=3
    )
    # 各chが自分にとって最良の露光から独立に復元される。
    assert abs(out[0, 0, 0] - 1.4) < 0.05    # R: 短露光から(1.0超も可)
    assert abs(out[0, 0, 1] - 0.5) < 0.02    # G: 中間、基準
    assert abs(out[0, 0, 2] - 0.01) < 0.004  # B: 最長の非飽和露光から(高SNR)


def test_capture_hdr_linear_saturated_long_exposure_must_not_corrupt_bright():
    """飽和した長露光の現像値が trust_high をすり抜けても明chを壊さない。

    2026-08-12 の色グリッド実測で顕在化した事故の再現: 単色 R255
    （R=0.4, G=B≈0.0004）で暗い G/B が黒潰れ判定→長露光ブラケット
    ×2/×4/×8 が走る。×4 以降 R はセンサ飽和するが、現像値は 1.0 に
    張り付かず 0.7 程度に落ちて飽和判定 (< 0.96) をすり抜ける。
    旧マージ（各フレーム自身の値で書き込み先を判定）はこの 0.7 を採用して
    R = 0.7/8 ≈ 0.087 に崩壊させた。基準露光アンカーの新マージでは
    R は基準値 0.4 のまま、G/B は長露光から回復される。

    落ちる誤実装: 書き込み先をフレーム自身の値 (< trust_high) だけで
    決める（旧実装に戻すと R が 0.087 になり落ちる）。
    """
    base_r, dark = 0.4, 0.004
    saturated_developed = 0.7  # センサ飽和後の現像値（< trust_high=0.96）

    def capture_at(scale):
        if scale == 1.0:
            values = [base_r, dark, dark]
        else:
            grown = base_r * scale
            values = [
                grown if grown < 0.96 else saturated_developed,
                dark * scale,
                dark * scale,
            ]
        return np.array([[values]], dtype=np.float32)

    out = ilut.capture_hdr_linear(
        capture_at, enabled=True, trust_low=10, trust_high=245, max_stops=3
    )
    assert abs(out[0, 0, 0] - base_r) < 1e-6   # 基準値のまま（旧実装は 0.0875）
    assert abs(out[0, 0, 1] - dark) < 1e-6     # 暗chは長露光から回復
    assert abs(out[0, 0, 2] - dark) < 1e-6


def test_capture_hdr_linear_prefers_longest_unsaturated():
    # 全チャンネルが暗い場合、low を最初に超えた露光ではなく、飽和しない
    # 範囲で最長の露光(最高SNR)を選ぶこと。読み出しノイズ相当の加法offset
    # を混ぜ、露光が長いほど回復値が真値に近づく合成カメラで、最長露光が
    # 選ばれたことを回復精度から判定する。
    true = np.array([[[0.01, 0.01, 0.01]]], dtype=np.float32)
    offset = 0.004  # 露光に依らない加法ノイズ; 回復後は offset/ratio

    def capture_at(scale):
        return np.clip(true * scale + offset, 0.0, 1.0).astype(np.float32)

    out = ilut.capture_hdr_linear(
        capture_at, enabled=True, trust_low=10, trust_high=245, max_stops=3,
    )
    # 掃引は ×4 (値 0.044 > low) で止まる。×4 から回復していれば誤差は
    # offset/4 = 0.001、×2 を選ぶ誤実装なら 0.002 になり落ちる。
    assert abs(out[0, 0, 2] - 0.01) < 0.0015


def test_capture_hdr_linear_stops_lengthening_before_predicted_clip():
    """明chが飽和すると予測されるスケールでは長露光を撮らない。

    飽和チャンネルは現像色行列で他chに混入するため、フレームごと使えない
    （2026-08-12 実測: R255 の ×8 露光で G 0.0005→0.03）。基準 R=0.4 なら
    ×2 (0.8 < 0.96) までで打ち切り、×4/×8 は撮影しないこと。

    落ちる誤実装: ガード無しで max_stops まで撮る（撮影回数 5 になる）。
    """
    base = np.array([[[0.4, 0.004, 0.004]]], dtype=np.float32)
    scales = []

    def capture_at(scale):
        scales.append(scale)
        grown = np.clip(base * scale, 0.0, 1.0)
        return grown.astype(np.float32)

    out = ilut.capture_hdr_linear(
        capture_at, enabled=True, trust_low=10, trust_high=245, max_stops=3
    )
    assert scales == [1.0, 2.0]  # 基準 + ×2 のみ（×4 は 1.6 ≥ 0.96 予測で中止）
    assert abs(out[0, 0, 0] - 0.4) < 1e-6
    assert abs(out[0, 0, 1] - 0.004) < 1e-6  # ×2 のクリーンなフレームから回復


def test_tv_to_seconds_parses_canon_formats():
    assert abs(ilut.tv_to_seconds("1/125") - 0.008) < 1e-6
    assert abs(ilut.tv_to_seconds("1/15") - 1 / 15) < 1e-9
    assert ilut.tv_to_seconds('0"5') == 0.5   # Canon 秒表記
    assert ilut.tv_to_seconds('1"5') == 1.5
    assert ilut.tv_to_seconds('4"') == 4.0
    assert ilut.tv_to_seconds('30"') == 30.0
    assert ilut.tv_to_seconds("2") == 2.0
    # Canon 1/3-stop annotated menu forms parse to their base seconds.
    assert abs(ilut.tv_to_seconds("1/10 (1/3)") - 0.1) < 1e-9
    assert abs(ilut.tv_to_seconds("1/6 (1/3)") - 1 / 6) < 1e-9
    assert ilut.tv_to_seconds('0"3 (1/3)') == 0.3
    # Canon 1/3-stop annotated entries are EXCLUDED from flicker-free (their
    # displayed fraction is rounded; the real exposure isn't an exact multiple).
    assert not ilut.is_flicker_free("1/10 (1/3)", 60.0)
    assert not ilut.is_flicker_free("1/6 (1/3)", 60.0)


def test_scale_shutter_tv():
    assert ilut.scale_shutter_tv("1/15", 0.5) == "1/30"   # 露光半分
    assert ilut.scale_shutter_tv("1/15", 2.0) == "1/8"    # 露光倍
    assert ilut.scale_shutter_tv("1/15", 1.0) == "1/15"
    assert ilut.scale_shutter_tv("2", 0.5) == "1"


def test_scale_shutter_tv_snaps_to_realizable_value():
    # 以前は非対応値(1/120, 1/2)を返し EDSDK が弾いていた段を、
    # 最も近い実在 Canon Tv にスナップする（tv=1/15, 3段ブラケット相当）。
    assert ilut.scale_shutter_tv("1/15", 0.125) == "1/125"
    assert ilut.scale_shutter_tv("1/15", 8.0) == '0"5'
    # 返り値は必ず実在の標準 Tv（tv_to_seconds で解釈可能）
    assert ilut.scale_shutter_tv("1/15", 0.125) in ilut._STANDARD_TV_DISPLAYS


def test_scale_shutter_tv_snaps_to_supported_menu():
    # カメラの受理メニューを与えると、その中の最も近い値へスナップする。
    supported = ["1/30", "1/15", "1/8", '0"5']
    assert ilut.scale_shutter_tv("1/15", 0.125, supported=supported) == "1/30"
    assert ilut.scale_shutter_tv("1/15", 8.0, supported=supported) == '0"5'
    # 候補外の露光でも、必ず与えたメニュー内の値を返す。
    assert ilut.scale_shutter_tv("1/15", 4.0, supported=supported) in supported


def test_is_flicker_free_60hz():
    # 1/60秒の整数倍はフリッカーフリー
    for tv in ("1/60", "1/30", "1/20", "1/15", "1/10", "1/6", "1/5", "1/4", '0"5', "1"):
        assert ilut.is_flicker_free(tv, 60.0), tv
    # 1/60秒の整数倍でない値はNG
    for tv in ("1/100", "1/125", "1/80", "1/50", "1/40", "1/13", "1/8"):
        assert not ilut.is_flicker_free(tv, 60.0), tv
    # 1周期より速い露光もNG
    assert not ilut.is_flicker_free("1/8000", 60.0)
    # 制約無効(0/None)なら常にTrue
    assert ilut.is_flicker_free("1/100", 0)
    assert ilut.is_flicker_free("1/100", None)


def test_snap_to_flicker_free_tv():
    # 1/100(非整数倍) -> 最も近いフリッカーフリー(最速は1/60)
    assert ilut.snap_to_flicker_free_tv("1/100", refresh_hz=60.0) == "1/60"
    # 既にフリッカーフリーならそのまま
    assert ilut.snap_to_flicker_free_tv("1/15", refresh_hz=60.0) == "1/15"
    assert ilut.snap_to_flicker_free_tv("1/45", refresh_hz=60.0) == "1/60"
    # メニューを与えたら、その中のフリッカーフリー値から選ぶ
    supported = ["1/30", "1/15", "1/8", '0"5']  # 1/8 はフリッカーフリーでない
    assert ilut.snap_to_flicker_free_tv("1/8", supported=supported, refresh_hz=60.0) == "1/15"
    # 制約無効ならそのまま
    assert ilut.snap_to_flicker_free_tv("1/100", refresh_hz=0) == "1/100"


def test_scale_shutter_tv_flicker_free_60hz():
    # HDRブラケットも60Hzの整数倍のみに制限される
    assert ilut.scale_shutter_tv("1/15", 0.5, refresh_hz=60.0) == "1/30"
    assert ilut.scale_shutter_tv("1/15", 0.25, refresh_hz=60.0) == "1/60"
    # 2倍: 目標2/15=0.133 -> フリッカーフリーの最近傍は1/10 (1/8は非整数倍で除外)
    assert ilut.scale_shutter_tv("1/15", 2.0, refresh_hz=60.0) == "1/10"
    # 返り値は必ずフリッカーフリー
    assert ilut.is_flicker_free(
        ilut.scale_shutter_tv("1/15", 0.25, refresh_hz=60.0), 60.0
    )


# ── Task 5: build ────────────────────────────────────────────────────


def _build_from_synth(gammas, black, num_levels=17, white=0.9, **kwargs):
    # 設定に依存しないよう信頼境界(8bit画素値)・解像度・系統を明示（既定と同値）
    kwargs.setdefault("trust_low", 10)
    kwargs.setdefault("trust_high", 245)
    kwargs.setdefault("lut_resolution", 256)
    kwargs.setdefault("source", "channel_ramp")
    # 生成も構築も同じ単一系統に揃える（混在させない）
    samples = ilut.generate_calibration_patterns(
        6, 6, num_levels=num_levels, source=kwargs["source"], dtype=np.uint8
    )

    def capture_fn(sample):
        return _synth_frame(sample.input_rgb, gammas, black, white=white)

    data = ilut.measure_calibration_samples(samples, capture_fn)
    return ilut.build_inverse_lut(data, **kwargs)


def test_build_inverse_lut_inverts_gamma_per_channel():
    gammas = (2.2, 2.0, 1.8)
    black = (10 / 255, 8 / 255, 6 / 255)
    lut = _build_from_synth(gammas, black, source="channel_ramp", lut_resolution=256)

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
        data, source="gray", lut_resolution=64, trust_low=10, trust_high=242
    )
    # 飽和ノットを除外しても NaN なし・単調・[0,1]・黒引きが保たれる。
    # 白は飽和値(0.99)ではなく信頼できる最上位(p=0.8)で正規化されるため上限は ~0.8。
    assert not np.any(np.isnan(lut.lut))
    for c in range(3):
        assert np.all(np.diff(lut.lut[c]) >= -1e-6)
        assert lut.lut[c, 0] < 1e-3
        assert 0.6 < lut.lut[c, -1] < 0.95


def test_build_inverse_lut_source_gray_recovers_gamma():
    gammas = (2.2, 2.2, 2.2)
    black = (0.02, 0.02, 0.02)
    lut = _build_from_synth(gammas, black, source="gray")
    assert lut.lut.shape[0] == 3
    assert np.max(np.abs(lut.lut[0] - lut.t_grid ** (1 / 2.2))) < 0.05


def test_build_inverse_lut_uses_only_selected_source_when_data_mixed():
    # 混在データを与えても、選んだ系統の計測のみが構築に使われる。
    # gray は γ=2.2、channel は恒等(γ=1) を計測したことにして分離を確認。
    levels = np.linspace(0.0, 1.0, 9)
    measurements = []
    for v in levels:
        gray_val = 0.02 + 0.88 * (v ** 2.2)  # gray: γ=2.2
        measurements.append(
            ilut.CalibrationMeasurement(
                (v, v, v), (gray_val, gray_val, gray_val), (0, 1, 2), "gray"
            )
        )
        chan_val = 0.02 + 0.88 * v  # channel: 恒等 γ=1
        measurements.append(
            ilut.CalibrationMeasurement(
                (v, 0.0, 0.0), (chan_val, np.nan, np.nan), (0,), "channel"
            )
        )
    data = ilut.CalibrationData(measurements=measurements, meta={})

    lut_gray = ilut.build_inverse_lut(data, source="gray", lut_resolution=128)
    lut_chan = ilut.build_inverse_lut(data, source="channel_ramp", lut_resolution=128)

    # gray 系統は γ=2.2 の逆（t^(1/2.2)）を復元
    assert np.max(np.abs(lut_gray.lut[0] - lut_gray.t_grid ** (1 / 2.2))) < 0.05
    # channel 系統は恒等（t^1）を復元
    assert np.max(np.abs(lut_chan.lut[0] - lut_chan.t_grid)) < 0.05


def test_build_inverse_lut_rejects_invalid_source():
    import pytest

    data = ilut.CalibrationData(
        measurements=[
            ilut.CalibrationMeasurement((0.0, 0.0, 0.0), (0.02, 0.02, 0.02), (0, 1, 2), "gray"),
            ilut.CalibrationMeasurement((1.0, 1.0, 1.0), (0.9, 0.9, 0.9), (0, 1, 2), "gray"),
        ],
        meta={},
    )
    with pytest.raises(ValueError):
        ilut.build_inverse_lut(data, source="both")


# ── Task 6: apply / save-load ────────────────────────────────────────


def test_apply_inverse_lut_linearizes_response():
    gammas = (2.2, 2.0, 1.8)
    black = (0.02, 0.02, 0.02)
    lut = _build_from_synth(gammas, black, source="channel_ramp")

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

    for name in (
        "namedWindow",
        "setWindowProperty",
        "moveWindow",
        "resizeWindow",
        "imshow",
        "destroyWindow",
    ):
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
source = "channel_ramp"
lut_resolution = 64
roi_width = 0
roi_height = 0
inverse_lut_path = "{out_lut.as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    gammas = (2.2, 2.2, 2.2)
    black = (0.02, 0.02, 0.02)

    def fake_run_capture(samples, **kwargs):
        def capture_fn(sample):
            return _synth_frame(sample.input_rgb, gammas, black)
        return ilut.measure_calibration_samples(samples, capture_fn)

    monkeypatch.setattr(module.ilut, "run_calibration_capture", fake_run_capture)

    module.main(["build_inverse_lut.py", "--config", str(config_path)])

    assert out_lut.exists()
    loaded = ilut.load_inverse_lut(out_lut)
    assert loaded.lut.shape == (3, 64)
    reload_config(_REPO_ROOT / "config.toml")  # 既定へ戻す
