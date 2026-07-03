"""Measured inverse LUT for projector response linearization.

Build per-channel 1D inverse lookup tables that cancel the projector's
own input-to-output nonlinearity (gamma), using the camera as a linear
light meter. This is a measured replacement for analytic inverse-gamma
correction; it is channel-independent and global (not per-pixel).

プロジェクタ応答線形化のための測定ベース逆LUT。

カメラをリニアな測光器として用い、プロジェクタ固有の入力→出力
非線形性（ガンマ）を打ち消すチャンネル別 1D 逆LUTを構築する。
解析的な逆ガンマ補正の測定版であり、チャンネル独立かつグローバル
（ピクセルごとではない）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from src.python.camera import create_camera_backend
from src.python.config import get_config


# ── Data structures ──────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationSample:
    """One projection pattern and how its capture should be recorded.

    Attributes:
        input_rgb: Projector input color in [0, 1].
            プロジェクタ入力色（[0, 1]）。
        image: Solid-color pattern image, shape (H, W, 3), RGB.
            単色パターン画像。形状 (H, W, 3)、RGB。
        kind: Sampling family, "gray" or "channel".
            サンプリング系統。"gray" または "channel"。
        recorded_channels: Camera channels to record for this sample.
            この計測で記録するカメラチャンネル。

    1つの投影パターンと、その撮影で記録すべき内容。
    """

    input_rgb: Tuple[float, float, float]
    image: np.ndarray
    kind: str
    recorded_channels: Tuple[int, ...]


@dataclass(frozen=True)
class CalibrationMeasurement:
    """A recorded measurement for one calibration sample.

    Attributes:
        input_rgb: Projector input color in [0, 1].
            プロジェクタ入力色（[0, 1]）。
        measured_rgb: Linear camera value per channel in [0, 1].
            Unrecorded channels are NaN.
            チャンネルごとのリニアカメラ計測値（[0, 1]）。
            未記録チャンネルは NaN。
        recorded_channels: Channels that were recorded.
            記録されたチャンネル。
        kind: Sampling family, "gray" or "channel".
            サンプリング系統。"gray" または "channel"。

    1つのキャリブレーションサンプルの記録計測値。
    """

    input_rgb: Tuple[float, float, float]
    measured_rgb: Tuple[float, float, float]
    recorded_channels: Tuple[int, ...]
    kind: str


@dataclass
class CalibrationData:
    """Collected calibration measurements plus metadata.

    収集されたキャリブレーション計測値とメタデータ。
    """

    measurements: List[CalibrationMeasurement]
    meta: dict = field(default_factory=dict)


@dataclass
class InverseLut:
    """A per-channel inverse LUT over normalized output t in [0, 1].

    Attributes:
        lut: Array (3, R). ``lut[c, i]`` is the projector input that
            produces normalized output ``t_grid[i]`` for channel c.
            配列 (3, R)。``lut[c, i]`` はチャンネル c で正規化出力
            ``t_grid[i]`` を生むプロジェクタ入力。
        t_grid: Array (R,), uniform samples of normalized output in [0, 1].
            配列 (R,)。[0, 1] の正規化出力の一様サンプル。
        knots: Per-channel (p, n) knot arrays used to build the curve
            (or ``None`` for a degenerate/identity channel).
            曲線構築に使ったチャンネルごとの (p, n) ノット配列
            （縮退/恒等チャンネルは ``None``）。
        meta: Build metadata (source, trust bounds, ...).
            構築メタデータ（source, 信頼境界, ...）。

    正規化出力 t（[0, 1]）に対するチャンネル別逆LUT。
    """

    lut: np.ndarray
    t_grid: np.ndarray
    knots: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ── Pattern generation ───────────────────────────────────────────────


def _make_solid_image(
    rgb01: Sequence[float],
    x_size: int,
    y_size: int,
    dtype: np.typing.DTypeLike,
) -> np.ndarray:
    """Create a solid-color RGB image at the given normalized color.

    指定した正規化色の単色 RGB 画像を生成する。
    """
    image = np.zeros((y_size, x_size, 3), dtype=np.float32)
    image[:, :, 0] = float(rgb01[0])
    image[:, :, 1] = float(rgb01[1])
    image[:, :, 2] = float(rgb01[2])
    if dtype == np.uint8:
        return (image * 255.0).round().astype(np.uint8)
    if dtype == np.uint16:
        return (image * 65535.0).round().astype(np.uint16)
    if dtype == np.float32:
        return image
    raise ValueError(
        "Unsupported dtype. Supported types are np.uint8, np.uint16, and np.float32."
    )


def generate_calibration_patterns(
    x_size: int,
    y_size: int,
    num_levels: Optional[int] = None,
    include_gray: bool = True,
    include_channel_ramps: bool = True,
    dtype: np.typing.DTypeLike = np.uint8,
) -> List[CalibrationSample]:
    """Generate gray and per-channel calibration ramp patterns.

    Produce a gray ramp ``(v, v, v)`` (recording all three camera
    channels) and per-channel ramps ``(v, 0, 0)`` / ``(0, v, 0)`` /
    ``(0, 0, v)`` (each recording only the projecting channel), sampled
    at ``num_levels`` equally spaced levels in [0, 1].

    グレーランプと各チャンネル単独ランプのキャリブレーションパターンを生成する。

    グレーランプ ``(v, v, v)``（カメラ 3 チャンネルすべてを記録）と、
    各チャンネル単独ランプ ``(v, 0, 0)`` / ``(0, v, 0)`` / ``(0, 0, v)``
    （投影しているチャンネルのみ記録）を、[0, 1] 等間隔の
    ``num_levels`` レベルで生成する。

    Args:
        x_size: Pattern width in pixels. パターン幅（ピクセル）。
        y_size: Pattern height in pixels. パターン高さ（ピクセル）。
        num_levels: Number of levels per ramp. Default from config.
            各ランプのレベル数。既定は設定から。
        include_gray: Whether to include the gray ramp.
            グレーランプを含めるか。
        include_channel_ramps: Whether to include per-channel ramps.
            各チャンネル単独ランプを含めるか。
        dtype: Output image dtype (uint8/uint16/float32).
            出力画像の dtype（uint8/uint16/float32）。

    Returns:
        List of :class:`CalibrationSample`.
            :class:`CalibrationSample` のリスト。
    """
    if num_levels is None:
        num_levels = get_config().inverse_lut.num_levels
    num_levels = max(2, int(num_levels))
    levels = np.linspace(0.0, 1.0, num_levels)

    samples: List[CalibrationSample] = []
    if include_gray:
        for v in levels:
            rgb = (float(v), float(v), float(v))
            samples.append(
                CalibrationSample(
                    input_rgb=rgb,
                    image=_make_solid_image(rgb, x_size, y_size, dtype),
                    kind="gray",
                    recorded_channels=(0, 1, 2),
                )
            )
    if include_channel_ramps:
        for channel in range(3):
            for v in levels:
                rgb_list = [0.0, 0.0, 0.0]
                rgb_list[channel] = float(v)
                rgb = (rgb_list[0], rgb_list[1], rgb_list[2])
                samples.append(
                    CalibrationSample(
                        input_rgb=rgb,
                        image=_make_solid_image(rgb, x_size, y_size, dtype),
                        kind="channel",
                        recorded_channels=(channel,),
                    )
                )
    return samples


# ── Measurement / reduction ──────────────────────────────────────────


def reduce_frame(frame: np.ndarray, reference_region: float = 0.2) -> np.ndarray:
    """Average the central region of a captured frame into one RGB value.

    撮影フレームの中央領域を平均し、1組の RGB 値に集約する。

    Args:
        frame: Linear RGB frame of shape (H, W, 3).
            形状 (H, W, 3) のリニア RGB フレーム。
        reference_region: Central fraction to average (0.0-1.0).
            平均する中央領域の割合（0.0〜1.0）。

    Returns:
        Mean RGB value of shape (3,), dtype float32.
            形状 (3,) の平均 RGB 値（float32）。
    """
    frame = np.asarray(frame, dtype=np.float32)
    height, width = frame.shape[:2]
    fraction = float(np.clip(reference_region, 0.0, 1.0))
    region_h = max(1, int(round(height * fraction)))
    region_w = max(1, int(round(width * fraction)))
    y0 = (height - region_h) // 2
    x0 = (width - region_w) // 2
    patch = frame[y0 : y0 + region_h, x0 : x0 + region_w, :3]
    return patch.reshape(-1, 3).mean(axis=0).astype(np.float32)


def measure_calibration_samples(
    samples: Sequence[CalibrationSample],
    capture_fn: Callable[[CalibrationSample], np.ndarray],
    reference_region: float = 0.2,
) -> CalibrationData:
    """Capture each sample once (dedup by input) and record measurements.

    各サンプルを撮影し（入力で重複排除して 1 回のみ）、記録方針に従い計測する。

    Args:
        samples: Calibration samples to measure.
            計測するキャリブレーションサンプル。
        capture_fn: Callable returning a linear RGB frame (H, W, 3) in
            [0, 1] for a given sample.
            サンプルに対しリニア RGB フレーム (H, W, 3)[0, 1] を返す callable。
        reference_region: Central fraction averaged per capture.
            撮影ごとに平均する中央領域の割合。

    Returns:
        A :class:`CalibrationData` with one measurement per sample.
            サンプルごとに 1 計測を持つ :class:`CalibrationData`。
    """
    cache: dict[Tuple[float, float, float], np.ndarray] = {}
    measurements: List[CalibrationMeasurement] = []
    for sample in samples:
        key = tuple(round(float(v), 6) for v in sample.input_rgb)
        if key not in cache:
            frame = capture_fn(sample)
            cache[key] = reduce_frame(frame, reference_region)
        mean = cache[key]
        measured = np.full(3, np.nan, dtype=np.float32)
        for channel in sample.recorded_channels:
            measured[channel] = mean[channel]
        measurements.append(
            CalibrationMeasurement(
                input_rgb=(
                    float(sample.input_rgb[0]),
                    float(sample.input_rgb[1]),
                    float(sample.input_rgb[2]),
                ),
                measured_rgb=(float(measured[0]), float(measured[1]), float(measured[2])),
                recorded_channels=sample.recorded_channels,
                kind=sample.kind,
            )
        )
    return CalibrationData(measurements=measurements, meta={"reference_region": reference_region})


# ── Capture orchestration (HDR provisional) ──────────────────────────


def capture_hdr_linear(
    capture_once_fn: Callable[[], np.ndarray],
    *,
    enabled: bool = False,
    trust_low: float = 10.0 / 255.0,
    trust_high: float = 245.0 / 255.0,
) -> np.ndarray:
    """Capture a linear frame, optionally via simple HDR (provisional).

    リニアフレームを撮影する（任意で簡易 HDR、仮実装）。

    Currently returns a single exposure regardless of ``enabled``.

    TODO(HDR): When enabled, bracket exposures — if a pixel saturates
    (>= trust_high) halve the exposure time and double the pixel value;
    if it is crushed (<= trust_low) double the exposure time and halve
    the pixel value; mask untrustworthy pixels per exposure and merge.
    Requires switching the Canon ``tv`` per shot (temporary config
    override).

    TODO(HDR): 有効時は露光ブラケット — サチる (>= trust_high) なら
    露光時間を半分・画素値 2 倍、黒潰れ (<= trust_low) なら露光時間を
    倍・画素値 1/2。露光ごとに非信頼画素をマスクしてマージする。
    ショットごとに Canon ``tv`` を切り替える（config 一時上書き）必要がある。

    Args:
        capture_once_fn: Callable returning one linear RGB frame.
            1 枚のリニア RGB フレームを返す callable。
        enabled: Whether HDR bracketing is enabled (currently no-op).
            HDR ブラケットを有効にするか（現状は無効動作）。
        trust_low: Lower trustworthy bound (reserved for HDR).
            信頼下限（HDR 用に予約）。
        trust_high: Upper trustworthy bound (reserved for HDR).
            信頼上限（HDR 用に予約）。

    Returns:
        Linear RGB frame of shape (H, W, 3).
            形状 (H, W, 3) のリニア RGB フレーム。
    """
    # NOTE: Provisional single-exposure implementation. See TODO(HDR).
    return capture_once_fn()


def run_calibration_capture(
    samples: Sequence[CalibrationSample],
    *,
    config=None,
    camera_backend=None,
    show_window: bool = True,
    window_name: str = "inverse_lut_calibration",
) -> CalibrationData:
    """Project each pattern and capture its linear response.

    各パターンを投影し、そのリニア応答を撮影する。

    Projects the *raw* (non-inverse-gamma) pattern and captures with
    ``camera.capture_linear_rgb()``, so the projector's native response
    is measured.

    生パターン（逆ガンマ非適用）を投影し ``camera.capture_linear_rgb()``
    で撮影するため、プロジェクタ固有の応答を測定する。

    Args:
        samples: Calibration samples to project and capture.
            投影・撮影するキャリブレーションサンプル。
        config: App configuration. Defaults to :func:`get_config`.
            アプリ設定。既定は :func:`get_config`。
        camera_backend: Camera backend. Defaults to one created from config.
            カメラバックエンド。既定は設定から生成。
        show_window: Whether to open the projector window (disable in tests
            with mocked ``cv2``).
            投影ウィンドウを開くか（テストではモック cv2 で無効化）。
        window_name: OpenCV window name. OpenCV ウィンドウ名。

    Returns:
        A :class:`CalibrationData` with the recorded measurements.
            記録された計測値を持つ :class:`CalibrationData`。
    """
    cfg = config if config is not None else get_config()
    ilut_cfg = cfg.inverse_lut
    if camera_backend is None:
        camera_backend = create_camera_backend(cfg.camera)

    if show_window:
        proj = cfg.projector
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(window_name, proj.pos_x, proj.pos_y)

    def capture_fn(sample: CalibrationSample) -> np.ndarray:
        def once() -> np.ndarray:
            if show_window:
                bgr = cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, bgr)
                cv2.waitKey(cfg.camera.wait_key_ms)
            return camera_backend.capture_linear_rgb()

        return capture_hdr_linear(
            once,
            enabled=ilut_cfg.hdr_enabled,
            trust_low=ilut_cfg.trust_low,
            trust_high=ilut_cfg.trust_high,
        )

    try:
        data = measure_calibration_samples(
            samples, capture_fn, ilut_cfg.reference_region
        )
    finally:
        if show_window:
            cv2.destroyWindow(window_name)

    data.meta["source"] = ilut_cfg.source
    data.meta["hdr_enabled"] = ilut_cfg.hdr_enabled
    return data


# ── Inverse LUT construction ─────────────────────────────────────────


def _strictly_increasing(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop points where x does not strictly increase (keep first).

    x が狭義単調増加しない点を除去する（最初を残す）。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.concatenate(([True], np.diff(x) > 0.0))
    return x[keep], y[keep]


def _channel_observations(
    measurements: Sequence[CalibrationMeasurement],
    channel: int,
    source: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect (input[c], measured[c]) pairs for a channel and source.

    指定チャンネル・ソースの (入力[c], 計測[c]) 対を収集する。
    """
    inputs: List[float] = []
    values: List[float] = []
    for m in measurements:
        if channel not in m.recorded_channels:
            continue
        if source == "gray" and m.kind != "gray":
            continue
        if source == "channel_ramp" and m.kind != "channel":
            continue
        value = m.measured_rgb[channel]
        if value is None or np.isnan(value):
            continue
        inputs.append(float(m.input_rgb[channel]))
        values.append(float(value))
    return np.asarray(inputs, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _aggregate_by_level(
    inputs: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Group observations by unique input level and average values.

    観測を一意な入力レベルでグルーピングし、値を平均する。
    """
    unique_inputs = np.unique(inputs)
    mean_values = np.array(
        [values[inputs == u].mean() for u in unique_inputs], dtype=np.float64
    )
    return unique_inputs, mean_values


def _build_channel_lut(
    inputs: np.ndarray,
    values: np.ndarray,
    t_grid: np.ndarray,
    trust_low: float,
    trust_high: float,
) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Build one channel's inverse LUT from aggregated (p, m) knots.

    集約済み (p, m) ノットから 1 チャンネルの逆LUTを構築する。

    Returns the LUT samples over *t_grid* and the (p, n) fit knots, or an
    identity LUT with ``None`` knots for degenerate inputs.

    *t_grid* 上の LUT サンプルと (p, n) フィットノットを返す。縮退時は
    恒等 LUT と ``None`` ノットを返す。
    """
    if inputs.size < 2:
        return t_grid.copy(), None

    order = np.argsort(inputs)
    p = inputs[order]
    m = values[order]

    has_black = p[0] == 0.0
    m0 = m[0] if has_black else float(m.min())

    trusted = (m >= trust_low) & (m <= trust_high)
    if has_black:
        trusted[0] = True  # 黒アンカーは常に保持
    if int(trusted.sum()) < 2:
        return t_grid.copy(), None

    m_white = float(m[trusted].max())
    if m_white - m0 <= 1e-8:
        return t_grid.copy(), None

    n = np.clip((m - m0) / (m_white - m0), 0.0, 1.0)
    p_fit = p[trusted]
    n_fit = n[trusted]

    # 端点を強制: p=0 -> n=0, p=1 -> n=1
    if p_fit[0] != 0.0:
        p_fit = np.concatenate(([0.0], p_fit))
        n_fit = np.concatenate(([0.0], n_fit))
    else:
        n_fit[0] = 0.0
    if p_fit[-1] != 1.0:
        p_fit = np.concatenate((p_fit, [1.0]))
        n_fit = np.concatenate((n_fit, [1.0]))
    else:
        n_fit[-1] = 1.0

    # 非減少化 → 狭義単調に整形
    n_fit = np.maximum.accumulate(n_fit)
    p_fit, n_fit = _strictly_increasing(p_fit, n_fit)
    if p_fit.size < 2:
        return t_grid.copy(), None

    # 前向き応答を PCHIP で滑らかにし、密サンプルから反転
    forward = PchipInterpolator(p_fit, n_fit)
    dense_count = max(int(t_grid.size) * 4, 1024)
    p_dense = np.linspace(0.0, 1.0, dense_count)
    n_dense = np.clip(forward(p_dense), 0.0, 1.0)
    n_dense = np.maximum.accumulate(n_dense)

    xp, fp = _strictly_increasing(n_dense, p_dense)
    lut_c = np.clip(np.interp(t_grid, xp, fp), 0.0, 1.0)
    return lut_c, (p_fit, n_fit)


def build_inverse_lut(
    data: CalibrationData,
    *,
    source: Optional[str] = None,
    lut_resolution: Optional[int] = None,
    trust_low: Optional[float] = None,
    trust_high: Optional[float] = None,
) -> InverseLut:
    """Build a per-channel inverse LUT from calibration measurements.

    キャリブレーション計測値からチャンネル別逆LUTを構築する。

    For each channel: aggregate ``(input[c], measured[c])`` by input
    level, subtract black and normalize by white, enforce monotonicity,
    and invert onto a uniform ``t`` grid in [0, 1].

    各チャンネルについて ``(入力[c], 計測[c])`` を入力レベルで集約し、
    黒引き・白正規化・単調化を行い、[0, 1] の一様 ``t`` グリッドへ反転する。

    Args:
        data: Calibration measurements.
            キャリブレーション計測値。
        source: "both" / "channel_ramp" / "gray". Default from config.
            "both" / "channel_ramp" / "gray"。既定は設定から。
        lut_resolution: Number of LUT entries. Default from config.
            LUT エントリ数。既定は設定から。
        trust_low: Lower trustworthy bound. Default from config.
            信頼下限。既定は設定から。
        trust_high: Upper trustworthy bound. Default from config.
            信頼上限。既定は設定から。

    Returns:
        The built :class:`InverseLut`.
            構築された :class:`InverseLut`。
    """
    cfg = get_config().inverse_lut
    source = cfg.source if source is None else source
    lut_resolution = cfg.lut_resolution if lut_resolution is None else int(lut_resolution)
    trust_low = cfg.trust_low if trust_low is None else float(trust_low)
    trust_high = cfg.trust_high if trust_high is None else float(trust_high)

    t_grid = np.linspace(0.0, 1.0, lut_resolution)
    lut = np.zeros((3, lut_resolution), dtype=np.float32)
    knots: list = []
    for channel in range(3):
        inputs, values = _channel_observations(data.measurements, channel, source)
        if inputs.size == 0:
            lut[channel] = t_grid.astype(np.float32)
            knots.append(None)
            continue
        agg_inputs, agg_values = _aggregate_by_level(inputs, values)
        lut_c, knot = _build_channel_lut(
            agg_inputs, agg_values, t_grid, trust_low, trust_high
        )
        lut[channel] = lut_c.astype(np.float32)
        knots.append(knot)

    return InverseLut(
        lut=lut,
        t_grid=t_grid.astype(np.float32),
        knots=knots,
        meta={"source": source, "trust_low": trust_low, "trust_high": trust_high},
    )


# ── Apply / save / load ──────────────────────────────────────────────


def apply_inverse_lut(
    image: np.ndarray,
    lut: InverseLut,
    dtype: np.typing.DTypeLike = np.float32,
) -> np.ndarray:
    """Apply a per-channel inverse LUT to an RGB image.

    RGB 画像にチャンネル別逆LUTを適用する。

    Maps normalized target values in [0, 1] through ``L_c`` per channel
    to projector input values. Drop-in analog of
    ``apply_inverse_gamma_correction``.

    正規化ターゲット値（[0, 1]）をチャンネルごとに ``L_c`` で写像し、
    プロジェクタ入力値へ変換する。``apply_inverse_gamma_correction`` の
    差し替え相当。

    Args:
        image: RGB image (H, W, 3). uint8/uint16 are normalized to [0, 1].
            RGB 画像 (H, W, 3)。uint8/uint16 は [0, 1] に正規化。
        lut: The inverse LUT to apply.
            適用する逆LUT。
        dtype: Output dtype. float32 stays in [0, 1]; uint8/uint16 scale.
            出力 dtype。float32 は [0, 1]、uint8/uint16 はスケール。

    Returns:
        Projector-input image (H, W, 3) in the requested dtype.
            要求 dtype のプロジェクタ入力画像 (H, W, 3)。
    """
    array = np.asarray(image)
    if array.dtype == np.uint8:
        norm = array.astype(np.float32) / 255.0
    elif array.dtype == np.uint16:
        norm = array.astype(np.float32) / 65535.0
    else:
        norm = array.astype(np.float32)

    out = np.empty(norm.shape, dtype=np.float32)
    for channel in range(3):
        out[..., channel] = np.interp(
            norm[..., channel], lut.t_grid, lut.lut[channel]
        )
    out = np.clip(out, 0.0, 1.0)

    if dtype == np.uint8:
        return (out * 255.0).round().astype(np.uint8)
    if dtype == np.uint16:
        return (out * 65535.0).round().astype(np.uint16)
    return out.astype(dtype)


def save_inverse_lut(path, lut: InverseLut) -> Path:
    """Save an inverse LUT to an .npz file.

    逆LUTを .npz ファイルへ保存する。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        lut=np.asarray(lut.lut, dtype=np.float32),
        t_grid=np.asarray(lut.t_grid, dtype=np.float32),
        meta=json.dumps(lut.meta),
    )
    return path


def load_inverse_lut(path) -> InverseLut:
    """Load an inverse LUT from an .npz file.

    逆LUTを .npz ファイルから読み込む。
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        lut = np.asarray(data["lut"], dtype=np.float32)
        t_grid = np.asarray(data["t_grid"], dtype=np.float32)
        meta = json.loads(str(data["meta"])) if "meta" in data else {}
    return InverseLut(lut=lut, t_grid=t_grid, knots=[], meta=meta)
