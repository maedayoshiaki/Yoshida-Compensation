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
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

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


VALID_SOURCES: Tuple[str, ...] = ("channel_ramp", "gray")
"""Allowed sampling methods. Exactly one is used per run; never mixed.

許容されるサンプリング系統。1 回の実行で一方のみ使用し、混在させない。
"""


def normalize_source(source: str) -> str:
    """Validate and normalize a sampling-method name.

    サンプリング系統名を検証・正規化する。

    Args:
        source: Sampling method, "channel_ramp" or "gray".
            サンプリング系統。"channel_ramp" または "gray"。

    Returns:
        The lowercased, stripped source name.
            小文字化・トリム済みの系統名。

    Raises:
        ValueError: If ``source`` is not one of :data:`VALID_SOURCES`
            (e.g. the removed "both").
            ``source`` が :data:`VALID_SOURCES` にない場合（廃止された
            "both" など）。
    """
    normalized = str(source).strip().lower()
    if normalized not in VALID_SOURCES:
        raise ValueError(
            f"Invalid source {source!r}. Expected one of {VALID_SOURCES} "
            "(the two sampling methods are mutually exclusive; 'both' is no "
            "longer supported)."
        )
    return normalized


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
    source: Optional[str] = None,
    dtype: np.typing.DTypeLike = np.uint8,
) -> List[CalibrationSample]:
    """Generate calibration ramp patterns for a single sampling method.

    Depending on ``source``, produce **either** a gray ramp ``(v, v, v)``
    (recording all three camera channels) **or** per-channel ramps
    ``(v, 0, 0)`` / ``(0, v, 0)`` / ``(0, 0, v)`` (each recording only the
    projecting channel), sampled at ``num_levels`` equally spaced levels in
    [0, 1]. Only one method is generated so the build sees a single,
    self-consistent set of observations per channel (the two methods are
    never mixed, which would make the inverse-curve fit inconsistent).

    単一のサンプリング系統のキャリブレーションパターンを生成する。

    ``source`` に応じて、グレーランプ ``(v, v, v)``（カメラ 3 チャンネル
    すべてを記録）**または** 各チャンネル単独ランプ ``(v, 0, 0)`` /
    ``(0, v, 0)`` / ``(0, 0, v)``（投影しているチャンネルのみ記録）の
    **どちらか一方**を、[0, 1] 等間隔の ``num_levels`` レベルで生成する。
    一方の系統のみを生成するため、構築時にチャンネルごとの観測が一貫する
    （両系統を混ぜると逆曲線フィットが不整合になるため混在させない）。

    Args:
        x_size: Pattern width in pixels. パターン幅（ピクセル）。
        y_size: Pattern height in pixels. パターン高さ（ピクセル）。
        num_levels: Number of levels per ramp. Default from config.
            各ランプのレベル数。既定は設定から。
        source: Sampling method, "channel_ramp" or "gray". Default from config.
            サンプリング系統。"channel_ramp" または "gray"。既定は設定から。
        dtype: Output image dtype (uint8/uint16/float32).
            出力画像の dtype（uint8/uint16/float32）。

    Returns:
        List of :class:`CalibrationSample` for the selected method.
            選択した系統の :class:`CalibrationSample` のリスト。

    Raises:
        ValueError: If ``source`` is not one of :data:`VALID_SOURCES`.
            ``source`` が :data:`VALID_SOURCES` にない場合。
    """
    if num_levels is None or source is None:
        cfg = get_config().inverse_lut
        if num_levels is None:
            num_levels = cfg.num_levels
        if source is None:
            source = cfg.source
    source = normalize_source(source)
    num_levels = max(2, int(num_levels))
    levels = np.linspace(0.0, 1.0, num_levels)

    samples: List[CalibrationSample] = []
    if source == "gray":
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
    else:  # "channel_ramp"
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


def reduce_frame(
    frame: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Average a rectangular ROI of a captured frame into one RGB value.

    撮影フレームの矩形 ROI を平均し、1組の RGB 値に集約する。

    Args:
        frame: Linear RGB frame of shape (H, W, 3).
            形状 (H, W, 3) のリニア RGB フレーム。
        roi: Region of interest as ``(x, y, width, height)`` in pixels.
            ``None`` (or a non-positive ``width``/``height``) averages the
            whole frame. The rectangle is clipped to the frame bounds; a
            fully out-of-bounds ROI falls back to the whole frame.
            関心領域 ``(x, y, width, height)``（ピクセル）。``None``（または
            ``width``/``height`` が非正）なら全フレームを平均。矩形はフレーム
            境界にクリップし、完全に範囲外なら全フレームにフォールバックする。

    Returns:
        Mean RGB value of shape (3,), dtype float32.
            形状 (3,) の平均 RGB 値（float32）。
    """
    frame = np.asarray(frame, dtype=np.float32)
    height, width = frame.shape[:2]

    patch = frame[:, :, :3]
    if roi is not None:
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        if w > 0 and h > 0:
            x0 = max(0, min(x, width))
            y0 = max(0, min(y, height))
            x1 = max(x0, min(x + w, width))
            y1 = max(y0, min(y + h, height))
            candidate = frame[y0:y1, x0:x1, :3]
            if candidate.size > 0:
                patch = candidate
    return patch.reshape(-1, 3).mean(axis=0).astype(np.float32)


def measure_calibration_samples(
    samples: Sequence[CalibrationSample],
    capture_fn: Callable[[CalibrationSample], np.ndarray],
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> CalibrationData:
    """Capture each sample once (dedup by input) and record measurements.

    各サンプルを撮影し（入力で重複排除して 1 回のみ）、記録方針に従い計測する。

    Args:
        samples: Calibration samples to measure.
            計測するキャリブレーションサンプル。
        capture_fn: Callable returning a linear RGB frame (H, W, 3) in
            [0, 1] for a given sample.
            サンプルに対しリニア RGB フレーム (H, W, 3)[0, 1] を返す callable。
        roi: Measurement ROI ``(x, y, width, height)`` in pixels averaged per
            capture. ``None`` averages the whole frame. See :func:`reduce_frame`.
            撮影ごとに平均する計測 ROI ``(x, y, width, height)``（ピクセル）。
            ``None`` は全フレーム平均。:func:`reduce_frame` を参照。

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
            cache[key] = reduce_frame(frame, roi)
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
    return CalibrationData(
        measurements=measurements,
        meta={"roi": tuple(int(v) for v in roi) if roi is not None else None},
    )


# ── Capture orchestration (HDR provisional) ──────────────────────────


# Standard Canon Tv (shutter-speed) values as display strings accepted by
# EDSDK's parser, spanning 1/16000s .. 30s. Used to snap a scaled exposure
# onto a realizable value when the camera's own menu is not supplied.
# EDSDK が受理する標準 Canon Tv（シャッター）表示値（1/16000〜30秒）。
# カメラ実機のメニューが無いとき、スケール後露光を実在値へスナップするのに使う。
_STANDARD_TV_DISPLAYS: Tuple[str, ...] = (
    "1/16000", "1/12800", "1/10000", "1/8000", "1/6400", "1/6000",
    "1/5000", "1/4000", "1/3200", "1/3000", "1/2500", "1/2000",
    "1/1600", "1/1500", "1/1250", "1/1000", "1/800", "1/750",
    "1/640", "1/500", "1/400", "1/350", "1/320", "1/250",
    "1/200", "1/180", "1/160", "1/125", "1/100", "1/90",
    "1/80", "1/60", "1/50", "1/45", "1/40", "1/30",
    "1/25", "1/20", "1/15", "1/13", "1/10", "1/8",
    "1/6", "1/5", "1/4", '0"3', '0"4', '0"5', '0"6', '0"7', '0"8',
    "1", '1"3', '1"5', '1"6', "2", '2"5', "3", '3"2',
    '4"', '5"', '6"', '8"', '10"', '13"', '15"', '20"', '25"', '30"',
)


def tv_to_seconds(tv: str) -> float:
    """Convert a Canon shutter (``tv``) display string to seconds.

    Canon シャッター（``tv``）表示文字列を秒に変換する。

    Accepts fraction form ``"1/125"``, Canon seconds notation ``0"5`` /
    ``1"5`` / ``4"`` / ``30"``, and plain integer/decimal seconds ``"2"``.

    ``"1/125"`` の分数、``0"5`` / ``1"5`` / ``4"`` / ``30"`` の Canon 秒表記、
    ``"2"`` の整数/小数秒を受け付ける。

    Args:
        tv: Shutter-speed display string. シャッタースピード表示文字列。

    Returns:
        Exposure time in seconds. 露光時間（秒）。
    """
    text = str(tv).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    if '"' in text:
        whole, _, frac = text.partition('"')
        whole = whole or "0"
        if frac:  # e.g. 0"5 -> 0.5, 3"2 -> 3.2
            return float(f"{int(whole)}.{frac}")
        return float(whole)  # e.g. 4" -> 4, 30" -> 30
    return float(text)


def scale_shutter_tv(
    tv: str,
    scale: float,
    *,
    supported: Optional[Sequence[str]] = None,
) -> str:
    """Scale a Canon shutter (``tv``) and snap to the nearest realizable value.

    Canon シャッター（``tv``）を露光係数でスケールし、実現可能な最も近い値へスナップする。

    ``scale = 0.5`` halves the exposure time (faster shutter); ``scale =
    2.0`` doubles it. The scaled target is snapped to the nearest value in
    ``supported`` -- the camera's actual Tv menu, e.g.
    ``list_supported()["Tv"]`` -- or, when not given, to the nearest
    standard Canon Tv. This guarantees a value the camera can accept and
    avoids an EDSDK ``ValueError`` that would abort calibration mid-run.

    ``scale = 0.5`` は露光時間を半分（速いシャッター）、``scale = 2.0`` は倍に
    する。スケール後の目標秒を ``supported``（カメラ実機の Tv メニュー、例
    ``list_supported()["Tv"]``）— 未指定なら標準 Canon Tv — の中で最も近い値へ
    スナップする。これによりカメラが受理できる値を保証し、キャリブレーション
    途中で ``ValueError`` により中断するのを防ぐ。

    Args:
        tv: Base shutter-speed string. 基準シャッタースピード文字列。
        scale: Exposure-time multiplier. 露光時間の倍率。
        supported: Candidate Tv display strings to snap onto (the camera's
            menu). Defaults to standard Canon Tv values.
            スナップ先候補の Tv 表示文字列（カメラのメニュー）。既定は標準
            Canon Tv 値。

    Returns:
        The nearest realizable shutter-speed display string.
            最も近い実現可能なシャッタースピード表示文字列。
    """
    target_seconds = tv_to_seconds(tv) * float(scale)
    candidates = list(supported) if supported else list(_STANDARD_TV_DISPLAYS)

    best_display: Optional[str] = None
    best_error = float("inf")
    for display in candidates:
        try:
            seconds = tv_to_seconds(display)
        except (ValueError, TypeError):
            continue  # skip non-numeric entries such as "bulb"
        error = abs(seconds - target_seconds)
        if error < best_error:
            best_error = error
            best_display = display

    if best_display is None:
        return str(tv).strip()
    return best_display


def _frame_and_ratio(ret, nominal_scale: float) -> Tuple[np.ndarray, float]:
    """Unpack a capture return into ``(frame, exposure_ratio)``.

    撮影戻り値を ``(フレーム, 露光比)`` に分解する。

    A ``capture_at_fn`` may return either a bare frame or a
    ``(frame, actual_ratio)`` tuple. For a bare frame the nominal ``scale``
    is used as the ratio (backward-compatible); for a tuple the *actual*
    achieved exposure ratio is used, which matters because a camera can
    only realize discrete shutter speeds.

    ``capture_at_fn`` は素のフレーム、または ``(frame, actual_ratio)``
    タプルを返し得る。素のフレームなら公称 ``scale`` を露光比として使い
    （後方互換）、タプルなら実効露光比を使う。カメラは離散的な
    シャッター速度しか実現できないため、この区別が効いてくる。
    """
    if isinstance(ret, tuple):
        frame, ratio = ret
        return np.asarray(frame, dtype=np.float32), float(ratio)
    return np.asarray(ret, dtype=np.float32), float(nominal_scale)


def capture_hdr_linear(
    capture_at_fn: Callable[[float], "np.ndarray | Tuple[np.ndarray, float]"],
    *,
    enabled: bool = False,
    trust_low: float = 10.0,
    trust_high: float = 245.0,
    max_stops: int = 3,
) -> np.ndarray:
    """Capture a linear frame, optionally via simple exposure-bracket HDR.

    リニアフレームを撮影する（任意で簡易な露光ブラケット HDR）。

    With ``enabled=False`` this simply returns the base exposure
    ``capture_at_fn(1.0)``. With ``enabled=True`` it brackets exposures:
    where the base pixel saturates (>= ``trust_high``) it shortens the
    exposure (``scale`` 0.5, 0.25, ...) and rescales the pixel by
    ``1/scale``; where the base pixel is crushed (<= ``trust_low``) it
    lengthens the exposure (``scale`` 2, 4, ...) and rescales by
    ``1/scale``. Trustworthy pixels replace untrustworthy ones, extending
    dynamic range in base-exposure linear units (values may exceed 1.0).

    ``enabled=False`` では基準露光 ``capture_at_fn(1.0)`` をそのまま返す。
    ``enabled=True`` では露光をブラケットする: 基準画素がサチる
    (>= ``trust_high``) 箇所は露光を短く（``scale`` 0.5, 0.25, ...）して
    画素を ``1/scale`` 倍、黒潰れ (<= ``trust_low``) 箇所は露光を長く
    （``scale`` 2, 4, ...）して ``1/scale`` 倍する。信頼できる画素で
    非信頼画素を置き換え、基準露光のリニア単位でダイナミックレンジを
    拡張する（値は 1.0 を超え得る）。

    Recovery divides each bracketed frame by its *actual* exposure ratio
    (see :func:`_frame_and_ratio`), not the nominal ``scale``, so discrete
    shutter snapping does not bias the merged values.

    回復は各ブラケットフレームを公称 ``scale`` ではなく *実効* 露光比
    （:func:`_frame_and_ratio` 参照）で割るため、離散シャッターへの
    スナップがマージ値を偏らせない。

    Args:
        capture_at_fn: Callable mapping an exposure ``scale`` to a linear
            RGB frame (H, W, 3) in [0, 1], or to a ``(frame, actual_ratio)``
            tuple when the realized exposure ratio differs from ``scale``.
            露光 ``scale`` をリニア RGB フレーム (H, W, 3)[0, 1] に写す callable。
            実現露光比が ``scale`` と異なる場合は ``(frame, actual_ratio)``
            タプルを返してもよい。
        enabled: Whether to bracket exposures.
            露光ブラケットを行うか。
        trust_low: Lower trustworthy bound as an 8-bit pixel value (0-255).
            信頼下限（8bit画素値 0〜255）。
        trust_high: Upper trustworthy bound as an 8-bit pixel value (0-255).
            信頼上限（8bit画素値 0〜255）。
        max_stops: Max exposure stops to bracket in each direction.
            各方向にブラケットする最大露光段数。

    Returns:
        Merged linear RGB frame of shape (H, W, 3).
            マージされたリニア RGB フレーム (H, W, 3)。
    """
    base, _ = _frame_and_ratio(capture_at_fn(1.0), 1.0)
    if not enabled:
        return base

    low = float(trust_low) / 255.0
    high = float(trust_high) / 255.0
    result = base.copy()

    # Saturated pixels: shorten exposure until they come below trust_high.
    scale = 1.0
    raw = base
    for _ in range(int(max_stops)):
        if not np.any(raw >= high):
            break
        scale *= 0.5
        raw, ratio = _frame_and_ratio(capture_at_fn(scale), scale)
        recovered = raw / ratio
        mask = (result >= high) & (raw <= high)
        result = np.where(mask, recovered, result)

    # Crushed pixels: lengthen exposure until they rise above trust_low.
    scale = 1.0
    raw = base
    for _ in range(int(max_stops)):
        if not np.any(raw <= low):
            break
        scale *= 2.0
        raw, ratio = _frame_and_ratio(capture_at_fn(scale), scale)
        recovered = raw / ratio
        mask = (result <= low) & (raw >= low)
        result = np.where(mask, recovered, result)

    return result.astype(np.float32)


def _query_supported_shutter_speeds(camera_backend: Any) -> Optional[List[str]]:
    """Return the camera's accepted Tv display strings, or ``None``.

    カメラが受理する Tv 表示文字列の一覧を返す（取得不可なら ``None``）。

    Uses a backend's optional ``supported_shutter_speeds()`` (no capture).
    Falls back to ``None`` -- and thus to standard Canon Tv snapping -- for
    backends that do not expose it or when the query fails.

    バックエンドの任意メソッド ``supported_shutter_speeds()``（撮影なし）を
    使う。非対応バックエンドや照会失敗時は ``None``（＝標準 Canon Tv への
    スナップ）にフォールバックする。
    """
    getter = getattr(camera_backend, "supported_shutter_speeds", None)
    if not callable(getter):
        return None
    try:
        values = list(getter())
    except Exception as exc:  # noqa: BLE001 - HW/SDK errors must stay non-fatal
        print(
            f"inverse_lut: could not query supported shutter speeds ({exc}); "
            "falling back to standard Canon Tv values.",
            file=sys.stderr,
        )
        return None
    return values or None


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

    # Simple HDR needs per-shot exposure control. Only the Canon backend
    # supports scaling the shutter (tv); otherwise fall back to single exposure.
    hdr_enabled = ilut_cfg.hdr_enabled
    backend_name = cfg.camera.backend.strip().lower()
    if hdr_enabled and backend_name != "canon_edsdk":
        print(
            "inverse_lut: HDR enabled but backend "
            f"'{cfg.camera.backend}' does not support exposure scaling; "
            "falling back to single exposure.",
            file=sys.stderr,
        )
        hdr_enabled = False

    # Query the body's actual Tv menu once so bracket exposures snap onto
    # values it accepts (prevents an EDSDK ValueError mid-calibration).
    supported_tv = (
        _query_supported_shutter_speeds(camera_backend) if hdr_enabled else None
    )
    base_seconds = tv_to_seconds(cfg.camera.tv)

    if show_window:
        proj = cfg.projector
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(window_name, proj.pos_x, proj.pos_y)

    def capture_fn(sample: CalibrationSample) -> np.ndarray:
        def capture_at(scale: float):
            if show_window:
                bgr = cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, bgr)
                cv2.waitKey(cfg.camera.wait_key_ms)
            if scale == 1.0:
                return camera_backend.capture_linear_rgb()
            # HDR bracket: rebuild the backend with a scaled shutter snapped
            # to a realizable Tv, and report the *actual* exposure ratio so
            # the merge does not assume the nominal scale was achieved.
            scaled_tv = scale_shutter_tv(cfg.camera.tv, scale, supported=supported_tv)
            actual_ratio = tv_to_seconds(scaled_tv) / base_seconds
            scaled_camera_cfg = replace(cfg.camera, tv=scaled_tv)
            frame = create_camera_backend(scaled_camera_cfg).capture_linear_rgb()
            return frame, actual_ratio

        return capture_hdr_linear(
            capture_at,
            enabled=hdr_enabled,
            trust_low=ilut_cfg.trust_low,
            trust_high=ilut_cfg.trust_high,
            max_stops=ilut_cfg.hdr_max_stops,
        )

    roi = (
        ilut_cfg.roi_x,
        ilut_cfg.roi_y,
        ilut_cfg.roi_width,
        ilut_cfg.roi_height,
    )
    try:
        data = measure_calibration_samples(samples, capture_fn, roi)
    finally:
        if show_window:
            cv2.destroyWindow(window_name)

    data.meta["source"] = ilut_cfg.source
    data.meta["hdr_enabled"] = hdr_enabled
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
        source: Sampling method, "channel_ramp" or "gray". Default from config.
            Only measurements of the matching method are used; the two are
            never mixed.
            サンプリング系統。"channel_ramp" または "gray"。既定は設定から。
            一致する系統の計測のみを使い、両系統を混在させない。
        lut_resolution: Number of LUT entries. Default from config.
            LUT エントリ数。既定は設定から。
        trust_low: Lower trustworthy bound as an 8-bit pixel value (0-255).
            Default from config.
            信頼下限（8bit画素値 0〜255）。既定は設定から。
        trust_high: Upper trustworthy bound as an 8-bit pixel value (0-255).
            Default from config.
            信頼上限（8bit画素値 0〜255）。既定は設定から。

    Returns:
        The built :class:`InverseLut`.
            構築された :class:`InverseLut`。
    """
    cfg = get_config().inverse_lut
    source = normalize_source(cfg.source if source is None else source)
    lut_resolution = cfg.lut_resolution if lut_resolution is None else int(lut_resolution)
    trust_low = cfg.trust_low if trust_low is None else float(trust_low)
    trust_high = cfg.trust_high if trust_high is None else float(trust_high)

    # 8-bit pixel-value bounds -> normalized [0, 1] for comparison with
    # linear measured values.
    trust_low_norm = trust_low / 255.0
    trust_high_norm = trust_high / 255.0

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
            agg_inputs, agg_values, t_grid, trust_low_norm, trust_high_norm
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
