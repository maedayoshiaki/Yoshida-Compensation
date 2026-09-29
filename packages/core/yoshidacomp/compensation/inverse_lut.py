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

import gc
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from yoshidacomp.config import get_config

# NOTE: ``yoshidacomp.capture`` is imported lazily inside
# :func:`run_calibration_capture`, the only place that needs it. At module
# scope it would drag the hardware layer (and its optional camera SDKs) into
# every pure-computation consumer, since ``compensation.pipeline`` imports
# this module. Same deferral pattern as the calibration scripts.
# 注: ``yoshidacomp.capture`` は唯一の利用箇所である
# :func:`run_calibration_capture` の内部で遅延importする。module scope に
# 置くと、``compensation.pipeline`` が本モジュールをimportする都合で、純粋
# 計算だけの利用者にもハードウェア層（と任意のカメラSDK）が付いてくる。
# calibration スクリプトと同じ遅延パターン。
#
# NOTE: ``scipy.interpolate.PchipInterpolator`` is imported lazily inside
# :func:`_build_channel_lut`, the only place that needs it (LUT
# *construction*; ``apply_inverse_lut`` -- the hot path every LUT
# linearizer call goes through -- uses only ``np.interp``). At module scope
# it cost ~1.5s of ``scipy.interpolate`` import time, which was most of
# what ``import yoshidacomp.compensation.pipeline`` still cost after the
# torch fixes (see
# ``.superpowers/sdd/2026-08-02-target-and-lut/task-1-report.md``). Same
# deferral pattern as ``yoshidacomp.capture`` above.
# 注: ``scipy.interpolate.PchipInterpolator`` は唯一の利用箇所である
# :func:`_build_channel_lut`（LUT*構築*時のみ。全linearizer呼び出しが通る
# ホットパスの ``apply_inverse_lut`` は ``np.interp`` のみを使う）の内部で
# 遅延importする。module scope に置くと ``scipy.interpolate`` の import に
# 約1.5秒かかり、torch対応後に残っていた ``import
# yoshidacomp.compensation.pipeline`` のコストの大半を占めていた（詳細は
# 上記レポート）。``yoshidacomp.capture`` と同じ遅延パターン。


# Wall-clock settle time between projecting a new pattern and capturing it, so
# the projector's display has fully switched before the shutter fires. Applied
# once per pattern (on the base exposure); HDR brackets keep the same pattern on
# screen and only change the shutter, so they need no extra settle.
# パターンを投影してから撮影するまでの整定待ち（実時間）。プロジェクタの表示が
# 完全に切り替わってからシャッターを切るための待機。パターンごとに1回（基準露光時）
# 適用する。HDRブラケットは同じパターンを表示したままシャッターだけ変えるので
# 追加の待機は不要。
_PROJECTION_SETTLE_SECONDS = 0.3


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
    # Drop a Canon fraction-stop annotation like " (1/3)" so third-stop menu
    # entries (e.g. '1/10 (1/3)') parse to their base seconds.
    if "(" in text:
        text = text.split("(", 1)[0].strip()
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


def is_flicker_free(tv: str, refresh_hz: Optional[float], tol: float = 0.03) -> bool:
    """Whether a shutter ``tv`` integrates whole projector refresh cycles.

    シャッター ``tv`` がプロジェクタのリフレッシュ周期の整数個を積分するか。

    Photographing a projector requires the exposure time to be an integer
    multiple of the refresh period (``1 / refresh_hz``); otherwise the frame
    captures a partial refresh and shows flicker/banding. Returns ``True``
    when ``exposure * refresh_hz`` is a positive near-integer within ``tol``,
    or when ``refresh_hz`` is falsy/non-positive (constraint disabled).

    プロジェクタ撮影では露光時間がリフレッシュ周期（``1 / refresh_hz``）の
    整数倍である必要がある。そうでないと部分的なリフレッシュを撮ってしまい
    フリッカー/バンディングが出る。``exposure * refresh_hz`` が ``tol`` 以内の
    正の（ほぼ）整数なら ``True``。``refresh_hz`` が偽値/非正なら制約無効で
    ``True``。
    """
    if not refresh_hz or refresh_hz <= 0:
        return True
    # Canon 1/3-stop menu entries (e.g. '1/6 (1/3)') display a rounded nominal
    # fraction whose ACTUAL exposure differs (1/6 shown, ~1/6.35 s real), so we
    # cannot trust them to be an exact refresh multiple -- exclude them and use
    # only clean, un-annotated shutter speeds.
    if "(" in str(tv):
        return False
    try:
        cycles = tv_to_seconds(tv) * float(refresh_hz)
    except (ValueError, TypeError):
        return False
    return cycles >= 1.0 - tol and abs(cycles - round(cycles)) <= tol


def _nearest_tv(candidates: Sequence[str], target_seconds: float) -> Optional[str]:
    """Return the candidate Tv whose seconds is nearest ``target_seconds``.

    ``target_seconds`` に最も近い秒を持つ候補 Tv を返す（解釈不能な候補は無視）。
    """
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
    return best_display


def scale_shutter_tv(
    tv: str,
    scale: float,
    *,
    supported: Optional[Sequence[str]] = None,
    refresh_hz: Optional[float] = None,
) -> str:
    """Scale a Canon shutter (``tv``) and snap to the nearest realizable value.

    Canon シャッター（``tv``）を露光係数でスケールし、実現可能な最も近い値へスナップする。

    ``scale = 0.5`` halves the exposure time (faster shutter); ``scale =
    2.0`` doubles it. The scaled target is snapped to the nearest value in
    ``supported`` -- the camera's actual Tv menu, e.g.
    ``list_supported()["Tv"]`` -- or, when not given, to the nearest
    standard Canon Tv. This guarantees a value the camera can accept and
    avoids an EDSDK ``ValueError`` that would abort calibration mid-run. When
    ``refresh_hz`` is given, only flicker-free candidates (exposure = integer
    multiple of ``1 / refresh_hz``) are considered, so the shutter stays
    synced to the projector.

    ``scale = 0.5`` は露光時間を半分（速いシャッター）、``scale = 2.0`` は倍に
    する。スケール後の目標秒を ``supported``（カメラ実機の Tv メニュー、例
    ``list_supported()["Tv"]``）— 未指定なら標準 Canon Tv — の中で最も近い値へ
    スナップする。これによりカメラが受理できる値を保証し、キャリブレーション
    途中で ``ValueError`` により中断するのを防ぐ。``refresh_hz`` を与えると、
    フリッカーフリー候補（露光が ``1 / refresh_hz`` の整数倍）のみを対象にし、
    シャッターをプロジェクタに同期させる。

    Args:
        tv: Base shutter-speed string. 基準シャッタースピード文字列。
        scale: Exposure-time multiplier. 露光時間の倍率。
        supported: Candidate Tv display strings to snap onto (the camera's
            menu). Defaults to standard Canon Tv values.
            スナップ先候補の Tv 表示文字列（カメラのメニュー）。既定は標準
            Canon Tv 値。
        refresh_hz: Projector refresh rate (Hz) to stay flicker-free with. If
            falsy/non-positive, the constraint is disabled.
            フリッカーフリーを保つプロジェクタのリフレッシュレート（Hz）。
            偽値/非正なら制約無効。

    Returns:
        The nearest realizable shutter-speed display string.
            最も近い実現可能なシャッタースピード表示文字列。
    """
    target_seconds = tv_to_seconds(tv) * float(scale)
    candidates = list(supported) if supported else list(_STANDARD_TV_DISPLAYS)
    if refresh_hz and refresh_hz > 0:
        flicker_free = [c for c in candidates if is_flicker_free(c, refresh_hz)]
        if flicker_free:
            candidates = flicker_free

    best_display = _nearest_tv(candidates, target_seconds)
    return best_display if best_display is not None else str(tv).strip()


def snap_to_flicker_free_tv(
    tv: str,
    supported: Optional[Sequence[str]] = None,
    refresh_hz: float = 60.0,
) -> str:
    """Snap a shutter ``tv`` to the nearest flicker-free realizable value.

    シャッター ``tv`` を、最も近いフリッカーフリーな実現可能値へスナップする。

    Picks the nearest value in ``supported`` (or standard Canon Tv) whose
    exposure is an integer multiple of the ``refresh_hz`` period. Returns
    ``tv`` unchanged when the constraint is disabled or no flicker-free
    candidate exists.

    ``supported``（または標準 Canon Tv）の中で、露光が ``refresh_hz`` 周期の
    整数倍になる最も近い値を選ぶ。制約が無効、または候補が無い場合は ``tv``
    をそのまま返す。
    """
    if not refresh_hz or refresh_hz <= 0:
        return str(tv).strip()
    candidates = list(supported) if supported else list(_STANDARD_TV_DISPLAYS)
    flicker_free = [c for c in candidates if is_flicker_free(c, refresh_hz)]
    best_display = _nearest_tv(flicker_free, tv_to_seconds(tv))
    return best_display if best_display is not None else str(tv).strip()


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


def _merge_hdr_exposures(
    exposures: Sequence[Tuple[np.ndarray, float]],
    *,
    trust_low: float,
    trust_high: float,
) -> np.ndarray:
    """Merge already captured exposures using the HDR selection policy.

    The write TARGET is decided from the BASE exposure, never from the
    bracketed frame's own value:

    - cells inside the trust band ``[low, high)`` at base keep the base value,
    - only base-dark cells (``<= low``) are recovered from longer exposures
      (they cannot saturate there: ``low * 2^max_stops < high``),
    - only base-saturated cells (``>= high``) are recovered from shorter
      exposures (falling back to the shortest one if every frame saturates).

    2026-08-12: 以前は「そのフレームで飽和していないセル」を露光比の大きい
    フレームで上書きしていたが、センサ飽和した長露光の現像値が 1.0 に
    張り付かず trust_high を下回ってすり抜け、÷露光比で明チャンネルが
    崩壊する（例: 単色 R255 が 0.7/8 ≈ 0.087 と記録される）。sample.py の
    2026-08-07 の修正と同じく、書き込み先を基準露光で判定して飽和判定への
    依存を断つ。中間帯のセルが長露光の SNR を得られなくなる代償は受け入れる
    （sample.py の HDR マージと同じ意味論に揃える）。
    """
    if not exposures:
        raise ValueError("At least one exposure is required for HDR merge.")

    low = float(trust_low) / 255.0
    high = float(trust_high) / 255.0
    base = np.asarray(exposures[0][0], dtype=np.float32)
    base_dark = base <= low
    base_saturated = base >= high
    result = base.copy()
    chosen = np.zeros(base.shape, dtype=np.float32)
    for frame, ratio in sorted(exposures[1:], key=lambda item: -item[1]):
        frame = np.asarray(frame, dtype=np.float32)
        if ratio > 1.0:
            take = base_dark & (frame < high) & (ratio > chosen)
        elif ratio < 1.0:
            take = base_saturated & (frame < high) & (chosen == 0.0)
        else:
            continue
        result = np.where(take, frame / ratio, result)
        chosen = np.where(take, np.float32(ratio), chosen)
    shortest_frame, shortest_ratio = min(exposures, key=lambda item: item[1])
    unrecovered = base_saturated & (chosen == 0.0)
    result = np.where(
        unrecovered,
        np.asarray(shortest_frame, dtype=np.float32) / shortest_ratio,
        result,
    )
    return result.astype(np.float32)


def capture_hdr_linear(
    capture_at_fn: Callable[[float], "np.ndarray | Tuple[np.ndarray, float]"],
    *,
    enabled: bool = False,
    trust_low: float = 10.0,
    trust_high: float = 245.0,
    max_stops: int = 3,
    channels: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Capture a linear frame, optionally via simple exposure-bracket HDR.

    リニアフレームを撮影する（任意で簡易な露光ブラケット HDR）。

    With ``enabled=False`` this simply returns the base exposure
    ``capture_at_fn(1.0)``. With ``enabled=True`` it brackets exposures
    (shorten while any gated channel saturates, lengthen while any is
    crushed) and then merges **independently per channel**, anchored on the
    base exposure: cells inside the trust band keep the base value, cells
    dark at base (``<= trust_low``) are recovered from the longest
    trustworthy long exposure, and cells saturated at base
    (``>= trust_high``) are recovered from a shorter exposure. Each channel
    judges its own state, so a bright channel is never held back by a dark
    one; values may exceed 1.0 (dynamic range beyond the base clip).

    ``enabled=False`` では基準露光 ``capture_at_fn(1.0)`` をそのまま返す。
    ``enabled=True`` では露光をブラケット（対象chのいずれかがサチる間は短く、
    いずれかが黒潰れの間は長く）してから、**基準露光を基点にチャンネル独立**で
    マージする: 信頼帯内のセルは基準値のまま、基準で暗い (``<= trust_low``)
    セルだけ長露光から、基準で飽和 (``>= trust_high``) したセルだけ短露光から
    回復する（詳細と理由は :func:`_merge_hdr_exposures`）。各chが自分の状態を
    判定するので、明るいchが暗いchに引きずられない。値は 1.0 を超え得る
    （基準クリップを超えるダイナミックレンジを拡張する）。

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
        channels: Channels to consider when deciding whether to bracket and
            which values to recover. Defaults to all channels. For
            single-channel calibration ramps pass the recorded channel(s) so
            the always-dark non-projected channels do not force needless
            long-exposure brackets, and only the measured channel is merged.
            ブラケット要否の判定と回復対象にするチャンネル。既定は全チャンネル。
            単独チャンネルランプでは記録チャンネルを渡すことで、投影していない
            常時暗のチャンネルが不要な長露光ブラケットを誘発するのを防ぎ、
            測定対象のチャンネルだけをマージする。

    Returns:
        Merged linear RGB frame of shape (H, W, 3).
            マージされたリニア RGB フレーム (H, W, 3)。
    """
    base = np.asarray(_frame_and_ratio(capture_at_fn(1.0), 1.0)[0], dtype=np.float32)
    if not enabled:
        return base

    low = float(trust_low) / 255.0
    high = float(trust_high) / 255.0

    # Per-channel gate (default: all). The always-dark non-projected channels of
    # a single-channel ramp are excluded so they neither drive brackets nor are
    # merged.
    channel_gate = np.zeros(base.shape[-1], dtype=bool)
    if channels is None:
        channel_gate[:] = True
    else:
        for c in channels:
            channel_gate[int(c)] = True

    # Capture the bracket sweep. Shorten while ANY gated channel is still
    # saturated; lengthen while ANY gated channel is still crushed. This only
    # decides HOW MANY exposures to grab (each is a whole-frame shot); the merge
    # below is judged fully independently per channel.
    exposures = [(base, 1.0)]
    raw = base
    scale = 1.0
    for _ in range(int(max_stops)):
        if not np.any((raw >= high) & channel_gate):
            break
        scale *= 0.5
        frame, ratio = _frame_and_ratio(capture_at_fn(scale), scale)
        raw = np.asarray(frame, dtype=np.float32)
        exposures.append((raw, ratio))
    raw = base
    scale = 1.0
    # ANY channel (gated or not) predicted to saturate poisons the whole
    # frame: the RAW development's colour matrix mixes the clipped channel
    # into the others, so even the dark channels of that frame read wrong
    # (2026-08-12 実測: 単色 R255 の ×8 露光で G が 0.0005 → 0.03 に化けた)。
    # Stop lengthening BEFORE the brightest channel is predicted (linearly
    # from the base exposure) to clip; the crushed channels then recover from
    # the longest clean frame only.
    base_peak = float(np.max(base)) if base.size else 0.0
    for _ in range(int(max_stops)):
        if not np.any((raw <= low) & channel_gate):
            break
        if base_peak * (scale * 2.0) >= high:
            break
        scale *= 2.0
        frame, ratio = _frame_and_ratio(capture_at_fn(scale), scale)
        raw = np.asarray(frame, dtype=np.float32)
        exposures.append((raw, ratio))

    # Per-channel merge anchored on the BASE exposure: base-dark cells are
    # recovered from longer exposures, base-saturated cells from shorter ones,
    # everything else keeps the base value. See _merge_hdr_exposures for why
    # the bracketed frame's own value must not decide the write target.
    return _merge_hdr_exposures(
        exposures, trust_low=trust_low, trust_high=trust_high
    )


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
    debug_dir: Optional[str] = None,
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
        debug_dir: If set, save every captured ROI crop -- each pre-HDR
            exposure and the post-HDR merged result -- as PNGs, plus a
            ``captures.csv`` of ROI-mean values per stage, for debugging.
            設定すると、撮影した ROI クロップ（各露光=HDR前、及びHDRマージ後）を
            PNG 保存し、各段の ROI 平均値を ``captures.csv`` に残す（デバッグ用）。

    Returns:
        A :class:`CalibrationData` with the recorded measurements.
            記録された計測値を持つ :class:`CalibrationData`。
    """
    cfg = config if config is not None else get_config()
    ilut_cfg = cfg.inverse_lut
    if camera_backend is None:
        # Deferred: keeps the hardware layer out of pure-computation imports.
        # 遅延import: 純粋計算だけの利用者にハードウェア層を持ち込まない。
        from yoshidacomp.capture import create_camera_backend

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

    # Snap the base shutter to a flicker-free value (an integer multiple of
    # the projector refresh period) so the camera integrates whole refresh
    # cycles; otherwise the projection shows flicker/banding. HDR brackets are
    # likewise restricted to flicker-free values below.
    refresh_hz = float(getattr(cfg.projector, "refresh_rate", 0.0) or 0.0)
    base_tv = cfg.camera.tv
    if refresh_hz > 0:
        snapped_tv = snap_to_flicker_free_tv(cfg.camera.tv, refresh_hz=refresh_hz)
        if snapped_tv != base_tv:
            print(
                f"inverse_lut: shutter {cfg.camera.tv!r} is not a multiple of the "
                f"{refresh_hz:g}Hz projector refresh; using flicker-free "
                f"{snapped_tv!r} instead.",
                file=sys.stderr,
            )
        base_tv = snapped_tv
    base_seconds = tv_to_seconds(base_tv)

    # Reuse ONE camera session across all captures when the backend supports
    # it (a persistent EDSDK session). Opening a session per shot repeats
    # InitializeSDK/TerminateSDK, which destabilizes EDSDK during batch
    # capture (EDS_ERR_DEVICE_BUSY / EDS_ERR_INTERNAL_ERROR). Backends without
    # a context manager (e.g. OpenCV) run unchanged.
    session_cm = (
        camera_backend
        if hasattr(camera_backend, "__enter__") and hasattr(camera_backend, "__exit__")
        else nullcontext(camera_backend)
    )
    supported_tv: Optional[List[str]] = None
    roi = (
        ilut_cfg.roi_x,
        ilut_cfg.roi_y,
        ilut_cfg.roi_width,
        ilut_cfg.roi_height,
    )

    # Debug: save every capture (each pre-HDR exposure + the post-HDR merge).
    # Work on the ROI crop (not the (1,1,3) mean) so both are viewable images;
    # the crop is small enough (~tens of MB) that the HDR merge stays in memory.
    debug = debug_dir is not None
    debug_state = {"idx": 0, "rows": []}
    debug_path = Path(debug_dir) if debug_dir is not None else None
    if debug_path is not None:
        debug_path.mkdir(parents=True, exist_ok=True)

    def _roi_crop(frame: np.ndarray) -> np.ndarray:
        """Return a COPY of the frame's ROI rectangle (whole frame if unset).

        A copy (not a slice view) is essential: a view keeps the whole ~390 MB
        capture alive, and holding several across the HDR merge exhausts memory.
        """
        frame = np.asarray(frame, dtype=np.float32)
        height, width = frame.shape[:2]
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            x0 = max(0, min(int(x), width))
            y0 = max(0, min(int(y), height))
            x1 = max(x0, min(int(x) + int(rw), width))
            y1 = max(y0, min(int(y) + int(rh), height))
            crop = frame[y0:y1, x0:x1, :3]
            if crop.size > 0:
                return crop.copy()
        return frame[:, :, :3].copy()

    def _save_crop(crop: np.ndarray, name: str) -> str:
        # Save an 8-bit PNG (viewable) and return its path. Disk-backed on
        # purpose: this machine cannot hold many full-res crops in RAM, so we
        # write each capture out and read it back only when merging.
        assert debug_path is not None
        path = debug_path / f"{name}.png"
        img = np.clip(np.asarray(crop, dtype=np.float32), 0.0, 1.0)
        bgr = cv2.cvtColor((img * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
        return str(path)

    def _load_crop(path: str) -> np.ndarray:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read back debug crop: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    def _merge_saved_crops(exposures, low: float, high: float) -> np.ndarray:
        # Re-merge the saved full-res crops (read from disk, one at a time) into
        # a post-HDR image. Runs after a pattern's captures, when no RAW develop
        # is in flight, so peak memory is only a few crops. The measurement
        # itself uses the exact mean-HDR below -- this is for a viewable image.
        result = _load_crop(exposures[0][0])
        for path, ratio in exposures[1:]:
            crop = _load_crop(path)
            recovered = crop / ratio
            if ratio < 1.0:  # shorter exposure recovers highlights
                mask = (result >= high) & (crop <= high)
            else:  # longer exposure recovers shadows
                mask = (result <= low) & (crop >= low)
            result = np.where(mask, recovered, result)
        return result

    debug_records: dict[Tuple[float, float, float], dict] = {}

    def _sample_key(sample: CalibrationSample) -> Tuple[float, float, float]:
        return tuple(round(float(v), 6) for v in sample.input_rgb)

    def _debug_record(sample: CalibrationSample) -> dict:
        key = _sample_key(sample)
        record = debug_records.get(key)
        if record is None:
            r, g, b = (int(round(v * 255)) for v in sample.input_rgb)
            tag = (
                f"{debug_state['idx']:02d}_{sample.kind}_"
                f"r{r:03d}g{g:03d}b{b:03d}"
            )
            record = {"tag": tag, "n": 0, "exposures": []}
            debug_records[key] = record
            debug_state["idx"] += 1
        return record

    def _prepare_frame(
        sample: CalibrationSample,
        frame: np.ndarray,
        tv_label: str,
        ratio: float,
    ) -> np.ndarray:
        # Reduce immediately after deferred RAW development. This keeps only a
        # 1x1 RGB mean per exposure in memory even for a large calibration.
        if debug:
            record = _debug_record(sample)
            crop = _roi_crop(frame)
            safe_tv = tv_label.replace("/", "-").replace('"', "s").replace(" ", "")
            path = _save_crop(
                crop,
                f"{record['tag']}_pre{record['n']:02d}_tv{safe_tv}",
            )
            record["exposures"].append((path, float(ratio)))
            mean = crop.reshape(-1, 3).mean(axis=0)
            debug_state["rows"].append(
                (record["tag"], f"pre:{tv_label}", *[float(x) for x in mean])
            )
            record["n"] += 1
            return mean.reshape(1, 1, 3).astype(np.float32)
        return reduce_frame(frame, roi).reshape(1, 1, 3)

    def _finish_debug(sample: CalibrationSample, merged: np.ndarray) -> None:
        if not debug:
            return
        record = _debug_record(sample)
        mean = np.asarray(merged, dtype=np.float32).reshape(-1, 3).mean(axis=0)
        debug_state["rows"].append(
            (record["tag"], "post_hdr", *[float(x) for x in mean])
        )
        post = _merge_saved_crops(
            record["exposures"],
            ilut_cfg.trust_low / 255.0,
            ilut_cfg.trust_high / 255.0,
        )
        _save_crop(post, f"{record['tag']}_HDR")

    def _project(
        sample: CalibrationSample,
        *,
        wait_ms: int,
        settle: bool,
    ) -> None:
        if not show_window:
            return
        bgr = cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, bgr)
        cv2.waitKey(wait_ms)
        if settle:
            time.sleep(_PROJECTION_SETTLE_SECONDS)

    def capture_fn(sample: CalibrationSample) -> np.ndarray:
        """Legacy immediate path for non-trigger-capable camera backends."""

        def capture_at(scale: float):
            # HDR brackets retain the same pattern, so only the first exposure
            # needs the extra projector settle delay.
            _project(
                sample,
                wait_ms=cfg.camera.wait_key_ms,
                settle=scale == 1.0,
            )
            if not hdr_enabled:
                return _prepare_frame(
                    sample,
                    camera_backend.capture_linear_rgb(),
                    base_tv,
                    1.0,
                )
            # HDR bracket (Canon only): set the shutter explicitly per shot on
            # the persistent session -- base tv for the reference exposure, a
            # snapped realizable Tv for brackets -- and report the *actual*
            # exposure ratio so the merge does not assume the nominal scale.
            if scale == 1.0:
                return _prepare_frame(
                    sample,
                    camera_backend.capture_linear_rgb(tv=base_tv), base_tv, 1.0
                )
            scaled_tv = scale_shutter_tv(
                base_tv, scale, supported=supported_tv, refresh_hz=refresh_hz
            )
            actual_ratio = tv_to_seconds(scaled_tv) / base_seconds
            return (
                _prepare_frame(
                    sample,
                    camera_backend.capture_linear_rgb(tv=scaled_tv),
                    scaled_tv,
                    actual_ratio,
                ),
                actual_ratio,
            )

        merged = capture_hdr_linear(
            capture_at,
            enabled=hdr_enabled,
            trust_low=ilut_cfg.trust_low,
            trust_high=ilut_cfg.trust_high,
            max_stops=ilut_cfg.hdr_max_stops,
            # Only bracket for the channel(s) this sample records, so the
            # always-dark non-projected channels of a channel ramp don't force
            # useless long-exposure brackets.
            channels=sample.recorded_channels,
        )
        _finish_debug(sample, merged)
        return merged

    def _run_triggered_capture() -> CalibrationData:
        """Capture a pass first, then develop/reduce its RAW files."""
        unique: list[CalibrationSample] = []
        seen: set[Tuple[float, float, float]] = set()
        for sample in samples:
            key = _sample_key(sample)
            if key not in seen:
                seen.add(key)
                unique.append(sample)

        exposures: dict[
            Tuple[float, float, float], list[Tuple[np.ndarray, float]]
        ] = {_sample_key(sample): [] for sample in unique}

        def capture_pass(
            pass_samples: Sequence[CalibrationSample],
            tv_label: str,
            ratio: float,
        ) -> dict[Tuple[float, float, float], np.ndarray]:
            if not pass_samples:
                return {}
            print(
                f"inverse_lut: triggered capture {len(pass_samples)} pattern(s) "
                f"at Tv={tv_label}; RAW development follows the pass.",
                file=sys.stderr,
            )
            prepared: dict[Tuple[float, float, float], np.ndarray] = {}
            with camera_backend.triggered_batch(tv_label) as batch:
                capture_started = time.perf_counter()
                switch_wait_ms = int(
                    getattr(
                        cfg.camera,
                        "trigger_wait_key_ms",
                        cfg.camera.wait_key_ms,
                    )
                )
                for index, sample in enumerate(pass_samples):
                    # The first frame gets the full initialization wait.
                    # Later synchronous EDSDK triggers provide ample settle
                    # time while the previous RAW asset is transferred.
                    _project(
                        sample,
                        wait_ms=(
                            cfg.camera.wait_key_ms
                            if index == 0
                            else switch_wait_ms
                        ),
                        settle=False,
                    )
                    batch.trigger()
                capture_seconds = time.perf_counter() - capture_started
                print(
                    f"inverse_lut: trigger pass complete in "
                    f"{capture_seconds:.2f}s; developing RAW files.",
                    file=sys.stderr,
                )
                development_started = time.perf_counter()
                for sample, frame in zip(
                    pass_samples, batch.iter_linear_rgb(), strict=True
                ):
                    prepared[_sample_key(sample)] = _prepare_frame(
                        sample, frame, tv_label, ratio
                    )
                    # The developed frame is full-resolution (~390 MB).
                    # Release it before the next RAW is developed; otherwise
                    # the loop can keep two large arrays alive and eventually
                    # terminate the native RAW decoder.
                    del frame
                    gc.collect()
                development_seconds = time.perf_counter() - development_started
                print(
                    f"inverse_lut: RAW development complete in "
                    f"{development_seconds:.2f}s.",
                    file=sys.stderr,
                )
            return prepared

        # Full-resolution Canon RAW batches can terminate the RAW developer
        # when too many files are held in one EDSDK spool.  Keep the capture
        # order and exposure unchanged, but develop small triggered batches
        # before acquiring the next chunk.
        calibration_chunk_size = 4

        def capture_chunked(
            pass_samples: Sequence[CalibrationSample],
            tv_label: str,
            ratio: float,
        ) -> dict[Tuple[float, float, float], np.ndarray]:
            prepared: dict[Tuple[float, float, float], np.ndarray] = {}
            for start in range(0, len(pass_samples), calibration_chunk_size):
                chunk = pass_samples[start : start + calibration_chunk_size]
                prepared.update(capture_pass(chunk, tv_label, ratio))
            return prepared

        base_frames = capture_pass(unique, base_tv, 1.0)
        for sample in unique:
            key = _sample_key(sample)
            exposures[key].append((base_frames[key], 1.0))

        if hdr_enabled:
            low = float(ilut_cfg.trust_low) / 255.0
            high = float(ilut_cfg.trust_high) / 255.0

            def needs_short(sample: CalibrationSample, frame: np.ndarray) -> bool:
                return bool(
                    np.any(
                        frame[..., list(sample.recorded_channels)] >= high
                    )
                )

            def needs_long(sample: CalibrationSample, frame: np.ndarray) -> bool:
                return bool(
                    np.any(
                        frame[..., list(sample.recorded_channels)] <= low
                    )
                )

            pending = [
                sample
                for sample in unique
                if needs_short(sample, base_frames[_sample_key(sample)])
            ]
            used_short_tvs = {base_tv}
            for stop in range(1, int(ilut_cfg.hdr_max_stops) + 1):
                if not pending:
                    break
                scale = 0.5**stop
                tv_label = scale_shutter_tv(
                    base_tv,
                    scale,
                    supported=supported_tv,
                    refresh_hz=refresh_hz,
                )
                if tv_label in used_short_tvs:
                    continue
                used_short_tvs.add(tv_label)
                ratio = tv_to_seconds(tv_label) / base_seconds
                frames = capture_pass(pending, tv_label, ratio)
                for sample in pending:
                    exposures[_sample_key(sample)].append(
                        (frames[_sample_key(sample)], ratio)
                    )
                pending = [
                    sample
                    for sample in pending
                    if needs_short(sample, frames[_sample_key(sample)])
                ]

            pending = [
                sample
                for sample in unique
                if needs_long(sample, base_frames[_sample_key(sample)])
            ]
            used_long_tvs = {base_tv}
            for stop in range(1, int(ilut_cfg.hdr_max_stops) + 1):
                scale = 2.0**stop
                # ANY channel predicted (linearly from base) to clip poisons
                # the whole frame via the RAW development colour matrix, so
                # such samples must not be captured at this scale at all.
                # 詳細は capture_hdr_linear の長露光ガードのコメントを参照。
                pending = [
                    sample
                    for sample in pending
                    if float(np.max(base_frames[_sample_key(sample)])) * scale
                    < high
                ]
                if not pending:
                    break
                tv_label = scale_shutter_tv(
                    base_tv,
                    scale,
                    supported=supported_tv,
                    refresh_hz=refresh_hz,
                )
                if tv_label in used_long_tvs:
                    continue
                used_long_tvs.add(tv_label)
                ratio = tv_to_seconds(tv_label) / base_seconds
                frames = capture_pass(pending, tv_label, ratio)
                for sample in pending:
                    exposures[_sample_key(sample)].append(
                        (frames[_sample_key(sample)], ratio)
                    )
                pending = [
                    sample
                    for sample in pending
                    if needs_long(sample, frames[_sample_key(sample)])
                ]

        merged_by_key = {
            key: _merge_hdr_exposures(
                captured,
                trust_low=ilut_cfg.trust_low,
                trust_high=ilut_cfg.trust_high,
            )
            for key, captured in exposures.items()
        }
        for sample in unique:
            _finish_debug(sample, merged_by_key[_sample_key(sample)])

        data = measure_calibration_samples(
            samples,
            lambda sample: merged_by_key[_sample_key(sample)],
            roi,
        )
        data.meta["capture_mode"] = "edsdk_triggered_batch"
        return data

    with session_cm:
        # Query the body's actual Tv menu once (reusing the open session) so
        # bracket exposures snap onto values it accepts.
        if hdr_enabled:
            supported_tv = _query_supported_shutter_speeds(camera_backend)
        if show_window:
            proj = cfg.projector
            # Move the normal window onto the target display before entering
            # fullscreen. On this multi-monitor Windows setup, fullscreen-first
            # can pin the window to the primary monitor even after moveWindow.
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(window_name, proj.pos_x, proj.pos_y)
            cv2.resizeWindow(window_name, proj.width, proj.height)
            cv2.waitKey(50)
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        try:
            if callable(getattr(camera_backend, "triggered_batch", None)):
                data = _run_triggered_capture()
            else:
                data = measure_calibration_samples(samples, capture_fn, roi)
                data.meta["capture_mode"] = "immediate"
        finally:
            if show_window:
                cv2.destroyWindow(window_name)

    if debug_path is not None:
        csv_lines = ["pattern,stage,roi_R,roi_G,roi_B"]
        csv_lines += [
            f"{tag},{stage},{r:.6f},{g:.6f},{b:.6f}"
            for tag, stage, r, g, b in debug_state["rows"]
        ]
        (debug_path / "captures.csv").write_text("\n".join(csv_lines), encoding="utf-8")
        print(
            f"inverse_lut: saved {debug_state['idx']} patterns "
            f"({len(debug_state['rows'])} captures) to {debug_path}",
            file=sys.stderr,
        )

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
    from scipy.interpolate import PchipInterpolator

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
