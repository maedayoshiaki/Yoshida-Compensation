# 逆LUT（測定ベース プロジェクタ応答線形化）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** カメラをリニア測光器として使い、プロジェクタのチャンネル別応答（ガンマ特性）を測定して打ち消す、グローバルな逆LUTモジュールを独立実装する。

**Architecture:** 新規 `src/python/inverse_lut.py` に「キャリブレーションパターン生成 → 生パターン投影＆リニア撮影 → 入力レベル集約・黒引き・白正規化・単調化・反転 → LUT適用/保存」を実装。CLI は `examples/python/build_inverse_lut.py`。`config.py` に `[inverse_lut]` 設定を追加。`sample.py` と色補償パイプラインは変更しない。

**Tech Stack:** Python 3.13, numpy, scipy（PCHIP補間）, OpenCV（投影ウィンドウ）, 既存カメラバックエンド抽象層, pytest。

## Global Constraints

- Python `>=3.13`。依存は既存のもののみ（`numpy>=2.3.4`, `scipy>=1.16.3`, `opencv-contrib-python`, `torch`）。新規依存を追加しない。
- Docstring は既存規約（英語 → 日本語の順で併記、NumPy/Google スタイル）に合わせる。
- 設定は `config.py` の frozen dataclass 規約（`_build_section` / `_serialize_config_toml` / `generate_config_file`）に準拠。
- 画像はメモリ上 **RGB** 順で扱い、`cv2.imread`/`imwrite` の境界でのみ BGR 変換する。
- カメラバックエンドの `capture_linear_rgb()` は **リニアRGB [0,1] float32** を返す。LUT計測にはこれを使う（`sample.py` の `capture_image()`（sRGB uint8）は使わない）。
- テストは **ハードウェア非依存**（`monkeypatch`/`tmp_path`、カメラと `cv2` はモック）。既存 `tests/test_sample_mocked.py` の流儀に合わせる。
- 逆LUTは値域 [0,1] float で保持し、8bit 量子化は投影/保存の直前でのみ行う。
- LUTは **チャンネル独立・グローバル**（ピクセルごとではない）。

---

## File Structure

- Create: `src/python/inverse_lut.py` — コア（データ構造・生成・計測・構築・適用・保存）。
- Modify: `src/python/config.py` — `InverseLutConfig` 追加と `AppConfig`/ローダ/シリアライザ配線。
- Modify: `config.toml` — `[inverse_lut]` セクション追加。
- Create: `examples/python/build_inverse_lut.py` — CLI。
- Create: `tests/test_inverse_lut.py` — 全テスト。

---

## Task 1: 設定 `[inverse_lut]` の追加

**Files:**
- Modify: `src/python/config.py`
- Modify: `config.toml`
- Test: `tests/test_inverse_lut.py`

**Interfaces:**
- Consumes: 既存 `AppConfig`, `_build_section`, `_serialize_config_toml`, `load_config`, `reload_config`。
- Produces:
  - `InverseLutConfig(num_levels:int=17, source:str="both", lut_resolution:int=256, reference_region:float=0.2, trust_low:float=10/255, trust_high:float=245/255, hdr_enabled:bool=False, inverse_lut_path:str="data/inverse_lut.npz")`（frozen dataclass）
  - `AppConfig.inverse_lut: InverseLutConfig`

- [ ] **Step 1: Write the failing test**

`tests/test_inverse_lut.py` を新規作成し、以下を記述:

```python
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.python.config import AppConfig, InverseLutConfig, reload_config


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inverse_lut.py::test_inverse_lut_config_defaults -v`
Expected: FAIL — `ImportError: cannot import name 'InverseLutConfig'`

- [ ] **Step 3: Add `InverseLutConfig` dataclass**

`src/python/config.py` の `CompensationConfig` 定義の直後に追加:

```python
@dataclass(frozen=True)
class InverseLutConfig:
    """Configuration for the measured inverse LUT (projector linearization).

    Attributes:
        num_levels: Number of gradation points per ramp (gray and per-channel).
            各ランプ（グレー / 単独チャンネル）の階調数。
        source: Curve-building source. One of "both", "channel_ramp", "gray".
            曲線構築ソース。"both" / "channel_ramp" / "gray"。
        lut_resolution: Number of entries in the dense [0,1] inverse LUT.
            密な [0,1] 逆LUTのエントリ数。
        reference_region: Central fraction of the captured frame averaged
            into one measurement (0.0-1.0).
            1計測に集約する撮影フレーム中央領域の割合（0.0〜1.0）。
        trust_low: Lower trustworthy bound on measured values ([0,1]).
            Values at or below are treated as unreliable/saturated.
            計測値の信頼下限（[0,1]）。以下は非信頼/サチュ扱い。
        trust_high: Upper trustworthy bound on measured values ([0,1]).
            計測値の信頼上限（[0,1]）。以上は非信頼/サチュ扱い。
        hdr_enabled: Whether to use the (provisional) simple HDR capture.
            簡易HDR撮影（仮実装）を使うか。
        inverse_lut_path: Output path for the saved inverse LUT (.npz).
            保存する逆LUT（.npz）の出力先パス。

    測定ベース逆LUT（プロジェクタ線形化）の設定。
    """

    num_levels: int = 17
    source: str = "both"
    lut_resolution: int = 256
    reference_region: float = 0.2
    trust_low: float = 10.0 / 255.0
    trust_high: float = 245.0 / 255.0
    hdr_enabled: bool = False
    inverse_lut_path: str = "data/inverse_lut.npz"
```

- [ ] **Step 4: Wire into `AppConfig` and the loader/serializer**

`AppConfig` にフィールドを追加（`compensation` の直後）:

```python
    compensation: CompensationConfig = field(default_factory=CompensationConfig)
    inverse_lut: InverseLutConfig = field(default_factory=InverseLutConfig)
```

`load_config` の `return AppConfig(...)` に行を追加:

```python
    return AppConfig(
        projector=_build_section(ProjectorConfig, data, "projector"),
        camera=_build_section(CameraConfig, data, "camera"),
        paths=_build_section(PathsConfig, data, "paths"),
        compensation=_build_section(CompensationConfig, data, "compensation"),
        inverse_lut=_build_section(InverseLutConfig, data, "inverse_lut"),
    )
```

`_serialize_config_toml` の `sections` リストに追加:

```python
    sections = [
        ("projector", asdict(config.projector)),
        ("camera", asdict(config.camera)),
        ("paths", asdict(config.paths)),
        ("compensation", asdict(config.compensation)),
        ("inverse_lut", asdict(config.inverse_lut)),
    ]
```

- [ ] **Step 5: Add the `[inverse_lut]` section to `config.toml`**

`config.toml` 末尾（`[compensation]` セクションの後）に追加:

```toml

[inverse_lut]
num_levels = 17                 # 各ランプ（グレー / 単独チャンネル）の階調数
source = "both"                 # "both" | "channel_ramp" | "gray"
lut_resolution = 256            # 密な [0,1] 逆LUT のエントリ数
reference_region = 0.2          # 計測に使う中央領域の割合
trust_low = 0.0392156862745098  # 信頼下限（≈10/255）
trust_high = 0.9607843137254902 # 信頼上限（≈245/255）
hdr_enabled = false             # 簡易HDR（仮実装）
inverse_lut_path = "data/inverse_lut.npz"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_inverse_lut.py -v`
Expected: PASS（2 件）

- [ ] **Step 7: Commit**

```bash
git add src/python/config.py config.toml tests/test_inverse_lut.py
git commit -m "feat(inverse_lut): [inverse_lut] 設定セクションを追加"
```

---

## Task 2: キャリブレーションパターン生成

**Files:**
- Create: `src/python/inverse_lut.py`
- Test: `tests/test_inverse_lut.py`

**Interfaces:**
- Consumes: `get_config().inverse_lut.num_levels`（既定）。
- Produces:
  - `CalibrationSample`（frozen dataclass）: `input_rgb: tuple[float,float,float]`, `image: np.ndarray`, `kind: str`（"gray"|"channel"）, `recorded_channels: tuple[int,...]`。
  - `generate_calibration_patterns(x_size:int, y_size:int, num_levels:int|None=None, include_gray:bool=True, include_channel_ramps:bool=True, dtype=np.uint8) -> list[CalibrationSample]`

- [ ] **Step 1: Write the failing test**

`tests/test_inverse_lut.py` に追記（先頭の import 群に追加）:

```python
import numpy as np
from src.python import inverse_lut as ilut


def test_generate_calibration_patterns_counts_and_kinds():
    samples = ilut.generate_calibration_patterns(
        4, 3, num_levels=17, dtype=np.uint8
    )
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
    # 単独Rランプは G,B 入力が 0
    for s in r_ramp:
        assert s.input_rgb[1] == 0.0 and s.input_rgb[2] == 0.0

    # 画像形状と dtype、単色であること
    assert gray[0].image.shape == (3, 4, 3)
    assert gray[0].image.dtype == np.uint8
    top_gray = max(gray, key=lambda s: s.input_rgb[0])
    assert int(top_gray.image[0, 0, 0]) == 255  # 最大階調は 255

    # 全黒の distinct 投影は 1 つ（重複入力の確認）
    distinct_inputs = {tuple(round(v, 6) for v in s.input_rgb) for s in samples}
    assert len(distinct_inputs) == 1 + 4 * (17 - 1)  # 65
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inverse_lut.py::test_generate_calibration_patterns_counts_and_kinds -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.python.inverse_lut'`

- [ ] **Step 3: Create `src/python/inverse_lut.py` with dataclass + generator**

```python
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

import numpy as np

from src.python.config import get_config


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inverse_lut.py::test_generate_calibration_patterns_counts_and_kinds -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/inverse_lut.py tests/test_inverse_lut.py
git commit -m "feat(inverse_lut): キャリブレーションパターン生成を追加"
```

---

## Task 3: 撮影フレームの集約と計測

**Files:**
- Modify: `src/python/inverse_lut.py`
- Test: `tests/test_inverse_lut.py`

**Interfaces:**
- Consumes: `CalibrationSample`（Task 2）。
- Produces:
  - `CalibrationMeasurement`（frozen dataclass）: `input_rgb: tuple[float,float,float]`, `measured_rgb: tuple[float,float,float]`（未記録は NaN）, `recorded_channels: tuple[int,...]`, `kind: str`。
  - `CalibrationData`（dataclass）: `measurements: list[CalibrationMeasurement]`, `meta: dict`。
  - `reduce_frame(frame: np.ndarray, reference_region: float = 0.2) -> np.ndarray`（(3,) 平均）。
  - `measure_calibration_samples(samples, capture_fn: Callable[[CalibrationSample], np.ndarray], reference_region: float = 0.2) -> CalibrationData`。`capture_fn` は各サンプルに対しリニアフレーム (H,W,3)[0,1] を返す。同一 `input_rgb` は 1 回だけ撮影（キャッシュ）。

- [ ] **Step 1: Write the failing test**

`tests/test_inverse_lut.py` に追記:

```python
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
    assert np.allclose(mean, 1.0)  # 中央20% = 2x2 領域はすべて 1


def test_measure_records_only_recorded_channels_and_dedups():
    gammas = (2.2, 2.0, 1.8)
    black = (10 / 255, 8 / 255, 6 / 255)
    samples = ilut.generate_calibration_patterns(6, 6, num_levels=5, dtype=np.uint8)

    calls = {"n": 0}

    def capture_fn(sample):
        calls["n"] += 1
        return _synth_frame(sample.input_rgb, gammas, black)

    data = ilut.measure_calibration_samples(samples, capture_fn, reference_region=0.5)

    # 計測数 = サンプル数、撮影回数 = distinct 入力数
    assert len(data.measurements) == len(samples)
    distinct = {tuple(round(v, 6) for v in s.input_rgb) for s in samples}
    assert calls["n"] == len(distinct)

    # グレーは 3ch 記録、単独ランプは 1ch のみ（他は NaN）
    gray_m = next(m for m in data.measurements if m.kind == "gray" and m.input_rgb[0] > 0)
    assert all(not np.isnan(x) for x in gray_m.measured_rgb)
    r_m = next(
        m for m in data.measurements
        if m.kind == "channel" and m.recorded_channels == (0,) and m.input_rgb[0] > 0
    )
    assert not np.isnan(r_m.measured_rgb[0])
    assert np.isnan(r_m.measured_rgb[1]) and np.isnan(r_m.measured_rgb[2])

    # 記録値は合成応答と一致
    v = r_m.input_rgb[0]
    assert abs(r_m.measured_rgb[0] - (black[0] + (0.9 - black[0]) * v ** gammas[0])) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inverse_lut.py::test_measure_records_only_recorded_channels_and_dedups -v`
Expected: FAIL — `AttributeError: module 'src.python.inverse_lut' has no attribute 'measure_calibration_samples'`

- [ ] **Step 3: Add measurement data structures and functions**

`src/python/inverse_lut.py` の `generate_calibration_patterns` の後に追加:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_inverse_lut.py::test_reduce_frame_central_mean tests/test_inverse_lut.py::test_measure_records_only_recorded_channels_and_dedups -v`
Expected: PASS（2 件）

- [ ] **Step 5: Commit**

```bash
git add src/python/inverse_lut.py tests/test_inverse_lut.py
git commit -m "feat(inverse_lut): 撮影フレーム集約と計測記録を追加"
```

---

## Task 4: 撮影オーケストレーション（HDR 仮実装＋投影/カメラ配線）

**Files:**
- Modify: `src/python/inverse_lut.py`
- Test: `tests/test_inverse_lut.py`

**Interfaces:**
- Consumes: `CalibrationSample`, `measure_calibration_samples`（Task 3）, `get_config()`, 既存 `create_camera_backend`, `cv2`。
- Produces:
  - `capture_hdr_linear(capture_once_fn: Callable[[], np.ndarray], *, enabled: bool = False, trust_low: float = 10/255, trust_high: float = 245/255) -> np.ndarray`（現状は単一露光、TODOあり）。
  - `run_calibration_capture(samples, *, config=None, camera_backend=None, show_window: bool = True, window_name: str = "inverse_lut_calibration") -> CalibrationData`。

- [ ] **Step 1: Write the failing test**

`tests/test_inverse_lut.py` に追記:

```python
def test_run_calibration_capture_projects_and_measures(monkeypatch):
    # cv2 のウィンドウ系呼び出しを無効化
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
    data = ilut.run_calibration_capture(
        samples, camera_backend=camera, show_window=True
    )

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inverse_lut.py::test_run_calibration_capture_projects_and_measures -v`
Expected: FAIL — `AttributeError: ... has no attribute 'run_calibration_capture'`

- [ ] **Step 3: Add HDR stub and capture orchestration**

`src/python/inverse_lut.py` の import 群に追加:

```python
import cv2

from src.python.camera import create_camera_backend
```

`measure_calibration_samples` の後に追加:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_inverse_lut.py::test_run_calibration_capture_projects_and_measures tests/test_inverse_lut.py::test_capture_hdr_linear_stub_single_exposure -v`
Expected: PASS（2 件）

- [ ] **Step 5: Commit**

```bash
git add src/python/inverse_lut.py tests/test_inverse_lut.py
git commit -m "feat(inverse_lut): 撮影オーケストレーションとHDR仮実装を追加"
```

---

## Task 5: 逆LUT構築（黒引き・白正規化・単調化・反転）

**Files:**
- Modify: `src/python/inverse_lut.py`
- Test: `tests/test_inverse_lut.py`

**Interfaces:**
- Consumes: `CalibrationData`, `CalibrationMeasurement`（Task 3）, `scipy.interpolate.PchipInterpolator`。
- Produces:
  - `InverseLut`（dataclass）: `lut: np.ndarray (3, R)`（[0,1] float32）, `t_grid: np.ndarray (R,)`, `knots: list`, `meta: dict`。
  - `build_inverse_lut(data: CalibrationData, *, source: str|None=None, lut_resolution: int|None=None, trust_low: float|None=None, trust_high: float|None=None) -> InverseLut`。

- [ ] **Step 1: Write the failing test**

`tests/test_inverse_lut.py` に追記:

```python
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
    # 端点: L(0)=0（黒引き）, L(1)=1
    for c in range(3):
        assert abs(lut.lut[c, 0] - 0.0) < 1e-3
        assert abs(lut.lut[c, -1] - 1.0) < 1e-3
    # 単調非減少
    for c in range(3):
        assert np.all(np.diff(lut.lut[c]) >= -1e-6)
    # L(t) ≈ t^(1/gamma)
    for c, g in enumerate(gammas):
        t = lut.t_grid
        expected = t ** (1.0 / g)
        assert np.max(np.abs(lut.lut[c] - expected)) < 0.05


def test_build_inverse_lut_black_subtraction_maps_zero_to_zero():
    # 0,0,0 でカメラが 10/255 でも L(0)=0
    lut = _build_from_synth((2.2, 2.2, 2.2), (10 / 255, 10 / 255, 10 / 255))
    assert np.all(lut.lut[:, 0] < 1e-3)


def test_build_inverse_lut_excludes_saturated_knots():
    # 手組みデータ: 高入力側を trust_high 超で飽和させる
    measurements = []
    levels = np.linspace(0.0, 1.0, 6)
    for v in levels:
        m = 0.02 + 0.98 * v  # 線形応答
        measured = 0.99 if m >= 0.95 else m  # 上位をクリップ（飽和）
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
    # グレーのみでも各チャンネルが構築される（グレーは 3ch 記録のため）
    assert lut.lut.shape[0] == 3
    assert np.max(np.abs(lut.lut[0] - lut.t_grid ** (1 / 2.2))) < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inverse_lut.py::test_build_inverse_lut_inverts_gamma_per_channel -v`
Expected: FAIL — `AttributeError: ... has no attribute 'build_inverse_lut'`

- [ ] **Step 3: Add build helpers and `build_inverse_lut`**

`src/python/inverse_lut.py` の import 群に追加:

```python
from scipy.interpolate import PchipInterpolator
```

`run_calibration_capture` の後に追加:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_inverse_lut.py -k build_inverse_lut -v`
Expected: PASS（4 件）

- [ ] **Step 5: Commit**

```bash
git add src/python/inverse_lut.py tests/test_inverse_lut.py
git commit -m "feat(inverse_lut): 逆LUT構築（黒引き・正規化・単調化・反転）を追加"
```

---

## Task 6: LUT 適用と保存/読込

**Files:**
- Modify: `src/python/inverse_lut.py`
- Test: `tests/test_inverse_lut.py`

**Interfaces:**
- Consumes: `InverseLut`（Task 5）。
- Produces:
  - `apply_inverse_lut(image: np.ndarray, lut: InverseLut, dtype=np.float32) -> np.ndarray`。
  - `save_inverse_lut(path, lut: InverseLut) -> Path`。
  - `load_inverse_lut(path) -> InverseLut`。

- [ ] **Step 1: Write the failing test**

`tests/test_inverse_lut.py` に追記:

```python
def test_apply_inverse_lut_linearizes_response():
    gammas = (2.2, 2.0, 1.8)
    black = (0.02, 0.02, 0.02)
    lut = _build_from_synth(gammas, black, source="both")

    # リニアなターゲット t を入力 → L(t) を投影 → 応答 g で戻すと t に近い
    t = np.linspace(0, 1, 16, dtype=np.float32)
    target = np.stack([t, t, t], axis=-1).reshape(1, 16, 3)  # (1,16,3)
    projector_input = ilut.apply_inverse_lut(target, lut, dtype=np.float32)
    assert projector_input.shape == target.shape

    for c, g in enumerate(gammas):
        reproduced = projector_input[0, :, c] ** g  # 黒引き後の正規化応答 = p^g
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inverse_lut.py::test_apply_inverse_lut_linearizes_response -v`
Expected: FAIL — `AttributeError: ... has no attribute 'apply_inverse_lut'`

- [ ] **Step 3: Add apply and save/load functions**

`src/python/inverse_lut.py` の `build_inverse_lut` の後に追加:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_inverse_lut.py -v`
Expected: PASS（全件）

- [ ] **Step 5: Commit**

```bash
git add src/python/inverse_lut.py tests/test_inverse_lut.py
git commit -m "feat(inverse_lut): LUT適用と保存/読込を追加"
```

---

## Task 7: CLI スクリプト `build_inverse_lut.py`

**Files:**
- Create: `examples/python/build_inverse_lut.py`
- Test: `tests/test_inverse_lut.py`

**Interfaces:**
- Consumes: すべての公開 API（Task 2-6）, `reload_config`, `split_cli_config_path`。
- Produces: `main(argv: list[str] | None = None) -> None`（`--config`, `--load <npz>`, `--apply <img>` 対応）。

- [ ] **Step 1: Write the failing test**

`tests/test_inverse_lut.py` に追記:

```python
def test_cli_main_builds_and_saves_lut(tmp_path, monkeypatch):
    import importlib.util

    # cv2 ウィンドウ系を無効化
    for name in ("namedWindow", "setWindowProperty", "moveWindow", "imshow", "destroyWindow"):
        monkeypatch.setattr(ilut.cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(ilut.cv2, "waitKey", lambda *a, **k: 0)

    # 合成カメラで build_inverse_lut CLI をロード
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

    class FakeCamera:
        def capture_linear_rgb(self):
            # 直近に imshow された画像を使わず、固定パターンを返すため
            # run_calibration_capture をモックする方が確実
            raise AssertionError("should not be called")

    def fake_run_capture(samples, **kwargs):
        def capture_fn(sample):
            return _synth_frame(sample.input_rgb, gammas, black)
        return ilut.measure_calibration_samples(samples, capture_fn, reference_region=0.5)

    monkeypatch.setattr(module.ilut, "run_calibration_capture", fake_run_capture)

    module.main(["build_inverse_lut.py", "--config", str(config_path)])

    assert out_lut.exists()
    loaded = ilut.load_inverse_lut(out_lut)
    assert loaded.lut.shape == (3, 64)
    ilut_module_reset = reload_config(_REPO_ROOT / "config.toml")  # 既定へ戻す
    assert ilut_module_reset is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inverse_lut.py::test_cli_main_builds_and_saves_lut -v`
Expected: FAIL — `FileNotFoundError`（CLI 未作成）

- [ ] **Step 3: Create `examples/python/build_inverse_lut.py`**

```python
"""Build a measured inverse LUT (projector linearization) from capture.

Generate gray + per-channel calibration ramps, project the raw patterns
and capture their linear response, build a per-channel inverse LUT, and
save it. Optionally apply the LUT to an image for a quick demo.

撮影から測定ベース逆LUT（プロジェクタ線形化）を構築する。

グレー＋各チャンネル単独ランプを生成し、生パターンを投影してリニア応答を
撮影、チャンネル別逆LUTを構築して保存する。任意で画像に適用するデモも行う。

Usage:
    uv run python examples/python/build_inverse_lut.py [--config config.toml]
                                                       [--load lut.npz]
                                                       [--apply image.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.python import inverse_lut as ilut
from src.python.config import get_config, reload_config, split_cli_config_path


def _split_flag(argv: list[str], flag: str) -> tuple[list[str], str | None]:
    """Extract ``--flag <value>`` from argv. Returns (cleaned, value)."""
    cleaned: list[str] = []
    value: str | None = None
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            if i + 1 >= len(argv):
                raise ValueError(f"`{flag}` requires a value.")
            value = argv[i + 1]
            i += 2
            continue
        cleaned.append(argv[i])
        i += 1
    return cleaned, value


def main(argv: list[str] | None = None) -> None:
    """Run the inverse-LUT calibration/build/save workflow.

    逆LUTのキャリブレーション・構築・保存ワークフローを実行する。
    """
    if argv is None:
        argv = sys.argv

    try:
        clean_argv, config_path = split_cli_config_path(argv)
        clean_argv, load_path = _split_flag(clean_argv, "--load")
        clean_argv, apply_path = _split_flag(clean_argv, "--apply")
    except ValueError as exc:
        print(f"Invalid CLI arguments: {exc}")
        raise SystemExit(1)

    if len(clean_argv) != 1:
        print(
            "Usage: python examples/python/build_inverse_lut.py "
            "[--config <config.toml>] [--load <lut.npz>] [--apply <image>]"
        )
        raise SystemExit(1)

    default_config_path = _REPO_ROOT / "config.toml"
    reload_config(config_path if config_path is not None else default_config_path)
    cfg = get_config()
    ilut_cfg = cfg.inverse_lut

    if load_path is not None:
        lut = ilut.load_inverse_lut(load_path)
        print(f"Loaded inverse LUT from: {load_path}")
    else:
        samples = ilut.generate_calibration_patterns(
            cfg.projector.width,
            cfg.projector.height,
            num_levels=ilut_cfg.num_levels,
            dtype=np.uint8,
        )
        print(f"Projecting {len(samples)} calibration patterns...")
        data = ilut.run_calibration_capture(samples)
        lut = ilut.build_inverse_lut(data)
        out_path = ilut.save_inverse_lut(ilut_cfg.inverse_lut_path, lut)
        print(f"Saved inverse LUT to: {out_path}")

    if apply_path is not None:
        bgr = cv2.imread(str(apply_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Failed to load image: {apply_path}")
            raise SystemExit(1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        projector_input = ilut.apply_inverse_lut(rgb, lut, dtype=np.uint8)
        out_image_path = Path(apply_path).with_name(
            Path(apply_path).stem + "_inv_lut.png"
        )
        cv2.imwrite(
            str(out_image_path),
            cv2.cvtColor(projector_input, cv2.COLOR_RGB2BGR),
        )
        print(f"Applied inverse LUT, saved to: {out_image_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_inverse_lut.py -v`
Expected: PASS（全件）

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: 既存テスト＋新規テストがすべて PASS

- [ ] **Step 6: Commit**

```bash
git add examples/python/build_inverse_lut.py tests/test_inverse_lut.py
git commit -m "feat(inverse_lut): CLI build_inverse_lut.py を追加"
```

---

## Self-Review 結果

**1. Spec coverage（spec 各節 → タスク対応）:**
- §1 目的/セマンティクス（測定版逆ガンマ, チャンネル独立, グローバル） → Task 5 の構築ロジック、Task 6 の適用。
- §2 モジュール構成/データフロー → Task 2-7 全体。
- §3 サンプリング（グレー＋単独ランプ, num_levels=17, ~65 distinct） → Task 2。
- §4 撮影/計測/保管方法（生パターン投影・リニア撮影・中央領域平均・記録チャンネル・信頼レンジ・HDR仮実装） → Task 3（記録/集約）, Task 4（投影/HDR）, Task 5（信頼レンジ適用）。
- §5 LUT構築（集約・黒引き・白正規化・単調化・反転, source） → Task 5。
- §6 LUT適用 → Task 6。
- §7 設定/保存 → Task 1（config）, Task 6（save/load）。
- §8 テスト → 各タスクの Test ステップ（黒引き/単調/サチュ除外/保管方法/ソース統合/チャンネル別γ/save-load/枚数/中央平均）。
- §9 非目標 → 実装に含めない（sample.py 非変更、グローバルのみ、HDR仮実装、n³不使用）。
- §10 枚数見積り → Task 2 のテストで 65 を検証。

**2. Placeholder scan:** プレースホルダなし。各コードステップは実コードを含む。HDR は「仮実装（単一露光）＋TODO明記」で spec の意図どおり（欠落ではない）。

**3. Type consistency:** `CalibrationSample`/`CalibrationMeasurement`/`CalibrationData`/`InverseLut` のフィールド名、`generate_calibration_patterns`/`measure_calibration_samples`/`run_calibration_capture`/`build_inverse_lut`/`apply_inverse_lut`/`save_inverse_lut`/`load_inverse_lut` のシグネチャはタスク間で一致。`recorded_channels`, `measured_rgb`, `t_grid`, `lut` の命名も一貫。
