# coding: utf-8
"""Centralized configuration loader for the Yoshida-Compensation project.

Reads config.toml from the project root, provides typed dataclasses
for each section, and falls back to hardcoded defaults when the file
or individual fields are absent.

Yoshida-Compensation プロジェクトの集中型設定ローダー。

プロジェクトルートの config.toml を読み込み、各セクションに対応する
型付きデータクラスを提供する。ファイルまたは個々のフィールドが存在しない
場合はハードコードされたデフォルト値にフォールバックする。
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional

# Project root: two levels up from src/python/config.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.toml"


# ── Section dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True)
class ProjectorConfig:
    """Configuration for the projector device.

    Attributes:
        gamma: Gamma correction value for the projector display.
            プロジェクタディスプレイのガンマ補正値。
        width: Projector display width in pixels.
            プロジェクタの表示幅（ピクセル）。
        height: Projector display height in pixels.
            プロジェクタの表示高さ（ピクセル）。
        pos_x: Horizontal position of the projector window on the desktop.
            デスクトップ上のプロジェクタウィンドウの水平位置。
        pos_y: Vertical position of the projector window on the desktop.
            デスクトップ上のプロジェクタウィンドウの垂直位置。

    プロジェクタデバイスの設定。
    """

    gamma: float = 2.2
    width: int = 1920
    height: int = 1080
    pos_x: int = 5360
    pos_y: int = 0


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for the camera device.

    Attributes:
        av: Aperture value (F-stop) as a string.
            絞り値（F値）を文字列で指定。
        tv: Shutter speed as a string (e.g., "1/15").
            シャッタースピードを文字列で指定（例: "1/15"）。
        iso: ISO sensitivity as a string.
            ISO感度を文字列で指定。
        image_quality: Image quality setting for the camera (e.g., "LR" for low-res RAW).
            カメラの画質設定（例: "LR" は低解像度RAW）。
        wait_key_ms: Delay in milliseconds between projecting a pattern and capturing.
            パターン投影からキャプチャまでの待機時間（ミリ秒）。

    カメラデバイスの設定。
    """

    av: str = "8"
    tv: str = "1/15"
    iso: str = "400"
    image_quality: str = "LR"
    wait_key_ms: int = 200


@dataclass(frozen=True)
class PathsConfig:
    """File path configuration for input/output directories.

    Attributes:
        c2p_map: Path to the camera-to-projector pixel correspondence map.
            カメラ-プロジェクタ間のピクセル対応マップのパス。
        rpcc_matrix: Path to the RPCC (Root-Polynomial Color Correction) matrix file.
            RPCC（ルート多項式色補正）行列ファイルのパス。
        target_image_dir: Directory containing target images for compensation.
            補償対象の目標画像を格納するディレクトリ。
        linear_pattern_dir: Directory for storing linear projection patterns.
            リニア投影パターンを保存するディレクトリ。
        inv_gamma_pattern_dir: Directory for storing inverse gamma corrected patterns.
            逆ガンマ補正済みパターンを保存するディレクトリ。
        captured_image_dir: Directory for storing captured camera images.
            カメラでキャプチャした画像を保存するディレクトリ。
        compensation_image_dir: Directory for storing computed compensation images.
            計算された補償画像を保存するディレクトリ。
        inv_gamma_comp_dir: Directory for storing inverse gamma corrected compensation images.
            逆ガンマ補正済み補償画像を保存するディレクトリ。

    入出力ディレクトリのファイルパス設定。
    """

    c2p_map: str = ""
    rpcc_matrix: str = ""
    target_image_dir: str = "data/target_images"
    linear_pattern_dir: str = "data/linear_proj_patterns"
    inv_gamma_pattern_dir: str = "data/inv_gamma_proj_patterns"
    captured_image_dir: str = "data/captured_images"
    compensation_image_dir: str = "data/compensation_images"
    inv_gamma_comp_dir: str = "data/inv_gamma_comp_images"


@dataclass(frozen=True)
class CompensationConfig:
    """Configuration for photometric compensation parameters.

    Attributes:
        num_divisions: Number of discrete intensity levels per color channel
            for generating projection patterns. Total patterns = num_divisions^3.
            投影パターン生成時の各色チャネルの離散強度レベル数。
            総パターン数 = num_divisions^3。
        safety_margin: Fraction of available memory to use for batch processing
            (0.0 to 1.0).
            バッチ処理に使用する利用可能メモリの割合（0.0〜1.0）。
        min_batch_size: Minimum number of pixels to process per batch.
            バッチあたりの最小処理ピクセル数。
        use_gpu: Whether to use GPU acceleration for matrix computations.
            行列計算にGPUアクセラレーションを使用するかどうか。

    色補償パラメータの設定。
    """

    num_divisions: int = 3
    safety_margin: float = 0.5
    min_batch_size: int = 256
    use_gpu: bool = False


# ── Top-level config container ───────────────────────────────────────


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration container.

    Aggregates all section-specific configurations into a single
    immutable object.

    Attributes:
        projector: Projector device settings.
            プロジェクタデバイスの設定。
        camera: Camera device settings.
            カメラデバイスの設定。
        paths: Input/output file path settings.
            入出力ファイルパスの設定。
        compensation: Photometric compensation parameters.
            色補償パラメータ。

    アプリケーション全体の設定コンテナ。

    セクションごとの設定を単一の不変オブジェクトに集約する。
    """

    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    compensation: CompensationConfig = field(default_factory=CompensationConfig)


# ── Loading logic ────────────────────────────────────────────────────

_FRACTION_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*$")


def _parse_number(value: object) -> object:
    """Convert a fraction string (e.g., "1/15") to a float.

    Integer and float values are returned as-is. Non-fraction strings
    are also returned unchanged.

    文字列の分数表記（例: "1/15"）を float に変換する。

    int や float はそのまま返す。分数表記でない文字列もそのまま返す。

    Args:
        value: The value to parse. Can be a string, int, or float.
            パースする値。文字列、int、または float。

    Returns:
        The parsed numeric value, or the original value if no conversion
        is applicable.
        パースされた数値、または変換が適用されない場合は元の値。
    """
    if isinstance(value, str):
        m = _FRACTION_RE.match(value)
        if m:
            return float(Fraction(m.group(1)) / Fraction(m.group(2)))
    return value


def _build_section(cls: type, data: dict, key: str):
    """Build a dataclass instance from a TOML sub-dictionary, ignoring unknown keys.

    TOML サブ辞書からデータクラスインスタンスを構築する。
    未知のキーは無視される。

    Args:
        cls: The dataclass type to instantiate.
            インスタンス化するデータクラス型。
        data: The full TOML dictionary.
            TOML 辞書全体。
        key: The section key to extract from *data*.
            *data* から抽出するセクションキー。

    Returns:
        An instance of *cls* populated with values from the TOML section.
        TOML セクションの値で生成された *cls* のインスタンス。
    """
    section = data.get(key, {})
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    filtered = {}
    for k, v in section.items():
        if k not in field_types:
            continue
        # str 型フィールドには分数パースを適用しない
        if field_types[k] == "str" or field_types[k] is str:
            filtered[k] = v
        else:
            filtered[k] = _parse_number(v)
    return cls(**filtered)


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from a TOML file.

    Falls back to compiled-in defaults if the file does not exist
    or if individual fields are absent.

    TOML ファイルから設定を読み込む。

    ファイルが存在しない場合、または個々のフィールドが欠落している場合は
    コンパイル済みデフォルト値にフォールバックする。

    Args:
        config_path: Path to the TOML configuration file. Defaults to
            ``config.toml`` in the project root.
            TOML 設定ファイルのパス。デフォルトはプロジェクトルートの
            ``config.toml``。

    Returns:
        A fully populated :class:`AppConfig` instance.
        完全に設定された :class:`AppConfig` インスタンス。
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return AppConfig()

    with open(config_path, "rb") as f:
        data = dict(tomllib.load(f))

    return AppConfig(
        projector=_build_section(ProjectorConfig, data, "projector"),
        camera=_build_section(CameraConfig, data, "camera"),
        paths=_build_section(PathsConfig, data, "paths"),
        compensation=_build_section(CompensationConfig, data, "compensation"),
    )


# ── Module-level singleton ───────────────────────────────────────────

_config: Optional[AppConfig] = None


def get_config(config_path: Optional[Path] = None) -> AppConfig:
    """Return the cached AppConfig, loading it on first call.

    キャッシュされた AppConfig を返す。初回呼び出し時に読み込みを行う。

    Args:
        config_path: Optional path to the TOML file. Only used on
            the first call when the config is not yet loaded.
            オプションの TOML ファイルパス。設定がまだ読み込まれていない
            初回呼び出し時のみ使用される。

    Returns:
        The singleton :class:`AppConfig` instance.
        シングルトンの :class:`AppConfig` インスタンス。
    """
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reload_config(config_path: Optional[Path] = None) -> AppConfig:
    """Force-reload configuration (useful for tests).

    設定を強制的に再読み込みする（テスト時に便利）。

    Args:
        config_path: Optional path to the TOML file.
            オプションの TOML ファイルパス。

    Returns:
        The newly loaded :class:`AppConfig` instance.
        新しく読み込まれた :class:`AppConfig` インスタンス。
    """
    global _config
    _config = load_config(config_path)
    return _config
