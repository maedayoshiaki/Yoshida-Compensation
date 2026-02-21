# coding: utf-8
"""Centralized configuration loader for the Yoshida-Compensation project.

Reads config.toml from the project root, provides typed dataclasses
for each section, and falls back to hardcoded defaults when the file
or individual fields are absent.
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
    gamma: float = 2.1
    width: int = 1920
    height: int = 1080
    pos_x: int = 5360
    pos_y: int = 0


@dataclass(frozen=True)
class CameraConfig:
    av: str = "8"
    tv: str = "1/15"
    iso: str = "400"
    image_quality: str = "LR"
    wait_key_ms: int = 200


@dataclass(frozen=True)
class PathsConfig:
    c2p_map: str = ""
    rpcc_matrix: str = ""
    target_image_dir: str = "data/target_images"
    linear_pattern_dir: str = "data/linear_proj_patterns"
    inv_gamma_pattern_dir: str = "data/inv_gamma_proj_patterns"
    captured_image_dir: str = "data/captured_images"
    compensation_image_dir: str = "data/compensation_images"
    inv_gamma_comp_dir: str = "data/inv_gamma_comp_images"


@dataclass(frozen=True)
class GammaCorrectionConfig:
    default_gamma: float = 2.2


@dataclass(frozen=True)
class CompensationConfig:
    num_divisions: int = 3
    safety_margin: float = 0.5
    min_batch_size: int = 256
    use_gpu: bool = False
    gamma_correction: GammaCorrectionConfig = field(
        default_factory=GammaCorrectionConfig
    )


# ── Top-level config container ───────────────────────────────────────


@dataclass(frozen=True)
class AppConfig:
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    compensation: CompensationConfig = field(default_factory=CompensationConfig)


# ── Loading logic ────────────────────────────────────────────────────

_FRACTION_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*$")


def _parse_number(value: object) -> object:
    """文字列の分数表記 (例: "1/15") を float に変換する。

    int や float はそのまま返す。分数表記でない文字列もそのまま返す。
    """
    if isinstance(value, str):
        m = _FRACTION_RE.match(value)
        if m:
            return float(Fraction(m.group(1)) / Fraction(m.group(2)))
    return value


def _build_section(cls: type, data: dict, key: str):
    """Build a dataclass from a TOML sub-dict, ignoring unknown keys."""
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
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return AppConfig()

    with open(config_path, "rb") as f:
        data = dict(tomllib.load(f))

    # Build compensation section specially (has nested 'gamma_correction')
    comp_data = dict(data.get("compensation", {}))
    gc_data = comp_data.pop("gamma_correction", {})

    comp_valid = {
        f.name for f in CompensationConfig.__dataclass_fields__.values()
    } - {"gamma_correction"}
    comp_filtered = {
        k: _parse_number(v) for k, v in comp_data.items() if k in comp_valid
    }

    gc_valid = {f.name for f in GammaCorrectionConfig.__dataclass_fields__.values()}
    gc_filtered = {k: _parse_number(v) for k, v in gc_data.items() if k in gc_valid}

    comp_cfg = CompensationConfig(
        gamma_correction=GammaCorrectionConfig(**gc_filtered), **comp_filtered
    )

    return AppConfig(
        projector=_build_section(ProjectorConfig, data, "projector"),
        camera=_build_section(CameraConfig, data, "camera"),
        paths=_build_section(PathsConfig, data, "paths"),
        compensation=comp_cfg,
    )


# ── Module-level singleton ───────────────────────────────────────────

_config: Optional[AppConfig] = None


def get_config(config_path: Optional[Path] = None) -> AppConfig:
    """Return the cached AppConfig, loading it on first call."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reload_config(config_path: Optional[Path] = None) -> AppConfig:
    """Force-reload configuration (useful for tests)."""
    global _config
    _config = load_config(config_path)
    return _config
