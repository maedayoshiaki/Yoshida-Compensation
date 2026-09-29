# coding: utf-8
"""graycode.evaluation — 較正パラメータ不要の対応マップ自己整合性ベース評価 (A群)。

GrayCode で得たカメラ–プロジェクタ画素対応だけから「何 px ずれているか」を統計解析する。
内部・外部パラメータを推定していない構造化光システム向け。

指標 (:mod:`graycode.evaluation.metrics`):
  - A1 :func:`cycle_consistency`               往復整合性 (自己整合性)
  - A2 :func:`holdout_interpolation_residual`  補間ホールドアウト残差 (補間品質)
  - A3 :func:`known_pattern_error`             既知パターン絶対誤差
  - A4 :func:`epipolar_sampson`                F 行列 + Sampson 距離 (幾何整合性)

統計 (:mod:`~graycode.evaluation.stats`)、可視化 (:mod:`~graycode.evaluation.viz`)、
既知パターン (:mod:`~graycode.evaluation.patterns`)、レポート (:mod:`~graycode.evaluation.report`)。

手法・引用の詳細は ``docs/reprojection_eval_methods.md``。
"""
from . import io, metrics, patterns, project, report, stats, viz
from .metrics import (
    CycleResult,
    EpipolarResult,
    HoldoutResult,
    KnownPatternResult,
    cycle_consistency,
    epipolar_sampson,
    holdout_interpolation_residual,
    known_pattern_error,
    sampson_distance,
)
from .stats import ErrorStats, summarize

__all__ = [
    "io",
    "metrics",
    "patterns",
    "project",
    "report",
    "stats",
    "viz",
    "cycle_consistency",
    "holdout_interpolation_residual",
    "known_pattern_error",
    "epipolar_sampson",
    "sampson_distance",
    "CycleResult",
    "HoldoutResult",
    "KnownPatternResult",
    "EpipolarResult",
    "ErrorStats",
    "summarize",
]
