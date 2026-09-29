# coding: utf-8
"""評価指標の統計集約とロバスト統計 (A群 評価の共通基盤)。

再投影 / 整合性誤差 (画素単位) を統計要約する。文献の教訓 (Lang & Schlegl 2016,
Moreno & Taubin 2012) に従い、単一の RMSE で終わらせず **平均±標準偏差・中央値・
分位点・最大値・ロバスト指標 (MAD)** を併せて返す。

引用・背景は ``docs/reprojection_eval_methods.md`` / ``docs/reprojection_error_survey.md``。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def magnitudes(residual_vectors: np.ndarray) -> np.ndarray:
    """残差ベクトル ``(..., D)`` → ユークリッドノルム ``(...,)``。

    1 次元入力 (既にスカラ誤差) はそのまま絶対値を返す。
    """
    a = np.asarray(residual_vectors, dtype=np.float64)
    if a.ndim == 1:
        return np.abs(a)
    return np.linalg.norm(a, axis=-1)


def mad(x: np.ndarray) -> float:
    """中央絶対偏差 ``median(|x - median(x)|)``。外れ値に頑健な散らばり指標。"""
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    med = np.median(a)
    return float(np.median(np.abs(a - med)))


def mad_std(x: np.ndarray) -> float:
    """正規分布で標準偏差に一致するようスケールした MAD (``1.4826 * MAD``)。

    標準偏差の外れ値頑健な代替。デコード誤りなど裾の重い誤差分布で有効。
    """
    return float(1.4826 * mad(x))


def robust_inlier_mask(mag: np.ndarray, k: float = 3.0) -> np.ndarray:
    """MAD ベースの内れ値マスク: ``|x - median| <= k * mad_std``。

    RANSAC を使わない外れ値除去 (デコード誤り等) の既定手段。``k=3`` は約 3σ 相当。
    ``mad_std`` が 0 (全点同値) や NaN の場合は有限な全点を内れ値扱いにする。
    """
    a = np.asarray(mag, dtype=np.float64).ravel()
    finite = np.isfinite(a)
    med = np.median(a[finite]) if finite.any() else np.nan
    s = mad_std(a)
    if not np.isfinite(s) or s == 0.0:
        return finite
    return finite & (np.abs(a - med) <= k * s)


def huber_mean(x: np.ndarray, delta: float | None = None, iters: int = 50) -> float:
    """Huber M 推定による頑健な位置 (location) を IRLS で求める。

    ``delta=None`` のとき ``1.345 * mad_std`` (正規分布で 95% 効率) を用いる。
    二乗損失 (平均) と絶対損失 (中央値) の中間で、外れ値の影響を抑えつつ効率を保つ。
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    if delta is None:
        delta = 1.345 * mad_std(a)
    mu = float(np.median(a))
    if not np.isfinite(delta) or delta == 0.0:
        return mu
    for _ in range(iters):
        r = a - mu
        w = np.ones_like(r)
        big = np.abs(r) > delta
        w[big] = delta / np.abs(r[big])
        new_mu = float(np.sum(w * a) / np.sum(w))
        if abs(new_mu - mu) < 1e-12:
            mu = new_mu
            break
        mu = new_mu
    return mu


@dataclass(frozen=True)
class ErrorStats:
    """誤差大きさ (画素) の統計要約。``unit`` は単位ラベル (既定 "px")。"""

    n: int
    rmse: float
    mean: float
    std: float
    median: float
    p90: float
    p95: float
    p99: float
    max: float
    mad: float
    mad_std: float
    unit: str = "px"

    def to_dict(self) -> dict:
        return asdict(self)


def _nan_stats(unit: str) -> "ErrorStats":
    nan = float("nan")
    return ErrorStats(0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, unit)


def summarize(mag: np.ndarray, unit: str = "px") -> ErrorStats:
    """誤差大きさ配列 → :class:`ErrorStats`。NaN/inf は除外して集計する。

    入力が残差ベクトル ``(..., D)`` の場合は :func:`magnitudes` で大きさに変換してから渡す。
    """
    m = np.asarray(mag, dtype=np.float64).ravel()
    m = m[np.isfinite(m)]
    if m.size == 0:
        return _nan_stats(unit)
    return ErrorStats(
        n=int(m.size),
        rmse=float(np.sqrt(np.mean(m**2))),
        mean=float(np.mean(m)),
        std=float(np.std(m)),
        median=float(np.median(m)),
        p90=float(np.percentile(m, 90)),
        p95=float(np.percentile(m, 95)),
        p99=float(np.percentile(m, 99)),
        max=float(np.max(m)),
        mad=mad(m),
        mad_std=mad_std(m),
        unit=unit,
    )
