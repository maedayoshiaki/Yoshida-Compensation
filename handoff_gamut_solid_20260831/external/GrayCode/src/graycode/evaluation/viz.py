# coding: utf-8
"""誤差分布の可視化 (ヒストグラム / ヒートマップ / 誤差ベクトル場)。

matplotlib があれば PNG 図を出力し、無ければヒートマップは cv2 カラーマップで代替、
ヒストグラム/quiver はスキップ (警告) する。いずれの場合も**生の誤差配列 (.npy) と
統計サマリは常に保存**するので、後から任意のツールで作図できる。

文献の可視化慣行: 平面フィット距離のヒストグラム (Lang & Schlegl 2016, 256 ビン)、
誤差ベクトル場は系統的歪み/一様シフトの検出に有効。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL = True
except ImportError:  # pragma: no cover
    _MPL = False

try:
    import cv2

    _CV2 = True
except ImportError:  # pragma: no cover
    _CV2 = False


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_histogram(
    mag: np.ndarray, path: str | Path, bins: int = 256, title: str = "Error histogram"
) -> Optional[Path]:
    """誤差大きさのヒストグラムを PNG 保存。matplotlib 不在なら None。"""
    m = np.asarray(mag, dtype=np.float64).ravel()
    m = m[np.isfinite(m)]
    if not _MPL or m.size == 0:
        return None
    p = _ensure_dir(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(m, bins=bins, color="#3b7dd8", edgecolor="none")
    ax.axvline(np.median(m), color="#d83b3b", lw=1.2, label=f"median={np.median(m):.3f}")
    ax.axvline(np.mean(m), color="#3bd86b", lw=1.2, ls="--", label=f"mean={np.mean(m):.3f}")
    ax.set_xlabel("error [px]")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_heatmap(
    mag_grid: np.ndarray, path: str | Path, title: str = "Error heatmap", vmax: Optional[float] = None
) -> Optional[Path]:
    """画像平面上の誤差大きさをヒートマップ PNG 保存。

    matplotlib があれば colorbar 付き、無ければ cv2 の JET カラーマップで代替保存。
    NaN は黒 (cv2) / 既定色 (mpl) で表示。
    """
    g = np.asarray(mag_grid, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("mag_grid must be 2D (H,W)")
    finite = np.isfinite(g)
    if not finite.any():
        return None
    if vmax is None:
        vmax = float(np.percentile(g[finite], 99))
    p = _ensure_dir(path)
    if _MPL:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(g, cmap="turbo", vmin=0.0, vmax=vmax, origin="upper")
        ax.set_title(title)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        fig.colorbar(im, ax=ax, label="error [px]")
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        return p
    if _CV2:
        norm = np.clip(g / vmax, 0.0, 1.0)
        norm[~finite] = 0.0
        u8 = (norm * 255).astype(np.uint8)
        color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        color[~finite] = (0, 0, 0)
        cv2.imwrite(str(p), color)
        return p
    return None


def plot_quiver(
    residual_grid: np.ndarray,
    path: str | Path,
    step: int = 24,
    scale: Optional[float] = None,
    title: str = "Error vector field",
) -> Optional[Path]:
    """誤差ベクトル場 (quiver) を PNG 保存。系統的歪み/一様シフトの可視化に有効。

    Args:
        residual_grid: (H,W,2) の残差ベクトル ([dx, dy], NaN 可)。
        step: 矢印を間引く格子間隔 (px)。
        scale: matplotlib quiver の scale (None で自動)。
    """
    r = np.asarray(residual_grid, dtype=np.float64)
    if r.ndim != 3 or r.shape[2] < 2:
        raise ValueError("residual_grid must be (H,W,2)")
    if not _MPL:
        return None
    h, w = r.shape[:2]
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    u = r[::step, ::step, 0]
    v = r[::step, ::step, 1]
    finite = np.isfinite(u) & np.isfinite(v)
    if not finite.any():
        return None
    p = _ensure_dir(path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.quiver(
        xs[finite], ys[finite], u[finite], v[finite],
        np.hypot(u[finite], v[finite]),
        cmap="turbo", angles="xy", scale_units="xy",
        scale=scale, width=0.003,
    )
    ax.set_title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # image convention (y down)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def save_error_arrays(
    out_dir: str | Path, name: str, magnitude: np.ndarray, residual: Optional[np.ndarray] = None
) -> dict:
    """生の誤差配列 (.npy) を保存し、保存パスの dict を返す (常に実行可能、依存なし)。"""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    mp = d / f"{name}_magnitude.npy"
    np.save(mp, np.asarray(magnitude))
    paths["magnitude"] = str(mp)
    if residual is not None:
        rp = d / f"{name}_residual.npy"
        np.save(rp, np.asarray(residual))
        paths["residual"] = str(rp)
    return paths


def backend_status() -> dict:
    """利用可能な作図バックエンドを返す (ログ/レポート用)。"""
    return {"matplotlib": _MPL, "cv2": _CV2}
