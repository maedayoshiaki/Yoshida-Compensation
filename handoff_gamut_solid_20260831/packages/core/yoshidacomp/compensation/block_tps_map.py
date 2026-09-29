"""Node-grid thin-plate-spline inverse colour maps (Grundhöfer & Iwai style).

Fits, at each node of a spatial grid over the projector's active region, a
scattered 3-D interpolation from measured camera colour ``q`` to projector
input code ``p`` (the *inverse* map, fitted directly as in Grundhöfer &
Iwai, IEEE TIP 2015), and applies it to target images with bilinear
blending between the four surrounding nodes. This replaces BOTH the
per-channel 1-D linearisation and the 4x3 colour mixing matrix of the
classic pipeline with a single nonlinear map that can represent
channel crosstalk and non-additivity.

プロジェクタ有効領域上の空間ノード格子ごとに、計測カメラ色 ``q`` →
プロジェクタ入力コード ``p`` の3D散布補間（逆写像を直接 fit、TIP2015 流）を
構築し、適用時は周囲4ノードの評価を双線形ブレンドする。古典パイプラインの
「チャンネル毎1D線形化 + 4×3行列」を、クロストークと非加法性を表現できる
単一の非線形写像で置き換える。

設計書: docs/superpowers/specs/2026-08-12-color-grid-response-design.md §5
（モデル比較で TPS が行列系の約4倍良かったことを受けた補償側の実装）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Candidate regularisation weights tried by the automatic smoothing search.
# 自動 smoothing 探索で試す正則化重みの候補。
_SMOOTHING_CANDIDATES: Tuple[float, ...] = (0.0, 1e-6, 1e-4, 1e-2)


@dataclass
class BlockTpsMap:
    """A fitted node-grid inverse colour map.

    Attributes:
        node_y / node_x: Node coordinates in projector pixels, shapes
            ``(gy,)`` / ``(gx,)``, spanning the active region inclusively.
            ノード座標（プロジェクタ画素）。有効領域を両端含みで張る。
        codes: Projector input codes of the calibration patterns,
            shape ``(N, 3)``, float in [0, 1].
            校正パターンのプロジェクタ入力コード（(N,3), [0,1]）。
        node_samples: Measured camera colours per node,
            shape ``(gy, gx, N, 3)``.
            ノードごとの計測カメラ色（(gy,gx,N,3)）。
        smoothing: TPS regularisation weight actually used.
            実際に使った TPS 正則化重み。
        active_region: ``(x, y, width, height)`` the map covers.
            写像が張る有効領域。
    """

    node_y: np.ndarray
    node_x: np.ndarray
    codes: np.ndarray
    node_samples: np.ndarray
    smoothing: float
    active_region: Tuple[int, int, int, int]
    # Fitted per-node interpolators, filled by fit_block_tps_map. Not part of
    # the serialized state; save/load refits from node_samples.
    # ノード別の fit 済み補間器。直列化対象外（save/load は再フィット）。
    _interpolators: List[List[object]] = field(default_factory=list, repr=False)


def node_grid(
    active_region: Tuple[int, int, int, int], grid: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Node coordinates spanning the active region inclusively.

    有効領域を両端含みで張るノード座標を返す。
    """
    x, y, width, height = (int(v) for v in active_region)
    gy, gx = (int(v) for v in grid)
    if gy < 2 or gx < 2:
        raise ValueError(f"grid must be at least 2x2, got {grid}.")
    return (
        np.linspace(y, y + height - 1, gy),
        np.linspace(x, x + width - 1, gx),
    )


def extract_node_samples(
    frames: Sequence[np.ndarray],
    node_y: np.ndarray,
    node_x: np.ndarray,
    window: int = 15,
) -> np.ndarray:
    """Average a small window around every node in every frame.

    各フレームの各ノード周辺 ``window`` 画素の平均をとり
    ``(gy, gx, N, 3)`` を返す。フレームはプロジェクタ座標のリニアRGB。
    """
    half = max(1, int(window)) // 2
    gy, gx = len(node_y), len(node_x)
    samples = np.empty((gy, gx, len(frames), 3), dtype=np.float64)
    for n, frame in enumerate(frames):
        frame = np.asarray(frame, dtype=np.float32)
        height, width = frame.shape[:2]
        for iy, ny in enumerate(node_y):
            y0 = max(0, int(round(ny)) - half)
            y1 = min(height, int(round(ny)) + half + 1)
            for ix, nx in enumerate(node_x):
                x0 = max(0, int(round(nx)) - half)
                x1 = min(width, int(round(nx)) + half + 1)
                patch = frame[y0:y1, x0:x1, :3]
                samples[iy, ix, n] = patch.reshape(-1, 3).mean(axis=0)
    return samples


def _fit_one(
    samples_q: np.ndarray, codes: np.ndarray, smoothing: float
):
    """Fit one inverse TPS ``q -> p`` on a node's samples."""
    from scipy.interpolate import RBFInterpolator

    return RBFInterpolator(
        np.asarray(samples_q, dtype=np.float64),
        np.asarray(codes, dtype=np.float64),
        kernel="thin_plate_spline",
        smoothing=smoothing,
    )


def _loocv_error(
    samples_q: np.ndarray, codes: np.ndarray, smoothing: float
) -> float:
    """Leave-one-out RMSE of a node fit (used by the smoothing search)."""
    n = len(codes)
    errors = np.empty((n, 3), dtype=np.float64)
    index = np.arange(n)
    for i in range(n):
        keep = index != i
        model = _fit_one(samples_q[keep], codes[keep], smoothing)
        errors[i] = model(samples_q[i : i + 1])[0] - codes[i]
    return float(np.sqrt(np.mean(errors**2)))


def select_smoothing(
    node_samples: np.ndarray,
    codes: np.ndarray,
    candidates: Sequence[float] = _SMOOTHING_CANDIDATES,
    probe_nodes: int = 5,
    seed: int = 0,
) -> float:
    """Pick the smoothing with the lowest LOOCV error on a few probe nodes.

    数ノードの leave-one-out 誤差が最小になる正則化重みを選ぶ。
    8bit量子化ノイズを丸暗記しない程度の正則化を自動で決めるため。
    """
    gy, gx = node_samples.shape[:2]
    rng = np.random.default_rng(seed)
    flat = rng.choice(gy * gx, size=min(probe_nodes, gy * gx), replace=False)
    scores = []
    for smoothing in candidates:
        total = 0.0
        for node in flat:
            iy, ix = divmod(int(node), gx)
            total += _loocv_error(node_samples[iy, ix], codes, smoothing)
        scores.append(total / len(flat))
    return float(candidates[int(np.argmin(scores))])


def fit_block_tps_map(
    node_samples: np.ndarray,
    codes: np.ndarray,
    node_y: np.ndarray,
    node_x: np.ndarray,
    active_region: Tuple[int, int, int, int],
    smoothing: Optional[float] = None,
) -> BlockTpsMap:
    """Fit the inverse TPS at every node.

    全ノードで逆TPSを fit する。``smoothing=None`` は LOOCV による自動選択。
    """
    codes = np.asarray(codes, dtype=np.float64)
    if smoothing is None:
        smoothing = select_smoothing(node_samples, codes)
    gy, gx = node_samples.shape[:2]
    interpolators: List[List[object]] = []
    for iy in range(gy):
        row: List[object] = []
        for ix in range(gx):
            row.append(_fit_one(node_samples[iy, ix], codes, smoothing))
        interpolators.append(row)
    fitted = BlockTpsMap(
        node_y=np.asarray(node_y, dtype=np.float64),
        node_x=np.asarray(node_x, dtype=np.float64),
        codes=codes,
        node_samples=np.asarray(node_samples, dtype=np.float64),
        smoothing=float(smoothing),
        active_region=tuple(int(v) for v in active_region),
    )
    fitted._interpolators = interpolators
    return fitted


def apply_block_tps_map(
    tps_map: BlockTpsMap,
    target_linear: np.ndarray,
    clamp_inputs: bool = True,
) -> np.ndarray:
    """Map a target image (desired camera colours) to projector codes.

    目標画像（望むカメラ色、float [0,1]、プロジェクタ座標フルフレーム）を
    プロジェクタ入力コード（float32 [0,1]）へ写す。有効領域の外は 0。
    セルごとに周囲4ノードの逆TPSを評価し双線形ブレンドする。

    ``clamp_inputs=True``（既定）は、各ノードの評価前に目標色をそのノードの
    実測サンプル範囲（軸平行ボックス）へクランプする。散布補間は観測域の外で
    外挿が暴れ、ノードごとに違う方向へ暴れたものをブレンドすると斑点状の
    アーティファクトになる（2026-08-12 実測: nature 系ターゲットで緑の斑）。
    色域外の目標はどのみち到達不能なので、観測域境界での逆像に丸める。
    """
    target = np.asarray(target_linear, dtype=np.float64)
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError(
            f"target_linear must have shape (H, W, 3), got {target.shape}."
        )
    height, width = target.shape[:2]
    output = np.zeros((height, width, 3), dtype=np.float32)
    x0, y0, region_w, region_h = tps_map.active_region
    y_lo, y_hi = y0, min(y0 + region_h, height)
    x_lo, x_hi = x0, min(x0 + region_w, width)
    if y_lo >= y_hi or x_lo >= x_hi:
        return output

    node_y, node_x = tps_map.node_y, tps_map.node_x
    interp = tps_map._interpolators
    if not interp:
        raise ValueError(
            "BlockTpsMap has no fitted interpolators; build it with "
            "fit_block_tps_map (save/load must refit)."
        )

    ys = np.arange(y_lo, y_hi)
    xs = np.arange(x_lo, x_hi)
    # Cell index and fractional position of every row/column between nodes.
    iy_cell = np.clip(np.searchsorted(node_y, ys, side="right") - 1, 0, len(node_y) - 2)
    ix_cell = np.clip(np.searchsorted(node_x, xs, side="right") - 1, 0, len(node_x) - 2)
    ty = (ys - node_y[iy_cell]) / (node_y[iy_cell + 1] - node_y[iy_cell])
    tx = (xs - node_x[ix_cell]) / (node_x[ix_cell + 1] - node_x[ix_cell])
    ty = np.clip(ty, 0.0, 1.0)
    tx = np.clip(tx, 0.0, 1.0)

    sample_lo = tps_map.node_samples.min(axis=2)  # (gy, gx, 3)
    sample_hi = tps_map.node_samples.max(axis=2)

    for cy in range(len(node_y) - 1):
        row_mask = iy_cell == cy
        if not np.any(row_mask):
            continue
        rows = ys[row_mask]
        wy = ty[row_mask][:, None]
        for cx in range(len(node_x) - 1):
            col_mask = ix_cell == cx
            if not np.any(col_mask):
                continue
            cols = xs[col_mask]
            wx = tx[col_mask][None, :]
            block = target[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
            flat = block.reshape(-1, 3)

            def eval_corner(iy_node: int, ix_node: int) -> np.ndarray:
                colours = flat
                if clamp_inputs:
                    colours = np.clip(
                        flat,
                        sample_lo[iy_node, ix_node],
                        sample_hi[iy_node, ix_node],
                    )
                return interp[iy_node][ix_node](colours)

            corners = [
                eval_corner(cy, cx),
                eval_corner(cy, cx + 1),
                eval_corner(cy + 1, cx),
                eval_corner(cy + 1, cx + 1),
            ]
            weight_tl = ((1 - wy) * (1 - wx)).reshape(-1, 1)
            weight_tr = ((1 - wy) * wx).reshape(-1, 1)
            weight_bl = (wy * (1 - wx)).reshape(-1, 1)
            weight_br = (wy * wx).reshape(-1, 1)
            blended = (
                corners[0] * weight_tl
                + corners[1] * weight_tr
                + corners[2] * weight_bl
                + corners[3] * weight_br
            )
            output[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1] = np.clip(
                blended, 0.0, 1.0
            ).reshape(block.shape).astype(np.float32)
    return output


def save_block_tps_map(path, tps_map: BlockTpsMap):
    """Persist the map's samples (interpolators are refit on load)."""
    np.savez_compressed(
        path,
        node_y=tps_map.node_y,
        node_x=tps_map.node_x,
        codes=tps_map.codes,
        node_samples=tps_map.node_samples,
        smoothing=np.float64(tps_map.smoothing),
        active_region=np.asarray(tps_map.active_region, dtype=np.int64),
    )
    return path


def load_block_tps_map(path) -> BlockTpsMap:
    """Load a saved map and refit its interpolators."""
    data = np.load(path)
    return fit_block_tps_map(
        node_samples=data["node_samples"],
        codes=data["codes"],
        node_y=data["node_y"],
        node_x=data["node_x"],
        active_region=tuple(int(v) for v in data["active_region"]),
        smoothing=float(data["smoothing"]),
    )
