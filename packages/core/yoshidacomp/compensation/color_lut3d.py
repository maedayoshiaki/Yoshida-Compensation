"""Baked 3-D colour LUTs: fit a scattered inverse map once, apply fast.

A :class:`ColorLut3d` is a regular N^3 lattice over the normalized colour
cube [0, 1]^3 whose entries are projector input codes. It is baked from a
scattered thin-plate-spline inverse map (measured camera colour ->
projector code, Grundhöfer & Iwai style) and applied with trilinear
interpolation -- the 3-D analogue of the pipeline's per-channel 1-D
inverse LUT, and drop-in for the same pipeline position.

正規化色立方体 [0,1]³ 上の N³ 格子にプロジェクタ入力コードを焼き込んだ
3D LUT。散布TPS逆写像（計測カメラ色→投影コード）から一度だけ焼き、
適用は三線形補間で行う。チャンネル毎 1D 逆LUTの3D版であり、
パイプラインの同じ位置に差し替えて使う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class ColorLut3d:
    """A baked 3-D LUT over the normalized colour cube.

    Attributes:
        lattice: ``(N, N, N, 3)`` projector codes; axis order is
            ``lattice[i_r, i_g, i_b]`` for input ``(r, g, b)``.
            格子値（投影コード）。``lattice[i_r, i_g, i_b]`` が入力
            ``(r, g, b)`` に対応する。
        meta: Build metadata (source, smoothing, input clamp box, ...).
            構築メタデータ。
    """

    lattice: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.lattice.shape[0])


def bake_inverse_lut3d(
    measured: np.ndarray,
    codes: np.ndarray,
    size: int = 33,
    smoothing: float = 1e-4,
    clamp_box: bool = True,
) -> ColorLut3d:
    """Bake a 3-D LUT from scattered (measured colour, projector code) pairs.

    正規化済みの計測色 ``measured`` (N,3) と投影コード ``codes`` (N,3) の
    散布対から、逆写像 TPS を fit して ``size³`` 格子に焼く。
    ``clamp_box=True`` は格子点を観測色の軸平行ボックスへクランプしてから
    評価する（観測域外での散布補間の外挿は信用できない —
    docs/color_grid_response_20260812.md §7.3）。
    """
    from scipy.interpolate import RBFInterpolator

    measured = np.asarray(measured, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.float64)
    inverse = RBFInterpolator(
        measured, codes, kernel="thin_plate_spline", smoothing=smoothing
    )
    axis = np.linspace(0.0, 1.0, int(size))
    rr, gg, bb = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)
    lo = measured.min(axis=0)
    hi = measured.max(axis=0)
    if clamp_box:
        points = np.clip(points, lo, hi)
    lattice = np.clip(inverse(points), 0.0, 1.0).reshape(size, size, size, 3)
    return ColorLut3d(
        lattice=lattice.astype(np.float32),
        meta={
            "size": int(size),
            "smoothing": float(smoothing),
            "clamp_box": bool(clamp_box),
            "input_lo": [float(v) for v in lo],
            "input_hi": [float(v) for v in hi],
            "n_samples": int(len(codes)),
        },
    )


def bake_inverse_lut3d_linear(
    measured: np.ndarray,
    codes: np.ndarray,
    size: int = 33,
) -> ColorLut3d:
    """Bake a 3-D LUT with piecewise-LINEAR scattered interpolation.

    TPS の代わりに区分線形補間で焼く: 計測色の Delaunay 四面体分割上の
    重心座標補間（``LinearNDInterpolator``）。標本点は厳密に通り、
    平滑化パラメータは無い。凸包の外の格子点は最近傍値で埋める
    （観測域外の外挿はしない、という方針は TPS 版のクランプと同じ）。
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    measured = np.asarray(measured, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.float64)
    linear = LinearNDInterpolator(measured, codes)
    nearest = NearestNDInterpolator(measured, codes)
    axis = np.linspace(0.0, 1.0, int(size))
    rr, gg, bb = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)
    values = linear(points)
    outside = np.any(~np.isfinite(values), axis=1)
    if np.any(outside):
        values[outside] = nearest(points[outside])
    lattice = np.clip(values, 0.0, 1.0).reshape(size, size, size, 3)
    return ColorLut3d(
        lattice=lattice.astype(np.float32),
        meta={
            "size": int(size),
            "method": "piecewise_linear",
            "outside_hull_fraction": float(np.mean(outside)),
            "input_lo": [float(v) for v in measured.min(axis=0)],
            "input_hi": [float(v) for v in measured.max(axis=0)],
            "n_samples": int(len(codes)),
        },
    )


def apply_lut3d(
    image: np.ndarray, lut: ColorLut3d, dtype: np.typing.DTypeLike = np.float32
) -> np.ndarray:
    """Apply a baked 3-D LUT to an image with trilinear interpolation.

    画像 (H, W, 3)（uint8/uint16 は [0,1] に正規化、float はそのまま
    [0,1] にクリップ）へ三線形補間で 3D LUT を適用する。
    ``apply_inverse_lut`` と同じ入出力契約（uint8/uint16 はスケール）。
    """
    array = np.asarray(image)
    if array.ndim < 2 or array.shape[-1] != 3:
        raise ValueError(f"image must have shape (..., 3), got {array.shape}.")
    if array.dtype == np.uint8:
        norm = array.astype(np.float32) / 255.0
    elif array.dtype == np.uint16:
        norm = array.astype(np.float32) / 65535.0
    else:
        norm = np.clip(array.astype(np.float32), 0.0, 1.0)

    size = lut.size
    lattice = lut.lattice
    scaled = norm * (size - 1)
    idx = np.minimum(scaled.astype(np.int64), size - 2)
    frac = (scaled - idx).astype(np.float32)

    ir, ig, ib = idx[..., 0], idx[..., 1], idx[..., 2]
    fr = frac[..., 0:1]
    fg = frac[..., 1:2]
    fb = frac[..., 2:3]

    def corner(dr: int, dg: int, db: int) -> np.ndarray:
        return lattice[ir + dr, ig + dg, ib + db]

    c00 = corner(0, 0, 0) * (1 - fr) + corner(1, 0, 0) * fr
    c01 = corner(0, 0, 1) * (1 - fr) + corner(1, 0, 1) * fr
    c10 = corner(0, 1, 0) * (1 - fr) + corner(1, 1, 0) * fr
    c11 = corner(0, 1, 1) * (1 - fr) + corner(1, 1, 1) * fr
    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg
    out = c0 * (1 - fb) + c1 * fb
    out = np.clip(out, 0.0, 1.0)

    if dtype == np.uint8:
        return np.floor(out * 255.0 + 0.5).astype(np.uint8)
    if dtype == np.uint16:
        return np.floor(out * 65535.0 + 0.5).astype(np.uint16)
    return out.astype(dtype)


def save_lut3d(path, lut: ColorLut3d):
    """Save a baked LUT to .npz. 焼き込み済みLUTを保存する。"""
    import json as _json

    np.savez_compressed(
        path, lattice=lut.lattice, meta=_json.dumps(lut.meta)
    )
    return path


def load_lut3d(path) -> ColorLut3d:
    """Load a baked LUT from .npz. 保存済みLUTを読み込む。"""
    import json as _json

    data = np.load(path, allow_pickle=False)
    meta = _json.loads(str(data["meta"]))
    return ColorLut3d(lattice=np.asarray(data["lattice"]), meta=meta)


def normalize_reference_response(
    mean_responses: np.ndarray, codes: np.ndarray
) -> np.ndarray:
    """Black-subtract / white-normalize per-channel reference responses.

    基準応答（各校正色の空間平均カメラ色 (N,3)）を黒引き・白正規化して
    「線形意図色 c」に写す。codes==0 の行を黒、codes==1 の行を白とする。
    行列 fit の目標（t_k）と 3D LUT の入力域はこの c で揃える。
    """
    mean_responses = np.asarray(mean_responses, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.float64)
    black_rows = np.all(codes == 0.0, axis=1)
    white_rows = np.all(codes == 1.0, axis=1)
    if not black_rows.any() or not white_rows.any():
        raise ValueError(
            "normalize_reference_response requires black (0,0,0) and white "
            "(1,1,1) calibration samples."
        )
    black = mean_responses[black_rows].mean(axis=0)
    white = mean_responses[white_rows].mean(axis=0)
    denom = np.where(np.abs(white - black) < 1e-12, 1.0, white - black)
    return (mean_responses - black) / denom
