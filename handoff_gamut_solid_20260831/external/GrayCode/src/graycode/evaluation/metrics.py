# coding: utf-8
"""A群 評価指標 (内部・外部パラメータ不要、対応マップ自己整合性ベース)。

較正パラメータを推定していない構造化光 (GrayCode) システムでも、得られた画素対応
だけから「何 px ずれているか」を統計的に評価する 4 手法を実装する。

- A1 :func:`cycle_consistency`      — 往復 (cam→C2P→proj→P2C→cam) 整合性。**自己整合性**。
- A2 :func:`holdout_interpolation_residual` — 補間のホールドアウト交差検証。**補間品質**。
- A3 :func:`known_pattern_error`    — 既知パターンによる**絶対**誤差 (proj 平面)。
- A4 :func:`epipolar_sampson`       — F 行列 + Sampson 距離。**幾何整合性**。外れ値検出も兼ねる。

各指標が何を測り・何を測れないかは ``docs/reprojection_eval_methods.md`` 参照。座標は
すべて pixel-is-point 規約 (整数 = 画素中心、:mod:`graycode.coords`)。

引用: Hartley & Zisserman (Sampson §11.4.3/§12.4), Hartley & Sturm (triangulation),
Moreno & Taubin 2012 (局所ホモグラフィ→特徴の proj 座標), Zhang 2000 (サブピクセル
コーナー), Fischler & Bolles 1981 (RANSAC)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # 補間 (A2) と任意のサンプリングに使用
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.ndimage import map_coordinates

    _SCIPY = True
except ImportError:  # pragma: no cover
    _SCIPY = False

try:
    import cv2

    _CV2 = True
except ImportError:  # pragma: no cover
    _CV2 = False

from . import stats as _stats
from .stats import ErrorStats


# ── 共通ヘルパ ────────────────────────────────────────────────────────


def _to_value_grid(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    """(N,4) フラットマップ or (H,W,>=4) を ``(H, W, 2)`` の値グリッドにする。

    列 [2,3] を「値」とみなす。c2p なら [proj_x, proj_y]、p2c なら [cam_x, cam_y]。
    フラット (N,4) は decode/interpolate の行優先 (y 外、x 内) 順で reshape される。
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3 and a.shape[2] >= 4:
        g = a
    elif a.ndim == 2 and a.shape[1] >= 4:
        if a.shape[0] != height * width:
            raise ValueError(
                f"flat map has {a.shape[0]} rows but H*W={height * width}"
            )
        g = a.reshape(height, width, a.shape[1])
    else:
        raise ValueError(f"unsupported map shape {a.shape}")
    return g[:, :, 2:4]


def sample_map_bilinear(
    value_grid: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    """値グリッド ``(H,W,C)`` を pixel-is-point 連続座標 (xs, ys) で双線形サンプリング。

    整数インデックス i = 画素 i の中心 (pixel-is-point) なので、``map_coordinates`` の
    インデックスをそのまま使える (半画素オフセット不要)。範囲 ``[0,W-1]×[0,H-1]`` 外は
    NaN を返す (``mode="nearest"`` で端の NaN 汚染を避け、後段で自前マスク)。
    """
    if not _SCIPY:
        raise ImportError("scipy is required for bilinear sampling.")
    g = np.asarray(value_grid, dtype=np.float64)
    h, w = g.shape[:2]
    c = g.shape[2] if g.ndim == 3 else 1
    g = g.reshape(h, w, c)
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    idx = np.vstack([ys, xs])
    out = np.empty((xs.size, c), dtype=np.float64)
    for k in range(c):
        out[:, k] = map_coordinates(g[..., k], idx, order=1, mode="nearest")
    valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1) & np.isfinite(xs) & np.isfinite(ys)
    out[~valid] = np.nan
    return out


# ── A1: 往復 (サイクル) 整合性 ────────────────────────────────────────


@dataclass
class CycleResult:
    """A1 の結果。``residual`` は (camH,camW,2) のカメラ平面残差ベクトル (NaN=無効)。"""

    residual: np.ndarray  # (camH, camW, 2) [dx, dy] in camera px
    magnitude: np.ndarray  # (camH, camW)
    valid: np.ndarray  # (camH, camW) bool
    stats: ErrorStats


def cycle_consistency(
    dense_c2p: np.ndarray,
    dense_p2c: np.ndarray,
    cam_size: tuple[int, int],
    proj_size: tuple[int, int],
) -> CycleResult:
    """A1: 往復整合性。``cam →(密C2P)→ proj →(密P2C)→ cam`` の戻り誤差 (カメラ px)。

    各カメラ画素 ``c=(x,y)`` について ``p = C2P(c)``、``c' = P2C(p)`` (双線形) を求め、
    残差 ``c' - c`` を返す。**測れるのは 2 つの密マップ (補間ドメインが異なる) の相互
    整合性**であって、decode 自体の絶対精度ではない (両マップが同一 decode 由来)。
    サニティチェック・補間/反転の不整合検出に用いる。

    Args:
        dense_c2p: (camH*camW,4)=[cam_x,cam_y,proj_x,proj_y] or (camH,camW,4)。
        dense_p2c: (projH*projW,4)=[proj_x,proj_y,cam_x,cam_y] or (projH,projW,4)。
        cam_size: (camH, camW)。
        proj_size: (projH, projW)。
    """
    cam_h, cam_w = cam_size
    proj_h, proj_w = proj_size
    c2p = _to_value_grid(dense_c2p, cam_h, cam_w)  # (camH,camW,2) proj coords
    p2c = _to_value_grid(dense_p2c, proj_h, proj_w)  # (projH,projW,2) cam coords

    proj_x = c2p[..., 0]
    proj_y = c2p[..., 1]
    back = sample_map_bilinear(p2c, proj_x, proj_y).reshape(cam_h, cam_w, 2)

    ys, xs = np.mgrid[0:cam_h, 0:cam_w]
    src = np.stack([xs, ys], axis=-1).astype(np.float64)
    residual = back - src
    magnitude = np.linalg.norm(residual, axis=-1)
    valid = np.isfinite(magnitude)
    return CycleResult(
        residual=residual,
        magnitude=magnitude,
        valid=valid,
        stats=_stats.summarize(magnitude[valid], unit="px(cam)"),
    )


# ── A2: 補間のホールドアウト交差検証 ─────────────────────────────────


@dataclass
class HoldoutResult:
    """A2 の結果。``residual`` は test 点の残差ベクトル (proj px)、凸包外は除外済み。"""

    residual: np.ndarray  # (K,2) [dproj_x, dproj_y]
    test_points: np.ndarray  # (K,2) camera coords of evaluated points
    n_total_test: int
    n_extrapolated: int  # 凸包外 (補間できず NaN) で評価できなかった test 点数
    stats: ErrorStats


def holdout_interpolation_residual(
    raw_c2p: np.ndarray,
    *,
    test_frac: float = 0.1,
    seed: int = 0,
) -> HoldoutResult:
    """A2: 補間品質をホールドアウト交差検証で評価する (proj px)。

    生 decode 点を train/test に分割し、train から ``LinearNDInterpolator`` を構築して
    test 点の proj 座標を予測、真の decode 値との差を残差とする。Delaunay は既知点を
    厳密保存するため「密マップ vs 生 decode」では構造上ほぼ 0 になり無意味。よって
    **未知点での補間/穴埋め誤差**を測るホールドアウトを用いる (本プロジェクトの
    inpaint 監査と同じ問題意識)。凸包外で予測不能 (NaN) な test 点は別カウントし除外。

    Args:
        raw_c2p: (N,4)=[cam_x,cam_y,proj_x,proj_y] の**生 decode** 対応 (補間前)。
        test_frac: テストに回す割合 (0<frac<1)。
        seed: 分割の乱数シード (再現性確保)。
    """
    if not _SCIPY:
        raise ImportError("scipy is required for A2 holdout interpolation.")
    a = np.asarray(raw_c2p, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 4:
        raise ValueError("raw_c2p must be (N,4) [cam_x,cam_y,proj_x,proj_y]")
    a = a[np.isfinite(a[:, :4]).all(axis=1)]
    n = a.shape[0]
    if n < 8:
        raise ValueError("not enough decoded points for holdout (need >= 8)")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    if train_idx.size < 4:
        raise ValueError("not enough training points after split")

    train_pts = a[train_idx, 0:2]
    train_val = a[train_idx, 2:4]
    test_pts = a[test_idx, 0:2]
    test_val = a[test_idx, 2:4]

    lin = LinearNDInterpolator(train_pts, train_val)
    pred = lin(test_pts)  # (n_test, 2), NaN outside train convex hull

    inside = np.isfinite(pred).all(axis=1)
    n_extrap = int((~inside).sum())
    residual = pred[inside] - test_val[inside]
    mag = np.linalg.norm(residual, axis=-1) if residual.size else residual
    return HoldoutResult(
        residual=residual,
        test_points=test_pts[inside],
        n_total_test=int(n_test),
        n_extrapolated=n_extrap,
        stats=_stats.summarize(mag, unit="px(proj)"),
    )


# ── A3: 既知パターンによる絶対誤差 ───────────────────────────────────


@dataclass
class KnownPatternResult:
    """A3 の結果。``residual`` は各特徴の proj 平面残差 (proj px)。``ids`` は特徴 ID。"""

    residual: np.ndarray  # (K,2) [dproj_x, dproj_y]
    ids: np.ndarray  # (K,)
    cam_points: np.ndarray  # (K,2) camera coords (sub-pixel)
    pred_proj: np.ndarray  # (K,2) GrayCode-predicted projector coords
    true_proj: np.ndarray  # (K,2) ground-truth projector coords (by design)
    stats: ErrorStats


def known_pattern_error(
    detected_features: np.ndarray,
    true_proj_coords: dict,
    dense_c2p: np.ndarray,
    cam_size: tuple[int, int],
) -> KnownPatternResult:
    """A3: 既知パターンを使った**絶対**誤差 (proj 平面、px)。

    既知パターン (例 ChArUco/市松) の各特徴は、画像を自作したため **真のプロジェクタ
    座標が設計値として既知**。カメラ画像でサブピクセル検出した特徴位置で GrayCode 対応
    マップを参照し、得られた proj 座標と真値との差を測る。較正不要で、自己整合性 (A1)
    より強い「真値からのずれ」に到達する (上限はカメラ側検出精度、~0.1-0.2px)。

    Moreno & Taubin 2012 が*較正*で使う「特徴のカメラ座標→プロジェクタ座標」変換を
    *評価*に転用したもの。

    Args:
        detected_features: (M,3)=[cam_x, cam_y, feature_id]。カメラでのサブピクセル検出。
        true_proj_coords: {feature_id: (proj_x, proj_y)}。設計上の真のプロジェクタ座標。
        dense_c2p: GrayCode の密 C2P マップ ((camH*camW,4) or (camH,camW,4))。
        cam_size: (camH, camW)。
    """
    feats = np.asarray(detected_features, dtype=np.float64)
    if feats.ndim != 2 or feats.shape[1] < 3:
        raise ValueError("detected_features must be (M,3) [cam_x,cam_y,feature_id]")
    cam_h, cam_w = cam_size
    c2p = _to_value_grid(dense_c2p, cam_h, cam_w)  # (camH,camW,2) proj coords

    rows = []
    for cam_x, cam_y, fid in feats:
        key = int(round(fid))
        if key not in true_proj_coords:
            continue
        tx, ty = true_proj_coords[key]
        rows.append((cam_x, cam_y, key, float(tx), float(ty)))
    if not rows:
        raise ValueError("no detected feature matched true_proj_coords keys")

    cam_pts = np.array([[r[0], r[1]] for r in rows], dtype=np.float64)
    ids = np.array([r[2] for r in rows], dtype=np.int64)
    true_proj = np.array([[r[3], r[4]] for r in rows], dtype=np.float64)

    pred_proj = sample_map_bilinear(c2p, cam_pts[:, 0], cam_pts[:, 1])
    finite = np.isfinite(pred_proj).all(axis=1)

    residual = pred_proj[finite] - true_proj[finite]
    mag = np.linalg.norm(residual, axis=-1) if residual.size else residual
    return KnownPatternResult(
        residual=residual,
        ids=ids[finite],
        cam_points=cam_pts[finite],
        pred_proj=pred_proj[finite],
        true_proj=true_proj[finite],
        stats=_stats.summarize(mag, unit="px(proj)"),
    )


# ── A4: エピポーラ整合性 (F 行列 + Sampson 距離) ──────────────────────


@dataclass
class EpipolarResult:
    """A4 の結果。``sampson`` は全対応の Sampson 距離 (px)。``F`` は推定基礎行列。"""

    F: np.ndarray  # (3,3) fundamental matrix (proj^T F cam = 0)
    sampson: np.ndarray  # (N,) Sampson distance per correspondence [px]
    ransac_inlier_mask: np.ndarray  # (N,) bool, RANSAC inliers (fit subset only marked)
    n_fit: int  # number of points used to fit F
    inlier_ratio: float  # RANSAC inlier ratio over the fit subset
    stats_all: ErrorStats  # stats over all correspondences
    stats_inliers: ErrorStats  # stats over MAD-robust inliers


def sampson_distance(corr: np.ndarray, F: np.ndarray) -> np.ndarray:
    """対応 ``(N,4)=[cam_x,cam_y,proj_x,proj_y]`` と F から Sampson 距離 (px) を返す。

    ``proj^T F cam = 0`` の規約。Sampson 距離は真の再投影誤差の 1 次近似
    (Hartley & Zisserman §11.4.3): ``d = |x2^T F x1| / sqrt((Fx1)_x^2+(Fx1)_y^2
    +(F^T x2)_x^2+(F^T x2)_y^2)``。
    """
    c = np.asarray(corr, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] < 4:
        raise ValueError("corr must be (N,4) [cam_x,cam_y,proj_x,proj_y]")
    x1 = np.column_stack([c[:, 0], c[:, 1], np.ones(len(c))])  # cam
    x2 = np.column_stack([c[:, 2], c[:, 3], np.ones(len(c))])  # proj
    F = np.asarray(F, dtype=np.float64)
    Fx1 = x1 @ F.T  # (N,3) = F x1 per row
    Ftx2 = x2 @ F  # (N,3) = F^T x2 per row
    num = np.einsum("nk,nk->n", x2, Fx1) ** 2  # (x2^T F x1)^2
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.sqrt(num / denom)
    d[~np.isfinite(d)] = np.nan
    return d


def epipolar_sampson(
    correspondences: np.ndarray,
    *,
    ransac_thresh: float = 1.0,
    confidence: float = 0.999,
    max_fit: int = 20000,
    seed: int = 0,
    inlier_k: float = 3.0,
) -> EpipolarResult:
    """A4: 対応から基礎行列 F を RANSAC 推定し、各点の Sampson 距離 (px) を返す。

    較正なしで「幾何的に何 px ずれているか」を測れる、最も学術標準に近い指標。
    エピポーラ幾何に矛盾する対応 (デコード誤り等) が大きな Sampson 距離として現れ、
    外れ値検出を兼ねる。F は up-to-scale なので**絶対スケールは測れない** (相対整合性)。

    Args:
        correspondences: (N,4)=[cam_x,cam_y,proj_x,proj_y]。
        ransac_thresh: cv2 RANSAC のエピポーラ距離閾値 (px)。
        max_fit: F 推定に使う最大点数 (超過分はランダム抽出)。Sampson は全点で評価。
        seed: 抽出の乱数シード。
        inlier_k: stats_inliers 用の MAD 内れ値係数。
    """
    if not _CV2:
        raise ImportError("opencv (cv2) is required for A4 fundamental matrix.")
    c = np.asarray(correspondences, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] < 4:
        raise ValueError("correspondences must be (N,4) [cam_x,cam_y,proj_x,proj_y]")
    c = c[np.isfinite(c[:, :4]).all(axis=1)]
    n = c.shape[0]
    if n < 8:
        raise ValueError("need >= 8 correspondences to estimate F")

    if n > max_fit:
        rng = np.random.default_rng(seed)
        fit_idx = rng.choice(n, size=max_fit, replace=False)
    else:
        fit_idx = np.arange(n)
    cam_fit = np.ascontiguousarray(c[fit_idx, 0:2], dtype=np.float64)
    proj_fit = np.ascontiguousarray(c[fit_idx, 2:4], dtype=np.float64)

    F, mask = cv2.findFundamentalMat(
        cam_fit, proj_fit, cv2.FM_RANSAC, ransac_thresh, confidence
    )
    if F is None or F.shape != (3, 3):
        raise RuntimeError("fundamental matrix estimation failed")

    mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(fit_idx), bool)
    full_mask = np.zeros(n, dtype=bool)
    full_mask[fit_idx[mask]] = True

    sampson = sampson_distance(c, F)
    robust_in = _stats.robust_inlier_mask(sampson, k=inlier_k)
    return EpipolarResult(
        F=F,
        sampson=sampson,
        ransac_inlier_mask=full_mask,
        n_fit=int(len(fit_idx)),
        inlier_ratio=float(mask.mean()) if mask.size else float("nan"),
        stats_all=_stats.summarize(sampson, unit="px"),
        stats_inliers=_stats.summarize(sampson[robust_in], unit="px"),
    )
