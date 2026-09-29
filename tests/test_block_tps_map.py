"""Tests for yoshidacomp.compensation.block_tps_map (hardware-independent).

合成の空間変動 + 非加法応答で、ノード格子TPS逆写像の fit/apply を検査する。
「正しい写像なら復元できる」と「線形逆写像では復元できない」の両方向を
必ず検査する（AGENTS.md 第8章）。
"""

from __future__ import annotations

import numpy as np
import pytest

from yoshidacomp.compensation import block_tps_map as btm


ACTIVE = (8, 6, 64, 48)  # (x, y, w, h) inside a 60x80 frame
FRAME_H, FRAME_W = 60, 80

V = np.array(
    [
        [0.55, 0.10, 0.03],
        [0.12, 0.60, 0.07],
        [0.04, 0.08, 0.45],
    ]
).T
F = np.array([0.012, 0.018, 0.010])


def _gain(height: int, width: int) -> np.ndarray:
    """Smooth per-channel spatial gain field (H, W, 3)."""
    yy, xx = np.mgrid[0:height, 0:width]
    base = 0.7 + 0.3 * (xx / max(1, width - 1))
    tilt = 1.0 + 0.1 * (yy / max(1, height - 1))
    return np.stack([base * tilt, base, base / tilt], axis=-1)


def _forward(codes: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Non-additive forward model C(P; x, y) for solid code frames.

    codes: (..., 3) in [0,1]. gain: (H, W, 3). Returns (..., H, W, 3)?
    ここでは単色フレーム用: codes (3,) -> (H, W, 3)。
    """
    g = np.asarray(codes, dtype=np.float64) ** 2.2
    cross = 0.25 * np.array(
        [g[0] * g[1], g[0] * g[2], g[1] * g[2]]
    ).sum()
    base = g @ V + F - 0.15 * cross
    return gain * base[None, None, :]


def _forward_pixelwise(code_image: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Pixel-wise forward model for a code image (H, W, 3)."""
    g = np.asarray(code_image, dtype=np.float64) ** 2.2
    cross = 0.25 * (
        g[..., 0] * g[..., 1] + g[..., 0] * g[..., 2] + g[..., 1] * g[..., 2]
    )
    base = g @ V + F - 0.15 * cross[..., None]
    return gain * base


@pytest.fixture(scope="module")
def fitted_map():
    # 実運用と同じ 6³=216 色の校正セット（4³ では逆写像の補間誤差が
    # 実データでも合成でも大きいことを 2026-08-12 に実測済み）。
    gain = _gain(FRAME_H, FRAME_W)
    levels = np.linspace(0.0, 1.0, 6)
    codes = np.array(
        [(r, g, b) for r in levels for g in levels for b in levels]
    )
    frames = [_forward(code, gain) for code in codes]
    node_y, node_x = btm.node_grid(ACTIVE, (5, 5))
    samples = btm.extract_node_samples(frames, node_y, node_x, window=3)
    tps_map = btm.fit_block_tps_map(
        samples, codes, node_y, node_x, ACTIVE, smoothing=0.0
    )
    return tps_map, gain, codes, samples


def test_node_grid_spans_active_region():
    """ノードが有効領域の両端をちょうど覆う。

    落ちる誤実装: off-by-one で端のセルが外挿になる。
    """
    node_y, node_x = btm.node_grid(ACTIVE, (5, 7))
    assert node_x[0] == ACTIVE[0] and node_x[-1] == ACTIVE[0] + ACTIVE[2] - 1
    assert node_y[0] == ACTIVE[1] and node_y[-1] == ACTIVE[1] + ACTIVE[3] - 1
    assert len(node_y) == 5 and len(node_x) == 7


def test_tps_recovers_codes_from_achievable_targets(fitted_map):
    """達成可能な目標色なら元のコードをほぼ復元する。

    落ちる誤実装: 順写像を fit している（逆になっていない）、ノードの
    双線形ブレンドの重み取り違え（空間勾配で復元誤差が跳ねる）。
    """
    tps_map, gain, _, _ = fitted_map
    x, y, w, h = ACTIVE
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
    truth = np.stack(
        [
            0.15 + 0.6 * (xx / FRAME_W),
            0.20 + 0.5 * (yy / FRAME_H),
            0.55 - 0.3 * (xx / FRAME_W),
        ],
        axis=-1,
    )
    target = _forward_pixelwise(truth, gain)
    recovered = btm.apply_block_tps_map(tps_map, target)
    region = np.s_[y : y + h, x : x + w]
    err = recovered[region] - truth[region]
    rmse = float(np.sqrt(np.mean(err**2)))
    assert rmse < 0.03
    assert float(np.max(np.abs(err))) < 0.10


def test_linear_inverse_cannot_recover_non_additive_codes(fitted_map):
    """同じデータの線形逆写像（[q,1]→p 最小二乗）では復元誤差が大きい。

    落ちる誤実装: TPS が実は線形回帰（両者の誤差が同水準になり、
    上のテストの閾値も本テストの比も破れる）。
    """
    tps_map, gain, codes, samples = fitted_map
    x, y, w, h = ACTIVE
    # 中央ノードの線形逆写像
    iy, ix = 2, 2
    q = samples[iy, ix]
    design = np.concatenate([q, np.ones((len(q), 1))], axis=1)
    weights, *_ = np.linalg.lstsq(design, codes, rcond=None)

    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
    truth = np.stack(
        [
            0.15 + 0.6 * (xx / FRAME_W),
            0.20 + 0.5 * (yy / FRAME_H),
            0.55 - 0.3 * (xx / FRAME_W),
        ],
        axis=-1,
    )
    target = _forward_pixelwise(truth, gain)
    region = np.s_[y : y + h, x : x + w]
    flat = target[region].reshape(-1, 3)
    linear_rec = np.concatenate(
        [flat, np.ones((len(flat), 1))], axis=1
    ) @ weights
    linear_rmse = float(
        np.sqrt(np.mean((linear_rec - truth[region].reshape(-1, 3)) ** 2))
    )

    tps_rec = btm.apply_block_tps_map(tps_map, target)
    tps_rmse = float(np.sqrt(np.mean((tps_rec[region] - truth[region]) ** 2)))
    # 線形写像は空間変動も非加法性も持てないので大きく劣る
    assert tps_rmse < 0.5 * linear_rmse
    assert linear_rmse > 0.03


def test_apply_masks_outside_active_region(fitted_map):
    """有効領域の外は 0 のまま。

    落ちる誤実装: フルフレームに外挿して周縁にゴミが出る。
    """
    tps_map, gain, _, _ = fitted_map
    target = np.full((FRAME_H, FRAME_W, 3), 0.2)
    out = btm.apply_block_tps_map(tps_map, target)
    x, y, w, h = ACTIVE
    assert np.all(out[: y, :] == 0)
    assert np.all(out[y + h :, :] == 0)
    assert np.all(out[:, : x] == 0)
    assert np.all(out[:, x + w :] == 0)
    assert np.any(out[y : y + h, x : x + w] > 0)


def test_output_is_clipped_unit_float(fitted_map):
    """出力は float32 [0,1]（範囲外の目標色でも溢れない）。

    落ちる誤実装: TPS 外挿値をそのまま返す（負や 1 超の投影コード）。
    """
    tps_map, _, _, _ = fitted_map
    target = np.full((FRAME_H, FRAME_W, 3), 5.0)  # 到達不能な明るさ
    out = btm.apply_block_tps_map(tps_map, target)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_clamp_inputs_bounds_extrapolation(fitted_map):
    """範囲外の目標色はノード観測域の境界での逆像に丸められる。

    落ちる誤実装: clamp_inputs が no-op（範囲外入力で clamp 有無の結果が
    一致してしまう）、クランプ境界の lo/hi 取り違え。
    """
    tps_map, gain, _, _ = fitted_map
    x, y, w, h = ACTIVE
    wild = np.full((FRAME_H, FRAME_W, 3), 3.0)  # 観測域のはるか外
    clamped = btm.apply_block_tps_map(tps_map, wild, clamp_inputs=True)
    raw = btm.apply_block_tps_map(tps_map, wild, clamp_inputs=False)
    region = np.s_[y : y + h, x : x + w]
    assert not np.allclose(clamped[region], raw[region])
    # クランプ後の入力は各ノードの最大サンプル（=フル白駆動の応答）なので、
    # 逆像はフル駆動コード付近に丸まる
    assert float(np.mean(clamped[region])) > 0.8

    # 観測域内の目標ではクランプは no-op
    inside = _forward_pixelwise(
        np.full((FRAME_H, FRAME_W, 3), 0.5), gain
    )
    a = btm.apply_block_tps_map(tps_map, inside, clamp_inputs=True)
    b = btm.apply_block_tps_map(tps_map, inside, clamp_inputs=False)
    np.testing.assert_allclose(a[region], b[region], atol=1e-6)


def test_save_load_roundtrip(tmp_path, fitted_map):
    """保存→読込→再フィットで適用結果が一致する。

    落ちる誤実装: smoothing や active_region の直列化漏れ。
    """
    tps_map, gain, _, _ = fitted_map
    path = tmp_path / "map.npz"
    btm.save_block_tps_map(path, tps_map)
    loaded = btm.load_block_tps_map(path)
    target = _forward_pixelwise(
        np.full((FRAME_H, FRAME_W, 3), 0.4), gain
    )
    a = btm.apply_block_tps_map(tps_map, target)
    b = btm.apply_block_tps_map(loaded, target)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_select_smoothing_returns_candidate(fitted_map):
    """自動 smoothing 探索は候補のどれかを返す。

    落ちる誤実装: LOOCV の index 取り違えで例外/候補外の値。
    """
    _, _, codes, samples = fitted_map
    smoothing = btm.select_smoothing(samples, codes, probe_nodes=2)
    assert smoothing in btm._SMOOTHING_CANDIDATES
