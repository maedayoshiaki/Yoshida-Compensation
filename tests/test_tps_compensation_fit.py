"""Tests for the fitting helpers in generate_tps_and_matrix_compensation.py.

合成データで画素別行列 fit（正方向・ガンマ合成）と行列場アップサンプルを
検査する。両方向検査（正しいモデル族なら残差≈0 / モデル族の外なら残差大）を
守る（AGENTS.md 第8章）。
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.python.generate_tps_and_matrix_compensation import (
    GAMMA,
    fit_pixel_matrices,
    upsample_matrices,
)


def test_fit_recovers_known_affine_inverse():
    """画素別アフィン順方向 q = p·V + c で生成したデータなら残差≈0。

    順方向がアフィン可逆なら逆写像もアフィンなので、fit は各画素で
    厳密に解ける。落ちる誤実装: 正規方程式の添字取り違え（einsum の
    i/j）、バイアス列の欠落（オフセット c で残差が浮く）。
    """
    rng = np.random.default_rng(0)
    height, width, n = 6, 7, 40
    p = rng.uniform(0.0, 1.0, size=(n, 3))
    v = rng.uniform(0.2, 0.8, size=(height, width, 3, 3))
    v += 2.0 * np.eye(3)  # 可逆性を確保する対角優位
    c = rng.uniform(0.0, 0.1, size=(height, width, 3))
    q = np.einsum("nk,hwkj->nhwj", p, v) + c[None]
    fitted = fit_pixel_matrices(q, p)
    design = np.concatenate([q, np.ones((n, height, width, 1))], axis=-1)
    predicted = np.einsum("nhwi,hwij->nhwj", design, fitted.astype(np.float64))
    assert np.max(np.abs(predicted - p[:, None, None, :])) < 1e-4


def test_fit_leaves_residual_for_non_affine_data():
    """非アフィン（ガンマ合成入り）データではアフィン fit に残差が残る。

    落ちる誤実装: fit が訓練データを丸暗記して返す no-op。
    """
    rng = np.random.default_rng(1)
    height, width, n = 4, 4, 60
    p = rng.uniform(0.0, 1.0, size=(n, 3))
    v = np.array([[0.6, 0.1, 0.05], [0.1, 0.7, 0.05], [0.05, 0.1, 0.5]])
    q = np.einsum("nk,kj->nj", p**GAMMA, v)
    q_field = np.broadcast_to(q[:, None, None, :], (n, height, width, 3)).copy()
    fitted = fit_pixel_matrices(q_field, p)
    design = np.concatenate(
        [q_field, np.ones((n, height, width, 1))], axis=-1
    )
    predicted = np.einsum("nhwi,hwij->nhwj", design, fitted.astype(np.float64))
    rmse = float(np.sqrt(np.mean((predicted - p[:, None, None, :]) ** 2)))
    assert rmse > 0.02  # ガンマの曲率はアフィンでは書けない


def test_gamma_feature_fit_recovers_gamma_affine_model():
    """q = V·p^2.2 型の合成データは「codes**GAMMA を目標に fit → 出力を
    ^(1/GAMMA)」の合成で残差≈0 になる。

    落ちる誤実装: GAMMA 適用の向き取り違え（fit 目標と適用側が食い違うと
    残差が浮く）。
    """
    rng = np.random.default_rng(2)
    height, width, n = 3, 5, 60
    p = rng.uniform(0.05, 1.0, size=(n, 3))
    v = np.array([[0.6, 0.1, 0.05], [0.1, 0.7, 0.05], [0.05, 0.1, 0.5]])
    q = np.einsum("nk,kj->nj", p**GAMMA, v)
    q_field = np.broadcast_to(q[:, None, None, :], (n, height, width, 3)).copy()
    fitted_t = fit_pixel_matrices(q_field, p**GAMMA)
    design = np.concatenate(
        [q_field, np.ones((n, height, width, 1))], axis=-1
    )
    t_hat = np.einsum("nhwi,hwij->nhwj", design, fitted_t.astype(np.float64))
    p_hat = np.clip(t_hat, 0.0, 1.0) ** (1.0 / GAMMA)
    assert np.max(np.abs(p_hat - p[:, None, None, :])) < 1e-3


def test_upsample_preserves_constant_field_and_shape():
    """定数行列場はアップサンプルしても不変で、形状は (H, W, 4, 3)。

    落ちる誤実装: reshape の軸順取り違え（12係数の並びが壊れる）。
    """
    matrices = np.tile(
        np.arange(12, dtype=np.float32).reshape(1, 1, 4, 3), (5, 6, 1, 1)
    )
    up = upsample_matrices(matrices, (24, 20))
    assert up.shape == (20, 24, 4, 3)
    np.testing.assert_allclose(
        up[10, 12], np.arange(12, dtype=np.float32).reshape(4, 3), atol=1e-6
    )
