"""Tests for yoshidacomp.compensation.color_lut3d (hardware-independent).

3D LUT の焼き込み・三線形適用・正規化を合成データで検査する。
「正しい写像なら復元できる」と「格子/軸の取り違えなら落ちる」の両方向を
意識した検査にする（AGENTS.md 第8章）。
"""

from __future__ import annotations

import numpy as np
import pytest

from yoshidacomp.compensation import color_lut3d as cl3


def _grid_codes(levels: int = 6) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, levels)
    return np.array([(r, g, b) for r in axis for g in axis for b in axis])


def _forward(codes: np.ndarray) -> np.ndarray:
    """滑らかで非加法な合成順応答（正規化済み: 黒=0, 白=1 になるよう構成）。"""
    g = np.asarray(codes, dtype=np.float64) ** 2.2
    v = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
    q = g @ v
    cross = 0.2 * np.stack(
        [g[:, 0] * g[:, 1], g[:, 1] * g[:, 2], g[:, 0] * g[:, 2]], axis=1
    )
    q = q + cross
    white = _white_of(v)
    return q / white


def _white_of(v: np.ndarray) -> np.ndarray:
    g = np.ones(3)
    return g @ v + 0.2 * np.array([1.0, 1.0, 1.0])


def test_bake_and_apply_inverts_forward_model():
    """焼いたLUTは達成可能な色に対して元コードをほぼ復元する。

    落ちる誤実装: 格子の軸順取り違え（meshgrid indexing='xy'）、
    lattice[i_r,i_g,i_b] の添字順の混同 — 非対称な V で復元誤差が跳ねる。
    """
    codes = _grid_codes(6)
    measured = _forward(codes)
    lut = cl3.bake_inverse_lut3d(measured, codes, size=33, smoothing=0.0)
    # 中間域の検証点。逆写像は暗部（c ≲ 0.1）で勾配が急峻になり 216 点
    # からの補間誤差が増える（実データでも同様）ため、この復元検査は
    # 中間域で行う。暗部の精度限界は仕様であってバグではない。
    truth = np.array([[0.4, 0.6, 0.35], [0.7, 0.35, 0.5], [0.35, 0.8, 0.55]])
    q = _forward(truth)
    recovered = cl3.apply_lut3d(q[None].astype(np.float32), lut)[0]
    assert np.max(np.abs(recovered - truth)) < 0.05


def test_apply_is_exact_at_lattice_points():
    """格子点ちょうどの入力では格子値そのものが返る。

    落ちる誤実装: 補間係数の off-by-one（隣接セルの値が混ざる）。
    """
    rng = np.random.default_rng(0)
    lattice = rng.uniform(0.0, 1.0, size=(5, 5, 5, 3)).astype(np.float32)
    lut = cl3.ColorLut3d(lattice=lattice)
    axis = np.linspace(0.0, 1.0, 5)
    image = np.array([[[axis[1], axis[3], axis[2]]]], dtype=np.float32)
    out = cl3.apply_lut3d(image, lut)
    np.testing.assert_allclose(out[0, 0], lattice[1, 3, 2], atol=1e-6)


def test_apply_is_trilinear_between_points():
    """恒等LUT（lattice=入力座標）なら任意入力がそのまま返る。

    落ちる誤実装: fr/fg/fb の軸対応の取り違え（恒等でもずれる）。
    """
    size = 9
    axis = np.linspace(0.0, 1.0, size)
    rr, gg, bb = np.meshgrid(axis, axis, axis, indexing="ij")
    lut = cl3.ColorLut3d(
        lattice=np.stack([rr, gg, bb], axis=-1).astype(np.float32)
    )
    rng = np.random.default_rng(1)
    image = rng.uniform(0.0, 1.0, size=(4, 6, 3)).astype(np.float32)
    out = cl3.apply_lut3d(image, lut)
    np.testing.assert_allclose(out, image, atol=1e-6)


def test_apply_handles_uint8_contract():
    """uint8 入力は /255 正規化、uint8 出力は half-up 量子化。

    落ちる誤実装: 1D LUT (apply_inverse_lut) と違う正規化契約。
    """
    size = 3
    axis = np.linspace(0.0, 1.0, size)
    rr, gg, bb = np.meshgrid(axis, axis, axis, indexing="ij")
    lut = cl3.ColorLut3d(
        lattice=np.stack([rr, gg, bb], axis=-1).astype(np.float32)
    )
    image = np.array([[[128, 64, 255]]], dtype=np.uint8)
    out = cl3.apply_lut3d(image, lut, dtype=np.uint8)
    np.testing.assert_array_equal(out[0, 0], [128, 64, 255])


def test_bake_linear_inverts_forward_model():
    """区分線形焼き込みでも達成可能な中間色は元コードへ復元できる。

    落ちる誤実装: 凸包外の NaN を放置（格子に NaN が残り出力が壊れる）、
    LinearND と Nearest の引数順取り違え。
    """
    codes = _grid_codes(6)
    measured = _forward(codes)
    lut = cl3.bake_inverse_lut3d_linear(measured, codes, size=33)
    assert np.all(np.isfinite(lut.lattice))
    truth = np.array([[0.4, 0.6, 0.35], [0.7, 0.35, 0.5], [0.35, 0.8, 0.55]])
    q = _forward(truth)
    recovered = cl3.apply_lut3d(q[None].astype(np.float32), lut)[0]
    assert np.max(np.abs(recovered - truth)) < 0.06


def test_bake_linear_is_exact_for_affine_forward():
    """順方向がアフィンなら区分線形補間の逆写像は（格子解像度の範囲で）
    ほぼ厳密になる。

    落ちる誤実装: 補間の入出力（measured/codes）の取り違え。
    """
    codes = _grid_codes(4)
    v = np.array([[0.7, 0.2, 0.1], [0.15, 0.7, 0.15], [0.1, 0.2, 0.7]])
    measured = codes @ v
    measured = measured / measured.max(axis=0)  # 正規化相当
    lut = cl3.bake_inverse_lut3d_linear(measured, codes, size=65)
    truth = np.array([[0.3, 0.5, 0.4], [0.6, 0.25, 0.45]])
    q = truth @ v / (np.ones(3) @ v)
    recovered = cl3.apply_lut3d(q[None].astype(np.float32), lut)[0]
    assert np.max(np.abs(recovered - truth)) < 0.02


def test_save_load_roundtrip(tmp_path):
    """保存→読込で格子とメタが一致する。"""
    codes = _grid_codes(4)
    lut = cl3.bake_inverse_lut3d(_forward(codes), codes, size=9, smoothing=1e-6)
    path = tmp_path / "lut3d.npz"
    cl3.save_lut3d(path, lut)
    loaded = cl3.load_lut3d(path)
    np.testing.assert_allclose(loaded.lattice, lut.lattice, atol=1e-7)
    assert loaded.meta["size"] == 9


def test_normalize_reference_response_endpoints():
    """黒→0、白→1 にチャンネル別で正規化される。

    落ちる誤実装: 黒引き忘れ（黒が非0）、白正規化のチャンネル取り違え。
    """
    codes = _grid_codes(2)  # 8色（黒白含む）
    responses = _forward(codes) * np.array([0.5, 0.3, 0.4]) + np.array(
        [0.01, 0.02, 0.015]
    )
    c = cl3.normalize_reference_response(responses, codes)
    black_row = np.all(codes == 0.0, axis=1)
    white_row = np.all(codes == 1.0, axis=1)
    np.testing.assert_allclose(c[black_row][0], [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(c[white_row][0], [1, 1, 1], atol=1e-12)


def test_normalize_requires_black_and_white():
    """黒白の校正サンプルが無ければ明示的に失敗する。"""
    codes = np.array([[0.5, 0.5, 0.5], [0.2, 0.4, 0.6]])
    with pytest.raises(ValueError):
        cl3.normalize_reference_response(np.ones((2, 3)), codes)
