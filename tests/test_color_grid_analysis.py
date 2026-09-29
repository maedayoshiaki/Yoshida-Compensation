"""Hardware-independent tests for examples/python/analyze_color_grid_response.py.

合成データで解析の数理部分を検査する。各テストが「どの誤実装なら落ちるか」を
docstring に書く（AGENTS.md 第8章）。フィット検査は「正しいモデル族なら残差≈0」
と「モデル族の外なら残差が大きい」の両方向を必ず検査する（片方向だけでは
no-op の fit が通り抜ける）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from examples.python import analyze_color_grid_response as acgr
from examples.python import measure_color_grid_response as mcgr


V_TRUE = np.array(
    [
        [0.50, 0.08, 0.02],
        [0.10, 0.60, 0.06],
        [0.05, 0.07, 0.40],
    ]
).T  # 列 = 出力チャンネル
F_TRUE = np.array([0.010, 0.020, 0.015])


def _g(v: np.ndarray) -> np.ndarray:
    return (np.asarray(v, dtype=np.float64) / 255.0) ** 2.2


def _synthetic_additive(inputs: np.ndarray) -> np.ndarray:
    """加法的な V·g(P)+f 合成応答。"""
    return _g(inputs) @ V_TRUE + F_TRUE


def _synthetic_non_additive(inputs: np.ndarray) -> np.ndarray:
    """滑らかな交差項で加法性を壊した合成応答。"""
    g = _g(inputs)
    cross = g[:, 0] * g[:, 1] + g[:, 0] * g[:, 2] + g[:, 1] * g[:, 2]
    return _synthetic_additive(inputs) - 0.15 * cross[:, None] * np.array(
        [1.0, 1.2, 0.8]
    )


@pytest.fixture(scope="module")
def grid_inputs() -> np.ndarray:
    return np.asarray(mcgr.build_raw_patch_set(), dtype=np.float64)


# ── 加法性 ────────────────────────────────────────────────────────────


def test_additivity_ratio_is_one_for_additive_data(grid_inputs):
    """加法的データでは mix/sum 比が 1 に張り付く。

    落ちる誤実装: 黒引き忘れ（f≠0 で比が系統的にずれる）、単独応答の
    補間軸の取り違え。
    """
    table = acgr.additivity_table(grid_inputs, _synthetic_additive(grid_inputs))
    ratio = table["ratio"]
    assert np.nanmax(np.abs(ratio - 1.0)) < 1e-6


def test_additivity_ratio_detects_non_additive_data(grid_inputs):
    """非加法データでは比が 1 から明確に外れる。

    落ちる誤実装: mix と sum を同じ配列から計算してしまう
    （常に比1を返す no-op）。
    """
    table = acgr.additivity_table(
        grid_inputs, _synthetic_non_additive(grid_inputs)
    )
    bright = np.all(table["inputs"] >= 153, axis=1)
    assert np.nanmean(table["ratio"][bright]) < 0.9


# ── モデル fit（双方向） ──────────────────────────────────────────────


def test_matrix_model_explains_additive_data(grid_inputs):
    """V·g(P)+f 合成データは gamma2.2+行列 fit で CV 残差 ≈ 0。

    落ちる誤実装: バイアス列の欠落（F_TRUE≠0 で残差が浮く）、
    特徴量の順序取り違え。
    """
    measured = _synthetic_additive(grid_inputs)
    predictions = acgr.kfold_predictions(
        grid_inputs,
        measured,
        lambda x, y: acgr.fit_matrix_model(x, y, [_g] * 3),
        k=5,
        seed=0,
    )
    rmse = float(np.sqrt(np.mean((predictions - measured) ** 2)))
    assert rmse < 1e-9


def test_matrix_model_fails_and_tps_succeeds_on_non_additive_data(grid_inputs):
    """非加法データでは行列モデルの残差が TPS より大きく残る。

    落ちる誤実装: TPS が実は行列回帰になっている（残差が同水準になる）、
    行列 fit が訓練データを丸暗記して返す（CV で露見）。
    """
    measured = _synthetic_non_additive(grid_inputs)
    matrix_pred = acgr.kfold_predictions(
        grid_inputs,
        measured,
        lambda x, y: acgr.fit_matrix_model(x, y, [_g] * 3),
        k=5,
        seed=0,
    )
    tps_pred = acgr.kfold_predictions(
        grid_inputs,
        measured,
        lambda x, y: acgr.fit_tps_model(x, y),
        k=5,
        seed=0,
    )
    matrix_rmse = float(np.sqrt(np.mean((matrix_pred - measured) ** 2)))
    tps_rmse = float(np.sqrt(np.mean((tps_pred - measured) ** 2)))
    assert matrix_rmse > 3.0 * tps_rmse
    assert matrix_rmse > 1e-3


def test_root_polynomial_captures_sqrt_cross_terms(grid_inputs):
    """M5 の特徴（√(g_i·g_j)）で張られる合成データは M5 で残差≈0、
    交差項なしでは残差が明確に残る。

    落ちる誤実装: cross_terms=True が特徴量を追加しない no-op（両 fit の
    残差が一致して前半の assert が落ちる）、交差項の組の取り違え。
    """
    g = _g(grid_inputs)
    cross = np.stack(
        [
            np.sqrt(g[:, 0] * g[:, 1]),
            np.sqrt(g[:, 0] * g[:, 2]),
            np.sqrt(g[:, 1] * g[:, 2]),
        ],
        axis=1,
    )
    measured = _synthetic_additive(grid_inputs) - cross @ np.diag(
        [0.10, 0.12, 0.08]
    )

    def fit_plain(x, y):
        return acgr.fit_matrix_model(x, y, [_g] * 3)

    def fit_cross(x, y):
        return acgr.fit_matrix_model(x, y, [_g] * 3, cross_terms=True)

    plain = acgr.kfold_predictions(grid_inputs, measured, fit_plain, k=5, seed=0)
    cross_pred = acgr.kfold_predictions(
        grid_inputs, measured, fit_cross, k=5, seed=0
    )
    rmse_plain = float(np.sqrt(np.mean((plain - measured) ** 2)))
    rmse_cross = float(np.sqrt(np.mean((cross_pred - measured) ** 2)))
    assert rmse_cross < 1e-9
    assert rmse_plain > 1e-3


# ── ΔE76 ─────────────────────────────────────────────────────────────


def test_delta_e76_zero_for_identical_and_large_for_black_white():
    """同一色で0、白黒間で大きい（>50）。

    落ちる誤実装: Lab 変換の白正規化忘れ（同一色でも非0になるスケール
    依存）、fy と fx の取り違え。
    """
    white = np.array([0.30, 0.31, 0.29])
    a = np.array([[0.1, 0.2, 0.15], [0.0, 0.0, 0.0]])
    b = np.array([[0.1, 0.2, 0.15], [0.30, 0.31, 0.29]])
    de = acgr.delta_e76(a, b, white)
    assert de[0] == pytest.approx(0.0, abs=1e-9)
    assert de[1] > 50.0


# ── ドリフト ──────────────────────────────────────────────────────────


def test_drift_factors_interpolate_between_sentinels():
    """番兵2点 (0%, +2%) の中間時刻で補正係数が +1% になる。

    落ちる誤実装: 時刻でなく index で補間する、チャンネル軸の取り違え。
    """
    sentinel_seconds = np.array([0.0, 100.0])
    relative = np.array([[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]])
    factors = acgr.drift_correction_factors(
        np.array([50.0]), sentinel_seconds, relative
    )
    assert factors[0, 1] == pytest.approx(1.01)
    assert factors[0, 0] == pytest.approx(1.0)
    assert factors[0, 2] == pytest.approx(1.0)


def test_sentinel_drift_reports_max_abs():
    """相対偏差の最大絶対値がチャンネル別に出る。

    落ちる誤実装: 平均でなく初回を基準にする（対称な揺れで max_abs が
    2倍になる）。
    """
    seconds = np.array([0.0, 1.0, 2.0])
    measured = np.array(
        [[0.10, 0.10, 0.10], [0.10, 0.11, 0.10], [0.10, 0.09, 0.10]]
    )
    drift = acgr.sentinel_drift(seconds, measured)
    assert drift["max_abs"][1] == pytest.approx(0.1, rel=1e-6)
    assert drift["max_abs"][0] == pytest.approx(0.0, abs=1e-12)


# ── 読み込み ──────────────────────────────────────────────────────────


def test_load_measurements_dedups_and_keeps_sentinels(tmp_path):
    """同一 (drive, 要求RGB) は最後の計測が勝ち、番兵は全件残る。

    落ちる誤実装: 最初の計測を採用する（再計測で埋めた値が無視される）、
    番兵まで重複排除する（ドリフト系列が1点になる）。
    """
    rows = [
        {
            "job_id": 0, "drive": "sentinel", "requested_rgb": [128, 128, 128],
            "projected_rgb": [128, 128, 128], "measured_mean": [0.1, 0.1, 0.1],
            "status": "ok", "timestamp": "2026-08-12T00:00:00+00:00",
        },
        {
            "job_id": 1, "drive": "raw", "requested_rgb": [51, 0, 0],
            "projected_rgb": [51, 0, 0], "measured_mean": [0.9, 0.9, 0.9],
            "status": "error", "timestamp": "2026-08-12T00:01:00+00:00",
        },
        {
            "job_id": 1, "drive": "raw", "requested_rgb": [51, 0, 0],
            "projected_rgb": [51, 0, 0], "measured_mean": [0.02, 0.0, 0.0],
            "status": "ok", "timestamp": "2026-08-12T00:02:00+00:00",
        },
        {
            "job_id": 1, "drive": "raw", "requested_rgb": [51, 0, 0],
            "projected_rgb": [51, 0, 0], "measured_mean": [0.03, 0.0, 0.0],
            "status": "ok", "timestamp": "2026-08-12T00:03:00+00:00",
        },
        {
            "job_id": 9, "drive": "sentinel", "requested_rgb": [128, 128, 128],
            "projected_rgb": [128, 128, 128], "measured_mean": [0.1, 0.1, 0.1],
            "status": "ok", "timestamp": "2026-08-12T00:04:00+00:00",
        },
    ]
    path = tmp_path / "measurements.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    data = acgr.load_measurements(path)
    raw_rows = [
        (d, r, m)
        for d, r, m in zip(data.drive, data.requested, data.measured)
        if d == "raw"
    ]
    assert len(raw_rows) == 1
    assert raw_rows[0][2] == (0.03, 0.0, 0.0)  # 最後の ok が勝つ
    assert sum(1 for d in data.drive if d == "sentinel") == 2
