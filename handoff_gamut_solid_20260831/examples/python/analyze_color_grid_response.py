"""Analyze the 3-D colour-grid response measurements.

Consumes ``measurements.jsonl`` produced by
``measure_color_grid_response.py`` and produces plots plus a
``summary.json`` quantifying:

1. sentinel drift over the capture session,
2. ramp linearity of the native (raw) and through-LUT responses,
3. additivity of the projector (mixed colour vs sum of single channels),
4. how well each forward-model class explains the measured response
   (per-channel 1-D linearisation + 3x3 matrix + offset vs full 3-D
   interpolation), via k-fold cross-validation.

``measure_color_grid_response.py`` が出力した ``measurements.jsonl`` を読み、
番兵ドリフト・ランプ線形性（生/LUT通し）・加法性・モデル別残差
（1D線形化+行列+オフセット vs 3D補間）をプロットと ``summary.json`` に
まとめる。撮影ハードウェアには依存しない（何度でも再実行可能）。

設計書: docs/superpowers/specs/2026-08-12-color-grid-response-design.md

Usage:
    uv run --no-sync python examples/python/analyze_color_grid_response.py \
        --input data/color_grid_response_20260812/measurements.jsonl \
        --output-dir data/color_grid_response_20260812/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.python.validate_inverse_lut import (  # noqa: E402
    linearity_metrics,
    normalize_response,
)

DRIVE_RAW = "raw"
DRIVE_LUT_GRAY = "lut_gray"
DRIVE_SENTINEL = "sentinel"

# Drift threshold above which the piecewise-linear sentinel correction is
# applied (relative, per channel). 番兵補正を適用する相対ドリフト閾値。
DRIFT_CORRECTION_THRESHOLD = 0.01


@dataclass
class Measurements:
    """Parsed measurement arrays, one row per successful capture."""

    drive: List[str] = field(default_factory=list)
    requested: List[Tuple[int, int, int]] = field(default_factory=list)
    projected: List[Tuple[int, int, int]] = field(default_factory=list)
    measured: List[Tuple[float, float, float]] = field(default_factory=list)
    seconds: List[float] = field(default_factory=list)  # since first capture

    def as_arrays(self):
        return (
            np.asarray(self.requested, dtype=np.float64),
            np.asarray(self.projected, dtype=np.float64),
            np.asarray(self.measured, dtype=np.float64),
            np.asarray(self.seconds, dtype=np.float64),
        )


def load_measurements(jsonl_path: Path) -> Measurements:
    """Load successful records; keep the LAST record per (drive, requested).

    status=="ok" の行だけ読み、同一 (drive, requested_rgb) は最後の計測を
    採用する（再実行で古い値を上書きする規約）。番兵は全件残す。
    """
    records: List[dict] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") != "ok":
            continue
        records.append(record)
    if not records:
        raise ValueError(f"No successful measurements in {jsonl_path}")

    t0 = min(datetime.fromisoformat(r["timestamp"]) for r in records)

    dedup: Dict[Tuple[str, Tuple[int, int, int]], dict] = {}
    sentinels: List[dict] = []
    for record in records:
        key = (record["drive"], tuple(record["requested_rgb"]))
        if record["drive"] == DRIVE_SENTINEL:
            sentinels.append(record)
        else:
            dedup[key] = record

    data = Measurements()
    for record in list(dedup.values()) + sentinels:
        data.drive.append(record["drive"])
        data.requested.append(tuple(int(v) for v in record["requested_rgb"]))
        data.projected.append(tuple(int(v) for v in record["projected_rgb"]))
        data.measured.append(tuple(float(v) for v in record["measured_mean"]))
        seconds = (
            datetime.fromisoformat(record["timestamp"]) - t0
        ).total_seconds()
        data.seconds.append(seconds)
    return data


# ── Sentinel drift ────────────────────────────────────────────────────


def sentinel_drift(
    seconds: np.ndarray, measured: np.ndarray
) -> Dict[str, np.ndarray]:
    """Relative sentinel deviation per channel vs the sentinel mean.

    番兵計測の相対偏差（チャンネル別、番兵平均に対する比 − 1）。
    """
    measured = np.asarray(measured, dtype=np.float64)
    mean = measured.mean(axis=0)
    relative = measured / np.where(np.abs(mean) < 1e-12, 1.0, mean) - 1.0
    order = np.argsort(seconds)
    return {
        "seconds": np.asarray(seconds, dtype=np.float64)[order],
        "relative": relative[order],
        "max_abs": np.max(np.abs(relative), axis=0),
    }


def drift_correction_factors(
    sample_seconds: np.ndarray,
    sentinel_seconds: np.ndarray,
    sentinel_relative: np.ndarray,
) -> np.ndarray:
    """Per-sample, per-channel divisors from piecewise-linear sentinel drift.

    番兵ドリフトの区分線形補間から、各計測に適用する除数 (N, 3) を返す。
    （測定値をこの係数で割ると番兵基準に正規化される。）
    """
    factors = np.ones((len(sample_seconds), 3), dtype=np.float64)
    for channel in range(3):
        factors[:, channel] = 1.0 + np.interp(
            sample_seconds,
            sentinel_seconds,
            sentinel_relative[:, channel],
        )
    return factors


# ── Colour difference ─────────────────────────────────────────────────

_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_D65_WHITE = _SRGB_TO_XYZ @ np.ones(3)


def delta_e76(
    rgb_a: np.ndarray, rgb_b: np.ndarray, white_rgb: np.ndarray
) -> np.ndarray:
    """CIE76 colour difference between linear camera-RGB arrays (N, 3).

    カメラのリニアRGBを「白正規化後は線形sRGB」とみなす近似で Lab 化して
    ΔE76 を計算する（比較目的の相対指標。絶対測色ではない）。
    """

    def to_lab(rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
        white = np.asarray(white_rgb, dtype=np.float64).reshape(3)
        norm = rgb / np.where(np.abs(white) < 1e-12, 1.0, white)
        xyz = norm @ _SRGB_TO_XYZ.T
        xyz_n = xyz / _D65_WHITE

        def f(t: np.ndarray) -> np.ndarray:
            delta = 6.0 / 29.0
            return np.where(
                t > delta**3,
                np.cbrt(np.maximum(t, 0.0)),
                t / (3 * delta**2) + 4.0 / 29.0,
            )

        fx, fy, fz = f(xyz_n[:, 0]), f(xyz_n[:, 1]), f(xyz_n[:, 2])
        return np.stack(
            [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=1
        )

    lab_a = to_lab(rgb_a)
    lab_b = to_lab(rgb_b)
    return np.sqrt(np.sum((lab_a - lab_b) ** 2, axis=1))


# ── Axis extraction and 1-D linearisation curves ─────────────────────


def extract_axis(
    inputs: np.ndarray, measured: np.ndarray, channel: int
) -> Tuple[np.ndarray, np.ndarray]:
    """All points on a single-channel axis (other channels zero), sorted.

    単チャンネル軸上（他chが0）の全計測を入力値順に返す。
    """
    mask = np.ones(len(inputs), dtype=bool)
    for other in range(3):
        if other != channel:
            mask &= inputs[:, other] == 0
    levels = inputs[mask, channel]
    order = np.argsort(levels)
    return levels[order], measured[mask][order]


def extract_gray(
    inputs: np.ndarray, measured: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """All gray-diagonal points (r == g == b), sorted by level."""
    mask = (inputs[:, 0] == inputs[:, 1]) & (inputs[:, 1] == inputs[:, 2])
    levels = inputs[mask, 0]
    order = np.argsort(levels)
    return levels[order], measured[mask][order]


def build_linearizer(
    levels: np.ndarray, response: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """1-D normalized forward response g(v) from a measured curve.

    黒引き・白正規化した順方向応答 g: [0,255] -> [0,1] を np.interp で返す。
    単調でない区間もそのまま使う（診断用途のため補正しない）。
    """
    levels = np.asarray(levels, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    black, white = response[0], response[-1]
    denom = white - black if abs(white - black) > 1e-12 else 1.0
    normalized = (response - black) / denom

    def g(v: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(v, dtype=np.float64), levels, normalized)

    return g


# ── Forward models ────────────────────────────────────────────────────


def fit_matrix_model(
    inputs: np.ndarray,
    measured: np.ndarray,
    linearizers: Sequence[Callable[[np.ndarray], np.ndarray]],
    cross_terms: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit ``C = V·g(P) + f`` (optionally + root-polynomial cross terms).

    ``C = V·g(P) + f`` を最小二乗で fit して予測関数を返す。
    ``cross_terms=True`` は √(g_i·g_j) の3特徴を加える（2次 root-polynomial）。
    """

    def features(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        cols = [linearizers[c](points[:, c]) for c in range(3)]
        if cross_terms:
            cols.append(np.sqrt(np.maximum(cols[0] * cols[1], 0.0)))
            cols.append(np.sqrt(np.maximum(cols[0] * cols[2], 0.0)))
            cols.append(np.sqrt(np.maximum(cols[1] * cols[2], 0.0)))
        cols.append(np.ones(len(points)))
        return np.stack(cols, axis=1)

    design = features(inputs)
    weights, *_ = np.linalg.lstsq(design, np.asarray(measured), rcond=None)

    def predict(points: np.ndarray) -> np.ndarray:
        return features(points) @ weights

    return predict


def fit_tps_model(
    inputs: np.ndarray, measured: np.ndarray, smoothing: float = 0.0
) -> Callable[[np.ndarray], np.ndarray]:
    """Thin-plate-spline scattered interpolation (Grundhöfer & Iwai style).

    薄板スプライン散布補間（TIP2015 流）。入力は [0,1]³ に正規化する。
    """
    from scipy.interpolate import RBFInterpolator

    interpolator = RBFInterpolator(
        np.asarray(inputs, dtype=np.float64) / 255.0,
        np.asarray(measured, dtype=np.float64),
        kernel="thin_plate_spline",
        smoothing=smoothing,
    )

    def predict(points: np.ndarray) -> np.ndarray:
        return interpolator(
            np.asarray(points, dtype=np.float64).reshape(-1, 3) / 255.0
        )

    return predict


def kfold_predictions(
    inputs: np.ndarray,
    measured: np.ndarray,
    fit_fn: Callable[[np.ndarray, np.ndarray], Callable],
    k: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Out-of-fold predictions for every point (fair across model classes).

    k-fold 交差検証の out-of-fold 予測を全点分返す。補間系モデルが
    「データ点をそのまま返して残差0」になる不公平を防ぐ。
    """
    n = len(inputs)
    rng = np.random.default_rng(seed)
    fold_of = np.repeat(np.arange(k), (n + k - 1) // k)[:n]
    rng.shuffle(fold_of)
    predictions = np.empty_like(np.asarray(measured, dtype=np.float64))
    for fold in range(k):
        test = fold_of == fold
        model = fit_fn(inputs[~test], measured[~test])
        predictions[test] = model(inputs[test])
    return predictions


def residual_metrics(
    measured: np.ndarray, predicted: np.ndarray, white_rgb: np.ndarray
) -> Dict[str, float]:
    """RMSE / max abs error (linear RGB) and ΔE76 mean / p95 / max."""
    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - measured
    de = delta_e76(predicted, measured, white_rgb)
    return {
        "rmse_linear": float(np.sqrt(np.mean(error**2))),
        "max_abs_linear": float(np.max(np.abs(error))),
        "delta_e76_mean": float(np.mean(de)),
        "delta_e76_p95": float(np.percentile(de, 95)),
        "delta_e76_max": float(np.max(de)),
    }


# ── Additivity ────────────────────────────────────────────────────────


def additivity_table(
    inputs: np.ndarray, measured: np.ndarray
) -> Dict[str, np.ndarray]:
    """Mixed-colour response vs sum of single-channel responses.

    各混合色 (r,g,b)（非ゼロが2ch以上）について、
    ``mix = f(r,g,b) − f(0)`` と ``sum = Σ_c f(単独c) − 3·f(0)`` を比較する。
    単独応答はグリッド軸の実測から補間する。
    """
    black_mask = np.all(inputs == 0, axis=1)
    if not np.any(black_mask):
        raise ValueError("Additivity requires a (0,0,0) measurement.")
    black = measured[black_mask].mean(axis=0)

    singles = []
    for channel in range(3):
        levels, response = extract_axis(inputs, measured, channel)
        singles.append((levels, response - black))

    mixed_mask = np.sum(inputs > 0, axis=1) >= 2
    mix_inputs = inputs[mixed_mask]
    mix_measured = measured[mixed_mask] - black

    predicted_sum = np.zeros_like(mix_measured)
    for channel in range(3):
        levels, response = singles[channel]
        for out in range(3):
            predicted_sum[:, out] += np.interp(
                mix_inputs[:, channel], levels, response[:, out]
            )

    denom = np.where(np.abs(predicted_sum) < 1e-6, np.nan, predicted_sum)
    return {
        "inputs": mix_inputs,
        "mix": mix_measured,
        "sum": predicted_sum,
        "ratio": mix_measured / denom,
    }


# ── Main ──────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kfold", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=2.2)
    return parser.parse_args()


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = _parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_measurements(args.input)
    requested, projected, measured, seconds = data.as_arrays()
    drive = np.asarray(data.drive)
    summary: Dict = {"input": str(args.input), "n_records": int(len(drive))}

    # ── 1. Sentinel drift ────────────────────────────────────────────
    sentinel_mask = drive == DRIVE_SENTINEL
    drift_applied = False
    if np.sum(sentinel_mask) >= 2:
        drift = sentinel_drift(seconds[sentinel_mask], measured[sentinel_mask])
        summary["sentinel"] = {
            "count": int(np.sum(sentinel_mask)),
            "max_abs_relative_drift": [float(v) for v in drift["max_abs"]],
        }
        fig, ax = plt.subplots(figsize=(8, 4))
        for channel, (name, color) in enumerate(
            zip("RGB", ("tab:red", "tab:green", "tab:blue"))
        ):
            ax.plot(
                drift["seconds"] / 60.0,
                drift["relative"][:, channel] * 100.0,
                marker="o",
                color=color,
                label=name,
            )
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel("minutes since first capture")
        ax.set_ylabel("sentinel deviation vs mean [%]")
        ax.set_title("Sentinel (128,128,128) drift over the session")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "drift_sentinels.png", dpi=120)
        plt.close(fig)

        if np.max(drift["max_abs"]) > DRIFT_CORRECTION_THRESHOLD:
            factors = drift_correction_factors(
                seconds, drift["seconds"], drift["relative"]
            )
            measured = measured / factors
            drift_applied = True
    summary["drift_correction_applied"] = drift_applied

    raw_mask = drive == DRIVE_RAW
    lut_mask = drive == DRIVE_LUT_GRAY
    raw_in = requested[raw_mask]
    raw_meas = measured[raw_mask]
    white_mask = np.all(raw_in == 255, axis=1)
    white_rgb = raw_meas[white_mask].mean(axis=0)
    summary["white_response"] = [float(v) for v in white_rgb]

    # ── 2. Native ramps and crosstalk ────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), sharey=False)
    ramp_specs = [("R", 0), ("G", 1), ("B", 2)]
    for ax, (name, channel) in zip(axes, ramp_specs):
        levels, response = extract_axis(raw_in, raw_meas, channel)
        for out, (oname, color) in enumerate(
            zip("RGB", ("tab:red", "tab:green", "tab:blue"))
        ):
            ax.plot(
                levels, response[:, out], marker="o", ms=3, color=color,
                label=f"measured {oname}",
            )
        ax.set_title(f"{name}-only drive (native)")
        ax.set_xlabel("input code value")
        ax.set_ylabel("linear response")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    levels_g, response_g = extract_gray(raw_in, raw_meas)
    ax = axes[3]
    for out, (oname, color) in enumerate(
        zip("RGB", ("tab:red", "tab:green", "tab:blue"))
    ):
        ax.plot(
            levels_g, response_g[:, out], marker="o", ms=3, color=color,
            label=f"measured {oname}",
        )
    reference = response_g[0, 1] + (response_g[-1, 1] - response_g[0, 1]) * (
        (levels_g / 255.0) ** args.gamma
    )
    ax.plot(levels_g, reference, "k--", lw=1, label=f"gamma {args.gamma:g} (G)")
    ax.set_title("gray drive (native)")
    ax.set_xlabel("input code value")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Native response: single-channel and gray ramps")
    fig.tight_layout()
    fig.savefig(out_dir / "ramps_native.png", dpi=120)
    plt.close(fig)

    # Native linearity metrics (how nonlinear the raw response is).
    native_metrics = {}
    for name, channel in ramp_specs:
        levels, response = extract_axis(raw_in, raw_meas, channel)
        normalized = normalize_response(response)
        native_metrics[f"{name.lower()}_ramp"] = {
            k: [float(x) for x in v]
            for k, v in linearity_metrics(levels / 255.0, normalized).items()
        }
    normalized = normalize_response(response_g)
    native_metrics["gray_ramp"] = {
        k: [float(x) for x in v]
        for k, v in linearity_metrics(levels_g / 255.0, normalized).items()
    }
    summary["native_linearity"] = native_metrics

    # ── 3. Through-LUT linearity (the actual question) ───────────────
    if np.any(lut_mask):
        lut_in = requested[lut_mask]
        lut_meas = measured[lut_mask]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        glevels, gresp = extract_gray(lut_in, lut_meas)
        gnorm = normalize_response(gresp)
        ax = axes[0]
        ax.plot([0, 255], [0, 1], "k--", lw=1, label="ideal (linear)")
        for out, (oname, color) in enumerate(
            zip("RGB", ("tab:red", "tab:green", "tab:blue"))
        ):
            ax.plot(
                glevels, gnorm[:, out], marker="o", ms=4, color=color,
                label=f"measured {oname}",
            )
        ax.set_title("gray ramp through gray-fit inverse LUT")
        ax.set_xlabel("requested code value")
        ax.set_ylabel("measured (normalized)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        lut_metrics = {
            "gray_ramp": {
                k: [float(x) for x in v]
                for k, v in linearity_metrics(glevels / 255.0, gnorm).items()
            }
        }
        ax = axes[1]
        ax.plot([0, 255], [0, 1], "k--", lw=1, label="ideal (linear)")
        for name, channel, color in (
            ("R", 0, "tab:red"),
            ("G", 1, "tab:green"),
            ("B", 2, "tab:blue"),
        ):
            levels, response = extract_axis(lut_in, lut_meas, channel)
            if len(levels) < 3:
                continue
            # Include gray endpoints (0 and 255) which live on the diagonal.
            normalized = (response[:, channel] - gresp[0, channel]) / (
                gresp[-1, channel] - gresp[0, channel]
            )
            ax.plot(
                levels, normalized, marker="s", ms=4, color=color,
                label=f"{name}-only drive, ch {name}",
            )
            lut_metrics[f"{name.lower()}_axis_dev"] = {
                "max_dev": float(
                    np.max(np.abs(normalized - levels / 255.0))
                )
            }
        ax.set_title("single-channel points through gray-fit LUT")
        ax.set_xlabel("requested code value")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.suptitle("Linearised (through-LUT) response vs ideal diagonal")
        fig.tight_layout()
        fig.savefig(out_dir / "lut_linearity.png", dpi=120)
        plt.close(fig)
        summary["lut_linearity"] = lut_metrics

    # ── 4. Additivity ────────────────────────────────────────────────
    add = additivity_table(raw_in, raw_meas)
    ratio = add["ratio"]
    finite = np.isfinite(ratio)
    summary["additivity"] = {
        "n_mixed": int(len(ratio)),
        "ratio_mean": [
            float(np.nanmean(ratio[:, c])) for c in range(3)
        ],
        "ratio_p05": [
            float(np.nanpercentile(ratio[:, c], 5)) for c in range(3)
        ],
        "ratio_p95": [
            float(np.nanpercentile(ratio[:, c], 95)) for c in range(3)
        ],
    }

    # White vs sum of primaries at the gray grid levels (Seime & Hardeberg).
    gray_levels = [51, 102, 153, 204, 255]
    white_vs_sum = []
    black = raw_meas[np.all(raw_in == 0, axis=1)].mean(axis=0)
    for level in gray_levels:
        gray_rows = raw_in[
            (raw_in[:, 0] == level)
            & (raw_in[:, 1] == level)
            & (raw_in[:, 2] == level)
        ]
        if len(gray_rows) == 0:
            continue
        gray_value = raw_meas[
            (raw_in[:, 0] == level)
            & (raw_in[:, 1] == level)
            & (raw_in[:, 2] == level)
        ].mean(axis=0) - black
        sum_value = np.zeros(3)
        for channel in range(3):
            mask = np.ones(len(raw_in), dtype=bool)
            for other in range(3):
                mask &= raw_in[:, other] == (level if other == channel else 0)
            sum_value += raw_meas[mask].mean(axis=0) - black
        white_vs_sum.append(
            {
                "level": level,
                "gray": [float(v) for v in gray_value],
                "sum_of_primaries": [float(v) for v in sum_value],
                "ratio": [
                    float(g / s) if abs(s) > 1e-9 else None
                    for g, s in zip(gray_value, sum_value)
                ],
            }
        )
    summary["white_vs_sum_of_primaries"] = white_vs_sum

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for channel, (name, color) in enumerate(
        zip("RGB", ("tab:red", "tab:green", "tab:blue"))
    ):
        ax.plot(
            [row["level"] for row in white_vs_sum],
            [row["ratio"][channel] for row in white_vs_sum],
            marker="o",
            color=color,
            label=name,
        )
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="additive")
    ax.set_xlabel("gray level")
    ax.set_ylabel("gray / sum of primaries")
    ax.set_title("White vs sum of primaries (Seime & Hardeberg check)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    for channel, (name, color) in enumerate(
        zip("RGB", ("tab:red", "tab:green", "tab:blue"))
    ):
        values = ratio[:, channel][finite[:, channel]]
        ax.hist(
            values, bins=40, range=(0.4, 1.6), alpha=0.5, color=color,
            label=f"{name} (mean {np.nanmean(ratio[:, channel]):.3f})",
        )
    ax.axvline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("mix / sum-of-singles ratio")
    ax.set_ylabel("mixed grid points")
    ax.set_title(f"Additivity over {len(ratio)} mixed colours")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "additivity.png", dpi=120)
    plt.close(fig)

    # ── 5. Model comparison (k-fold CV on the raw set) ───────────────
    gamma = args.gamma

    def gamma_lin(v: np.ndarray) -> np.ndarray:
        return (np.asarray(v, dtype=np.float64) / 255.0) ** gamma

    def make_channel_linearizers(inputs, meas):
        lins = []
        for channel in range(3):
            levels, response = extract_axis(inputs, meas, channel)
            lins.append(build_linearizer(levels, response[:, channel]))
        return lins

    def make_gray_linearizers(inputs, meas):
        levels, response = extract_gray(inputs, meas)
        return [
            build_linearizer(levels, response[:, channel])
            for channel in range(3)
        ]

    model_fits: Dict[str, Callable] = {
        "M1_gamma22_matrix": lambda x, y: fit_matrix_model(
            x, y, [gamma_lin] * 3
        ),
        "M2_channel_lut_matrix": lambda x, y: fit_matrix_model(
            x, y, make_channel_linearizers(x, y)
        ),
        "M3_gray_lut_matrix": lambda x, y: fit_matrix_model(
            x, y, make_gray_linearizers(x, y)
        ),
        "M5_root_polynomial": lambda x, y: fit_matrix_model(
            x, y, make_gray_linearizers(x, y), cross_terms=True
        ),
        "M4_tps": lambda x, y: fit_tps_model(x, y, smoothing=0.0),
    }
    model_results: Dict[str, Dict] = {}
    for name, fit_fn in model_fits.items():
        predictions = kfold_predictions(
            raw_in, raw_meas, fit_fn, k=args.kfold, seed=0
        )
        model_results[name] = residual_metrics(
            raw_meas, predictions, white_rgb
        )
        model_results[name]["cv"] = f"{args.kfold}-fold"
    summary["model_comparison_cv"] = model_results

    # Axis+gray-only fit -> evaluated on mixed colours (extrapolation view).
    axis_mask = (np.sum(raw_in > 0, axis=1) <= 1) | (
        (raw_in[:, 0] == raw_in[:, 1]) & (raw_in[:, 1] == raw_in[:, 2])
    )
    mixed_mask = ~axis_mask
    axisfit_results: Dict[str, Dict] = {}
    for name, fit_fn in model_fits.items():
        if name == "M4_tps":
            continue  # TPS on a degenerate axis-only set is ill-posed
        model = fit_fn(raw_in[axis_mask], raw_meas[axis_mask])
        predicted = model(raw_in[mixed_mask])
        axisfit_results[name] = residual_metrics(
            raw_meas[mixed_mask], predicted, white_rgb
        )
    summary["model_comparison_axis_fit"] = axisfit_results

    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = list(model_results.keys())
    x = np.arange(len(names))
    ax.bar(
        x - 0.2,
        [model_results[n]["delta_e76_mean"] for n in names],
        width=0.4,
        label="ΔE76 mean (CV)",
        color="tab:blue",
    )
    ax.bar(
        x + 0.2,
        [model_results[n]["delta_e76_p95"] for n in names],
        width=0.4,
        label="ΔE76 p95 (CV)",
        color="tab:orange",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("ΔE76")
    ax.set_title("Forward-model residuals (10-fold CV, 321 raw colours)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=120)
    plt.close(fig)

    # Residual slice heatmaps for the best matrix model vs TPS.
    for model_name in ("M3_gray_lut_matrix", "M4_tps"):
        predictions = kfold_predictions(
            raw_in, raw_meas, model_fits[model_name], k=args.kfold, seed=0
        )
        de = delta_e76(predictions, raw_meas, white_rgb)
        fig, axes = plt.subplots(1, 6, figsize=(21, 3.6))
        vmax = np.percentile(de, 98)
        for ax, b_level in zip(axes, (0, 51, 102, 153, 204, 255)):
            image = np.full((6, 6), np.nan)
            for i, r_level in enumerate((0, 51, 102, 153, 204, 255)):
                for j, g_level in enumerate((0, 51, 102, 153, 204, 255)):
                    mask = (
                        (raw_in[:, 0] == r_level)
                        & (raw_in[:, 1] == g_level)
                        & (raw_in[:, 2] == b_level)
                    )
                    if np.any(mask):
                        image[j, i] = de[mask][0]
            im = ax.imshow(
                image, origin="lower", vmin=0, vmax=vmax, cmap="viridis"
            )
            ax.set_title(f"B={b_level}")
            ax.set_xlabel("R level idx")
            if b_level == 0:
                ax.set_ylabel("G level idx")
        fig.colorbar(im, ax=axes, shrink=0.85, label="ΔE76 (CV)")
        fig.suptitle(f"Residual over the RGB cube: {model_name}")
        fig.savefig(out_dir / f"residual_slices_{model_name}.png", dpi=120)
        plt.close(fig)

    # ── 6. Through-LUT measurements vs raw-data prediction ──────────
    if np.any(lut_mask):
        tps_all = fit_tps_model(raw_in, raw_meas, smoothing=0.0)
        predicted = tps_all(projected[lut_mask])
        summary["lut_check_vs_raw_prediction"] = residual_metrics(
            measured[lut_mask], predicted, white_rgb
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nPlots and summary written to: {out_dir}")


if __name__ == "__main__":
    main()
