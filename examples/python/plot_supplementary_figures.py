"""Supplementary characterization figures recommended by the survey.

Creates, from EXISTING measurement data, the three figure types the
visualization survey (docs/projector_io_visualization_survey_20260815.md)
identified as missing from our set:

1. ``chromaticity_tracking.png`` — camera-referred chromaticity
   trajectories of the R/G/B/gray ramps, black-corrected pair
   (Seime & Hardeberg Fig.4/5 style). CAMERA-referred (r, g) =
   (R, G)/(R+G+B); no colorimetric calibration is claimed.
2. ``additivity_wyble.png`` — gray-ramp measured vs sum of single-channel
   ramps overlaid, with the difference on a secondary axis
   (Wyble & Rosen Fig.3 style), one panel per output channel.
3. ``session_stability.png`` — sentinel (128,128,128) relative drift vs
   minutes since run start for every capture session of 08-12/13.
   (A true warm-up series needs a cold start; not available remotely.)

可視化調査で欠落とされた3種を既存データから作る補助図スクリプト。
①カメラ基準色度軌跡（黒補正前後の対）②Wyble式加法性（差分第2軸）
③セッション安定性（全ランの番兵時系列）。

Usage:
    uv run --no-sync python examples/python/plot_supplementary_figures.py \
        --response data/color_grid_response_20260812_r2/measurements.jsonl \
        --output-dir data/color_grid_response_20260812_r2/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.python.plot_line_proportionality import load_lines  # noqa: E402

RAMP_STYLE = {
    "R": ("tab:red", "o"),
    "G": ("tab:green", "s"),
    "B": ("tab:blue", "^"),
    "RGB": ("black", "D"),
}


def _chromaticity(values: np.ndarray) -> np.ndarray:
    total = values.sum(axis=-1, keepdims=True)
    total = np.where(np.abs(total) < 1e-9, 1.0, total)
    return values[..., :2] / total


def plot_chromaticity_tracking(lines, out_path):
    import matplotlib.pyplot as plt

    black = lines["R"][1][0]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.3))
    for ax, corrected in zip(axes, (False, True)):
        for name, (color, marker) in RAMP_STYLE.items():
            levels, q = lines[name]
            values = q - black if corrected else q
            keep = levels >= (16 if corrected else 32)  # 黒近傍は比が不定
            xy = _chromaticity(values[keep])
            ax.plot(
                xy[:, 0], xy[:, 1], color=color, marker=marker, ms=5,
                lw=1.3, label=f"{name} ramp",
            )
            # 矢印: レベル減少方向（Seime & Hardeberg の流儀）
            if len(xy) >= 2:
                ax.annotate(
                    "", xy=tuple(xy[0]), xytext=tuple(xy[1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.6),
                )
        ax.set_title(
            ("black-corrected" if corrected else "raw (flare included)")
            + " — arrows point toward LOWER drive level"
        )
        ax.set_xlabel("camera-referred r = R/(R+G+B)")
        ax.set_ylabel("camera-referred g = G/(R+G+B)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Channel chromaticity tracking (camera-referred; NOT colorimetric). "
        "Constant chromaticity = points collapse per ramp.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path.name}")


def plot_additivity_wyble(lines, out_path):
    import matplotlib.pyplot as plt

    black = lines["R"][1][0]
    levels_g, gray = lines["RGB"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for out_ch, ax in enumerate(axes):
        summed = np.zeros_like(levels_g)
        for single in ("R", "G", "B"):
            lv, q = lines[single]
            summed = summed + np.interp(
                levels_g, lv, q[:, out_ch] - black[out_ch]
            )
        measured = gray[:, out_ch] - black[out_ch]
        ax.plot(
            levels_g, measured, color="black", lw=2.0, marker="o", ms=4,
            label="gray ramp measured",
        )
        ax.plot(
            levels_g, summed, color="tab:orange", lw=1.8, ls="--", marker="s",
            ms=3.5, label="sum of R+G+B ramps",
        )
        ax.set_xlabel("input code (0-255)")
        if out_ch == 0:
            ax.set_ylabel("linear response (black-subtracted)")
        ax.set_title(f"output c_{'RGB'[out_ch]}")
        ax.grid(alpha=0.3)
        twin = ax.twinx()
        twin.plot(
            levels_g, measured - summed, color="tab:purple", lw=1.2,
            label="difference (measured − sum)",
        )
        twin.axhline(0.0, color="tab:purple", lw=0.6, ls=":")
        twin.set_ylabel("difference", color="tab:purple")
        twin.tick_params(axis="y", colors="tab:purple")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    fig.suptitle(
        "Additivity check, Wyble & Rosen style: gray-ramp response vs sum of "
        "single-channel responses (difference on secondary axis)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path.name}")


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=None)


def plot_session_stability(out_path):
    import matplotlib.pyplot as plt

    runs = [
        ("response r1 (settle 0.3s)",
         _REPO_ROOT / "data/color_grid_response_20260812/measurements.jsonl",
         "drive", "sentinel", "measured_mean"),
        ("response r2 (settle 2s)",
         _REPO_ROOT / "data/color_grid_response_20260812_r2/measurements.jsonl",
         "drive", "sentinel", "measured_mean"),
        ("tps calibration",
         _REPO_ROOT / "data/tps_calibration_20260812/manifest.jsonl",
         "drive", "sentinel", "roi_mean"),
        ("lut3d validation (tps)",
         _REPO_ROOT / "data/lut3d_validation_20260813/measurements.jsonl",
         "set", "sentinel", "measured_mean_q"),
        ("lut3d validation (linear)",
         _REPO_ROOT / "data/lut3d_validation_20260813_linear/measurements.jsonl",
         "set", "sentinel", "measured_mean_q"),
        # EPSON (2026-08-17): 存在するものだけ描かれる
        ("EPSON response (settle 2s)",
         _REPO_ROOT / "data/epson_color_grid_response_20260817/measurements.jsonl",
         "drive", "sentinel", "measured_mean"),
        ("EPSON tps calibration",
         _REPO_ROOT / "data/epson_tps_calibration_20260817/manifest.jsonl",
         "drive", "sentinel", "roi_mean"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = plt.get_cmap("tab10")
    for run_index, (label, path, kind_key, kind, value_key) in enumerate(runs):
        if not path.exists():
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        start = None
        times, values = [], []
        for row in rows:
            if row.get("status") not in (None, "ok"):
                continue
            ts = row.get("timestamp")
            if ts is None:
                continue
            t = _parse_ts(ts)
            if start is None:
                start = t
            if row.get(kind_key) == kind:
                times.append((t - start).total_seconds() / 60.0)
                values.append(row[value_key])
        if len(values) < 2:
            continue
        values = np.asarray(values, dtype=float)
        rel = values / values.mean(axis=0) - 1.0
        mean_rel = rel.mean(axis=1) * 100.0
        ax.plot(
            times, mean_rel, marker="o", ms=4, lw=1.5,
            color=colors(run_index), label=label,
        )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("minutes since run start")
    ax.set_ylabel("sentinel (128,128,128) deviation vs run mean [%]")
    ax.set_title(
        "Session stability across 08-12/13 capture runs (channel-mean "
        "relative deviation; NOT a cold-start warm-up series)"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path.name}")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = load_lines(args.response)
    plot_chromaticity_tracking(lines, out_dir / "chromaticity_tracking.png")
    plot_additivity_wyble(lines, out_dir / "additivity_wyble.png")
    plot_session_stability(out_dir / "session_stability.png")


if __name__ == "__main__":
    main()
