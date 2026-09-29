"""Input->output relations along the 7 drive lines, vs gamma / 1-D LUT models.

From the colour-grid response study (r2), extracts the seven input lines
R, G, B, RG, RB, GB, RGB(gray) and draws:

1. ``line_responses.png``   - measured outputs per line, overlaid with the
   additive prediction composed from the single-channel lines and with an
   endpoint-scaled gamma-2.2 shape.
2. ``line_shapes.png``      - per OUTPUT channel, the normalized response
   shapes of every line that drives it, plus the gamma-2.2 curve. A 1-D
   LUT (channel_ramp or gray fit) is exactly ONE curve per channel, so the
   spread of this family is the error floor of any per-channel model.
3. ``line_additivity.png``  - measured / (sum of singles) ratio curves for
   RG, RB, GB, RGB per output channel (the level-resolved proportionality).

7本の入力ライン（R,G,B,RG,RB,GB,グレー）の入出力関係を、ガンマ2.2 と
1D LUT 系モデルの前提（チャンネル毎の単一形状+加法性）と比較する3枚組。

Usage:
    uv run --no-sync python examples/python/plot_line_proportionality.py \
        --input data/color_grid_response_20260812_r2/measurements.jsonl \
        --output-dir data/color_grid_response_20260812_r2/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

LINES: Dict[str, Tuple[int, ...]] = {
    "R": (0,),
    "G": (1,),
    "B": (2,),
    "RG": (0, 1),
    "RB": (0, 2),
    "GB": (1, 2),
    "RGB": (0, 1, 2),
}
LINE_COLORS = {
    "R": "tab:red", "G": "tab:green", "B": "tab:blue",
    "RG": "olive", "RB": "purple", "GB": "teal", "RGB": "black",
}
OUT_COLORS = ("tab:red", "tab:green", "tab:blue")


def load_lines(jsonl_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """{line: (levels (M,), measured (M,3))} sorted by level; raw drive only."""
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw = {}
    for r in rows:
        if r.get("status") == "ok" and r.get("drive") == "raw":
            raw[tuple(r["requested_rgb"])] = np.asarray(r["measured_mean"])
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, channels in LINES.items():
        levels, values = [], []
        for rgb, q in raw.items():
            active = [rgb[c] for c in channels]
            inactive = [rgb[c] for c in range(3) if c not in channels]
            if len(set(active)) == 1 and all(v == 0 for v in inactive):
                if name != "RGB" and active[0] == 0 and len(channels) < 3:
                    pass  # (0,0,0) belongs to every line; keep it
                levels.append(active[0])
                values.append(q)
        order = np.argsort(levels)
        out[name] = (
            np.asarray(levels, float)[order],
            np.asarray(values)[order],
        )
    return out


def additive_prediction(
    lines: Dict[str, Tuple[np.ndarray, np.ndarray]],
    name: str,
    levels: np.ndarray,
) -> np.ndarray:
    """Sum of the single-channel responses (black-subtracted) at ``levels``."""
    black = lines["R"][1][0]  # level-0 measurement (shared by all lines)
    total = np.zeros((len(levels), 3))
    for channel, single in ((0, "R"), (1, "G"), (2, "B")):
        if channel in LINES[name]:
            lv, q = lines[single]
            for out_ch in range(3):
                total[:, out_ch] += np.interp(levels, lv, q[:, out_ch] - black[out_ch])
    return total + black


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=2.2)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = load_lines(args.input)
    black = lines["R"][1][0]

    # ── 1. per-line measured vs additive vs gamma ───────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    for panel, name in enumerate(LINES):
        ax = axes[panel // 4, panel % 4]
        levels, q = lines[name]
        pred = additive_prediction(lines, name, levels)
        for out_ch in range(3):
            ax.plot(
                levels, q[:, out_ch], color=OUT_COLORS[out_ch], lw=1.8,
                marker="o", ms=3,
                label=f"measured c_{'RGB'[out_ch]}" if panel == 0 else None,
            )
            if len(LINES[name]) > 1:
                ax.plot(
                    levels, pred[:, out_ch], color=OUT_COLORS[out_ch],
                    lw=1.4, ls="--", alpha=0.8,
                    label="additive sum of singles" if panel == 0 and out_ch == 0 else None,
                )
            # endpoint-scaled gamma shape for driven channels
            if out_ch in LINES[name]:
                gamma_curve = black[out_ch] + (q[-1, out_ch] - black[out_ch]) * (
                    (levels / 255.0) ** args.gamma
                )
                ax.plot(
                    levels, gamma_curve, color="0.3", lw=1.0, ls=":",
                    label=f"gamma {args.gamma:g} shape" if panel == 0 and out_ch == LINES[name][0] else None,
                )
        ax.set_title(f"{name} line", fontsize=12)
        ax.grid(alpha=0.3)
        if panel // 4 == 1:
            ax.set_xlabel("input code (0-255)")
        if panel % 4 == 0:
            ax.set_ylabel("measured linear response")
    axes[1, 3].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 3].legend(handles, labels, loc="center", fontsize=11)
    fig.suptitle(
        "Seven drive lines: measured outputs (solid) vs additive composition "
        "of single channels (dashed) vs endpoint-scaled gamma shape (dotted)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "line_responses.png", dpi=110)
    plt.close(fig)
    print("saved line_responses.png")

    # ── 2. normalized shapes per output channel ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for out_ch, ax in enumerate(axes):
        x = np.linspace(0, 255, 100)
        ax.plot(
            x, (x / 255.0) ** args.gamma, color="0.2", lw=2.2, ls=":",
            label=f"gamma {args.gamma:g}",
        )
        for name in LINES:
            if out_ch not in LINES[name]:
                continue
            levels, q = lines[name]
            y = q[:, out_ch] - black[out_ch]
            denom = y[-1] if abs(y[-1]) > 1e-9 else 1.0
            ax.plot(
                levels, y / denom, color=LINE_COLORS[name], lw=1.8,
                marker="o", ms=3.5, label=f"{name} line",
            )
        ax.set_title(
            f"output c_{'RGB'[out_ch]}: normalized shape per line\n"
            "(a per-channel 1-D LUT = exactly ONE of these curves)"
        )
        ax.set_xlabel("input code (0-255)")
        if out_ch == 0:
            ax.set_ylabel("response (black-subtracted, endpoint-normalized)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Shape families: if a single 1-D curve per channel sufficed, "
        "each panel's curves would coincide",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "line_shapes.png", dpi=120)
    plt.close(fig)
    print("saved line_shapes.png")

    # ── 3. additivity ratio curves ──────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.2), sharey=True)
    for panel, name in enumerate(("RG", "RB", "GB", "RGB")):
        ax = axes[panel]
        levels, q = lines[name]
        pred = additive_prediction(lines, name, levels)
        for out_ch in LINES[name]:
            num = q[:, out_ch] - black[out_ch]
            den = pred[:, out_ch] - black[out_ch]
            valid = np.abs(den) > 2e-3
            ax.plot(
                levels[valid], num[valid] / den[valid],
                color=OUT_COLORS[out_ch], lw=1.8, marker="o", ms=3.5,
                label=f"c_{'RGB'[out_ch]}",
            )
        ax.axhline(1.0, color="k", lw=1, ls="--")
        ax.set_title(f"{name}: measured / additive-sum")
        ax.set_xlabel("input code (0-255)")
        if panel == 0:
            ax.set_ylabel("proportionality ratio")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(0.4, 2.6)
    fig.suptitle(
        "Level-resolved additivity: ratio of the measured mixed-line response "
        "to the sum of its single-channel responses (1.0 = additive)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "line_additivity.png", dpi=120)
    plt.close(fig)
    print("saved line_additivity.png")


if __name__ == "__main__":
    main()
