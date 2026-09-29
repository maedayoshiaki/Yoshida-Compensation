"""Per-channel input -> output response families from the 6^3 calibration.

A 3x3 matrix of panels: column = swept INPUT channel, row = measured
OUTPUT channel. Each panel shows 36 curves -- one per combination of the
two non-swept input channels -- coloured by that background colour.
Diagonal panels are the own-channel response family (its vertical spread
IS the non-additivity a per-channel 1-D LUT cannot represent); off-
diagonal panels are crosstalk versus input. The pure single-channel curve
(others = 0) is drawn bold solid, the others-at-255 curve bold dashed.

6³校正から「入力チャンネル別の掃引に対する3出力の応答」を 3×3 パネルで
描く（列=掃引する入力ch、行=計測出力ch、各パネルに他2chの背景36通りの
曲線ファミリー）。対角パネルの縦の広がりが非加法性そのもの、非対角は
クロストークの入力依存。他ch=0 の純単独駆動を太実線、他ch=255 を太破線で
強調する。

Usage:
    uv run --no-sync python examples/python/plot_channel_response_families.py \
        --calibration-dir data/tps_calibration_20260812 \
        --output-dir data/lut3d_validation_20260813/analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.python.plot_gamut_solid_3d import load_solid  # noqa: E402


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    codes, c, index, n = load_solid(args.calibration_dir)
    grid = np.full((n, n, n, 3), np.nan)
    for k in range(len(codes)):
        grid[tuple(index[k])] = c[k]
    levels = np.unique(codes[:, 0])
    code_255 = (levels * 255).round().astype(int)

    input_names = ("input R", "input G", "input B")
    output_names = ("output c_R", "output c_G", "output c_B")

    fig, axes = plt.subplots(3, 3, figsize=(17, 15), sharex=True)
    for col, sweep_axis in enumerate(range(3)):
        others = [a for a in range(3) if a != sweep_axis]
        for row, out_ch in enumerate(range(3)):
            ax = axes[row, col]
            for ia in range(n):
                for ib in range(n):
                    sel = [None, None, None]
                    sel[sweep_axis] = slice(None)
                    sel[others[0]] = ia
                    sel[others[1]] = ib
                    curve = grid[tuple(sel)][:, out_ch]
                    background = np.zeros(3)
                    background[others[0]] = levels[ia]
                    background[others[1]] = levels[ib]
                    line_color = np.clip(background ** (1 / 2.2), 0.0, 0.85)
                    is_zero = ia == 0 and ib == 0
                    is_full = ia == n - 1 and ib == n - 1
                    ax.plot(
                        code_255,
                        curve,
                        color="black" if is_zero else line_color,
                        lw=2.8 if (is_zero or is_full) else 0.9,
                        ls="--" if is_full else "-",
                        alpha=1.0 if (is_zero or is_full) else 0.55,
                        zorder=5 if (is_zero or is_full) else 2,
                    )
            if row == col:
                # 参照: 対角パネルには「掃引chの寄与が加法的なら全曲線が
                # 平行移動になる」ことの目安として、純単独駆動の曲線の形が
                # 見やすいよう薄いグリッドのみ（追加の参照線は引かない）。
                ax.set_facecolor((0.97, 0.97, 0.97))
            ax.grid(alpha=0.3)
            if row == 2:
                ax.set_xlabel(f"{input_names[col]} (code 0-255)")
            if col == 0:
                ax.set_ylabel(output_names[row])
            if row == 0:
                ax.set_title(
                    f"sweep {input_names[col]}\n"
                    "(36 curves = backgrounds of the other 2 channels)",
                    fontsize=11,
                )
    handles = [
        plt.Line2D([0], [0], color="black", lw=2.8, label="others = 0 (single-channel drive)"),
        plt.Line2D(
            [0], [0], color=(0.85, 0.85, 0.85), lw=2.8, ls="--",
            label="others = 255",
        ),
        plt.Line2D(
            [0], [0], color=(0.6, 0.4, 0.2), lw=1.0, alpha=0.6,
            label="other backgrounds (line colour = background colour)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10)
    fig.suptitle(
        "Channel response families: swept input (columns) vs measured "
        "outputs (rows).\nVertical spread of a family = non-additivity; "
        "off-diagonal panels = crosstalk.",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.savefig(out_dir / "channel_response_families.png", dpi=110)
    plt.close(fig)
    print("saved channel_response_families.png")


if __name__ == "__main__":
    main()
