"""Six-dimensional views of the projector's input->output colour map.

The map (r, g, b) -> (c_R, c_G, c_B) is a 3D->3D function: its graph is a
6-dimensional object. Two encodings that show all six coordinates at once:

1. 3-D displacement fields: point position = the 3 input coordinates,
   arrow = the 3 output coordinates (as a displacement). Variant "raw"
   draws input code -> measured output; variant "residual" draws
   (per-channel 1-D gray-linearized prediction) -> measured output, so the
   arrows that remain are exactly the 3-D structure no per-channel 1-D LUT
   can remove (crosstalk + non-additivity).
2. Slice small-multiples: one panel per input-B level; per panel a quiver
   over the input (R, G) plane with arrows (Δc_R, Δc_G) and point colour
   Δc_B -- all six dimensions, quantitatively readable.

入力(r,g,b)→出力(c_R,c_G,c_B) は3D→3D写像でグラフは6次元。6成分を同時に
描く2つの符号化: ①3D変位場（位置=入力3、矢印=出力3。residual 版は
チャンネル毎1Dグレー線形化の予測→実測の変位で、1D LUT で消せない3D成分
だけが残る）②入力Bレベル別の小多重クイバー（矢印=(Δc_R,Δc_G)、点色=Δc_B）。

Usage:
    uv run --no-sync python examples/python/plot_io_map_6d.py \
        --calibration-dir data/tps_calibration_20260812 \
        --output-dir data/lut3d_validation_20260813/analysis [--gif]
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


def gray_linearizers(grid: np.ndarray, levels: np.ndarray):
    """Per-channel 1-D response from the gray diagonal.

    グレー対角の実測から、チャンネル毎の1D応答 g_ch(v) を作る
    （現行パイプラインの「1D線形化で説明できる成分」の定義）。
    """
    n = grid.shape[0]
    diag = np.array([grid[i, i, i] for i in range(n)])

    def make(channel):
        return lambda v: np.interp(v, levels, diag[:, channel])

    return [make(ch) for ch in range(3)]


def ideal_1d_prediction(codes: np.ndarray, linearizers) -> np.ndarray:
    """(N,3) per-channel 1-D prediction for every input code."""
    return np.stack(
        [linearizers[ch](codes[:, ch]) for ch in range(3)], axis=1
    )


def _draw_displacement(ax, origins, targets, colors, title):
    for k in range(len(origins)):
        ax.plot(
            [origins[k, 0], targets[k, 0]],
            [origins[k, 1], targets[k, 1]],
            [origins[k, 2], targets[k, 2]],
            color=colors[k], lw=0.9, alpha=0.85,
        )
    ax.scatter(
        origins[:, 0], origins[:, 1], origins[:, 2],
        c=colors, s=10, depthshade=False, edgecolors="none",
    )
    ax.scatter(
        targets[:, 0], targets[:, 1], targets[:, 2],
        c=colors, s=22, depthshade=False, edgecolors="black", linewidths=0.3,
    )
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_title(title, fontsize=10)
    ax.set_box_aspect((1, 1, 1))


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    codes, c, index, n = load_solid(args.calibration_dir)
    grid = np.full((n, n, n, 3), np.nan)
    for k in range(len(codes)):
        grid[tuple(index[k])] = c[k]
    levels = np.unique(codes[:, 0])
    linearizers = gray_linearizers(grid, levels)
    ideal = ideal_1d_prediction(codes, linearizers)
    display = np.clip(codes ** (1 / 2.2), 0, 1)

    # ── 1. 3-D displacement fields (raw and 1-D-residual) ──────────────
    for name, origins, targets, title in (
        (
            "io_map_raw",
            codes,
            c,
            "input code -> measured output (all 6 dims: position=input, "
            "arrow=output)",
        ),
        (
            "io_map_residual",
            ideal,
            c,
            "1-D gray-linearized prediction -> measured output\n"
            "(remaining arrows = crosstalk + non-additivity, "
            "invisible to any per-channel LUT)",
        ),
    ):
        fig = plt.figure(figsize=(9.5, 8.5))
        ax = fig.add_subplot(111, projection="3d")
        _draw_displacement(ax, origins, targets, display, title)
        ax.view_init(elev=22, azim=40)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=110)
        plt.close(fig)
        print(f"saved {name}.png")

    if args.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig = plt.figure(figsize=(7.2, 6.6))
        ax = fig.add_subplot(111, projection="3d")
        _draw_displacement(
            ax, ideal, c, display,
            "1-D-linearized prediction -> measured (3-D-only residual)",
        )

        def update(frame):
            ax.view_init(elev=20, azim=frame * 4)
            return ()

        anim = FuncAnimation(fig, update, frames=90, interval=80)
        gif_path = out_dir / "io_map_residual_rotation.gif"
        anim.save(gif_path, writer=PillowWriter(fps=12), dpi=80)
        plt.close(fig)
        print(f"saved {gif_path}")

    # ── 2. Slice small-multiples: all 6 dims on paper ───────────────────
    residual = c - ideal
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    vmax = float(np.percentile(np.abs(residual[:, 2]), 98))
    scatter = None
    for panel, b_idx in enumerate(range(n)):
        ax = axes[panel // 3, panel % 3]
        mask = index[:, 2] == b_idx
        rr = codes[mask, 0]
        gg = codes[mask, 1]
        d = residual[mask]
        scatter = ax.scatter(
            rr, gg, c=d[:, 2], cmap="coolwarm", vmin=-vmax, vmax=vmax,
            s=520, edgecolors="0.4", linewidths=0.5, zorder=1,
        )
        ax.quiver(
            rr, gg, d[:, 0], d[:, 1],
            angles="xy", scale_units="xy", scale=1.0,
            width=0.006, color="black", zorder=2,
        )
        ax.set_title(
            f"input B = {int(round(levels[b_idx] * 255))}", fontsize=11
        )
        ax.set_xlabel("input R (code, 0-1)")
        if panel % 3 == 0:
            ax.set_ylabel("input G (code, 0-1)")
        ax.set_xlim(-0.15, 1.25)
        ax.set_ylim(-0.15, 1.25)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    fig.colorbar(
        scatter, ax=axes, shrink=0.75,
        label="Δc_B (residual of the B output after 1-D linearization)",
    )
    fig.suptitle(
        "6-dim I/O map, sliced by input B: arrows = (Δc_R, Δc_G), "
        "colour = Δc_B — the residual after per-channel 1-D linearization "
        "(arrow length 0.1 = 10% of white)",
        fontsize=13,
    )
    fig.savefig(out_dir / "io_map_slices_residual.png", dpi=110)
    plt.close(fig)
    print("saved io_map_slices_residual.png")


if __name__ == "__main__":
    main()
