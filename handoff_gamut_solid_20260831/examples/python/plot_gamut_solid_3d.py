"""3-D visualization of the projector's input->output colour relationship.

Draws, in the normalized output colour space ``c`` (black=0, white=1 per
channel), the deformed solid that the 6^3 input-code lattice actually
produces: wireframe lines along the input lattice's R/G/B directions, the
ideal unit cube for reference, the gray-axis trajectory (the code-207 jump
shows as a gap), and the three single-channel edges (their bending away
from the reference axes IS the crosstalk). Optionally overlays a
validation run as requested->achieved displacement lines.

入力6³格子が実測出力色空間（正規化 c）で作る「変形立方体」を描く。
理想の単位立方体（灰色）との差が入出力関係の歪みそのもの:
グレー対角の切れ目=207跳び、単ch稜線の曲がり=クロストーク、
全体の内側への湾曲=ガンマ。検証ラン（要求→達成の変位線）も重畳できる。

Usage:
    uv run --no-sync python examples/python/plot_gamut_solid_3d.py \
        --calibration-dir data/tps_calibration_20260812 \
        [--validation data/lut3d_validation_20260813/measurements.jsonl] \
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

from yoshidacomp.compensation import color_lut3d as cl3


def load_solid(calibration_dir: Path, active_region=None):
    """(codes (N,3), normalized outputs c (N,3), lattice index (N,3)).

    ``active_region`` は校正時の投影ROI（省略時は metadata.json から読む）。
    """
    import json

    from examples.python.generate_tps_and_matrix_compensation import (
        _load_calibration,
    )

    frames, codes = _load_calibration(calibration_dir)
    meta = json.loads(
        (calibration_dir / "metadata.json").read_text(encoding="utf-8")
    )
    ds = int(meta.get("downsample", 4))
    if active_region is None:
        active_region = meta["projector"]["active_region"]
    x0, y0, aw, ah = (int(v) for v in active_region)
    region = np.s_[y0 // ds : (y0 + ah) // ds, x0 // ds : (x0 + aw) // ds]
    mean_q = frames[:, region[0], region[1], :].reshape(len(frames), -1, 3).mean(
        axis=1
    )
    c = cl3.normalize_reference_response(mean_q, codes)
    levels = np.unique(codes[:, 0])
    index = np.stack(
        [np.searchsorted(levels, codes[:, k]) for k in range(3)], axis=1
    )
    return codes, c, index, len(levels)


def _build_grid(codes, c, index, n):
    grid = np.full((n, n, n, 3), np.nan)
    colors = np.full((n, n, n, 3), np.nan)
    for k in range(len(codes)):
        i, j, l = index[k]
        grid[i, j, l] = c[k]
        # 認識しやすいよう入力色を sRGB 風に表示（表示用のみ）
        colors[i, j, l] = np.clip(codes[k] ** (1 / 2.2), 0, 1)
    return grid, colors


def _draw_reference_cube(ax):
    for s, e in (
        ((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, 0, 1)),
        ((1, 0, 0), (1, 1, 0)), ((1, 0, 0), (1, 0, 1)),
        ((0, 1, 0), (1, 1, 0)), ((0, 1, 0), (0, 1, 1)),
        ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1)),
        ((1, 1, 0), (1, 1, 1)), ((1, 0, 1), (1, 1, 1)), ((0, 1, 1), (1, 1, 1)),
    ):
        ax.plot(*zip(s, e), color="0.55", lw=1.0, ls="--", alpha=0.9)


def _draw_solid(ax, codes, c, index, n, validation=None, shell_alpha=0.6):
    grid, colors = _build_grid(codes, c, index, n)
    _draw_reference_cube(ax)

    # Outer shell of the input cube rendered as 6 coloured surfaces.
    # 入力立方体の外殻6面をサーフェスで描く（内部格子は描かない）。
    def face(sel_grid, sel_color):
        fc = 0.25 * (
            sel_color[:-1, :-1] + sel_color[1:, :-1]
            + sel_color[:-1, 1:] + sel_color[1:, 1:]
        )
        rgba = np.concatenate(
            [fc, np.full((*fc.shape[:2], 1), shell_alpha)], axis=-1
        )
        ax.plot_surface(
            sel_grid[..., 0], sel_grid[..., 1], sel_grid[..., 2],
            facecolors=rgba, shade=False,
            edgecolor=(0.25, 0.25, 0.25, 0.5), linewidth=0.4,
            rcount=n, ccount=n,
        )

    for axis in range(3):
        for side in (0, n - 1):
            slicer = [slice(None)] * 3
            slicer[axis] = side
            face(grid[tuple(slicer)], colors[tuple(slicer)])

    # gray diagonal (the code-207 jump shows as one long segment)
    diag = np.array([grid[i, i, i] for i in range(n)])
    ax.plot(
        diag[:, 0], diag[:, 1], diag[:, 2],
        color="black", lw=3.0, marker="o", ms=5, label="gray axis", zorder=10,
    )
    for axis, color, label in (
        (0, "darkred", "R-only edge"),
        (1, "darkgreen", "G-only edge"),
        (2, "navy", "B-only edge"),
    ):
        sel = np.array(
            [grid[tuple(np.eye(3, dtype=int)[axis] * i)] for i in range(n)]
        )
        ax.plot(
            sel[:, 0], sel[:, 1], sel[:, 2],
            color=color, lw=2.2, marker="s", ms=3.5, label=label, zorder=10,
        )

    if validation is not None:
        requested, achieved = validation
        inside = np.all(requested <= 1.0, axis=1)
        for mask, color, label in (
            (inside, "magenta", "validation (in domain)"),
            (~inside, "darkorange", "validation (out of domain)"),
        ):
            if not np.any(mask):
                continue
            first = True
            for k in np.where(mask)[0]:
                ax.plot(
                    [requested[k, 0], achieved[k, 0]],
                    [requested[k, 1], achieved[k, 1]],
                    [requested[k, 2], achieved[k, 2]],
                    color=color, lw=1.2, alpha=0.9, zorder=9,
                )
                ax.scatter(
                    *requested[k], marker="x", color=color, s=24, zorder=9,
                    label=label if first else None,
                )
                first = False

    ax.set_xlabel("c_R (normalized output)")
    ax.set_ylabel("c_G")
    ax.set_zlabel("c_B")
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1.4)
    ax.set_zlim(0, 1.4)
    ax.set_box_aspect((1, 1, 1))


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    codes, c, index, n = load_solid(args.calibration_dir)

    validation = None
    if args.validation is not None:
        from examples.python.analyze_lut3d_validation import (
            load_rows,
            normalize_with_anchors,
        )

        data = normalize_with_anchors(load_rows(args.validation))
        keep = data["sets"] != "sentinel"
        validation = (data["requested"][keep], data["achieved"][keep])

    for name, elev, azim in (
        ("view_iso", 22, 40),
        ("view_rg", 18, -35),
        ("view_top", 62, 25),
    ):
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        _draw_solid(ax, codes, c, index, n)
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(
            "Measured output solid of the 6^3 input lattice "
            "(gray dashed = ideal cube)"
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"gamut_solid_{name}.png", dpi=110)
        plt.close(fig)
        print(f"saved gamut_solid_{name}.png")

    if validation is not None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        _draw_solid(
            ax, codes, c, index, n, validation=validation, shell_alpha=0.18
        )
        ax.view_init(elev=22, azim=40)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(
            "3D LUT validation: requested (x) -> achieved, over the "
            "measured solid"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "gamut_solid_validation.png", dpi=110)
        plt.close(fig)
        print("saved gamut_solid_validation.png")

    if args.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig = plt.figure(figsize=(7, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        _draw_solid(ax, codes, c, index, n)
        ax.legend(loc="upper left", fontsize=7)
        ax.set_title("Input lattice -> measured output solid")

        def update(frame):
            ax.view_init(elev=20, azim=frame * 4)
            return ()

        anim = FuncAnimation(fig, update, frames=90, interval=80)
        gif_path = out_dir / "gamut_solid_rotation.gif"
        anim.save(gif_path, writer=PillowWriter(fps=12), dpi=80)
        plt.close(fig)
        print(f"saved {gif_path}")


if __name__ == "__main__":
    main()
