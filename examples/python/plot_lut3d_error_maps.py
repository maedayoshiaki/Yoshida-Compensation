"""Error-visualization suite for the baked 3-D LUT (and its TPS source).

Three figures aimed specifically at WHERE / WHICH WAY / HOW MUCH the
colour map misses:

1. ``lut3d_error_slices.png`` — predicted achieved-colour error field
   over the requested-colour cube, sliced by c_B: for a dense grid of
   requests c the LUT's code p = LUT(c) is pushed through the reference
   FORWARD map (fit from the same calibration, accurate to dE~3) and the
   round-trip error dE(F(LUT(c)), c) is drawn as a heatmap. The reachable
   solid's cross-section is outlined (outside it the error is gamut, not
   map). Hardware validation points are overlaid with the SAME colour
   scale, so model-predicted and measured errors can be compared at a
   glance.
2. ``lut3d_error_ab_plane.png`` — requested -> achieved displacement
   arrows in the a*b* chroma plane plus a lightness panel: shows the
   systematic DIRECTION of the misses (hue/chroma bias), which a scalar
   dE hides.
3. ``lut3d_error_cdf.png`` — per-set cumulative distributions of the
   hardware-measured dE.

3D LUT / TPS の「ズレ」可視化3点セット: ①要求色立方体上の予測誤差場の
スライス（到達殻断面つき、実測検証点を同一色尺度で重畳）②a*b*面の変位
矢印（外し方の方向）③実測ΔEのセット別CDF。

Usage:
    uv run --no-sync python examples/python/plot_lut3d_error_maps.py \
        --calibration-dir data/tps_calibration_20260812 \
        --lut data/tps_compensation_20260812/inverse_lut3d.npz \
        --validation data/lut3d_validation_20260813/measurements.jsonl \
        --output-dir data/lut3d_validation_20260813/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.python.analyze_color_grid_response import delta_e76  # noqa: E402
from yoshidacomp.compensation import color_lut3d as cl3  # noqa: E402


def _to_lab(c: np.ndarray) -> np.ndarray:
    """Normalized c (white=1) -> Lab via the linear-sRGB assumption."""
    srgb_to_xyz = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    white_xyz = srgb_to_xyz @ np.ones(3)
    xyz = np.clip(c, 0.0, None) @ srgb_to_xyz.T / white_xyz

    def f(t):
        d = 6.0 / 29.0
        return np.where(t > d**3, np.cbrt(np.maximum(t, 0)), t / (3 * d**2) + 4 / 29)

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    return np.stack(
        [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1
    )


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import Delaunay

    from examples.python.analyze_lut3d_validation import (
        load_rows,
        normalize_with_anchors,
    )
    from examples.python.generate_tps_and_matrix_compensation import (
        _load_calibration,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reference data and maps. 校正から基準の c と順写像を作る。
    frames, codes = _load_calibration(args.calibration_dir)
    calib_meta = json.loads(
        (args.calibration_dir / "metadata.json").read_text(encoding="utf-8")
    )
    ds = int(calib_meta.get("downsample", 4))
    x0, y0, aw, ah = (int(v) for v in calib_meta["projector"]["active_region"])
    region = np.s_[y0 // ds : (y0 + ah) // ds, x0 // ds : (x0 + aw) // ds]
    mean_q = frames[:, region[0], region[1], :].reshape(len(frames), -1, 3).mean(
        axis=1
    )
    del frames
    c_cal = cl3.normalize_reference_response(mean_q, codes)
    forward = RBFInterpolator(
        codes, c_cal, kernel="thin_plate_spline", smoothing=1e-6
    )
    hull = Delaunay(c_cal)
    lut = cl3.load_lut3d(args.lut)

    data = normalize_with_anchors(load_rows(args.validation))
    val_sets = data["sets"]
    keep = val_sets != "sentinel"
    val_req = data["requested"][keep]
    val_ach = data["achieved"][keep]
    val_sets = val_sets[keep]
    val_de = delta_e76(val_ach, val_req, np.ones(3))

    # ── 1. predicted error field, sliced by c_B ─────────────────────────
    grid_n = 61
    axis_rg = np.linspace(0.0, 1.3, grid_n)
    b_levels = (0.1, 0.3, 0.5, 0.7, 0.9, 1.1)
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    vmax = 25.0
    mesh = None
    for panel, b in enumerate(b_levels):
        ax = axes[panel // 3, panel % 3]
        rr, gg = np.meshgrid(axis_rg, axis_rg)
        req = np.stack([rr, gg, np.full_like(rr, b)], axis=-1).reshape(-1, 3)
        p = cl3.apply_lut3d(
            req.reshape(1, -1, 3).astype(np.float32), lut
        ).reshape(-1, 3)
        p8 = np.round(p * 255) / 255
        achieved = forward(p8)
        de = delta_e76(achieved, req, np.ones(3)).reshape(grid_n, grid_n)
        inside = (hull.find_simplex(req) >= 0).reshape(grid_n, grid_n)
        de_masked = np.where(inside, de, np.nan)
        mesh = ax.pcolormesh(
            rr, gg, de_masked, cmap="magma_r", vmin=0, vmax=vmax, shading="auto"
        )
        # 殻の外は薄灰で示す（そこは色域誤差でありLUTの誤差ではない）
        ax.contourf(
            rr, gg, (~inside).astype(float), levels=[0.5, 1.5],
            colors=["0.88"], alpha=1.0, zorder=0,
        )
        ax.contour(
            rr, gg, inside.astype(float), levels=[0.5],
            colors="k", linewidths=1.0,
        )
        near = np.abs(val_req[:, 2] - b) < 0.09
        if np.any(near):
            ax.scatter(
                val_req[near, 0], val_req[near, 1], c=val_de[near],
                cmap="magma_r", vmin=0, vmax=vmax, s=90,
                edgecolors="cyan", linewidths=1.5, zorder=3,
            )
        ax.set_title(f"requested c_B = {b:.1f}", fontsize=11)
        ax.set_xlabel("requested c_R")
        if panel % 3 == 0:
            ax.set_ylabel("requested c_G")
        ax.set_aspect("equal")
    fig.colorbar(mesh, ax=axes, shrink=0.75, label="round-trip ΔE76 (predicted)")
    fig.suptitle(
        "3D LUT round-trip error field ΔE(F(LUT(c)), c), sliced by c_B — "
        "black line = reachable-solid cross-section, grey = out of gamut, "
        "cyan-edged dots = hardware validation (same colour scale)",
        fontsize=13,
    )
    fig.savefig(out_dir / "lut3d_error_slices.png", dpi=110)
    plt.close(fig)
    print("saved lut3d_error_slices.png")

    # ── 2. a*b* displacement + lightness panel ─────────────────────────
    lab_req = _to_lab(val_req)
    lab_ach = _to_lab(val_ach)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    ax = axes[0]
    styles = {
        "gray": ("o", "black"),
        "single": ("s", "tab:purple"),
        "random_box": ("^", "tab:blue"),
        "held_out": ("D", "tab:green"),
    }
    for set_name, (marker, color) in styles.items():
        mask = val_sets == set_name
        if not np.any(mask):
            continue
        for k in np.where(mask)[0]:
            ax.annotate(
                "",
                xy=(lab_ach[k, 1], lab_ach[k, 2]),
                xytext=(lab_req[k, 1], lab_req[k, 2]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.8),
            )
        ax.scatter(
            lab_req[mask, 1], lab_req[mask, 2], marker=marker, s=30,
            color=color, label=set_name, zorder=3,
        )
    ax.axhline(0, color="0.7", lw=0.8)
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_xlabel("a* (− green   /   + red)")
    ax.set_ylabel("b* (− blue   /   + yellow)")
    ax.set_title("chroma plane: requested (marker) → achieved (arrow head)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")

    ax = axes[1]
    for set_name, (marker, color) in styles.items():
        mask = val_sets == set_name
        if not np.any(mask):
            continue
        ax.scatter(
            lab_req[mask, 0], lab_ach[mask, 0] - lab_req[mask, 0],
            marker=marker, s=34, color=color, label=set_name,
        )
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("requested L*")
    ax.set_ylabel("achieved L* − requested L*")
    ax.set_title("lightness error vs requested lightness")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.suptitle(
        "Direction of the misses (hardware validation, Lab via "
        "linear-sRGB assumption)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "lut3d_error_ab_plane.png", dpi=120)
    plt.close(fig)
    print("saved lut3d_error_ab_plane.png")

    # ── 3. per-set dE CDF ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for set_name, (marker, color) in styles.items():
        mask = val_sets == set_name
        if not np.any(mask):
            continue
        values = np.sort(val_de[mask])
        ax.step(
            values, np.arange(1, len(values) + 1) / len(values),
            where="post", color=color, lw=2,
            label=f"{set_name} (n={mask.sum()}, median {np.median(values):.1f})",
        )
    ax.set_xlabel("measured ΔE76")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Hardware-measured ΔE distribution by validation set")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "lut3d_error_cdf.png", dpi=120)
    plt.close(fig)
    print("saved lut3d_error_cdf.png")


if __name__ == "__main__":
    main()
