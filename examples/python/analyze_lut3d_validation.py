"""Analyze a validate_lut3d.py run: requested vs achieved colours.

``measurements.jsonl`` を読み、当該ランの黒/白アンカーで達成色を正規化して
要求色と突き合わせる。セット別（gray / single / random_box / held_out）の
誤差、グレー軸の線形性、番兵ドリフトを数値と PNG にまとめる。
撮影ハードウェアには依存しない（再実行可能）。

Usage:
    uv run --no-sync python examples/python/analyze_lut3d_validation.py \
        --input data/lut3d_validation_20260813/measurements.jsonl \
        --output-dir data/lut3d_validation_20260813/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.python.analyze_color_grid_response import delta_e76  # noqa: E402


def load_rows(jsonl_path: Path) -> List[dict]:
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No measurements in {jsonl_path}")
    return rows


def normalize_with_anchors(rows: List[dict]) -> Dict[str, np.ndarray]:
    """Normalize measured q with the run's own black/white anchors."""
    black = np.mean(
        [r["measured_mean_q"] for r in rows if r["set"] == "anchor_black"],
        axis=0,
    )
    white = np.mean(
        [r["measured_mean_q"] for r in rows if r["set"] == "anchor_white"],
        axis=0,
    )
    denom = np.where(np.abs(white - black) < 1e-12, 1.0, white - black)
    data = [r for r in rows if r["set"] not in ("anchor_black", "anchor_white")]
    measured_q = np.array([r["measured_mean_q"] for r in data])
    return {
        "sets": np.array([r["set"] for r in data]),
        "requested": np.array([r["requested_c"] for r in data]),
        "achieved": (measured_q - black) / denom,
        "black": black,
        "white": white,
    }


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.input)
    data = normalize_with_anchors(rows)
    sets, requested, achieved = data["sets"], data["requested"], data["achieved"]
    white = data["white"]

    summary: Dict = {
        "input": str(args.input),
        "black_anchor": [float(v) for v in data["black"]],
        "white_anchor": [float(v) for v in white],
    }

    # Sentinel drift (raw mid gray; requested_c is the (-1,-1,-1) marker).
    sentinel_mask = sets == "sentinel"
    if np.any(sentinel_mask):
        s = achieved[sentinel_mask]
        rel = s / s.mean(axis=0) - 1.0
        summary["sentinel_max_abs_relative"] = [
            float(v) for v in np.max(np.abs(rel), axis=0)
        ]

    valid = ~sentinel_mask
    error = achieved[valid] - requested[valid]
    de = delta_e76(achieved[valid], requested[valid], np.ones(3))
    summary["overall"] = {
        "n": int(valid.sum()),
        "rmse_c": float(np.sqrt(np.mean(error**2))),
        "p95_abs_c": float(np.percentile(np.abs(error), 95)),
        "max_abs_c": float(np.max(np.abs(error))),
        "delta_e76_mean": float(de.mean()),
        "delta_e76_p95": float(np.percentile(de, 95)),
    }
    for set_name in ("gray", "single", "random_box", "held_out"):
        mask = sets == set_name
        if not np.any(mask):
            continue
        err = achieved[mask] - requested[mask]
        de_s = delta_e76(achieved[mask], requested[mask], np.ones(3))
        summary[set_name] = {
            "n": int(mask.sum()),
            "rmse_c": float(np.sqrt(np.mean(err**2))),
            "p95_abs_c": float(np.percentile(np.abs(err), 95)),
            "max_abs_c": float(np.max(np.abs(err))),
            "delta_e76_mean": float(de_s.mean()),
        }

    # ── plots ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    gray_mask = sets == "gray"
    order = np.argsort(requested[gray_mask][:, 0])
    levels = requested[gray_mask][order, 0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    for channel, (name, color) in enumerate(
        zip("RGB", ("tab:red", "tab:green", "tab:blue"))
    ):
        ax.plot(
            levels, achieved[gray_mask][order, channel],
            marker="o", ms=4, color=color, label=f"achieved {name}",
        )
    ax.set_xlabel("requested c")
    ax.set_ylabel("achieved c")
    ax.set_title("gray axis through 3D LUT")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    markers = {"gray": "o", "single": "s", "random_box": "^", "held_out": "D"}
    for set_name, marker in markers.items():
        mask = sets == set_name
        if not np.any(mask):
            continue
        for channel, color in enumerate(("tab:red", "tab:green", "tab:blue")):
            ax.scatter(
                requested[mask][:, channel],
                achieved[mask][:, channel],
                s=14, marker=marker, color=color, alpha=0.6,
                label=set_name if channel == 0 else None,
            )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("requested c (per channel)")
    ax.set_ylabel("achieved c")
    ax.set_title("all validation colours")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    names = [n for n in ("gray", "single", "random_box", "held_out") if n in summary]
    ax.bar(
        np.arange(len(names)) - 0.2,
        [summary[n]["rmse_c"] for n in names],
        width=0.4, label="RMSE (c units)", color="tab:blue",
    )
    ax.bar(
        np.arange(len(names)) + 0.2,
        [summary[n]["p95_abs_c"] for n in names],
        width=0.4, label="p95 |err|", color="tab:orange",
    )
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_title("achieved-colour error by set")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle("3D LUT hardware validation: requested vs achieved colour")
    fig.tight_layout()
    fig.savefig(out_dir / "lut3d_validation.png", dpi=120)
    plt.close(fig)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
