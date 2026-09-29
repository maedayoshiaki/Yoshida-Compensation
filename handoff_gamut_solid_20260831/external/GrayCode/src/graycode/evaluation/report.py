# coding: utf-8
"""評価結果の集約レポート出力 (JSON + CSV サマリ)。

各指標の :class:`~graycode.evaluation.stats.ErrorStats` をまとめて人間可読/機械可読に
保存する。文献の教訓に従い、**複数指標を併記**し再投影/整合性誤差を単独で結論しない。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .stats import ErrorStats

# 何を測り・何を測れないか (レポートに併記して誤解を防ぐ)
METRIC_SEMANTICS: dict[str, str] = {
    "A1_cycle": "self-consistency of the two dense maps (NOT decode accuracy)",
    "A2_holdout": "interpolation/hole-filling quality on held-out decoded points",
    "A3_known_pattern": "ABSOLUTE error vs known projector coords (up to camera detection precision)",
    "A4_epipolar_sampson": "geometric (epipolar) consistency in px; up-to-scale (relative)",
}


def _stats_row(name: str, st: ErrorStats) -> dict:
    d = {"metric": name, "measures": METRIC_SEMANTICS.get(name, "")}
    d.update(st.to_dict())
    return d


def write_report(
    out_dir: str | Path,
    stats_by_metric: Mapping[str, ErrorStats],
    *,
    extra: Mapping | None = None,
    basename: str = "eval_report",
) -> dict[str, str]:
    """指標名→ErrorStats の dict を JSON と CSV に保存し、保存パスを返す。

    Args:
        out_dir: 出力ディレクトリ。
        stats_by_metric: {"A1_cycle": ErrorStats, ...}。
        extra: 追加メタ情報 (RANSAC inlier 率など) を JSON に同梱。
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)

    rows = [_stats_row(name, st) for name, st in stats_by_metric.items()]
    payload = {"metrics": rows}
    if extra:
        payload["extra"] = dict(extra)

    json_path = d / f"{basename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = d / f"{basename}.csv"
    fields = [
        "metric", "measures", "unit", "n", "rmse", "mean", "std",
        "median", "p90", "p95", "p99", "max", "mad", "mad_std",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            f.write(
                ",".join(
                    (f'"{r[k]}"' if k == "measures" else str(r.get(k, "")))
                    for k in fields
                )
                + "\n"
            )
    return {"json": str(json_path), "csv": str(csv_path)}
