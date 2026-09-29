# coding: utf-8
"""``python -m graycode.evaluation`` の CLI。

サブコマンド:
  - ``eval`` (既定): 保存済み対応マップから A1/A2/A4 (入力があれば A3) を計算し、統計
    レポート (JSON/CSV) と図を出力。入力は graycode の ``result_*.npy`` か、2dsr-prc の
    ``ProjectorCameraMap`` ``.npz`` (``--p2c-npz``)。後者により 2dsr-prc 側ディレクトリで
    GrayCode を撮った後そのまま評価できる。
  - ``gen-pattern``: A3 用既知パターン画像 + 真座標 JSON を生成 (``--project`` で投影も)。
  - ``project``: 任意画像を projector-controller で投影。

入力が無い指標はスキップ (警告) する。座標規約は pixel-is-point (:mod:`graycode.coords`)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..config import get_config, reload_config
from . import io, metrics, patterns, project, report, viz


def _load_c2p(path: Path) -> np.ndarray:
    from ..interpolate_c2p import load_c2p_numpy_array

    return load_c2p_numpy_array(str(path))


def _load_p2c(path: Path) -> np.ndarray:
    from ..interpolate_p2c import load_p2c_numpy_array

    return load_p2c_numpy_array(str(path))


# ── データソース解決 (graycode .npy ファイル or 2dsr-prc .npz) ─────────


def _get_correspondences(args, out_dir: Path):
    """評価対象の対応 (N,4)=[cam_x,cam_y,proj_x,proj_y] と proj_size, conv を返す。"""
    if args.p2c_npz:
        corr, proj_size, conv = io.load_projector_camera_map_npz(args.p2c_npz)
        return corr, proj_size, conv
    p = Path(args.raw_c2p) if args.raw_c2p else (out_dir / "result_c2p.npy")
    if p.exists():
        return _load_c2p(p), None, None
    return None, None, None


def _get_dense_c2p(args, out_dir: Path, corr) -> "np.ndarray | None":
    """A1/A3 用の密 C2P。明示ファイル > npz から補間生成 > 既定ファイル の順。

    ``--no-extrapolate`` のとき凸包外は NaN のまま (投影領域マスク外を穴埋めしない)。
    """
    if args.dense_c2p:
        return _load_c2p(Path(args.dense_c2p))
    if args.p2c_npz and corr is not None and args.cam_h and args.cam_w:
        from ..interpolate_c2p import interpolate_c2p_delaunay

        extrap = not getattr(args, "no_extrapolate", False)
        print(f"[info] building dense C2P from npz via Delaunay "
              f"({args.cam_h}x{args.cam_w}, extrapolate={extrap})...")
        return interpolate_c2p_delaunay(
            args.cam_h, args.cam_w, np.asarray(corr, np.float64), extrapolate=extrap
        )
    default = out_dir / "result_c2p_compensated_delaunay.npy"
    return _load_c2p(default) if default.exists() else None


def _get_dense_p2c(args, out_dir: Path, corr, proj_h: int, proj_w: int) -> "np.ndarray | None":
    """A1 用の密 P2C。明示ファイル > npz > 既定ファイル の順。

    ``--no-extrapolate`` のときは npz の対応から Delaunay で再構築し凸包外を NaN にする
    (生の疎 p2c でも動く)。外挿オン (既定) かつ既に密な npz なら列順を戻すだけ。
    """
    if args.dense_p2c:
        return _load_p2c(Path(args.dense_p2c))
    if args.p2c_npz and corr is not None:
        if getattr(args, "no_extrapolate", False):
            from ..interpolate_p2c import interpolate_p2c_delaunay

            print(f"[info] building dense P2C from npz via Delaunay "
                  f"({proj_h}x{proj_w}, extrapolate=False)...")
            # corr is [cam_x,cam_y,proj_x,proj_y]; interpolate_p2c_delaunay expects
            # [proj_x,proj_y,cam_x,cam_y] — reorder before passing (else points/values swap).
            p2c_rows = io.p2c_grid_from_correspondences(corr)  # [proj_x,proj_y,cam_x,cam_y]
            return interpolate_p2c_delaunay(
                proj_h, proj_w, np.asarray(p2c_rows, np.float64), extrapolate=False
            )
        return io.p2c_grid_from_correspondences(corr)  # [proj,proj,cam,cam] dense
    default = out_dir / "result_p2c_compensated_delaunay.npy"
    return _load_p2c(default) if default.exists() else None


def _run_eval(args: argparse.Namespace) -> int:
    if args.config:
        reload_config(Path(args.config))
    cfg = get_config()
    out_dir = Path(args.output_dir or cfg.paths.output_dir)
    fig_dir = out_dir / "eval_figures"

    corr, proj_size_npz, conv = _get_correspondences(args, out_dir)
    if conv and conv != "pixel-is-point":
        print(f"[warn] npz coord_convention={conv!r} (expected 'pixel-is-point'); "
              "proj coords may be offset ~0.5px — migrate the map first.")

    proj_h = args.proj_h or (proj_size_npz[0] if proj_size_npz else cfg.pipeline.proj_height)
    proj_w = args.proj_w or (proj_size_npz[1] if proj_size_npz else cfg.pipeline.proj_width)

    metrics_sel = {m.strip().lower() for m in args.metrics.split(",") if m.strip()}
    stats_out: dict = {}
    extra: dict = {"backends": viz.backend_status(), "source": "npz" if args.p2c_npz else "npy"}

    # ── A1: cycle consistency ──
    if "a1" in metrics_sel:
        if not (args.cam_h and args.cam_w):
            print("[A1] skipped: --cam-h/--cam-w required for cycle consistency")
        else:
            dc2p = _get_dense_c2p(args, out_dir, corr)
            dp2c = _get_dense_p2c(args, out_dir, corr, proj_h, proj_w)
            if dc2p is None or dp2c is None:
                print("[A1] skipped: need dense C2P and dense P2C "
                      "(provide --dense-c2p/--dense-p2c, or --p2c-npz with --cam-h/--cam-w)")
            else:
                res = metrics.cycle_consistency(dc2p, dp2c, (args.cam_h, args.cam_w), (proj_h, proj_w))
                stats_out["A1_cycle"] = res.stats
                viz.save_error_arrays(fig_dir, "A1_cycle", res.magnitude, res.residual)
                viz.plot_histogram(res.magnitude[res.valid], fig_dir / "A1_cycle_hist.png",
                                   title="A1 cycle consistency [px]")
                viz.plot_heatmap(res.magnitude, fig_dir / "A1_cycle_heatmap.png",
                                 title="A1 cycle consistency [px]")
                viz.plot_quiver(res.residual, fig_dir / "A1_cycle_quiver.png",
                                title="A1 cycle residual field")
                print(f"[A1] cycle consistency: RMSE={res.stats.rmse:.4f}px "
                      f"median={res.stats.median:.4f}px n={res.stats.n}")

    # ── A2: holdout interpolation residual ──
    if "a2" in metrics_sel:
        if corr is None:
            print("[A2] skipped: no correspondences (need result_c2p.npy or --p2c-npz)")
        else:
            res = metrics.holdout_interpolation_residual(
                np.asarray(corr, np.float64), test_frac=args.test_frac, seed=args.seed
            )
            stats_out["A2_holdout"] = res.stats
            mag = np.linalg.norm(res.residual, axis=-1) if res.residual.size else res.residual
            viz.save_error_arrays(fig_dir, "A2_holdout", mag, res.residual)
            viz.plot_histogram(mag, fig_dir / "A2_holdout_hist.png",
                               title="A2 holdout interpolation residual [px]")
            extra["A2_n_extrapolated"] = res.n_extrapolated
            print(f"[A2] holdout interp: RMSE={res.stats.rmse:.4f}px "
                  f"median={res.stats.median:.4f}px n={res.stats.n} "
                  f"(extrapolated/skipped={res.n_extrapolated})")

    # ── A4: epipolar Sampson ──
    if "a4" in metrics_sel:
        if corr is None:
            print("[A4] skipped: no correspondences (need result_c2p.npy or --p2c-npz)")
        else:
            res = metrics.epipolar_sampson(
                np.asarray(corr, np.float64), ransac_thresh=args.ransac_thresh, seed=args.seed
            )
            stats_out["A4_epipolar_sampson"] = res.stats_all
            viz.save_error_arrays(fig_dir, "A4_sampson", res.sampson)
            viz.plot_histogram(res.sampson, fig_dir / "A4_sampson_hist.png",
                               title="A4 epipolar Sampson distance [px]")
            extra["A4_inlier_ratio"] = res.inlier_ratio
            extra["A4_n_fit"] = res.n_fit
            extra["A4_stats_inliers"] = res.stats_inliers.to_dict()
            print(f"[A4] epipolar Sampson: RMSE={res.stats_all.rmse:.4f}px "
                  f"median={res.stats_all.median:.4f}px "
                  f"RANSAC inlier ratio={res.inlier_ratio:.3f}")

    # ── A3: known pattern absolute error ──
    if "a3" in metrics_sel:
        coords = ids = None
        true_coords: dict | None = None
        img = None
        if not (args.pattern_image and args.true_coords and args.cam_h and args.cam_w):
            print("[A3] skipped: requires --pattern-image, --true-coords, --cam-h, --cam-w")
        else:
            import cv2

            img = cv2.imread(str(args.pattern_image), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[A3] skipped: cannot read pattern image {args.pattern_image}")
            else:
                with open(args.true_coords, encoding="utf-8") as f:
                    tc_raw = json.load(f)
                true_coords = {int(k): (float(v[0]), float(v[1])) for k, v in tc_raw.items()}
                if args.pattern_type == "checkerboard":
                    inner = None
                    if not args.inner_size:
                        print("[A3] skipped: --inner-size WxH required for checkerboard")
                    else:
                        try:
                            iw, ih = (int(x) for x in args.inner_size.lower().split("x"))
                            inner = (iw, ih)
                        except ValueError:
                            print("[A3] skipped: --inner-size must be WxH (e.g. 9x6)")
                    if inner is not None:
                        coords, ids = patterns.detect_checkerboard_corners(img, inner)
                else:
                    board = patterns.make_charuco_board(args.squares_x, args.squares_y, 1.0, 0.75)
                    coords, ids = patterns.detect_charuco_corners(img, board)

        if ids is not None and len(ids) > 0 and true_coords is not None:
            dc2p = _get_dense_c2p(args, out_dir, corr)
            if dc2p is None:
                print("[A3] skipped: need dense C2P (--dense-c2p, or --p2c-npz with --cam-h/--cam-w)")
            else:
                feats = np.column_stack([coords, ids.astype(np.float64)])
                res = metrics.known_pattern_error(feats, true_coords, dc2p, (args.cam_h, args.cam_w))
                stats_out["A3_known_pattern"] = res.stats
                mag = np.linalg.norm(res.residual, axis=-1) if res.residual.size else res.residual
                viz.save_error_arrays(fig_dir, "A3_known_pattern", mag, res.residual)
                viz.plot_histogram(mag, fig_dir / "A3_known_pattern_hist.png",
                                   title="A3 known-pattern absolute error [px]")
                print(f"[A3] known pattern: RMSE={res.stats.rmse:.4f}px "
                      f"median={res.stats.median:.4f}px n={res.stats.n}")
        elif ids is not None and len(ids) == 0:
            print("[A3] skipped: no features detected in pattern image")

    if not stats_out:
        print("No metric produced results. Check inputs / --metrics.")
        return 1

    paths = report.write_report(out_dir, stats_out, extra=extra)
    print(f"\nReport: {paths['json']}\n        {paths['csv']}")
    print(f"Figures: {fig_dir}")
    return 0


def _gen_pattern(args: argparse.Namespace):
    cfg = get_config()
    out_dir = Path(args.output_dir or cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proj_h = args.proj_h or cfg.pipeline.proj_height
    proj_w = args.proj_w or cfg.pipeline.proj_width
    import cv2

    if args.pattern_type == "checkerboard":
        img, true_coords = patterns.generate_checkerboard_pattern(
            proj_w, proj_h, args.squares_x, args.squares_y, args.square_px
        )
    else:
        img, true_coords, _ = patterns.generate_charuco_pattern(
            proj_w, proj_h, args.squares_x, args.squares_y, args.square_px
        )
    img_path = out_dir / f"eval_pattern_{args.pattern_type}.png"
    json_path = out_dir / f"eval_pattern_{args.pattern_type}_true_coords.json"
    cv2.imwrite(str(img_path), img)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({str(k): [v[0], v[1]] for k, v in true_coords.items()}, f, indent=2)
    print(f"Saved pattern image : {img_path}")
    print(f"Saved true coords   : {json_path}  ({len(true_coords)} features)")
    return img_path


def _run_gen_pattern(args: argparse.Namespace) -> int:
    if args.config:
        reload_config(Path(args.config))
    img_path = _gen_pattern(args)
    if args.project:
        print(f"Projecting via projector-controller (display={args.display})...")
        project.project_image(img_path, display=args.display, fullscreen=True,
                              duration=args.duration)
    else:
        print("Project this image, capture with GrayCode, then run "
              "`python -m graycode.evaluation eval --metrics a3 ...`.")
    return 0


def _run_project(args: argparse.Namespace) -> int:
    if args.config:
        reload_config(Path(args.config))
    position = None
    if args.x is not None and args.y is not None:
        position = (args.x, args.y)
    size = (args.width, args.height) if (args.width and args.height) else None
    project.project_image(
        args.image, display=args.display, fullscreen=args.fullscreen,
        position=position, size=size, duration=args.duration, fit_mode=args.fit_mode,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m graycode.evaluation",
                                description="GrayCode A-group reprojection/consistency evaluation")
    sub = p.add_subparsers(dest="mode")

    e = sub.add_parser("eval", help="run evaluation metrics from saved maps")
    e.add_argument("--config", default=None)
    e.add_argument("--output-dir", default=None)
    e.add_argument("--metrics", default="a2,a4",
                   help="comma list of a1,a2,a3,a4 (default a2,a4)")
    e.add_argument("--p2c-npz", default=None,
                   help="2dsr-prc ProjectorCameraMap .npz (overrides result_*.npy inputs)")
    e.add_argument("--proj-h", type=int, default=None)
    e.add_argument("--proj-w", type=int, default=None)
    e.add_argument("--cam-h", type=int, default=None)
    e.add_argument("--cam-w", type=int, default=None)
    e.add_argument("--raw-c2p", default=None)
    e.add_argument("--dense-c2p", default=None)
    e.add_argument("--dense-p2c", default=None)
    e.add_argument("--no-extrapolate", action="store_true",
                   help="A1/A3: don't fill outside the decoded convex hull (leave NaN); "
                        "evaluate only within the projection region")
    e.add_argument("--test-frac", type=float, default=0.1)
    e.add_argument("--ransac-thresh", type=float, default=1.0)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--pattern-image", default=None, help="A3: camera capture of known pattern")
    e.add_argument("--true-coords", default=None, help="A3: JSON {id:[x,y]} of true projector coords")
    e.add_argument("--pattern-type", choices=["charuco", "checkerboard"], default="charuco")
    e.add_argument("--inner-size", default=None, help="A3 checkerboard inner corners WxH")
    e.add_argument("--squares-x", type=int, default=12)
    e.add_argument("--squares-y", type=int, default=8)

    g = sub.add_parser("gen-pattern", help="generate A3 known pattern + true-coords JSON")
    g.add_argument("--config", default=None)
    g.add_argument("--output-dir", default=None)
    g.add_argument("--proj-h", type=int, default=None)
    g.add_argument("--proj-w", type=int, default=None)
    g.add_argument("--pattern-type", choices=["charuco", "checkerboard"], default="charuco")
    g.add_argument("--squares-x", type=int, default=12)
    g.add_argument("--squares-y", type=int, default=8)
    g.add_argument("--square-px", type=int, default=120)
    g.add_argument("--project", action="store_true",
                   help="also project the pattern via projector-controller")
    g.add_argument("--display", type=int, default=1, help="projector display index for --project")
    g.add_argument("--duration", type=float, default=None, help="projection seconds (None=until closed)")

    pr = sub.add_parser("project", help="project an image via projector-controller")
    pr.add_argument("--config", default=None)
    pr.add_argument("--image", required=True)
    pr.add_argument("--display", type=int, default=1)
    pr.add_argument("--fullscreen", action="store_true", default=True)
    pr.add_argument("--no-fullscreen", dest="fullscreen", action="store_false")
    pr.add_argument("--x", type=int, default=None)
    pr.add_argument("--y", type=int, default=None)
    pr.add_argument("--width", type=int, default=None)
    pr.add_argument("--height", type=int, default=None)
    pr.add_argument("--duration", type=float, default=None)
    pr.add_argument("--fit-mode", default="native",
                    choices=["native", "contain", "cover", "stretch"])
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "gen-pattern":
        return _run_gen_pattern(args)
    if args.mode == "project":
        return _run_project(args)
    if args.mode == "eval":
        return _run_eval(args)
    if args.mode is None:
        return _run_eval(parser.parse_args(["eval", *argv]))
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
