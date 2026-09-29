"""Build a dense P2C geometry map with the current GrayCode package.

Gray-code frames are projected and triggered first. Canon RAW development and
decoding happen only after the complete projection sequence has finished.

2026-08-10: capture goes through ``sample.capture_projection_patterns`` (the
same function and settings as colour-matrix calibration and evaluation), and
the decoder input is LINEAR grayscale quantized to uint8 once. No sRGB /
gamma encoding anywhere in the computation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yoshidacomp.geometry.graycode_path import ensure_on_syspath

ensure_on_syspath()

from graycode.decode import decode_c2p
from graycode.gen_graycode import generate_expanded_patterns
from graycode.interpolate_p2c import interpolate_p2c_delaunay

from examples.python import sample
from yoshidacomp.compensation.photometric_compensation import quantize_unit_image
from yoshidacomp.config import reload_config
from yoshidacomp.compensation.projection_region import mask_to_active_region, resolve_active_region


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height-step", type=int, default=4)
    parser.add_argument("--width-step", type=int, default=4)
    parser.add_argument("--black-threshold", type=int, default=20)
    parser.add_argument("--white-threshold", type=int, default=5)
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=1.0,
        help="Scale captured camera coordinates to full-resolution coordinates.",
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Project GrayCode over the full projector frame, ignoring the configured active region.",
    )
    parser.add_argument(
        "--derive-roi",
        action="store_true",
        help="Derive a square projector ROI from the decoded GrayCode-valid region.",
    )
    parser.add_argument(
        "--roi-inset",
        type=int,
        default=8,
        help="Inset in projector pixels from the largest inscribed decoded square.",
    )
    return parser.parse_args()


def _save_patterns(output_dir: Path, patterns: list[np.ndarray]) -> None:
    pattern_dir = output_dir / "patterns"
    pattern_dir.mkdir(parents=True, exist_ok=True)
    for index, pattern in enumerate(patterns):
        cv2.imwrite(str(pattern_dir / f"pattern_{index:02d}.png"), pattern)


def _capture_triggered_grayscale(
    patterns: list[np.ndarray],
    *,
    cfg,
    output_dir: Path,
) -> tuple[list[np.ndarray], float, float]:
    """Capture GrayCode frames with the shared projection-capture function.

    Uses ``sample.capture_projection_patterns`` — identical function,
    timing parameters, and RAW development path as colour-matrix
    calibration and evaluation capture. Returns LINEAR grayscale frames
    quantized to uint8 once (no sRGB / gamma encoding anywhere).
    """
    proj = cfg.projector
    rgb_patterns = [
        cv2.cvtColor(pattern, cv2.COLOR_GRAY2RGB)
        if pattern.ndim == 2
        else pattern
        for pattern in patterns
    ]

    window_name = "projection_window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, proj.pos_x, proj.pos_y)
    cv2.resizeWindow(window_name, proj.width, proj.height)
    cv2.waitKey(50)
    cv2.setWindowProperty(
        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )
    if hasattr(cv2, "WND_PROP_TOPMOST"):
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    capture_started = time.perf_counter()
    try:
        captures = sample.capture_projection_patterns(
            rgb_patterns,
            cfg=cfg,
            window_name=window_name,
        )
    finally:
        cv2.destroyWindow(window_name)
    total_seconds = time.perf_counter() - capture_started
    if captures is None:
        raise RuntimeError("GrayCode capture failed.")

    capture_dir = output_dir / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    grayscale_frames: list[np.ndarray] = []
    for index in range(len(captures)):
        frame = captures[index]
        gray_linear = cv2.cvtColor(
            np.asarray(frame, dtype=np.float32), cv2.COLOR_RGB2GRAY
        )
        gray = quantize_unit_image(gray_linear, np.uint8)
        grayscale_frames.append(gray)
        cv2.imwrite(
            str(capture_dir / f"capture_{index:02d}.png"),
            gray,
        )
        # Full-resolution linear frames are ~390MB each; release as we go.
        captures[index] = None

    if len(grayscale_frames) != len(patterns):
        raise RuntimeError(
            "GrayCode frame count mismatch: "
            f"{len(grayscale_frames)} != {len(patterns)}"
        )
    # Development happens inside the shared capture function; report the
    # combined wall time under trigger_capture and 0.0 for development.
    return grayscale_frames, total_seconds, 0.0


def _derive_square_roi(
    c2p: np.ndarray,
    *,
    projector_width: int,
    projector_height: int,
    inset: int,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray, dict[str, int]]:
    """Derive a conservative square ROI from decoded projector coordinates."""
    valid = np.zeros((projector_height, projector_width), dtype=np.uint8)
    projector_xy = np.rint(np.asarray(c2p, dtype=np.float32)[:, 2:4]).astype(np.int32)
    inside = (
        (projector_xy[:, 0] >= 0)
        & (projector_xy[:, 0] < projector_width)
        & (projector_xy[:, 1] >= 0)
        & (projector_xy[:, 1] < projector_height)
    )
    valid[projector_xy[inside, 1], projector_xy[inside, 0]] = 1
    valid = cv2.morphologyEx(
        valid,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
    )
    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(valid, 8)
    if num_components <= 1:
        raise RuntimeError("Could not find a connected decoded GrayCode region.")
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest_label).astype(np.uint8)
    component = cv2.morphologyEx(
        component,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
    )
    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(int(np.argmax(distance)), distance.shape)
    radius = int(np.floor(float(distance[cy, cx])))
    half_side = radius - int(inset)
    if half_side < 32:
        raise RuntimeError(
            "Decoded GrayCode region is too small for a stable square ROI: "
            f"radius={radius}, inset={inset}."
        )
    side = int(2 * half_side)
    x = int(cx - half_side)
    y = int(cy - half_side)
    if x < 0 or y < 0 or x + side > projector_width or y + side > projector_height:
        raise RuntimeError(
            "Derived square ROI does not fit the projector frame: "
            f"roi={(x, y, side, side)}, frame={(projector_width, projector_height)}"
        )
    component_area = int(np.count_nonzero(component))
    stats_out = {
        "connected_components": int(num_components - 1),
        "largest_component_area": component_area,
        "decoded_valid_pixels": int(np.count_nonzero(valid)),
        "inscribed_radius": radius,
        "center_x": int(cx),
        "center_y": int(cy),
    }
    return (x, y, side, side), valid, component, stats_out


def main() -> None:
    args = _parse_args()
    cfg = reload_config(args.config)
    if cfg.camera.half_size:
        raise ValueError(
            "GrayCode calibration requires full-resolution RAW; "
            "camera.half_size must be false."
        )
    if args.camera_scale != 1.0:
        raise ValueError(
            "Full-resolution RAW requires --camera-scale 1.0."
        )
    proj = cfg.projector
    configured_region = resolve_active_region(proj)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = generate_expanded_patterns(
        proj.height,
        proj.width,
        args.height_step,
        args.width_step,
    )
    if args.full_frame or args.derive_roi:
        projection_region = (0, 0, int(proj.width), int(proj.height))
    else:
        projection_region = configured_region
    patterns = [
        mask_to_active_region(pattern, projection_region) for pattern in generated
    ]
    _save_patterns(output_dir, patterns)
    print(
        f"Generated {len(patterns)} GrayCode frames "
        f"(step={args.width_step}x{args.height_step})."
    )

    frames, capture_seconds, development_seconds = _capture_triggered_grayscale(
        patterns,
        cfg=cfg,
        output_dir=output_dir,
    )
    bit_images = frames[:-2]
    white = frames[-2]
    black = frames[-1]

    decode_started = time.perf_counter()
    c2p_list, camera_shape = decode_c2p(
        bit_images,
        white,
        black,
        proj_height=proj.height,
        proj_width=proj.width,
        height_step=args.height_step,
        width_step=args.width_step,
        black_threshold=args.black_threshold,
        white_threshold=args.white_threshold,
        debug_dir=output_dir / "decode_debug",
    )
    decode_seconds = time.perf_counter() - decode_started
    if not c2p_list:
        raise RuntimeError("GrayCode decoding produced no correspondences.")

    scale = float(args.camera_scale)
    c2p = np.asarray(
        [
            [cam_x * scale, cam_y * scale, proj_x, proj_y]
            for (cam_x, cam_y), (proj_x, proj_y) in c2p_list
        ],
        dtype=np.float32,
    )
    p2c_sparse = c2p[:, [2, 3, 0, 1]]
    np.save(output_dir / "c2p_sparse_full_camera.npy", c2p)
    np.save(output_dir / "p2c_sparse_full_camera.npy", p2c_sparse)

    roi_stats: dict[str, int] | None = None
    decoded_valid_mask: np.ndarray | None = None
    decoded_component: np.ndarray | None = None
    if args.derive_roi:
        active_region, decoded_valid_mask, decoded_component, roi_stats = _derive_square_roi(
            c2p,
            projector_width=int(proj.width),
            projector_height=int(proj.height),
            inset=int(args.roi_inset),
        )
        cv2.imwrite(
            str(output_dir / "decoded_projector_valid_mask.png"),
            decoded_valid_mask * 255,
        )
        preview = np.zeros((int(proj.height), int(proj.width), 3), dtype=np.uint8)
        preview[decoded_component.astype(bool)] = (80, 80, 80)
        x0, y0, w0, h0 = active_region
        cv2.rectangle(
            preview,
            (x0, y0),
            (x0 + w0 - 1, y0 + h0 - 1),
            (0, 255, 0),
            4,
        )
        cv2.putText(
            preview,
            f"derived ROI ({x0}, {y0}, {w0}, {h0})",
            (max(8, x0 + 8), max(32, y0 + 32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_dir / "derived_roi_preview.png"), preview)
        (output_dir / "derived_active_region.json").write_text(
            json.dumps(
                {
                    "active_region": list(active_region),
                    "roi_inset": int(args.roi_inset),
                    "stats": roi_stats,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Derived projector ROI from GrayCode-valid region: {active_region}")
    else:
        active_region = configured_region

    interpolation_started = time.perf_counter()
    p2c_dense, extrapolated = interpolate_p2c_delaunay(
        proj.height,
        proj.width,
        p2c_sparse,
        return_mask=True,
        extrapolate=True,
    )
    interpolation_seconds = time.perf_counter() - interpolation_started
    np.save(output_dir / "p2c_dense_unmasked_1920x1080.npy", p2c_dense)

    x, y, width, height = active_region
    active_mask = np.zeros((proj.height, proj.width), dtype=bool)
    active_mask[y : y + height, x : x + width] = True
    p2c_dense_active = p2c_dense.copy().reshape(proj.height, proj.width, 4)
    p2c_dense_active[~active_mask, 2:4] = np.nan
    p2c_dense_active = p2c_dense_active.reshape(-1, 4)
    np.save(output_dir / "p2c_dense_active_1920x1080.npy", p2c_dense_active)
    cv2.imwrite(
        str(output_dir / "p2c_extrapolated_mask.png"),
        extrapolated.reshape(proj.height, proj.width).astype(np.uint8) * 255,
    )

    extrapolated_image = extrapolated.reshape(proj.height, proj.width)
    active_extrapolated_fraction = float(
        np.mean(extrapolated_image[y : y + height, x : x + width])
    )
    metadata = {
        "graycode_source": "external/GrayCode/src/graycode",
        "projector_size": [proj.width, proj.height],
        "active_region": list(active_region),
        "configured_active_region": list(configured_region),
        "projected_full_frame": bool(args.full_frame or args.derive_roi),
        "roi_derived_from_decoded_graycode": bool(args.derive_roi),
        "roi_inset": int(args.roi_inset),
        "decoded_roi_stats": roi_stats,
        "camera_capture_shape": list(camera_shape),
        "camera_coordinate_scale": scale,
        "camera_full_shape": [
            int(round(camera_shape[0] * scale)),
            int(round(camera_shape[1] * scale)),
        ],
        "graycode_step": [args.width_step, args.height_step],
        "pattern_count": len(patterns),
        "correspondence_count": int(len(c2p)),
        "black_threshold": args.black_threshold,
        "white_threshold": args.white_threshold,
        "active_extrapolated_fraction": active_extrapolated_fraction,
        # このフィールドが無い実行は旧チェーン（独自チャンク撮影 + sRGB
        # エンコード済みグレースケールを decode に入力）である。
        "capture_pipeline": "sample.capture_projection_patterns (unified)",
        "decode_input": "linear_gray_uint8",
        "timing_seconds": {
            "trigger_capture": capture_seconds,
            "raw_development": development_seconds,
            "decode": decode_seconds,
            "p2c_interpolation": interpolation_seconds,
        },
        "timing_note": (
            "trigger_capture includes deferred RAW development (shared "
            "capture function); raw_development is reported as 0.0"
        ),
        "outputs": {
            "c2p_sparse": "c2p_sparse_full_camera.npy",
            "p2c_sparse": "p2c_sparse_full_camera.npy",
            "p2c_dense": "p2c_dense_active_1920x1080.npy",
            "p2c_dense_unmasked": "p2c_dense_unmasked_1920x1080.npy",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
