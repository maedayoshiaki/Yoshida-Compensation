"""Build a dense P2C geometry map with the current GrayCode package.

Gray-code frames are projected and triggered first. Canon RAW development and
decoding happen only after the complete projection sequence has finished.
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
_GRAYCODE_SRC = _REPO_ROOT / "external" / "GrayCode" / "src"
if str(_GRAYCODE_SRC) not in sys.path:
    sys.path.insert(0, str(_GRAYCODE_SRC))

from graycode.decode import decode_c2p
from graycode.gen_graycode import generate_expanded_patterns
from graycode.interpolate_p2c import interpolate_p2c_delaunay

from examples.python.sample import capture_to_srgb_uint8
from yoshidacomp.capture import create_camera_backend
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
    proj = cfg.projector
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

    camera = create_camera_backend(cfg.camera)
    grayscale_frames: list[np.ndarray] = []
    capture_dir = output_dir / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    try:
        with camera:
            triggered_batch = getattr(camera, "triggered_batch", None)
            if not callable(triggered_batch):
                raise RuntimeError(
                    "edsdk-python >= 0.2.0 triggered_batch() is required."
                )
            with triggered_batch(cfg.camera.tv) as batch:
                capture_started = time.perf_counter()
                for index, pattern in enumerate(patterns):
                    cv2.imshow(window_name, pattern)
                    cv2.waitKey(
                        cfg.camera.wait_key_ms
                        if index == 0
                        else cfg.camera.trigger_wait_key_ms
                    )
                    batch.trigger()
                capture_seconds = time.perf_counter() - capture_started
                print(
                    f"GrayCode trigger pass complete in {capture_seconds:.2f}s; "
                    "developing RAW files."
                )

                development_started = time.perf_counter()
                for index, frame in enumerate(batch.iter_linear_rgb()):
                    srgb = capture_to_srgb_uint8(frame)
                    gray = cv2.cvtColor(srgb, cv2.COLOR_RGB2GRAY)
                    grayscale_frames.append(gray)
                    cv2.imwrite(
                        str(capture_dir / f"capture_{index:02d}.png"),
                        gray,
                    )
                development_seconds = time.perf_counter() - development_started
                print(
                    "GrayCode deferred RAW development complete in "
                    f"{development_seconds:.2f}s."
                )
    finally:
        cv2.destroyWindow(window_name)

    if len(grayscale_frames) != len(patterns):
        raise RuntimeError(
            "GrayCode frame count mismatch: "
            f"{len(grayscale_frames)} != {len(patterns)}"
        )
    return grayscale_frames, capture_seconds, development_seconds


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
    active_region = resolve_active_region(proj)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = generate_expanded_patterns(
        proj.height,
        proj.width,
        args.height_step,
        args.width_step,
    )
    patterns = [
        mask_to_active_region(pattern, active_region) for pattern in generated
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
        "timing_seconds": {
            "trigger_capture": capture_seconds,
            "raw_development": development_seconds,
            "decode": decode_seconds,
            "p2c_interpolation": interpolation_seconds,
        },
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
