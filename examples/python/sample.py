"""Main sample implementation for the photometric compensation pipeline.

Provides camera integration, image warping, color correction, and the
full compensation workflow from pattern generation through final output.

色補償パイプラインのメインサンプル実装。

カメラ連携、画像ワーピング、色補正、およびパターン生成から最終出力までの
完全な補償ワークフローを提供する。
"""

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import cv2
from typing import List, Tuple, Optional, Literal, Sequence, cast
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yoshidacomp.geometry.graycode_path import ensure_on_syspath

ensure_on_syspath()

from graycode.warp_image import (
    PixelMapWarperTorch,
    AggregationMethod,
    InpaintMethod,
    SplatMethod,
)
from graycode.interpolate_c2p import load_c2p_numpy
from graycode.interpolate_p2c import load_p2c_numpy_array
from graycode.config import (
    reload_config as reload_graycode_config,
)

from yoshidacomp.compensation.color_mixing_matrix import (
    generate_projection_patterns,
    apply_inverse_gamma_correction,
    calc_color_mixing_matrices,
)
from yoshidacomp.capture import CameraCaptureError, create_camera_backend
from yoshidacomp.compensation.photometric_compensation import (
    calc_compensation_image,
)
from yoshidacomp.config import (
    get_config,
    reload_config,
    split_cli_config_path,
)
from yoshidacomp.compensation.projection_region import (
    embed_in_active_region,
    mask_to_active_region,
    resolve_active_region,
)
from yoshidacomp.compensation.pipeline import (
    SUPPORTED_TARGET_IMAGE_SUFFIXES,
    TargetImageSpace,
    capture_to_srgb_uint8,
    compensation_space_from_warp_method,
    linear_to_srgb,
    load_target_images as _load_target_images,
    prepare_target_image_for_compensation,
    resolve_target_image_space,
    resolve_warp_settings,
    srgb_to_linear,
)
from yoshidacomp.geometry.warp import (
    # invwarp_image is not referenced anywhere else in this file; it is kept
    # as a deliberate compatibility re-export because
    # tests/test_sample_mocked.py's _patch_warp_helper() monkeypatches
    # `sample.invwarp_image`, and four tests would fail with AttributeError
    # if this import were removed. Do not delete it as "unused".
    WarpMethod,
    center_rect,
    invwarp_image,
    warp_color_mixing_matrices_to_projector,
    warp_image,
)


WARP_DEBUG_DIR = Path("data") / "sample_warp_debug_DEBUG_ONLY"
WARP_ONLY_GROUP_DIRNAME = "warp_only_targets"
WARP_DEBUG_README = """# DEBUG ONLY

This folder stores warp-debug artifacts emitted by `examples/python/sample.py`.

- Generated automatically during normal sample execution
- Safe to delete and regenerate
- Not intended as final pipeline output
"""


def _sanitize_debug_name(name: str) -> str:
    stem = Path(name).stem
    sanitized = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem
    ).strip("_")
    return sanitized or "target"


def _coerce_rgb_uint8_for_debug(image: np.ndarray) -> np.ndarray:
    """Convert arbitrary RGB arrays into uint8 for debug image output."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Debug image output requires shape (H, W, 3).")

    if rgb.dtype == np.uint8:
        return rgb

    rgb = np.nan_to_num(rgb.astype(np.float32, copy=False), copy=False)
    if rgb.size > 0 and float(np.nanmax(rgb)) <= 1.0:
        rgb = rgb * 255.0
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def _save_rgb_debug_image(path: Path, image: np.ndarray) -> None:
    """Save an RGB image for debugging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _coerce_rgb_uint8_for_debug(image)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _write_warp_debug_readme(debug_dir: Path) -> None:
    """Create the marker README for the debug-only warp folder."""
    (debug_dir / "README.md").write_text(WARP_DEBUG_README, encoding="utf-8")


def _reset_debug_output_dir(debug_dir: Path) -> None:
    """Clear previous debug artifacts so each run regenerates them from scratch."""
    if debug_dir.exists():
        for child in debug_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    debug_dir.mkdir(parents=True, exist_ok=True)


def _clear_generated_pngs(output_dir: Path, pattern: str) -> None:
    """Remove previously generated PNG outputs matching the given pattern."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob(pattern):
        if path.is_file():
            path.unlink()


def _write_warp_debug_session(
    debug_dir: Path,
    warp_method: WarpMethod,
    target_image_space: TargetImageSpace,
    pixel_map_path: str,
    proj_width: int,
    proj_height: int,
) -> None:
    """Persist run-level metadata for warp debugging."""
    session = {
        "warp_method": warp_method,
        "target_image_space": target_image_space,
        "pixel_map_path": pixel_map_path,
        "projector_size": {
            "width": proj_width,
            "height": proj_height,
        },
    }
    (debug_dir / "session.json").write_text(
        json.dumps(session, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _warp_debug_target_dir(debug_dir: Path, target_idx: int, target_name: str) -> Path:
    """Return the per-target warp-debug directory."""
    target_dir = debug_dir / f"{target_idx:02d}_{_sanitize_debug_name(target_name)}"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _warp_only_group_dir(debug_dir: Path) -> Path:
    """Return the folder that groups warp-only outputs across targets."""
    group_dir = debug_dir / WARP_ONLY_GROUP_DIRNAME
    group_dir.mkdir(parents=True, exist_ok=True)
    return group_dir


def _projector_region_preview(
    proj_width: int,
    proj_height: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Visualize the centered projector region used by warping."""
    preview = np.zeros((proj_height, proj_width, 3), dtype=np.uint8)
    x, y, w, h = center_rect(image_width, image_height, proj_width, proj_height)
    if w > 0 and h > 0:
        cv2.rectangle(
            preview,
            (x, y),
            (x + w - 1, y + h - 1),
            (0, 255, 0),
            2,
        )
    return preview


def _save_warp_debug_bundle(
    target_dir: Path,
    step_index: int,
    stage_name: str,
    src_image: np.ndarray,
    warped_image: np.ndarray,
    proj_width: int,
    proj_height: int,
    region_image_width: int,
    region_image_height: int,
) -> list[str]:
    """Save input, output, and placement-region previews for one warp step."""
    base_name = f"{step_index:02d}_{stage_name}"
    saved_files = [
        f"{base_name}_input.png",
        f"{base_name}_output.png",
        f"{base_name}_region.png",
    ]
    _save_rgb_debug_image(target_dir / saved_files[0], src_image)
    _save_rgb_debug_image(target_dir / saved_files[1], warped_image)
    _save_rgb_debug_image(
        target_dir / saved_files[2],
        _projector_region_preview(
            proj_width,
            proj_height,
            region_image_width,
            region_image_height,
        ),
    )
    return saved_files


def _save_grouped_warp_only_image(
    group_dir: Path,
    target_idx: int,
    target_name: str,
    stage_name: str,
    warped_image: np.ndarray,
) -> str:
    """Save a warp-only image into the grouped warp-output folder."""
    file_name = (
        f"{target_idx:02d}_{_sanitize_debug_name(target_name)}_{stage_name}.png"
    )
    _save_rgb_debug_image(group_dir / file_name, warped_image)
    return file_name


def _ensure_matching_spatial_shape(
    image: np.ndarray,
    color_mixing_matrices: np.ndarray,
    context: str,
) -> None:
    """Validate that the compensation inputs share the same spatial size."""
    if image.shape[:2] != color_mixing_matrices.shape[:2]:
        raise ValueError(
            f"{context}: target_image shape {image.shape[:2]} does not match "
            f"color_mixing_matrices shape {color_mixing_matrices.shape[:2]}."
        )


def apply_rpcc_correction(
    linear_image: np.ndarray,
    RPCC_MATRIX: np.ndarray,
    degree: Literal[1, 2, 3] = 2,
) -> np.ndarray:
    """Apply Root-Polynomial Color Correction (RPCC) to an image.

    Expand the RGB values using root-polynomial expansion and apply the
    pre-computed RPCC matrix to convert camera RGB to XYZ color space.

    画像にルート多項式色補正（RPCC）を適用する。

    ルート多項式展開を用いて RGB 値を拡張し、事前計算された RPCC 行列を
    適用してカメラ RGB を XYZ 色空間に変換する。

    Args:
        linear_image: Linear camera image with shape ``(H, W, 3)``,
            values in range [0.0, 1.0].
            形状 ``(H, W, 3)``、値の範囲 [0.0, 1.0] のリニアカメラ画像。
        RPCC_MATRIX: Pre-computed RPCC transformation matrix with shape
            ``(3, N)``, where *N* depends on the degree (6 for degree=2,
            13 for degree=3).
            形状 ``(3, N)`` の事前計算された RPCC 変換行列。*N* は次数に
            依存する（degree=2 で 6、degree=3 で 13）。
        degree: Polynomial degree for expansion. Default is 2.
            多項式展開の次数。デフォルトは 2。

            - 1: Linear (R, G, B) / リニア
            - 2: Quadratic with cross terms / 交差項付き二次
              ``(R, G, B, sqrt(RG), sqrt(GB), sqrt(BR))``
            - 3: Cubic with additional terms / 追加項付き三次

    Returns:
        Color-corrected image in XYZ color space with shape
        ``(H, W, 3)``, values clipped to range [0.0, 1.0].
        形状 ``(H, W, 3)``、値が [0.0, 1.0] にクリップされた
        XYZ 色空間の色補正済み画像。

    Notes:
        The RPCC matrix expects expanded RGB input, so this function
        internally performs the same polynomial expansion on the image
        data.
        RPCC 行列は展開された RGB 入力を想定しているため、この関数は
        内部的に画像データに対して同じ多項式展開を行う。
    """
    import colour

    # Expand image data: [R, G, B] -> [R, G, B, sqrt(RG), sqrt(GB), sqrt(BR)]
    expanded_image = colour.characterisation.polynomial_expansion_Finlayson2015(
        linear_image, degree=degree, root_polynomial_expansion=True
    )

    # Matrix multiplication: XYZ = M @ Expanded_RGB
    corrected_xyz = colour.algebra.vecmul(RPCC_MATRIX, expanded_image)

    # Clip values to valid range (noise reduction)
    corrected_xyz = np.clip(corrected_xyz, 0.0, 1.0)

    return corrected_xyz


def capture_image() -> Optional[np.ndarray]:
    """Capture one LINEAR RGB frame from the configured camera.

    The whole compensation pipeline operates on linear light: captures
    are used as-is (no tone encoding). Use :func:`capture_to_srgb_uint8`
    only when writing a capture to disk for viewing.

    カメラから 1 フレームをリニア RGB のまま取得する。

    補償パイプライン全体はリニア光量で動作するため、撮影値はエンコード
    せずそのまま使う。ディスク保存（閲覧用）時のみ
    :func:`capture_to_srgb_uint8` でエンコードする。

    Returns:
        Linear RGB image, float32, shape ``(H, W, 3)``, values in
        [0, 1]; or ``None`` if capture fails.
        リニア RGB 画像（float32、形状 ``(H, W, 3)``、[0, 1]）。
        キャプチャに失敗した場合は ``None``。
    """
    try:
        cam_cfg = get_config().camera
        camera = create_camera_backend(cam_cfg)
        linear_rgb = camera.capture_linear_rgb()
        return np.clip(np.asarray(linear_rgb, dtype=np.float32), 0.0, 1.0)
    except CameraCaptureError as exc:
        print(f"Camera capture failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return None
    except Exception as exc:
        print(f"Unexpected error during capture: {exc}", file=sys.stderr)
        traceback.print_exc()
        return None


def capture_projection_patterns(
    patterns: Sequence[np.ndarray],
    *,
    cfg,
    window_name: str,
) -> Optional[list[np.ndarray]]:
    """Project and capture a calibration sequence.

    Canon EDSDK uses triggered capture to spool every encoded frame first and
    defer RAW development until the complete projection sequence has finished.
    Other backends retain the existing one-shot capture behavior.
    """
    backend_name = cfg.camera.backend.strip().lower()
    captured_images: list[np.ndarray] = []
    hdr_enabled = bool(getattr(cfg.camera, "hdr_enabled", False))

    def project(pattern: np.ndarray, wait_ms: int) -> None:
        bgr_pattern = cv2.cvtColor(pattern, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, bgr_pattern)
        cv2.waitKey(wait_ms)

    try:
        if backend_name == "canon_edsdk":
            camera = create_camera_backend(cfg.camera)
            with camera:
                triggered_batch = getattr(camera, "triggered_batch", None)
                if not callable(triggered_batch):
                    raise CameraCaptureError(
                        "edsdk-python >= 0.2.0 is required for fast triggered capture."
                    )
                switch_wait_ms = int(
                    getattr(
                        cfg.camera,
                        "trigger_wait_key_ms",
                        cfg.camera.wait_key_ms,
                    )
                )

                def capture_pass(
                    indices: Sequence[int],
                    tv_label: str,
                    consume,
                ) -> None:
                    """Trigger one Tv pass, then develop and consume it in order."""
                    if not indices:
                        return
                    print(
                        f"Triggered capture: {len(indices)} pattern(s) at "
                        f"Tv={tv_label} (first wait={cfg.camera.wait_key_ms}ms, "
                        f"switch wait={switch_wait_ms}ms)."
                    )
                    with triggered_batch(tv_label) as batch:
                        capture_started = time.perf_counter()
                        for pass_index, pattern_index in enumerate(indices):
                            project(
                                patterns[pattern_index],
                                (
                                    cfg.camera.wait_key_ms
                                    if pass_index == 0
                                    else switch_wait_ms
                                ),
                            )
                            batch.trigger()
                        capture_seconds = time.perf_counter() - capture_started
                        print(
                            f"Triggered pass complete in {capture_seconds:.2f}s; "
                            "starting deferred RAW development."
                        )
                        development_started = time.perf_counter()
                        for pattern_index, frame in zip(
                            indices,
                            batch.iter_linear_rgb(),
                            strict=True,
                        ):
                            consume(
                                pattern_index,
                                np.clip(
                                    np.asarray(frame, dtype=np.float32),
                                    0.0,
                                    1.0,
                                ),
                            )
                        development_seconds = (
                            time.perf_counter() - development_started
                        )
                        print(
                            f"Deferred RAW development complete in "
                            f"{development_seconds:.2f}s."
                        )

                if hdr_enabled:
                    from yoshidacomp.compensation import inverse_lut as hdr_utils

                    ilut_cfg = cfg.inverse_lut
                    low = float(ilut_cfg.trust_low) / 255.0
                    high = float(ilut_cfg.trust_high) / 255.0
                    max_stops = int(ilut_cfg.hdr_max_stops)
                    base_tv = str(cfg.camera.tv)
                    base_seconds = hdr_utils.tv_to_seconds(base_tv)
                    supported_tv = hdr_utils._query_supported_shutter_speeds(
                        camera
                    )
                    refresh_hz = float(cfg.projector.refresh_rate)
                    roi = (
                        int(ilut_cfg.roi_x),
                        int(ilut_cfg.roi_y),
                        int(ilut_cfg.roi_width),
                        int(ilut_cfg.roi_height),
                    )

                    def gate_crop(frame: np.ndarray) -> np.ndarray:
                        x, y, width, height = roi
                        frame_height, frame_width = frame.shape[:2]
                        x0 = max(0, min(x, frame_width))
                        y0 = max(0, min(y, frame_height))
                        x1 = max(x0, min(x + width, frame_width))
                        y1 = max(y0, min(y + height, frame_height))
                        crop = frame[y0:y1, x0:x1, :3]
                        return crop if crop.size else frame[..., :3]

                    frames: list[Optional[np.ndarray]] = [None] * len(patterns)

                    def keep_base(index: int, frame: np.ndarray) -> None:
                        frames[index] = frame

                    all_indices = list(range(len(patterns)))
                    capture_pass(all_indices, base_tv, keep_base)
                    base_frames = [
                        cast(np.ndarray, frame) for frame in frames
                    ]
                    short_pending = [
                        index
                        for index, frame in enumerate(base_frames)
                        if np.any(gate_crop(frame) >= high)
                    ]
                    long_pending = [
                        index
                        for index, frame in enumerate(base_frames)
                        if np.any(gate_crop(frame) <= low)
                    ]

                    # Reuse the base-frame arrays as the merged HDR output.
                    # Saturated base samples stay NaN until a shorter exposure
                    # recovers them. Longer trustworthy exposures overwrite the
                    # base to improve shadow SNR.
                    for frame in base_frames:
                        frame[frame >= high] = np.nan

                    used_tvs = {base_tv}
                    for stop in range(1, max_stops + 1):
                        if not long_pending:
                            break
                        tv_label = hdr_utils.scale_shutter_tv(
                            base_tv,
                            2.0**stop,
                            supported=supported_tv,
                            refresh_hz=refresh_hz,
                        )
                        if tv_label in used_tvs:
                            continue
                        used_tvs.add(tv_label)
                        ratio = hdr_utils.tv_to_seconds(tv_label) / base_seconds
                        next_pending: list[int] = []

                        def merge_long(index: int, frame: np.ndarray) -> None:
                            if np.any(gate_crop(frame) <= low):
                                next_pending.append(index)
                            trustworthy = frame < high
                            frame /= np.float32(ratio)
                            np.copyto(
                                base_frames[index],
                                frame,
                                where=trustworthy,
                            )

                        capture_pass(long_pending, tv_label, merge_long)
                        long_pending = next_pending

                    short_used_tvs = {base_tv}
                    for stop in range(1, max_stops + 1):
                        if not short_pending:
                            break
                        tv_label = hdr_utils.scale_shutter_tv(
                            base_tv,
                            0.5**stop,
                            supported=supported_tv,
                            refresh_hz=refresh_hz,
                        )
                        if tv_label in short_used_tvs:
                            continue
                        short_used_tvs.add(tv_label)
                        ratio = hdr_utils.tv_to_seconds(tv_label) / base_seconds
                        next_pending = []

                        def merge_short(index: int, frame: np.ndarray) -> None:
                            if np.any(gate_crop(frame) >= high):
                                next_pending.append(index)
                            trustworthy = frame < high
                            missing = np.isnan(base_frames[index])
                            frame /= np.float32(ratio)
                            np.copyto(
                                base_frames[index],
                                frame,
                                where=trustworthy & missing,
                            )

                        capture_pass(short_pending, tv_label, merge_short)
                        short_pending = next_pending

                    for frame in base_frames:
                        np.nan_to_num(frame, copy=False, nan=1.0)
                    captured_images.extend(base_frames)
                    print(
                        f"HDR merge complete: {len(captured_images)} pattern(s), "
                        f"base Tv={base_tv}, max_stops={max_stops}, "
                        f"gate_roi={roi}."
                    )
                else:
                    print(
                        f"Triggered capture: projecting and capturing "
                        f"{len(patterns)} pattern(s) before RAW development."
                    )

                    def append_frame(_index: int, frame: np.ndarray) -> None:
                        captured_images.append(frame)

                    capture_pass(
                        list(range(len(patterns))),
                        str(cfg.camera.tv),
                        append_frame,
                    )
        else:
            if hdr_enabled:
                print(
                    f"HDR projection capture is only available for canon_edsdk; "
                    f"falling back to single exposure for {backend_name!r}.",
                    file=sys.stderr,
                )
            for pattern in patterns:
                project(pattern, cfg.camera.wait_key_ms)
                captured_image = capture_image()
                if captured_image is None:
                    return None
                captured_images.append(captured_image)
    except CameraCaptureError as exc:
        print(f"Camera capture failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return None
    except Exception as exc:
        print(f"Unexpected error during capture: {exc}", file=sys.stderr)
        traceback.print_exc()
        return None

    if len(captured_images) != len(patterns):
        print(
            "Camera capture failed: triggered frame count does not match "
            f"the projected pattern count ({len(captured_images)} != {len(patterns)}).",
            file=sys.stderr,
        )
        return None
    return captured_images


def main(argv: list[str] | None = None):
    """Run the complete photometric compensation pipeline.

    Orchestrate the full workflow:

    1. Generate and save projection patterns (linear and inverse gamma
       corrected).
    2. Display patterns on projector and capture camera responses.
    3. Calculate color mixing matrices from captured images.
    4. Load target images and compute compensation images.
    5. Apply geometric warping and inverse gamma correction.
    6. Save all intermediate and final results.

    色補償パイプラインの完全な実行。

    以下のワークフロー全体を統括する:

    1. 投影パターンの生成と保存（リニアおよび逆ガンマ補正済み）。
    2. プロジェクタにパターンを表示し、カメラ応答をキャプチャ。
    3. キャプチャ画像からカラーミキシング行列を計算。
    4. 目標画像を読み込み、補償画像を計算。
    5. 幾何学的ワーピングと逆ガンマ補正を適用。
    6. すべての中間結果と最終結果を保存。
    """
    if argv is None:
        argv = sys.argv

    try:
        clean_argv, config_path = split_cli_config_path(argv)
    except ValueError as e:
        print(f"Invalid CLI arguments: {e}")
        return

    if len(clean_argv) != 1:
        print("Usage: python examples/python/sample.py [--config <config.toml>]")
        return

    default_config_path = Path(__file__).resolve().parents[2] / "config.toml"
    effective_config_path = (
        config_path if config_path is not None else default_config_path
    )
    # Yoshida config is the single source of truth. Mirror the same path to GrayCode.
    reload_config(effective_config_path)
    reload_graycode_config(effective_config_path)

    cfg = get_config()
    proj = cfg.projector
    paths = cfg.paths
    try:
        active_region = resolve_active_region(proj)
    except ValueError as e:
        print(f"Invalid projector active region: {e}")
        return
    try:
        warp_method, pixel_map_path = resolve_warp_settings(paths)
        target_image_space = resolve_target_image_space(paths)
    except ValueError as e:
        print(f"Invalid warp config: {e}")
        return

    # Generate projection patterns
    linear_model_patterns = generate_projection_patterns(
        proj.width, proj.height, dtype=np.dtype(np.uint8)
    )
    linear_proj_patterns = [
        mask_to_active_region(pattern, active_region)
        for pattern in linear_model_patterns
    ]
    inv_gamma_patterns = []

    # Create output directories
    os.makedirs(paths.linear_pattern_dir, exist_ok=True)
    os.makedirs(paths.inv_gamma_pattern_dir, exist_ok=True)

    # Save patterns to disk
    for i, pattern in enumerate(linear_proj_patterns):
        bgr_pattern = cv2.cvtColor(pattern, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(paths.linear_pattern_dir, f"pattern_{i:02d}.png"),
            bgr_pattern,
        )

        inv_gamma_pattern = apply_inverse_gamma_correction(
            pattern, gamma=1 / proj.gamma
        )
        inv_gamma_patterns.append(inv_gamma_pattern)
        bgr_inv_gamma_pattern = cv2.cvtColor(inv_gamma_pattern, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(paths.inv_gamma_pattern_dir, f"inv_gamma_pattern_{i:02d}.png"),
            bgr_inv_gamma_pattern,
        )

    # Display patterns and capture images
    # Create fullscreen window on projector display
    window_name = "projection_window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, proj.pos_x, proj.pos_y)
    cv2.resizeWindow(window_name, proj.width, proj.height)
    cv2.waitKey(50)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if hasattr(cv2, "WND_PROP_TOPMOST"):
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    os.makedirs(paths.captured_image_dir, exist_ok=True)
    try:
        captured_images = capture_projection_patterns(
            inv_gamma_patterns,
            cfg=cfg,
            window_name=window_name,
        )
        if captured_images is None:
            print("Aborting pipeline because image capture failed.", file=sys.stderr)
            return
    finally:
        cv2.destroyWindow(window_name)

    # Captures are saved only after the projection/capture pass. In Canon
    # trigger mode this keeps PNG encoding and other processing out of the
    # timing-critical projection loop.
    for index, captured_image in enumerate(captured_images):
        # Saved PNGs are sRGB-encoded for viewing only; the pipeline keeps
        # using the linear array in `captured_images`.
        bgr_captured = cv2.cvtColor(
            capture_to_srgb_uint8(captured_image), cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(
            os.path.join(
                paths.captured_image_dir,
                f"captured_image_{index:02d}.png",
            ),
            bgr_captured,
        )

    # Calculate color mixing matrices
    color_mixing_matrices = calc_color_mixing_matrices(
        # Projected frames are spatially masked, while their commanded RGB
        # values inside the active region are the original solid colors.
        proj_images=linear_model_patterns, captured_images=captured_images
    )
    np.save("data/color_mixing_matrices.npy", color_mixing_matrices)

    # Load target images
    target_images = _load_target_images(paths.target_image_dir)

    # Calculate and save compensation images
    compensation_output_dir = Path(paths.compensation_image_dir)
    inv_gamma_output_dir = Path(paths.inv_gamma_comp_dir)
    _clear_generated_pngs(compensation_output_dir, "compensation_image_*.png")
    _clear_generated_pngs(inv_gamma_output_dir, "inv_gamma_comp_image_*.png")
    _reset_debug_output_dir(WARP_DEBUG_DIR)
    _write_warp_debug_readme(WARP_DEBUG_DIR)
    _write_warp_debug_session(
        WARP_DEBUG_DIR,
        warp_method,
        target_image_space,
        pixel_map_path,
        proj.width,
        proj.height,
    )
    grouped_warp_only_dir = _warp_only_group_dir(WARP_DEBUG_DIR)
    cam_width = captured_images[0].shape[1]
    cam_height = captured_images[0].shape[0]
    compensation_space = compensation_space_from_warp_method(warp_method)

    for idx, (target_name, target_uint8) in enumerate(target_images):
        # Targets on disk are sRGB-encoded; the matrices map LINEAR camera
        # values, so decode to linear light before any computation.
        target_image = srgb_to_linear(
            target_uint8.astype(np.float32) / 255.0
        ).astype(np.float32)
        if target_image_space == "projector":
            target_image = embed_in_active_region(
                target_image,
                (proj.width, proj.height),
                active_region,
            )
        target_debug_dir = _warp_debug_target_dir(WARP_DEBUG_DIR, idx, target_name)
        target_crop_rect = center_rect(
            target_image.shape[1],
            target_image.shape[0],
            proj.width,
            proj.height,
        )
        target_exceeds_projector = (
            target_image.shape[1] > proj.width or target_image.shape[0] > proj.height
        )
        _save_rgb_debug_image(target_debug_dir / "00_original_target.png", target_uint8)
        _save_rgb_debug_image(
            target_debug_dir / "00_original_target_region.png",
            _projector_region_preview(
                proj.width,
                proj.height,
                target_image.shape[1],
                target_image.shape[0],
            ),
        )
        debug_metadata: dict[str, object] = {
            "target_name": target_name,
            "target_image_space": target_image_space,
            "compensation_space": compensation_space,
            "target_shape": list(target_image.shape),
            "target_crop_rect": list(target_crop_rect),
            "target_exceeds_projector": target_exceeds_projector,
            "warps": [],
        }
        if target_exceeds_projector:
            print(
                "Warp debug warning: "
                f"target '{target_name}' exceeds projector size and will be clamped "
                f"to crop_rect={target_crop_rect}.",
                file=sys.stderr,
            )
        warp_debug_step = 1
        compensation_target_image = prepare_target_image_for_compensation(
            target_image,
            target_image_space,
            pixel_map_path,
            proj.width,
            proj.height,
            cam_width,
            cam_height,
            warp_method,
        )
        if target_image_space != compensation_space:
            prepare_stage_name = (
                "prepare_target_projector_to_camera"
                if target_image_space == "projector"
                else "prepare_target_camera_to_projector"
            )
            saved_files = _save_warp_debug_bundle(
                target_debug_dir,
                warp_debug_step,
                prepare_stage_name,
                target_image,
                compensation_target_image,
                proj.width,
                proj.height,
                target_image.shape[1],
                target_image.shape[0],
            )
            debug_metadata["warps"].append(
                {
                    "stage": prepare_stage_name,
                    "crop_rect": list(target_crop_rect),
                    "input_shape": list(target_image.shape),
                    "output_shape": list(compensation_target_image.shape),
                    "saved_files": saved_files,
                }
            )
            grouped_warp_only_file = _save_grouped_warp_only_image(
                grouped_warp_only_dir,
                idx,
                target_name,
                prepare_stage_name,
                compensation_target_image,
            )
            debug_metadata["grouped_warp_only_file"] = grouped_warp_only_file
            warp_debug_step += 1

        if compensation_space == "camera":
            _ensure_matching_spatial_shape(
                compensation_target_image,
                color_mixing_matrices,
                "c2p compensation",
            )
            before_warped_compensation_image = calc_compensation_image(
                target_image=compensation_target_image,
                color_mixing_matrices=color_mixing_matrices,
                dtype=np.uint8,
            )
            compensation_image = warp_image(
                before_warped_compensation_image,
                pixel_map_path,
                proj.width,
                proj.height,
                target_image.shape[1],
                target_image.shape[0],
                warp_method=warp_method,
            )
            saved_files = _save_warp_debug_bundle(
                target_debug_dir,
                warp_debug_step,
                "final_compensation_camera_to_projector",
                before_warped_compensation_image,
                compensation_image,
                proj.width,
                proj.height,
                target_image.shape[1],
                target_image.shape[0],
            )
            debug_metadata["warps"].append(
                {
                    "stage": "final_compensation_camera_to_projector",
                    "crop_rect": list(target_crop_rect),
                    "input_shape": list(before_warped_compensation_image.shape),
                    "output_shape": list(compensation_image.shape),
                    "saved_files": saved_files,
                }
            )
        else:
            warped_color_mixing_matrices = warp_color_mixing_matrices_to_projector(
                color_mixing_matrices,
                pixel_map_path,
                proj.width,
                proj.height,
                target_image.shape[1],
                target_image.shape[0],
                warp_method=warp_method,
            )
            _ensure_matching_spatial_shape(
                compensation_target_image,
                warped_color_mixing_matrices,
                "p2c compensation",
            )
            compensation_image = calc_compensation_image(
                target_image=compensation_target_image,
                color_mixing_matrices=warped_color_mixing_matrices,
                dtype=np.uint8,
            )

        if compensation_image is None or compensation_image.size == 0:
            return

        compensation_image = mask_to_active_region(
            compensation_image,
            active_region,
        )

        (target_debug_dir / "metadata.json").write_text(
            json.dumps(debug_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Apply inverse gamma correction for projector display
        inv_gamma_comp_image = apply_inverse_gamma_correction(
            compensation_image, gamma=1 / proj.gamma
        )

        if isinstance(inv_gamma_comp_image, torch.Tensor):
            inv_gamma_comp_image = inv_gamma_comp_image.cpu().numpy()

        # Save compensation images (convert RGB to BGR for OpenCV)
        bgr_comp = cv2.cvtColor(compensation_image, cv2.COLOR_RGB2BGR)
        bgr_inv_gamma_comp = cv2.cvtColor(inv_gamma_comp_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(compensation_output_dir / f"compensation_image_{idx:02d}.png"),
            bgr_comp,
        )
        cv2.imwrite(
            str(inv_gamma_output_dir / f"inv_gamma_comp_image_{idx:02d}.png"),
            bgr_inv_gamma_comp,
        )

    print(f"Warp debug images saved to: {WARP_DEBUG_DIR}")


if __name__ == "__main__":
    main()
