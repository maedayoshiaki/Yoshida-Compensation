"""Main sample implementation for the photometric compensation pipeline.

Provides camera integration, image warping, color correction, and the
full compensation workflow from pattern generation through final output.

色補償パイプラインのメインサンプル実装。

カメラ連携、画像ワーピング、色補正、およびパターン生成から最終出力までの
完全な補償ワークフローを提供する。
"""

import os
import numpy as np
from numpy import ndarray
import cv2
from typing import List, Tuple, Optional, overload, Literal
import rawpy
import torch
import colour
from external.GrayCode.src.python.warp_image import (
    PixelMapWarperTorch,
    AggregationMethod,
    InpaintMethod,
    SplatMethod,
)
from external.GrayCode.src.python.interpolate_c2p import load_c2p_numpy

# capture images from camera
from edsdk.camera_controller import CameraController

from src.python.color_mixing_matrix import (
    generate_projection_patterns,
    apply_inverse_gamma_correction,
    calc_color_mixing_matrices,
)
from src.python.photometric_compensation import (
    calc_compensation_image,
)
from src.python.config import get_config


def center_rect(
    image_width: int, image_height: int, proj_width: int, proj_height: int
) -> Tuple[int, int, int, int]:
    """Calculate the rectangle coordinates to center an image within the projector frame.

    プロジェクタフレーム内に画像を中央配置するための矩形座標を計算する。

    Args:
        image_width: Width of the image to be centered.
            中央配置する画像の幅。
        image_height: Height of the image to be centered.
            中央配置する画像の高さ。
        proj_width: Width of the projector display.
            プロジェクタディスプレイの幅。
        proj_height: Height of the projector display.
            プロジェクタディスプレイの高さ。

    Returns:
        A tuple ``(x_start, y_start, width, height)`` representing the
        centered rectangle.
        中央配置された矩形を表すタプル ``(x_start, y_start, width, height)``。
    """
    x_start = max((proj_width - image_width) // 2, 0)
    y_start = max((proj_height - image_height) // 2, 0)
    return x_start, y_start, image_width, image_height


def raw_to_rgb(raw_bytes) -> np.ndarray:
    """Convert RAW image bytes to a linear RGB numpy array.

    Process RAW camera data with specific settings optimized for
    photometric measurements: linear gamma, no auto brightness
    adjustment, 16-bit output, and raw color space preservation.

    RAW 画像バイトデータをリニア RGB の NumPy 配列に変換する。

    測光測定に最適化された設定で RAW カメラデータを処理する:
    リニアガンマ、自動明るさ調整なし、16 ビット出力、
    RAW 色空間の保持。

    Args:
        raw_bytes: RAW image data as bytes or file-like object.
            バイト列またはファイルライクオブジェクトとしての RAW 画像データ。

    Returns:
        Linear RGB image as numpy array with shape ``(H, W, 3)`` and
        dtype uint16.
        形状 ``(H, W, 3)``、dtype uint16 のリニア RGB 画像の NumPy 配列。

    Raises:
        ValueError: If the RAW to RGB conversion fails.
            RAW から RGB への変換に失敗した場合。
    """
    with rawpy.imread(raw_bytes) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            bright=1,
            output_bps=16,
            output_color=rawpy.ColorSpace.raw,
            use_camera_wb=True,
            use_auto_wb=False,
            no_auto_scale=False,
            user_sat=None,
        )
        return rgb
    if rgb is None:
        raise ValueError("Failed to convert RAW to RGB")


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
    # Expand image data: [R, G, B] -> [R, G, B, sqrt(RG), sqrt(GB), sqrt(BR)]
    expanded_image = colour.characterisation.polynomial_expansion_Finlayson2015(
        linear_image, degree=degree, root_polynomial_expansion=True
    )

    # Matrix multiplication: XYZ = M @ Expanded_RGB
    corrected_xyz = colour.algebra.vecmul(RPCC_MATRIX, expanded_image)

    # Clip values to valid range (noise reduction)
    corrected_xyz = np.clip(corrected_xyz, 0.0, 1.0)

    return corrected_xyz


def capture_image(
    RPCC_matrix: np.ndarray | None = None, RPCC_degree: Literal[1, 2, 3] = 2
) -> Optional[np.ndarray]:
    """Capture an image from the camera and apply color correction.

    Capture a RAW image from a Canon camera using the EDSDK, convert it
    to linear RGB, optionally apply RPCC color correction to XYZ, and
    then convert to sRGB for output.

    カメラから画像をキャプチャし、色補正を適用する。

    EDSDK を使用して Canon カメラから RAW 画像をキャプチャし、リニア RGB に
    変換した後、オプションで RPCC 色補正を XYZ に適用し、出力用に sRGB に
    変換する。

    Args:
        RPCC_matrix: Optional pre-computed RPCC transformation matrix.
            If provided, color correction will be applied.
            オプションの事前計算された RPCC 変換行列。指定された場合、
            色補正が適用される。
        RPCC_degree: Polynomial degree for RPCC expansion (1, 2, or 3).
            Default is 2.
            RPCC 展開の多項式次数（1、2、または 3）。デフォルトは 2。

    Returns:
        Captured image as BGR numpy array with shape ``(H, W, 3)`` and
        dtype uint8, or ``None`` if capture fails.
        形状 ``(H, W, 3)``、dtype uint8 の BGR NumPy 配列としての
        キャプチャ画像。キャプチャに失敗した場合は ``None``。
    """
    try:
        imgs = []
        cam_cfg = get_config().camera
        with CameraController() as camera:
            camera.set_properties(
                av=cam_cfg.av,
                tv=cam_cfg.tv,
                iso=cam_cfg.iso,
                image_quality=cam_cfg.image_quality,
            )
            imgs = camera.capture_numpy(raw_processor=raw_to_rgb)
            if imgs[0] is None:
                return None
        img = imgs[0]
        img = img.astype(np.float32) / 65535.0

        xyz_img: np.ndarray = img
        if RPCC_matrix is not None:
            try:
                xyz_img = apply_rpcc_correction(img, RPCC_matrix, degree=RPCC_degree)
            except ImportError:
                pass

        srgb_img = colour.XYZ_to_sRGB(xyz_img, apply_cctf_encoding=False)
        srgb_img = np.clip(srgb_img, 0.0, 1.0)
        srgb_img = (srgb_img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(srgb_img, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        return None


def warp_image(
    src_image: np.ndarray,
    pixel_map_path: str,
    proj_width: int,
    proj_height: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Warp an image from camera view to projector view using a pixel correspondence map.

    Perform forward warping to transform an image captured by the camera
    into the projector's coordinate system using pre-computed
    camera-to-projector mapping.

    ピクセル対応マップを使用してカメラビューからプロジェクタビューに画像をワーピングする。

    事前計算されたカメラ-プロジェクタ間マッピングを使用して、カメラで
    キャプチャされた画像をプロジェクタの座標系に順方向ワーピングで変換する。

    Args:
        src_image: Source image in camera coordinates with shape
            ``(H, W, 3)``.
            形状 ``(H, W, 3)`` のカメラ座標系のソース画像。
        pixel_map_path: Path to the camera-to-projector correspondence
            map (``.npy`` file).
            カメラ-プロジェクタ対応マップのパス（``.npy`` ファイル）。
        proj_width: Width of the projector display.
            プロジェクタディスプレイの幅。
        proj_height: Height of the projector display.
            プロジェクタディスプレイの高さ。
        image_width: Width of the target image region.
            目標画像領域の幅。
        image_height: Height of the target image region.
            目標画像領域の高さ。

    Returns:
        Warped image in projector coordinates with shape
        ``(proj_height, proj_width, 3)``.
        形状 ``(proj_height, proj_width, 3)`` のプロジェクタ座標系の
        ワーピング済み画像。
    """
    pixel_map = load_c2p_numpy(pixel_map_path)
    warper = PixelMapWarperTorch(pixel_map)
    crop_rect = center_rect(image_width, image_height, proj_width, proj_height)
    src_image_tensor = torch.from_numpy(src_image).permute(2, 0, 1)
    warped_image_tensor = warper.forward_warp(
        src_image_tensor,
        splat_method=SplatMethod.BILINEAR,
        crop_rect=crop_rect,
        inpaint=InpaintMethod.NONE,
        aggregation=AggregationMethod.MEAN,
    )
    warped_image = warped_image_tensor.permute(1, 2, 0).numpy()
    return warped_image


def invwarp_image(
    src_image: np.ndarray,
    pixel_map_path: str,
    proj_width: int,
    proj_height: int,
    cam_width: int,
    cam_height: int,
) -> np.ndarray:
    """Inverse warp an image from projector view to camera view.

    Perform backward warping to transform an image from projector
    coordinates into the camera's coordinate system using pre-computed
    mapping.

    プロジェクタビューからカメラビューに画像を逆ワーピングする。

    事前計算されたマッピングを使用して、プロジェクタ座標系の画像を
    カメラの座標系に逆方向ワーピングで変換する。

    Args:
        src_image: Source image in projector coordinates with shape
            ``(H, W, 3)``.
            形状 ``(H, W, 3)`` のプロジェクタ座標系のソース画像。
        pixel_map_path: Path to the camera-to-projector correspondence
            map (``.npy`` file).
            カメラ-プロジェクタ対応マップのパス（``.npy`` ファイル）。
        proj_width: Width of the projector display.
            プロジェクタディスプレイの幅。
        proj_height: Height of the projector display.
            プロジェクタディスプレイの高さ。
        cam_width: Width of the camera image.
            カメラ画像の幅。
        cam_height: Height of the camera image.
            カメラ画像の高さ。

    Returns:
        Inverse warped image in camera coordinates with shape
        ``(cam_height, cam_width, 3)``.
        形状 ``(cam_height, cam_width, 3)`` のカメラ座標系の
        逆ワーピング済み画像。
    """
    pixel_map = load_c2p_numpy(pixel_map_path)
    warper = PixelMapWarperTorch(pixel_map)
    src_rect = center_rect(
        src_image.shape[1], src_image.shape[0], proj_width, proj_height
    )
    src_image_tensor = torch.from_numpy(src_image).permute(2, 0, 1)
    invwarped_image_tensor = warper.backward_warp(
        src_image_tensor,
        dst_size=(cam_width, cam_height),
        src_rect=src_rect,
        inpaint=InpaintMethod.NONE,
    )
    invwarped_image = invwarped_image_tensor.permute(1, 2, 0).numpy()
    return invwarped_image


def main():
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
    cfg = get_config()
    proj = cfg.projector
    paths = cfg.paths

    # Generate projection patterns
    linear_proj_patterns = generate_projection_patterns(
        proj.width, proj.height, dtype=np.dtype(np.uint8)
    )
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
    captured_images = []

    # Create fullscreen window on projector display
    window_name = "projection_window"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(window_name, proj.pos_x, proj.pos_y)

    os.makedirs(paths.captured_image_dir, exist_ok=True)
    for pattern in inv_gamma_patterns:
        bgr_pattern = cv2.cvtColor(pattern, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, bgr_pattern)
        cv2.waitKey(cfg.camera.wait_key_ms)

        captured_image = capture_image()
        if captured_image is None:
            return

        captured_images.append(captured_image)
        bgr_captured = cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(
                paths.captured_image_dir,
                f"captured_image_{len(captured_images) - 1:02d}.png",
            ),
            bgr_captured,
        )

    cv2.destroyWindow(window_name)

    # Calculate color mixing matrices
    color_mixing_matrices = calc_color_mixing_matrices(
        proj_images=linear_proj_patterns, captured_images=captured_images
    )
    np.save("data/color_mixing_matrices.npy", color_mixing_matrices)

    # Load target images
    target_images = []
    for img_name in os.listdir(paths.target_image_dir):
        if img_name.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(paths.target_image_dir, img_name)
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            target_img_array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            target_images.append(target_img_array)

    # Calculate and save compensation images
    os.makedirs(paths.compensation_image_dir, exist_ok=True)
    os.makedirs(paths.inv_gamma_comp_dir, exist_ok=True)
    CAM_WIDTH = captured_images[0].shape[1]
    CAM_HEIGHT = captured_images[0].shape[0]

    for idx, target_image in enumerate(target_images):
        # Inverse warp target image to camera coordinates
        invwarped_target = invwarp_image(
            target_image,
            paths.c2p_map,
            proj.width,
            proj.height,
            CAM_WIDTH,
            CAM_HEIGHT,
        )

        # Calculate compensation image
        before_warped_compensation_image = calc_compensation_image(
            target_image=invwarped_target,
            color_mixing_matrices=color_mixing_matrices,
            dtype=np.uint8,
        )

        # Warp compensation image to projector coordinates
        compensation_image = warp_image(
            before_warped_compensation_image,
            paths.c2p_map,
            proj.width,
            proj.height,
            target_image.shape[1],
            target_image.shape[0],
        )

        if compensation_image is None or compensation_image.size == 0:
            return

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
            os.path.join(
                paths.compensation_image_dir,
                f"compensation_image_{idx:02d}.png",
            ),
            bgr_comp,
        )
        cv2.imwrite(
            os.path.join(
                paths.inv_gamma_comp_dir,
                f"inv_gamma_comp_image_{idx:02d}.png",
            ),
            bgr_inv_gamma_comp,
        )


if __name__ == "__main__":
    main()
