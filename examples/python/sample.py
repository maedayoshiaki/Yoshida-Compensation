import os
import numpy as np
from numpy import ndarray
import cv2
from typing import List, Tuple, Optional, overload, Literal
import rawpy
import torch
import colour
from external.Graycode.src.python.warp_image_torch import (
    PixelMapWarperTorch,
    AggregationMethod,
    InpaintMethod,
    SplatMethod,
)
from external.Graycode.src.python.interpolate_c2p import load_c2p_numpy

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

PROJ_GAMMA = 2.1
PROJ_WIDTH = 1920
PROJ_HEIGHT = 1080
PROJ_POS_X = 1920 + 3440  # プロジェクタの表示位置X座標
PROJ_POS_Y = 0  # プロジェクタの表示位置Y座標
C2P_MAP_PATH = "C:\\py_scripts\\CompenNeSt-plusplus\\data\\light10\\pos1\\normal\\cam\\raw\\sl\\correspondence_map.npy"
RPCC_MAT_PATH = "C:\\py_scripts\\CompenNeSt-plusplus\\data\\light10\\pos1\\normal\\EOSM6Mark2_colorchercker_sunlight_20251204.npy"


def center_rect(
    image_width: int, image_height: int, proj_width: int, proj_height: int
) -> Tuple[int, int, int, int]:
    x_start = max((proj_width - image_width) // 2, 0)
    y_start = max((proj_height - image_height) // 2, 0)
    return x_start, y_start, image_width, image_height


def raw_to_rgb(raw_bytes) -> np.ndarray:
    with rawpy.imread(raw_bytes) as raw:
        rgb = raw.postprocess(
            # 1. ガンマ補正を無効化 (Linear)
            gamma=(1, 1),
            # 2. 自動明るさ調整を無効化 (計測値を変えないため)
            no_auto_bright=True,
            bright=1,
            # 3. ビット深度を16bitにする
            output_bps=16,
            # 4. カメラの色空間のまま出力する
            #    rawpy.ColorSpace.sRGB (=1) にするとsRGB変換行列がかかってしまうため、
            #    rawpy.ColorSpace.raw (=0) を指定してセンサーの生の混ざり具合を保持する
            output_color=rawpy.ColorSpace.raw,
            # 5. ホワイトバランスはカメラの設定を使用する
            use_camera_wb=True,
            use_auto_wb=False,
            no_auto_scale=False,  # 自動スケールは有効のままにする
            user_sat=None,  # ハイライトクリップを行わない
        )
        return rgb
    if rgb is None:
        raise ValueError("Failed to convert RAW to RGB")


def apply_rpcc_correction(
    linear_image: np.ndarray,
    RPCC_MATRIX: np.ndarray,
    degree: Literal[1, 2, 3] = 2,
) -> np.ndarray:
    """
    Apply Root-Polynomial Color Correction (RPCC) to an image.
    (Root-Polynomial Color Correction (RPCC) を画像に適用する関数)

    This function expands the RGB values using root-polynomial expansion
    and applies the pre-computed RPCC matrix to convert camera RGB to XYZ.
    (RGB値をルート多項式展開で拡張し、事前計算されたRPCC行列を適用して
    カメラRGBをXYZに変換します)

    Parameters
    ----------
    linear_image : np.ndarray
        Linear camera image with shape (H, W, 3), values in range [0.0, 1.0].
        (リニアカメラ画像、形状 (H, W, 3)、値の範囲は [0.0, 1.0])
    RPCC_MATRIX : np.ndarray
        Pre-computed RPCC transformation matrix with shape (3, N),
        where N depends on the degree (6 for degree=2, 13 for degree=3).
        (事前計算されたRPCC変換行列、形状 (3, N)、Nは次数に依存)
    degree : Literal[1, 2, 3], optional
        Polynomial degree for expansion. Default is 2.
        - 1: Linear (R, G, B)
        - 2: Quadratic with cross terms (R, G, B, sqrt(RG), sqrt(GB), sqrt(BR))
        - 3: Cubic with additional terms
        (展開の多項式次数。デフォルトは2)

    Returns
    -------
    np.ndarray
        Color-corrected image in XYZ color space with shape (H, W, 3),
        values clipped to range [0.0, 1.0].
        (XYZ色空間の補正済み画像、形状 (H, W, 3)、値は [0.0, 1.0] にクリップ)

    Notes
    -----
    The RPCC matrix expects expanded RGB input, so this function internally
    performs the same polynomial expansion on the image data.
    (RPCC行列は拡張されたRGB入力を期待するため、この関数は内部で
    画像データに対して同じ多項式展開を実行します)
    """
    # Expand image data: [R, G, B] -> [R, G, B, sqrt(RG), sqrt(GB), sqrt(BR)]
    # (画像データの拡張処理)

    expanded_image = colour.characterisation.polynomial_expansion_Finlayson2015(
        linear_image, degree=degree, root_polynomial_expansion=True
    )

    print(f"Expanded Image Shape: {expanded_image.shape}")

    # Matrix multiplication: XYZ = M @ Expanded_RGB
    # (行列演算)
    corrected_xyz = colour.algebra.vecmul(RPCC_MATRIX, expanded_image)

    # Clip values to valid range (noise reduction)
    # (有効範囲にクリップ、ノイズ対策)
    corrected_xyz = np.clip(corrected_xyz, 0.0, 1.0)

    return corrected_xyz


def capture_image(
    RPCC_matrix: np.ndarray | None = None, RPCC_degree: Literal[1, 2, 3] = 2
) -> Optional[np.ndarray]:
    try:
        imgs = []
        with CameraController() as camera:
            camera.set_properties(av="8", tv="1/15", iso="400", image_quality="LR")
            imgs = camera.capture_numpy(raw_processor=raw_to_rgb)
            if imgs[0] is None:
                print(" Error: Captured image is None")
                return None
        img = imgs[0]
        print(f"Captured image shape: {img.shape}, dtype: {img.dtype}")
        img = img.astype(np.float32) / 65535.0  # Normalize to [0, 1]
        xyz_img: np.ndarray = img
        if RPCC_matrix is not None:
            try:
                xyz_img = apply_rpcc_correction(img, RPCC_matrix, degree=RPCC_degree)
            except ImportError:
                print(" Warning: 'colour' library not found, skipping RPCC correction")
        srgb_img = colour.XYZ_to_sRGB(xyz_img, apply_cctf_encoding=False)
        srgb_img = np.clip(srgb_img, 0.0, 1.0)
        srgb_img = (srgb_img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(srgb_img, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        print(f" Error capturing image: {e}")
        return None


def warp_image(
    src_image: np.ndarray,
    pixel_map_path: str,
    proj_width: int,
    proj_height: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
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
    # Generate Projection patterns
    linear_proj_patterns = generate_projection_patterns(
        PROJ_WIDTH, PROJ_HEIGHT, num_divisions=3, dtype=np.dtype(np.uint8)
    )
    inv_gamma_patterns = []
    # Save patterns to disk
    os.makedirs("data/linear_proj_patterns", exist_ok=True)
    os.makedirs("data/inv_gamma_proj_patterns", exist_ok=True)

    for i, pattern in enumerate(linear_proj_patterns):
        # pattern: H x W x 3, RGB(uint8) を想定
        # RGB -> BGR に変換して保存（ファイル自体はどちらでもよいが一貫させる）
        bgr_pattern = cv2.cvtColor(pattern, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join("data/linear_proj_patterns", f"pattern_{i:02d}.png"),
            bgr_pattern,
        )

        inv_gamma_pattern = apply_inverse_gamma_correction(
            pattern, gamma=1 / PROJ_GAMMA
        )
        inv_gamma_patterns.append(inv_gamma_pattern)
        bgr_inv_gamma_pattern = cv2.cvtColor(inv_gamma_pattern, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(
                "data/inv_gamma_proj_patterns", f"inv_gamma_pattern_{i:02d}.png"
            ),
            bgr_inv_gamma_pattern,
        )
        print(f"Saved pattern_{i:02d}.png and inv_gamma_pattern_{i:02d}.png")

    # Display patterns and capture images using OpenCV
    captured_images = []

    # プロジェクタ側モニタにフルスクリーン表示するためのウィンドウ
    window_name = "projection_window"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(
        window_name, PROJ_POS_X, PROJ_POS_Y
    )  # プロジェクタの位置にウィンドウを移動

    os.makedirs("data/captured_images", exist_ok=True)
    for pattern in inv_gamma_patterns:
        # pattern: H x W x 3, dtype=uint8 (RGB想定)
        # OpenCV は BGR なので変換
        bgr_pattern = cv2.cvtColor(pattern, cv2.COLOR_RGB2BGR)

        cv2.imshow(window_name, bgr_pattern)
        # プロジェクタが安定するまで短い待機（200ms）
        cv2.waitKey(200)

        # Capture image
        captured_image = capture_image()

        # For debugging without camera, read from disk
        # idx = len(captured_images)
        # captured_image = cv2.imread(
        #     f"data/captured_images/captured_image_{idx:02d}.png"
        # )
        # captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)

        if captured_image is None:
            print("Failed to capture image. Exiting.")
            return
        captured_images.append(captured_image)
        bgr_captured = cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(
                "data/captured_images",
                f"captured_image_{len(captured_images) - 1:02d}.png",
            ),
            bgr_captured,
        )
        print(f"Captured and saved captured_image_{len(captured_images) - 1:02d}.png")

    cv2.destroyWindow(window_name)
    print("Captured all images.")

    # After capturing all images, calculate color mixing matrices
    color_mixing_matrices = calc_color_mixing_matrices(
        proj_images=linear_proj_patterns, captured_images=captured_images
    )
    np.save("data/color_mixing_matrices.npy", color_mixing_matrices)
    print("Saved color_mixing_matrices.npy")

    # Load target image
    target_images = []
    target_image_folder_path = "data/target_images"
    for img_name in os.listdir(target_image_folder_path):
        if img_name.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(target_image_folder_path, img_name)
            # BGR で読み込まれるので RGB に変換
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            target_img_array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            target_images.append(target_img_array)

    print(f"Loaded {len(target_images)} target images.")
    # Calculate and save compensation images
    os.makedirs("data/compensation_images", exist_ok=True)
    os.makedirs("data/inv_gamma_comp_images", exist_ok=True)
    CAM_WIDTH = captured_images[0].shape[1]
    CAM_HEIGHT = captured_images[0].shape[0]
    for idx, target_image in enumerate(target_images):
        print(f"[DEBUG] target_image[{idx}] shape:", target_image.shape)
        print(f"[DEBUG] target_image[{idx}] dtype:", target_image.dtype)
        cv2.imwrite(
            f"data/debug/target_image_{idx:02d}.png",
            cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR),
        )
        print(f"[DEBUG] PROJ_WIDTH: {PROJ_WIDTH}, PROJ_HEIGHT: {PROJ_HEIGHT}")
        print(f"[DEBUG] CAM_WIDTH: {CAM_WIDTH}, CAM_HEIGHT: {CAM_HEIGHT}")
        invwarped_target = invwarp_image(
            target_image,
            C2P_MAP_PATH,
            PROJ_WIDTH,
            PROJ_HEIGHT,
            CAM_WIDTH,
            CAM_HEIGHT,
        )
        print(f"[DEBUG] invwarped_target[{idx}] dtype:", invwarped_target.dtype)
        cv2.imwrite(
            f"data/debug/invwarped_target_{idx:02d}.png",
            cv2.cvtColor(invwarped_target, cv2.COLOR_RGB2BGR),
        )
        print(f"[DEBUG] invwarped_target[{idx}] shape:", invwarped_target.shape)
        print(f"[DEBUG] color_mixing_matrices shape:", color_mixing_matrices.shape)

        # Calculate compensation image
        before_warped_compensation_image = calc_compensation_image(
            target_image=invwarped_target,
            color_mixing_matrices=color_mixing_matrices,
            dtype=np.uint8,
        )
        print(
            f"[DEBUG] before_warped_compensation_image[{idx}] shape:",
            before_warped_compensation_image.shape,
        )
        cv2.imwrite(
            f"data/debug/before_warped_comp_image_{idx:02d}.png",
            cv2.cvtColor(before_warped_compensation_image, cv2.COLOR_RGB2BGR),
        )

        # Warp compensation image to projector view
        compensation_image = warp_image(
            before_warped_compensation_image,
            C2P_MAP_PATH,
            PROJ_WIDTH,
            PROJ_HEIGHT,
            target_image.shape[1],
            target_image.shape[0],
        )
        print(
            f"[DEBUG] compensation_image[{idx}] shape:",
            None if compensation_image is None else compensation_image.shape,
        )
        print(f"[DEBUG] compensation_image[{idx}] dtype:", compensation_image.dtype)

        if compensation_image is None or compensation_image.size == 0:
            print("[ERROR] compensation_image is empty – aborting.")
            return
        inv_gamma_comp_image = apply_inverse_gamma_correction(
            compensation_image, gamma=1 / PROJ_GAMMA
        )
        # identify inv_gamma_comp_image's type for debugging
        print(
            f"[DEBUG] inv_gamma_comp_image[{idx}] type:",
            type(inv_gamma_comp_image),
        )
        if isinstance(inv_gamma_comp_image, torch.Tensor):
            inv_gamma_comp_image = inv_gamma_comp_image.cpu().numpy()

        # compensation_image: RGB を想定 -> 保存時に BGR に変換
        bgr_comp = cv2.cvtColor(compensation_image, cv2.COLOR_RGB2BGR)
        bgr_inv_gamma_comp = cv2.cvtColor(inv_gamma_comp_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            f"data/compensation_images/compensation_image_{idx:02d}.png",
            bgr_comp,
        )
        cv2.imwrite(
            f"data/inv_gamma_comp_images/inv_gamma_comp_image_{idx:02d}.png",
            bgr_inv_gamma_comp,
        )
        print(
            f"Saved compensation_image_{idx:02d}.png and inv_gamma_comp_image_{idx:02d}.png"
        )


if __name__ == "__main__":
    main()
