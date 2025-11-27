import os
import numpy as np
from numpy import ndarray
import cv2
from typing import List, Tuple, Optional, overload
import rawpy
import torch
from external.Graycode.src.python.warp_image_torch import PixelMapWarperTorch
from external.Graycode.src.python.interpolate_c2p import load_c2p_numpy

# capture images from camera
from edsdk.camera_controller import CameraController

from src.python.color_mixing_matrix import (
    generate_projection_patterns,
    apply_inverse_gamma_correction,
    calculate_color_mixing_matrix,
)
from src.python.photometric_compensation import (
    calculate_compensation_image,
)

PROJ_WIDTH = 1920
PROJ_HEIGHT = 1080
PROJ_POS_X = 1920 + 3440  # プロジェクタの表示位置X座標
PROJ_POS_Y = 0  # プロジェクタの表示位置Y座標
C2P_MAP_PATH = "C:\\py_scripts\\CompenNeSt-plusplus\\data\\light10\\pos1\\normal\\cam\\raw\\sl\\correspondence_map.npy"


@overload
def center_rect(
    image: ndarray, proj_width: int, proj_height: int
) -> Tuple[int, int, int, int]:
    """
    Calculate the top-left coordinates to center a rectangle of given width and height within the image.

    Parameters
    ----------
    image : numpy.ndarray
    proj_width : int
    proj_height : int

    Returns
    -------
    Tuple[x, y, width, height]
    """
    img_height, img_width = image.shape[:2]
    x_start = max((img_width - proj_width) // 2, 0)
    y_start = max((img_height - proj_height) // 2, 0)
    return x_start, y_start, proj_width, proj_height


@overload
def center_rect(
    image_width: int, image_height: int, proj_width: int, proj_height: int
) -> Tuple[int, int, int, int]:
    x_start = max((image_width - proj_width) // 2, 0)
    y_start = max((image_height - proj_height) // 2, 0)
    return x_start, y_start, proj_width, proj_height


def raw_to_rgb(raw_image_path: str) -> Optional[np.ndarray]:
    try:
        with rawpy.imread(raw_image_path) as raw:
            rgb: np.ndarray = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
                gamma=(1, 1),
            )
            return rgb
    except Exception as e:
        print(f" Error processing RAW image {raw_image_path}: {e}")
        return None


def capture_image() -> Optional[ndarray]:
    try:
        with CameraController() as camera:
            save_str_name = "temp_capture"
            camera.set_properties(av="8", tv="1/15", iso="400", image_quality="LR")
            save_paths = camera.capture(filename=save_str_name)

        img = raw_to_rgb(save_paths[0])

        if img is None:
            print(" Error: Captured image is None")
            return None
        os.remove(save_paths[0])  # remove temporary RAW file
        return img
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
    warped_image_tensor = warper.forward_warp(src_image_tensor, crop_rect=crop_rect)
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
    crop_rect = center_rect(src_image, proj_width, proj_height)
    src_image_tensor = torch.from_numpy(src_image).permute(2, 0, 1)
    invwarped_image_tensor = warper.backward_warp(
        src_image_tensor, dst_size=(cam_height, cam_width), src_rect=crop_rect
    )
    invwarped_image = invwarped_image_tensor.permute(1, 2, 0).numpy()
    return invwarped_image


def main():
    # Generate Projection patterns
    linear_proj_patterns = generate_projection_patterns(
        PROJ_WIDTH, PROJ_HEIGHT, num_divisions=4, dtype=np.dtype(np.uint8)
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

        inv_gamma_pattern = apply_inverse_gamma_correction(pattern, gamma=2.2)
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

    for pattern in inv_gamma_patterns:
        # pattern: H x W x 3, dtype=uint8 (RGB想定)
        # OpenCV は BGR なので変換
        bgr_pattern = cv2.cvtColor(pattern, cv2.COLOR_RGB2BGR)

        cv2.imshow(window_name, bgr_pattern)
        # プロジェクタが安定するまで短い待機（200ms）
        cv2.waitKey(200)

        # Capture image
        captured_image = capture_image()
        if captured_image is None:
            print("Failed to capture image. Exiting.")
            return
        captured_images.append(captured_image)

    cv2.destroyWindow(window_name)
    print("Captured all images.")

    # After capturing all images, calculate color mixing matrices
    color_mixing_matrices = calculate_color_mixing_matrix(
        proj_images=linear_proj_patterns, captured_images=captured_images
    )

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
    for idx, target_image in enumerate(target_images):
        invwarped_target = invwarp_image(
            target_image, C2P_MAP_PATH, PROJ_WIDTH, PROJ_HEIGHT
        )
        # Calculate compensation image
        before_warped_compensation_image = calculate_compensation_image(
            target_image=invwarped_target,
            color_mixing_matrices=color_mixing_matrices,
            dtype=np.dtype(np.uint8),
        )
        # Warp compensation image to projector view
        compensation_image = warp_image(
            before_warped_compensation_image,
            C2P_MAP_PATH,
            PROJ_WIDTH,
            PROJ_HEIGHT,
        )
        inv_gamma_comp_image = apply_inverse_gamma_correction(
            compensation_image, gamma=2.2
        )

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
