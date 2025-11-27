import os
import numpy as np
from numpy import ndarray
import cv2

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


def Capture() -> ndarray:
    with CameraController() as cam:
        cam.set_properties(av=8, tv=1 / 15, iso=100)
        captured_images = cam.capture_numpy()
    if captured_images is None or len(captured_images) == 0:
        raise RuntimeError("Failed to capture image from camera.")
    return captured_images[0]


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
        captured_image = Capture()
        captured_images.append(captured_image)
        print("Captured an image.")

    cv2.destroyWindow(window_name)

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
    for idx, target_image in enumerate(target_images):
        # Calculate compensation image
        compensation_image = calculate_compensation_image(
            target_image=target_image,
            color_mixing_matrices=color_mixing_matrices,
            dtype=np.dtype(np.uint8),
        )

        # compensation_image: RGB を想定 -> 保存時に BGR に変換
        bgr_comp = cv2.cvtColor(compensation_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            f"data/compensation_images/compensation_image_{idx:02d}.png",
            bgr_comp,
        )
        print(f"Saved compensation_image_{idx:02d}.png")


if __name__ == "__main__":
    main()
