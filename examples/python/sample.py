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
PROJ_POS_X = 1920 + 3440
PROJ_POS_Y = 0
C2P_MAP_PATH = "C:\\py_scripts\\CompenNeSt-plusplus\\data\\light10\\pos1\\normal\\cam\\raw\\sl\\correspondence_map.npy"
RPCC_MAT_PATH = "C:\\py_scripts\\CompenNeSt-plusplus\\data\\light10\\pos1\\normal\\EOSM6Mark2_colorchercker_sunlight_20251204.npy"


def center_rect(
    image_width: int, image_height: int, proj_width: int, proj_height: int
) -> Tuple[int, int, int, int]:
    """
    Calculate the rectangle coordinates to center an image within the projector frame.

    Args:
        image_width: Width of the image to be centered.
        image_height: Height of the image to be centered.
        proj_width: Width of the projector display.
        proj_height: Height of the projector display.

    Returns:
        A tuple (x_start, y_start, width, height) representing the centered rectangle.
    """
    x_start = max((proj_width - image_width) // 2, 0)
    y_start = max((proj_height - image_height) // 2, 0)
    return x_start, y_start, image_width, image_height


def raw_to_rgb(raw_bytes) -> np.ndarray:
    """
    Convert RAW image bytes to a linear RGB numpy array.

    This function processes RAW camera data with specific settings optimized
    for photometric measurements: linear gamma, no auto brightness adjustment,
    16-bit output, and raw color space preservation.

    Args:
        raw_bytes: RAW image data as bytes or file-like object.

    Returns:
        Linear RGB image as numpy array with shape (H, W, 3) and dtype uint16.

    Raises:
        ValueError: If the RAW to RGB conversion fails.
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
    """
    Apply Root-Polynomial Color Correction (RPCC) to an image.

    This function expands the RGB values using root-polynomial expansion
    and applies the pre-computed RPCC matrix to convert camera RGB to XYZ.

    Args:
        linear_image: Linear camera image with shape (H, W, 3), values in range [0.0, 1.0].
        RPCC_MATRIX: Pre-computed RPCC transformation matrix with shape (3, N),
            where N depends on the degree (6 for degree=2, 13 for degree=3).
        degree: Polynomial degree for expansion. Default is 2.
            - 1: Linear (R, G, B)
            - 2: Quadratic with cross terms (R, G, B, sqrt(RG), sqrt(GB), sqrt(BR))
            - 3: Cubic with additional terms

    Returns:
        Color-corrected image in XYZ color space with shape (H, W, 3),
        values clipped to range [0.0, 1.0].

    Notes:
        The RPCC matrix expects expanded RGB input, so this function internally
        performs the same polynomial expansion on the image data.
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
    """
    Capture an image from the camera and apply color correction.

    This function captures a RAW image from a Canon camera using the EDSDK,
    converts it to linear RGB, optionally applies RPCC color correction to XYZ,
    and then converts to sRGB for output.

    Args:
        RPCC_matrix: Optional pre-computed RPCC transformation matrix.
            If provided, color correction will be applied.
        RPCC_degree: Polynomial degree for RPCC expansion (1, 2, or 3). Default is 2.

    Returns:
        Captured image as BGR numpy array with shape (H, W, 3) and dtype uint8,
        or None if capture fails.
    """
    try:
        imgs = []
        with CameraController() as camera:
            camera.set_properties(av="8", tv="1/15", iso="400", image_quality="LR")
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
    """
    Warp an image from camera view to projector view using a pixel correspondence map.

    This function performs forward warping to transform an image captured by the camera
    into the projector's coordinate system using pre-computed camera-to-projector mapping.

    Args:
        src_image: Source image in camera coordinates with shape (H, W, 3).
        pixel_map_path: Path to the camera-to-projector correspondence map (.npy file).
        proj_width: Width of the projector display.
        proj_height: Height of the projector display.
        image_width: Width of the target image region.
        image_height: Height of the target image region.

    Returns:
        Warped image in projector coordinates with shape (proj_height, proj_width, 3).
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
    """
    Inverse warp an image from projector view to camera view.

    This function performs backward warping to transform an image from projector
    coordinates into the camera's coordinate system using pre-computed mapping.

    Args:
        src_image: Source image in projector coordinates with shape (H, W, 3).
        pixel_map_path: Path to the camera-to-projector correspondence map (.npy file).
        proj_width: Width of the projector display.
        proj_height: Height of the projector display.
        cam_width: Width of the camera image.
        cam_height: Height of the camera image.

    Returns:
        Inverse warped image in camera coordinates with shape (cam_height, cam_width, 3).
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
    """
    Main function for photometric compensation pipeline.

    This function orchestrates the complete photometric compensation workflow:
    1. Generate and save projection patterns (linear and inverse gamma corrected)
    2. Display patterns on projector and capture camera responses
    3. Calculate color mixing matrices from captured images
    4. Load target images and compute compensation images
    5. Apply geometric warping and inverse gamma correction
    6. Save all intermediate and final results
    """
    # Generate projection patterns
    linear_proj_patterns = generate_projection_patterns(
        PROJ_WIDTH, PROJ_HEIGHT, num_divisions=3, dtype=np.dtype(np.uint8)
    )
    inv_gamma_patterns = []

    # Create output directories
    os.makedirs("data/linear_proj_patterns", exist_ok=True)
    os.makedirs("data/inv_gamma_proj_patterns", exist_ok=True)

    # Save patterns to disk
    for i, pattern in enumerate(linear_proj_patterns):
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

    # Display patterns and capture images
    captured_images = []

    # Create fullscreen window on projector display
    window_name = "projection_window"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(window_name, PROJ_POS_X, PROJ_POS_Y)

    os.makedirs("data/captured_images", exist_ok=True)
    for pattern in inv_gamma_patterns:
        bgr_pattern = cv2.cvtColor(pattern, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, bgr_pattern)
        cv2.waitKey(200)

        captured_image = capture_image()
        if captured_image is None:
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

    cv2.destroyWindow(window_name)

    # Calculate color mixing matrices
    color_mixing_matrices = calc_color_mixing_matrices(
        proj_images=linear_proj_patterns, captured_images=captured_images
    )
    np.save("data/color_mixing_matrices.npy", color_mixing_matrices)

    # Load target images
    target_images = []
    target_image_folder_path = "data/target_images"
    for img_name in os.listdir(target_image_folder_path):
        if img_name.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(target_image_folder_path, img_name)
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            target_img_array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            target_images.append(target_img_array)

    # Calculate and save compensation images
    os.makedirs("data/compensation_images", exist_ok=True)
    os.makedirs("data/inv_gamma_comp_images", exist_ok=True)
    CAM_WIDTH = captured_images[0].shape[1]
    CAM_HEIGHT = captured_images[0].shape[0]

    for idx, target_image in enumerate(target_images):
        # Inverse warp target image to camera coordinates
        invwarped_target = invwarp_image(
            target_image,
            C2P_MAP_PATH,
            PROJ_WIDTH,
            PROJ_HEIGHT,
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
            C2P_MAP_PATH,
            PROJ_WIDTH,
            PROJ_HEIGHT,
            target_image.shape[1],
            target_image.shape[0],
        )

        if compensation_image is None or compensation_image.size == 0:
            return

        # Apply inverse gamma correction for projector display
        inv_gamma_comp_image = apply_inverse_gamma_correction(
            compensation_image, gamma=1 / PROJ_GAMMA
        )

        if isinstance(inv_gamma_comp_image, torch.Tensor):
            inv_gamma_comp_image = inv_gamma_comp_image.cpu().numpy()

        # Save compensation images (convert RGB to BGR for OpenCV)
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


if __name__ == "__main__":
    main()
