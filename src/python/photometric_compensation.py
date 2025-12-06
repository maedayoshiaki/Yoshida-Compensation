import numpy as np
from numpy import ndarray
import torch


def calc_compensation_image(
    target_image: ndarray,
    color_mixing_matrices: ndarray,
    dtype: np.typing.DTypeLike = np.float32,
) -> ndarray:
    """
    Calculate a compensation image using color mixing matrices.

    This function applies photometric compensation by transforming the target image
    through pixel-wise color mixing matrices. It supports batch processing using
    PyTorch for GPU acceleration when available.

    Args:
        target_image: Input target image with shape (H, W, 3).
            Supports uint8 (0-255), uint16 (0-65535), or float (0.0-1.0) formats.
        color_mixing_matrices: Color mixing matrices with shape (H, W, 4, 3).
            Each pixel has a 4x3 matrix that transforms [R, G, B, 1] to [R', G', B'].
        dtype: Output data type. Defaults to np.float32.
            If uint8, output is scaled to 0-255.
            If uint16, output is scaled to 0-65535.

    Returns:
        Compensation image with shape (H, W, 3) and the specified dtype.
        Values are clamped to the valid range [0, 1] before scaling.
    """
    height, width, _ = target_image.shape

    # Normalize target image to [0, 1] range
    target_dtype = target_image.dtype
    if target_dtype == np.uint8:
        norm_target_image = target_image.astype(np.float32) / 255.0
    elif target_dtype == np.uint16:
        norm_target_image = target_image.astype(np.float32) / 65535.0
    else:
        norm_target_image = target_image.astype(np.float32)

    # Convert to torch tensor for batch processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm_target_tensor = torch.from_numpy(norm_target_image).float().to(device)
    color_mixing_matrices_tensor = (
        torch.from_numpy(color_mixing_matrices)
        .float()
        .reshape(-1, color_mixing_matrices.shape[2], color_mixing_matrices.shape[3])
        .to(device)
    )

    # Reshape target tensor and append bias term for affine transformation
    num_pixels = height * width
    norm_target_tensor = norm_target_tensor.view(num_pixels, 3)
    bias = torch.ones((num_pixels, 1), device=device, dtype=norm_target_tensor.dtype)
    norm_target_tensor = torch.cat([norm_target_tensor, bias], dim=1)

    # Apply color mixing matrices via batch matrix multiplication
    compensation_tensor = torch.bmm(
        norm_target_tensor.unsqueeze(1),  # Shape: (num_pixels, 1, 4)
        color_mixing_matrices_tensor,
    ).squeeze(1)  # Shape: (num_pixels, 3)

    # Clamp to valid range and convert back to numpy
    compensation_tensor = torch.clamp(compensation_tensor, 0.0, 1.0)
    compensation_image = compensation_tensor.view(height, width, 3).cpu().numpy()

    # Scale to output dtype range
    if dtype == np.uint8:
        max_value = 255.0
    elif dtype == np.uint16:
        max_value = 65535.0
    else:
        max_value = 1.0

    compensation_image = compensation_image * max_value
    compensation_image = np.floor(compensation_image + 0.5).astype(dtype)

    return compensation_image
