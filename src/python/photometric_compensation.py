import numpy as np
from numpy import ndarray
from typing import List
import torch


def calculate_compensation_image(
    target_image: ndarray,
    color_mixing_matrices: ndarray,
    dtype: np.dtype = np.float32,
) -> ndarray:
    height, width, _ = target_image.shape

    # Normalize target image
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
        torch.from_numpy(np.array(color_mixing_matrices)).float().to(device)
    )

    num_pixels = height * width
    color_mixing_matrices_tensor = color_mixing_matrices_tensor.reshape(
        num_pixels, 4, 3
    )
    norm_target_tensor = norm_target_tensor.view(num_pixels, 3)
    bias = torch.ones((num_pixels, 1), device=device, dtype=norm_target_tensor.dtype)
    norm_target_tensor = torch.cat([norm_target_tensor, bias], dim=1)

    compensation_tensor = torch.bmm(
        norm_target_tensor.unsqueeze(1),
        color_mixing_matrices_tensor,
    ).squeeze(1)

    compensation_tensor = torch.clamp(compensation_tensor, 0.0, 1.0)
    compensation_image = compensation_tensor.view(height, width, 3).cpu().numpy()

    if dtype == np.uint8:
        compensation_image = np.rint(compensation_image * 255.0).astype(np.uint8)
    elif dtype == np.uint16:
        compensation_image = np.rint(compensation_image * 65535.0).astype(np.uint16)
    else:
        compensation_image = compensation_image.astype(dtype)

    return compensation_image
