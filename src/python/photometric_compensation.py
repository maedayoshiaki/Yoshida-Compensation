import numpy as np
from numpy import ndarray
from typing import List


def calculate_compensation_image(
    target_image: ndarray,
    color_mixing_matrices: List[ndarray],
) -> ndarray:
    """
    Calculate the compensation image using the target image and color mixing matrices.
    Input images should be numpy arrays representing the images.
    Each numpy array should have shape (height, width, 3(R, G, B)) for RGB images.

    Args:
        target_image (ndarray): The target image as a numpy array.
        color_mixing_matrices (List[ndarray]): List of color mixing matrices for each pixel.

    Returns:
        np.ndarray: The calculated compensation image.
    """
    height, width, _ = target_image.shape
    compensation_image = np.zeros((height, width, 3), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            M = color_mixing_matrices[y * width + x]
            C = np.append(target_image[y, x, :], 1.0)  # Add bias term
            P = M @ C
            compensation_image[y, x, :] = P[:3]

    np.clip(compensation_image, 0.0, 1.0, out=compensation_image)

    return compensation_image
