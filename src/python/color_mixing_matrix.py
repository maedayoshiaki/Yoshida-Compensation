import os
import numpy as np
import PIL.Image as Image
from numpy import ndarray
from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Type


def calculate_color_mixing_matrix(
    proj_images: List[ndarray], captured_images: List[ndarray]
) -> List[ndarray]:
    """
    Calculate the color mixing matrix based on projected and captured images.
    Input images should be lists of numpy arrays representing the images.
    Each numpy array should have shape (height, width, 3) for RGB images.

    Args:
        proj_images (List[ndarray]): List of projection images as numpy arrays.
        captured_images (List[ndarray]): List of captured images as numpy arrays.

    Returns:
        List[ndarray]: The calculated color mixing matrices for each pixel.
    """
    # Validate input images
    num_colors = proj_images[0].shape[0]
    if len(proj_images) != len(captured_images):
        raise ValueError(
            "The number of projected and captured images must be the same."
        )
    if (
        proj_images[0].shape[2] != captured_images[0].shape[2]
        or proj_images[0].shape[2] != 3
    ):
        raise ValueError("Images must have 3 color channels (RGB).")

    if proj_images[0].shape != captured_images[0].shape:
        raise ValueError("Projected and captured images must have the same dimensions.")

    # Normalize projection images if necessary
    if proj_images[0].dtype == np.uint8:
        for i in range(len(proj_images)):
            proj_images[i] = proj_images[i].astype(np.float32) / 255.0
            captured_images[i] = captured_images[i].astype(np.float32) / 255.0
    elif proj_images[0].dtype == np.uint16:
        for i in range(len(proj_images)):
            proj_images[i] = proj_images[i].astype(np.float32) / 65535.0
            captured_images[i] = captured_images[i].astype(np.float32) / 65535.0
    if proj_images[0].dtype != np.float32 and proj_images[0].dtype != np.float64:
        raise ValueError("Images must be of type uint8, uint16, float32, or float64.")

    # Normalize captured images if necessary
    if captured_images[0].dtype == np.uint8:
        for i in range(len(captured_images)):
            captured_images[i] = captured_images[i].astype(np.float32) / 255.0
    elif captured_images[0].dtype == np.uint16:
        for i in range(len(captured_images)):
            captured_images[i] = captured_images[i].astype(np.float32) / 65535.0

    color_mixing_matrices = []
    for y in range(proj_images[0].shape[0]):
        for x in range(proj_images[0].shape[1]):
            P = np.array([img[y, x, :] for img in proj_images])
            P = np.append(P, np.ones((P.shape[0], 1)), axis=1)  # Add bias term
            C = np.array([img[y, x, :] for img in captured_images])
            M, _, _, _ = np.linalg.lstsq(C, P, rcond=None)
            color_mixing_matrices.append(M)

    return color_mixing_matrices
