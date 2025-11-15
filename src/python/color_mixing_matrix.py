import os
import numpy as np
import PIL.Image as Image
from numpy import ndarray
from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Type


def generate_projection_patterns(
    x_size: int, y_size: int, num_divisions: int = 3, dtype: ndarray.dtype = np.uint8
) -> List[ndarray]:
    """
    Generate a set of RGB projection patterns covering a discrete color grid.
    This function produces all combinations of RGB values sampled uniformly
    over the range [0, 1] (then optionally mapped to 8-bit or 16-bit integer
    intensity levels), arranged as solid-color images of the specified size.
    The total number of generated patterns is ``num_divisions ** 3``, where
    each channel (R, G, B) takes one of ``num_divisions`` equally spaced
    values in [0, 1].
    Parameters
    ----------
    x_size : int
        Width (in pixels) of each generated pattern.
    y_size : int
        Height (in pixels) of each generated pattern.
    num_divisions : int, optional
        Number of discrete intensity levels per color channel. Values less
        than 2 are clamped to 2. The total number of patterns is
        ``num_divisions ** 3``. Default is 3.
    dtype : numpy.dtype, optional
        Desired data type of the output patterns. Supported types are
        ``numpy.uint8``, ``numpy.uint16``, and ``numpy.float32``. If ``numpy.uint8``, intensities
        in [0, 1] are scaled to [0, 255]; if ``numpy.uint16``, they are
        scaled to [0, 65535]. Default is ``numpy.uint8``.
    Returns
    -------
    List[numpy.ndarray]
        A list of 3D NumPy arrays of shape ``(y_size, x_size, 3)``, where the
        last dimension corresponds to the R, G, and B channels. Each array is
        a solid-color pattern representing one combination in the RGB grid.
    """

    num_divisions = max(2, num_divisions)
    patterns = []
    for r in range(num_divisions):  # B, G, R channels
        for g in range(num_divisions):
            for b in range(num_divisions):
                pattern = np.zeros((y_size, x_size, 3), dtype=np.float32)
                pattern[:, :, 0] = (b * 1.0) / (num_divisions - 1)
                pattern[:, :, 1] = (g * 1.0) / (num_divisions - 1)
                pattern[:, :, 2] = (r * 1.0) / (num_divisions - 1)
                if dtype == np.uint8:
                    pattern = (pattern * 255.0).astype(np.uint8)
                elif dtype == np.uint16:
                    pattern = (pattern * 65535.0).astype(np.uint16)
                elif dtype == np.float32:
                    pass  # keep as is
                else:
                    raise ValueError(
                        "Unsupported dtype. Supported types are np.uint8, np.uint16, and np.float32."
                    )
                patterns.append(pattern)

    return patterns


def apply_inverse_gamma_correction(image: ndarray, gamma: float = 2.2) -> ndarray:
    """
    Apply inverse gamma correction to an RGB image.
    This function adjusts the pixel intensities of the input image according
    to the specified gamma value, effectively linearizing the color values.
    Parameters
    ----------
    image : numpy.ndarray
        Input RGB image as a NumPy array of shape (height, width, 3).
        The pixel values should be in the range [0, 1].
    gamma : float, optional
        The gamma value to use for correction. Default is 2.2.
    Returns
    -------
    numpy.ndarray
        The gamma-corrected RGB image as a NumPy array of the same shape
        as the input.
    """
    image_dtype = image.dtype
    if image_dtype == np.uint8:
        _image = image.astype(np.float32) / 255.0
    elif image_dtype == np.uint16:
        _image = image.astype(np.float32) / 65535.0
    else:
        _image = image.astype(np.float32)

    corrected_image = np.power(_image, 1.0 / gamma)
    np.clip(corrected_image, 0.0, 1.0, out=corrected_image)

    if image_dtype == np.uint8:
        corrected_image = (corrected_image * 255).astype(np.uint8)
    elif image_dtype == np.uint16:
        corrected_image = (corrected_image * 65535).astype(np.uint16)
    else:
        pass

    return corrected_image


def calculate_color_mixing_matrix(
    proj_images: List[ndarray], captured_images: List[ndarray]
) -> List[ndarray]:
    # Validate input images
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

    # Normalize projection images
    proj_dtype = proj_images[0].dtype
    if proj_dtype == np.uint8:
        proj_images = [img.astype(np.float32) / 255.0 for img in proj_images]
    elif proj_dtype == np.uint16:
        proj_images = [img.astype(np.float32) / 65535.0 for img in proj_images]

    # Normalize captured images
    cap_dtype = captured_images[0].dtype
    if cap_dtype == np.uint8:
        captured_images = [img.astype(np.float32) / 255.0 for img in captured_images]
    elif cap_dtype == np.uint16:
        captured_images = [img.astype(np.float32) / 65535.0 for img in captured_images]

    color_mixing_matrices = []
    for y in range(proj_images[0].shape[0]):
        for x in range(proj_images[0].shape[1]):
            P = np.array([img[y, x, :] for img in proj_images])
            C = np.array([img[y, x, :] for img in captured_images])
            C = np.append(C, np.ones((C.shape[0], 1)), axis=1)  # Add bias term
            M, _, _, _ = np.linalg.lstsq(C, P, rcond=None)
            color_mixing_matrices.append(M)

    return color_mixing_matrices
