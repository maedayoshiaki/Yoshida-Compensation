import numpy as np
from numpy import ndarray
from typing import List
import torch


def generate_projection_patterns(
    x_size: int,
    y_size: int,
    num_divisions: int = 3,
    dtype: np.dtype = np.dtype(np.uint8),
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
    for r in range(num_divisions):  # R, G, B channels
        for g in range(num_divisions):
            for b in range(num_divisions):
                pattern = np.zeros((y_size, x_size, 3), dtype=np.float32)
                pattern[:, :, 0] = (r * 1.0) / (num_divisions - 1)
                pattern[:, :, 1] = (g * 1.0) / (num_divisions - 1)
                pattern[:, :, 2] = (b * 1.0) / (num_divisions - 1)
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


def apply_inverse_gamma_correction(
    image: ndarray | torch.Tensor,
    gamma: float = 2.2,
    device: str | torch.device = "cuda",
) -> ndarray | torch.Tensor:
    """
    Apply inverse gamma correction to an RGB image using PyTorch.
    This function adjusts the pixel intensities of the input image according
    to the specified gamma value, effectively linearizing the color values.
    Supports GPU acceleration when available.

    Parameters
    ----------
    image : numpy.ndarray or torch.Tensor
        Input RGB image as a NumPy array or torch.Tensor of shape (height, width, 3).
        The pixel values should be in the range [0, 1] for float types,
        [0, 255] for uint8, or [0, 65535] for uint16.
    gamma : float, optional
        The gamma value to use for correction. Default is 2.2.
    device : str or torch.device, optional
        Device to perform computation on. Default is "cuda".
        Automatically falls back to "cpu" if CUDA is not available.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The gamma-corrected RGB image. Returns the same type as input.
    """
    # Determine if input is numpy array
    is_numpy = isinstance(image, np.ndarray)

    # Convert to torch tensor if needed
    if is_numpy:
        image_dtype = image.dtype
        image_tensor = torch.from_numpy(image)
    else:
        image_tensor = image
        image_dtype = image_tensor.dtype

    # Determine device
    if isinstance(device, str):
        device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

    # Move to device
    image_tensor = image_tensor.to(device)

    # Normalize to [0, 1] range
    if image_dtype == np.uint8 or image_dtype == torch.uint8:
        _image = image_tensor.float() / 255.0
    elif image_dtype == np.uint16 or image_dtype == torch.uint16:
        _image = image_tensor.float() / 65535.0
    else:
        _image = image_tensor.float()

    # Apply gamma correction
    corrected_image = torch.pow(_image, 1.0 / gamma)
    corrected_image = torch.clamp(corrected_image, 0.0, 1.0)

    # Convert back to original dtype
    if image_dtype == np.uint8 or image_dtype == torch.uint8:
        corrected_image = (corrected_image * 255).to(torch.uint8)
    elif image_dtype == np.uint16 or image_dtype == torch.uint16:
        corrected_image = (corrected_image * 65535).to(torch.uint16)

    # Convert back to numpy if input was numpy
    if is_numpy:
        corrected_image = corrected_image.cpu().numpy()

    return corrected_image


def calculate_color_mixing_matrix(
    proj_images: List[ndarray], captured_images: List[ndarray]
) -> ndarray:
    """
    Estimate a per-pixel linear color mixing transformation between projected and captured RGB images.
    This function assumes a linear model per pixel of the form:
        P(x, y) ≈ C(x, y) @ M(x, y)[:3, :] + M(x, y)[3, :]
    where:
      - P(x, y) is the projected RGB color at pixel (x, y),
      - C(x, y) is the captured RGB color at pixel (x, y),
      - the last row of M is a bias term (per channel),
      - M(x, y) is a 4×3 matrix for each pixel, estimated via least-squares regression
        over the input image set.
    More concretely, for each pixel (x, y), the function solves:
        [C_r C_g C_b 1] @ M_pixel ≈ [P_r P_g P_b]
    using the normal equations with a pseudoinverse.
    Parameters
    ----------
    proj_images : List[numpy.ndarray]
        List of projected RGB images used as regression targets.
        Each image must be a 3-channel array of shape (H, W, 3).
        All images must share the same spatial dimensions and dtype.
        Supported dtypes are uint8 and uint16, which are internally normalized
        to [0, 1] as float32.
    captured_images : List[numpy.ndarray]
        List of corresponding captured RGB images, same length as `proj_images`.
        Each captured image must have the same shape and dtype as the
        corresponding projected image. Supported dtypes are uint8 and uint16,
        which are internally normalized to [0, 1] as float32.
    Returns
    -------
    numpy.ndarray
        A 4D array of shape (H, W, 4, 3), where for each pixel (h, w),
        `M[h, w]` is a 4×3 matrix encoding the linear color mixing model:
        the first 3 rows correspond to a 3×3 color transform applied to the
        captured RGB, and the last row is a bias term.
    Raises
    ------
    ValueError
        If the number of projected and captured images differs.
        If the images do not have 3 color channels (RGB).
        If projected and captured images do not share the same spatial dimensions.
    Notes
    -----
    - Computation is performed in PyTorch and uses GPU acceleration if available.
    - Normalization is based on the dtype of the first image in each list. All
      images within a list are assumed to share the same dtype.
    - The solution uses the Moore–Penrose pseudoinverse per pixel, which makes
      it robust to poorly conditioned or rank-deficient pixel-wise design matrices.
    """

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

    n = len(proj_images)
    height, width, _ = proj_images[0].shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert images to torch tensors (N, H, W, 3)
    P = torch.from_numpy(np.array(proj_images)).to(device=device, dtype=torch.float32)
    C = torch.from_numpy(np.array(captured_images)).to(
        device=device, dtype=torch.float32
    )

    num_pixels = height * width

    # reshape to (N, H*W, 3)
    P_batch = torch.reshape(P, (n, num_pixels, 3))
    C_batch = torch.reshape(C, (n, num_pixels, 3))

    # Add bias term to C_batch -> (N, H*W, 4)
    bias = torch.ones((n, num_pixels, 1), device=device, dtype=torch.float32)
    C_batch = torch.cat([C_batch, bias], dim=2)

    # Move pixel dimension first so we can solve per-pixel regressions in batch
    P_pixels = P_batch.permute(1, 0, 2)  # (H*W, N, 3)
    C_pixels = C_batch.permute(1, 0, 2)  # (H*W, N, 4)

    C_pixels_T = torch.transpose(C_pixels, 1, 2)  # (H*W, 4, N)
    CTC = torch.bmm(C_pixels_T, C_pixels)  # (H*W, 4, 4)
    CTP = torch.bmm(C_pixels_T, P_pixels)  # (H*W, 4, 3)

    # Solve normal equations per pixel
    CTC_inv = torch.linalg.pinv(CTC)  # (H*W, 4, 4)
    M_pixels = torch.bmm(CTC_inv, CTP)  # (H*W, 4, 3)

    M = torch.reshape(M_pixels, (height, width, 4, 3)).cpu().numpy()

    return M
