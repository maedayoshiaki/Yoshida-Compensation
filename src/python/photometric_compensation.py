import numpy as np
from numpy import ndarray
import torch


def calculate_compensation_image(
    target_image: ndarray,
    color_mixing_matrices: ndarray,
    dtype: np.dtype = np.dtype(np.float32),
) -> ndarray:
    """
    Compute a photometric compensation image using per-pixel color mixing matrices.
    This function normalizes the input target image, applies a per-pixel 4×3 color
    mixing matrix (including a bias term) using batched matrix multiplication on
    CPU or GPU via PyTorch, clamps the result to the valid range [0, 1], and
    finally converts it back to the requested output dtype.

    Parameters
    ----------
    target_image : numpy.ndarray
        Input target image of shape (H, W, 3). The dtype can be:
        - uint8   (values in [0, 255])
        - uint16  (values in [0, 65535])
        - float32/float64 (assumed already in [0.0, 1.0]).
    color_mixing_matrices : numpy.ndarray
        Array of per-pixel color mixing matrices. It must be broadcastable or
        reshaped to (H * W, 4, 3), where each 4×3 matrix transforms a pixel
        [R, G, B, 1]ᵀ into a compensated RGB value.
    dtype : numpy.dtype, optional
        Desired dtype of the output compensation image. Supported values are
        np.uint8, np.uint16, or a floating-point dtype such as np.float32.
        Default is np.float32.
    Returns
    -------
    numpy.ndarray
        Compensation image of shape (H, W, 3) with values:
        - in [0, 255] if dtype is np.uint8
        - in [0, 65535] if dtype is np.uint16
        - in [0.0, 1.0] for floating-point dtypes.
    Notes
    -----
    - If a CUDA-capable GPU is available, computations are performed on the GPU;
      otherwise, they fall back to CPU.
    - The function assumes that `color_mixing_matrices` has been precomputed to
      match the spatial resolution of `target_image`.
    """
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
