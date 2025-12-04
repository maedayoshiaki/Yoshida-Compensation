import numpy as np
from numpy import ndarray
import torch


def calc_compensation_image(
    target_image: ndarray,
    color_mixing_matrices: ndarray,
    dtype: np.typing.DTypeLike = np.float32,
) -> ndarray:
    # DEBUG: 入力の情報
    print(
        "[DEBUG] target_image.shape:", target_image.shape, "dtype:", target_image.dtype
    )
    print(
        "[DEBUG] color_mixing_matrices.shape:",
        color_mixing_matrices.shape,
        "dtype:",
        color_mixing_matrices.dtype,
    )

    height, width, _ = target_image.shape
    # Normalize target image
    target_dtype = target_image.dtype
    print("[DEBUG] target_dtype:", target_dtype)
    if target_dtype == np.uint8:
        # check max value is 255
        if target_image.max() <= 1:
            print(
                "[WARNING] target_image max value is <= 1 for uint8 dtype. Actual max:",
                target_image.max(),
            )
        norm_target_image = target_image.astype(np.float32) / 255.0
    elif target_dtype == np.uint16:
        if target_image.max() <= 1:
            print(
                "[WARNING] target_image max value is <= 1 for uint16 dtype. Actual max:",
                target_image.max(),
            )
        norm_target_image = target_image.astype(np.float32) / 65535.0
    else:
        norm_target_image = target_image.astype(np.float32)

    # DEBUG: 正規化後
    print(
        "[DEBUG] norm_target_image.shape:",
        norm_target_image.shape,
        "dtype:",
        norm_target_image.dtype,
    )
    print("[DEBUG] norm_target_image[0, 0, :]:", norm_target_image[0, 0, :])

    # Convert to torch tensor for batch processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    norm_target_tensor = torch.from_numpy(norm_target_image).float().to(device)
    color_mixing_matrices_tensor = (
        torch.from_numpy(color_mixing_matrices)
        .float()
        .reshape(-1, color_mixing_matrices.shape[2], color_mixing_matrices.shape[3])
        .to(device)
    )

    print(
        "[DEBUG] norm_target_tensor.shape:",
        norm_target_tensor.shape,
        "dtype:",
        norm_target_tensor.dtype,
    )
    print(
        "[DEBUG] color_mixing_matrices_tensor.shape:",
        color_mixing_matrices_tensor.shape,
        "dtype:",
        color_mixing_matrices_tensor.dtype,
    )

    num_pixels = height * width
    norm_target_tensor = norm_target_tensor.view(num_pixels, 3)
    bias = torch.ones((num_pixels, 1), device=device, dtype=norm_target_tensor.dtype)
    norm_target_tensor = torch.cat([norm_target_tensor, bias], dim=1)

    # DEBUG: 行列計算前
    print("[DEBUG] num_pixels:", num_pixels)
    print("[DEBUG] norm_target_tensor.shape (after cat):", norm_target_tensor.shape)
    print("[DEBUG] norm_target_tensor[0]:", norm_target_tensor[0])
    print("[DEBUG] color_mixing_matrices_tensor[0]:", color_mixing_matrices_tensor[0])

    compensation_tensor = torch.bmm(
        norm_target_tensor.unsqueeze(1),  # Shape: (num_pixels, 1, 4)
        color_mixing_matrices_tensor,
    ).squeeze(1)  # Shape: (num_pixels, 3)

    # DEBUG: 計算結果
    print("[DEBUG] compensation_tensor.shape:", compensation_tensor.shape)
    print("[DEBUG] compensation_tensor[0]:", compensation_tensor[0])

    compensation_tensor = torch.clamp(compensation_tensor, 0.0, 1.0)
    compensation_image = compensation_tensor.view(height, width, 3).cpu().numpy()

    print(
        "[DEBUG] compensation_image.shape:",
        compensation_image.shape,
        "dtype:",
        compensation_image.dtype,
    )
    print("[DEBUG] compensation_image[0, 0, :]:", compensation_image[0, 0, :])

    # if dtype == np.uint8:
    #     compensation_image = np.rint(compensation_image * 255.0).astype(np.uint8)
    # elif dtype == np.uint16:
    #     compensation_image = np.rint(compensation_image * 65535.0).astype(np.uint16)
    # else:
    #     compensation_image = compensation_image.astype(dtype)

    if dtype == np.uint8:
        max_value = 255.0
    elif dtype == np.uint16:
        max_value = 65535.0
    else:
        max_value = 1.0
    compensation_image = compensation_image * max_value
    compensation_image = np.floor(compensation_image + 0.5).astype(dtype)

    print(
        "[DEBUG] output compensation_image.shape:",
        compensation_image.shape,
        "dtype:",
        compensation_image.dtype,
    )

    return compensation_image
