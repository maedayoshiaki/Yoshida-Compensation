import numpy as np
from numpy import ndarray
from typing import List, Tuple
import torch
import psutil


def generate_projection_patterns(
    x_size: int,
    y_size: int,
    num_divisions: int = 3,
    dtype: np.typing.DTypeLike = np.uint8,
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

    # Apply inverse gamma correction
    corrected_image = torch.pow(_image, gamma)
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


def calc_color_mixing_matrices(
    proj_images: List[ndarray],
    captured_images: List[ndarray],
    ref_x: int = 0,
    ref_y: int = 0,
    safety_margin: float = 0.5,
    min_batch_size: int = 256,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Calculate the color mixing matrix that maps captured image colors to projected image colors.
    P(x, y) = C(x, y) * M(x, y), where P in R^3 is projected image color,
    C in R^4 is captured image color with bias, and M in R^(4x3) is color mixing matrix.


    Args:
        proj_images (List[ndarray]): projector input images (linear RGB)
        captured_images (List[ndarray]): captured images (linear RGB)
        ref_x (int): reference x coordinate for color sampling
        ref_y (int): reference y coordinate for color sampling

    Returns:
        np.ndarray: color mixing matrix of shape (H, W, 4, 3)
    """

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print("device:", device)

    # Preprocess projection images: normalize to [0, 1]
    norm_proj_images = []
    for img in proj_images:
        if img.dtype == np.uint8:
            norm_img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            norm_img = img.astype(np.float32) / 65535.0
        else:
            norm_img = img.astype(np.float32)
        norm_proj_images.append(norm_img)

    # Preprocess captured images: normalize to [0, 1]
    norm_captured_images = []
    for img in captured_images:
        if img.dtype == np.uint8:
            norm_img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            norm_img = img.astype(np.float32) / 65535.0
        else:
            norm_img = img.astype(np.float32)
        norm_captured_images.append(norm_img)

    # Preprocess captured images: transform to torch tensors (H, W, N, 4)
    # まずは CPU 上でテンソルを作成し、バッチ毎に GPU に転送する設計にする
    captured_tensor_cpu = torch.from_numpy(np.array(norm_captured_images)).permute(
        1, 2, 0, 3
    )  # Shape: (H, W, N, 3) on CPU
    bias = torch.ones(
        (*captured_tensor_cpu.shape[:3], 1),
        device="cpu",
        dtype=captured_tensor_cpu.dtype,
    )
    captured_tensor_cpu = torch.cat(
        [captured_tensor_cpu, bias], dim=3
    )  # Shape: (H, W, N, 4) on CPU

    H, W, N, _ = captured_tensor_cpu.shape
    num_pixels = H * W
    # Preprocess projection images: extract reference pixel colors
    proj_ref_colors = []
    for img in norm_proj_images:
        ref_color = img[ref_y, ref_x, :]  # Shape: (3,)
        proj_ref_colors.append(ref_color)

    # 参照色を (N, 3) のテンソルに変換
    proj_ref = torch.from_numpy(np.array(proj_ref_colors)).to(device)  # (N, 3)

    # CPU / GPU メモリから安全なバッチサイズを見積もる

    def estimate_safe_batch_size(
        num_patterns: int,
        safety: float,
        min_bs: int,
        use_cuda: bool,
    ) -> int:
        """利用可能メモリから安全なバッチサイズ (ピクセル数) を見積もる。"""

        # 利用可能メモリ取得（バイト）
        if use_cuda and torch.cuda.is_available():
            _device = torch.device("cuda")
            stats = torch.cuda.memory_stats(_device)
            # free = total - allocated - reserved などを単純化して近似
            total = torch.cuda.get_device_properties(_device).total_memory
            reserved = stats.get("reserved_bytes.all.current", 0)
            allocated = stats.get("allocated_bytes.all.current", 0)
            free_bytes = max(int(total - reserved - allocated), 0)
        else:
            vm = psutil.virtual_memory()
            free_bytes = int(vm.available)

        # 1ピクセルあたりに必要なおおよそのメモリを見積もる
        # captured_batch: (B, N, 4) float32 → B * N * 4 * 4 bytes
        # cap_pinv_batch: (B, N, 4) float32 → 同上
        # proj_ref_batch: (B, N, 3) float32 → B * N * 3 * 4 bytes
        # cmm_batch: (B, 4, 3) float32 → B * 4 * 3 * 4 bytes
        bytes_per_pixel = (num_patterns * 4 * 4 * 3 + num_patterns * 3 * 4) + 4 * 3 * 4
        bytes_per_pixel = max(bytes_per_pixel, 1)

        usable_bytes = max(int(free_bytes * safety), 1)
        max_batch = usable_bytes // bytes_per_pixel
        if max_batch < min_bs:
            return min_bs
        return int(max_batch)

    batch_size = estimate_safe_batch_size(
        num_patterns=N,
        safety=safety_margin,
        min_bs=min_batch_size,
        use_cuda=(device.type == "cuda"),
    )

    batch_size = max(min_batch_size, min(batch_size, num_pixels))
    print(f"num_pixels={num_pixels}, estimated batch_size={batch_size}")

    # 結果用テンソルを CPU 上に確保
    cmm_flat_all = torch.empty((num_pixels, 4, 3), dtype=torch.float32)

    # フラットビュー (CPU 上のまま)
    captured_flat_all = captured_tensor_cpu.reshape(num_pixels, N, 4)

    # バッチ処理
    idx = 0
    while idx < num_pixels:
        end = min(idx + batch_size, num_pixels)
        B = end - idx

        # (B, N, 4) を device に転送
        captured_batch = captured_flat_all[idx:end].to(device)

        # proj_ref を (B, N, 3) にブロードキャスト
        proj_ref_batch = proj_ref.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)

        cmm_batch = torch.linalg.lstsq(
            captured_batch, proj_ref_batch
        ).solution  # Shape: (B, 4, 3)

        # CPU に退避
        cmm_flat_all[idx:end] = cmm_batch.to("cpu", dtype=torch.float32)

        # メモリ開放
        del captured_batch, proj_ref_batch, cmm_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

        idx = end

    color_mixing_matrix = cmm_flat_all.reshape(H, W, 4, 3).numpy()

    return color_mixing_matrix
