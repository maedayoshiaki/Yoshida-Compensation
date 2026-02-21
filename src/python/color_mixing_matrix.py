"""Color calibration and gamma correction functions.

Provides tools for generating RGB projection patterns, applying inverse
gamma correction, and calculating per-pixel color mixing matrices for
projector-camera color compensation.

色キャリブレーションおよびガンマ補正機能モジュール。

RGB 投影パターンの生成、逆ガンマ補正の適用、およびプロジェクタ-カメラ間の
色補償のためのピクセルごとのカラーミキシング行列の計算ツールを提供する。
"""

import numpy as np
from numpy import ndarray
from typing import List, Optional
import torch
import psutil

from src.python.config import get_config


def generate_projection_patterns(
    x_size: int,
    y_size: int,
    num_divisions: Optional[int] = None,
    dtype: np.typing.DTypeLike = np.uint8,
) -> List[ndarray]:
    """Generate a set of RGB projection patterns covering a discrete color grid.

    Produce all combinations of RGB values sampled uniformly over the
    range [0, 1] (then optionally mapped to 8-bit or 16-bit integer
    intensity levels), arranged as solid-color images of the specified
    size. The total number of generated patterns is ``num_divisions ** 3``,
    where each channel (R, G, B) takes one of ``num_divisions`` equally
    spaced values in [0, 1].

    離散カラーグリッドを網羅する RGB 投影パターンのセットを生成する。

    [0, 1] の範囲で均等にサンプリングされた RGB 値のすべての組み合わせを
    生成し（オプションで 8 ビットまたは 16 ビット整数強度レベルにマッピング）、
    指定サイズの単色画像として配列する。生成されるパターンの総数は
    ``num_divisions ** 3`` で、各チャネル (R, G, B) は [0, 1] の
    等間隔の ``num_divisions`` 個の値のいずれかを取る。

    Parameters
    ----------
    x_size : int
        Width (in pixels) of each generated pattern.
        各パターンの幅（ピクセル）。
    y_size : int
        Height (in pixels) of each generated pattern.
        各パターンの高さ（ピクセル）。
    num_divisions : int, optional
        Number of discrete intensity levels per color channel. Values
        less than 2 are clamped to 2. The total number of patterns is
        ``num_divisions ** 3``. Default is read from config.
        各色チャネルの離散強度レベル数。2 未満の値は 2 にクランプされる。
        パターン総数は ``num_divisions ** 3``。デフォルトは設定から読み込む。
    dtype : numpy.dtype, optional
        Desired data type of the output patterns. Supported types are
        ``numpy.uint8``, ``numpy.uint16``, and ``numpy.float32``. If
        ``numpy.uint8``, intensities in [0, 1] are scaled to [0, 255];
        if ``numpy.uint16``, they are scaled to [0, 65535]. Default is
        ``numpy.uint8``.
        出力パターンのデータ型。``numpy.uint8``、``numpy.uint16``、
        ``numpy.float32`` に対応。``numpy.uint8`` の場合は [0, 255]、
        ``numpy.uint16`` の場合は [0, 65535] にスケーリング。
        デフォルトは ``numpy.uint8``。

    Returns
    -------
    list of numpy.ndarray
        A list of 3-D NumPy arrays of shape ``(y_size, x_size, 3)``,
        where the last dimension corresponds to the R, G, and B channels.
        Each array is a solid-color pattern representing one combination
        in the RGB grid.
        形状 ``(y_size, x_size, 3)`` の 3 次元 NumPy 配列のリスト。
        最後の次元は R、G、B チャネルに対応する。各配列は RGB グリッドの
        1 つの組み合わせを表す単色パターンである。

    Raises
    ------
    ValueError
        If *dtype* is not one of the supported types.
        *dtype* がサポートされている型のいずれでもない場合。
    """

    if num_divisions is None:
        num_divisions = get_config().compensation.num_divisions
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
    gamma: Optional[float] = None,
    device: str | torch.device = "cuda",
) -> ndarray | torch.Tensor:
    """Apply inverse gamma correction to an RGB image using PyTorch.

    Adjust the pixel intensities of the input image according to the
    specified gamma value, effectively linearizing the color values.
    Supports GPU acceleration when available.

    PyTorch を用いて RGB 画像に逆ガンマ補正を適用する。

    入力画像のピクセル強度を指定されたガンマ値に従って調整し、
    色値を実質的にリニアライズする。GPU が利用可能な場合は
    GPU アクセラレーションに対応する。

    Parameters
    ----------
    image : numpy.ndarray or torch.Tensor
        Input RGB image of shape ``(H, W, 3)``. Pixel values should be
        in [0, 1] for float types, [0, 255] for uint8, or [0, 65535]
        for uint16.
        形状 ``(H, W, 3)`` の入力 RGB 画像。ピクセル値は float 型で
        [0, 1]、uint8 で [0, 255]、uint16 で [0, 65535] の範囲。
    gamma : float, optional
        The gamma value to use for correction. Default is read from
        config (``projector.gamma``).
        補正に使用するガンマ値。デフォルトは設定から読み込む
        （``projector.gamma``）。
    device : str or torch.device, optional
        Device to perform computation on. Default is ``"cuda"``.
        Automatically falls back to ``"cpu"`` if CUDA is not available.
        計算を実行するデバイス。デフォルトは ``"cuda"``。
        CUDA が利用できない場合は自動的に ``"cpu"`` にフォールバックする。

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The gamma-corrected RGB image. Returns the same type as input.
        ガンマ補正された RGB 画像。入力と同じ型で返す。
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

    if gamma is None:
        gamma = get_config().projector.gamma

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
    safety_margin: Optional[float] = None,
    min_batch_size: Optional[int] = None,
    use_gpu: Optional[bool] = None,
) -> np.ndarray:
    """Calculate the color mixing matrix mapping captured colors to projected colors.

    For each pixel ``(x, y)``, solve the linear system
    ``P(x, y) = C(x, y) * M(x, y)`` via least squares, where *P* is
    the projected color vector (R^3), *C* is the captured color vector
    with bias (R^4), and *M* is the 4x3 color mixing matrix.

    キャプチャ色から投影色へのカラーミキシング行列を計算する。

    各ピクセル ``(x, y)`` について、最小二乗法で線形システム
    ``P(x, y) = C(x, y) * M(x, y)`` を解く。ここで *P* は投影色
    ベクトル (R^3)、*C* はバイアス付きキャプチャ色ベクトル (R^4)、
    *M* は 4x3 のカラーミキシング行列である。

    Args:
        proj_images: List of projector input images (linear RGB). Each
            image has shape ``(H, W, 3)``.
            プロジェクタ入力画像のリスト（リニア RGB）。各画像の形状は
            ``(H, W, 3)``。
        captured_images: List of captured camera images (linear RGB).
            Each image has shape ``(H, W, 3)``.
            キャプチャされたカメラ画像のリスト（リニア RGB）。各画像の
            形状は ``(H, W, 3)``。
        ref_x: Reference x coordinate for color sampling. Default is 0.
            色サンプリングの参照 x 座標。デフォルトは 0。
        ref_y: Reference y coordinate for color sampling. Default is 0.
            色サンプリングの参照 y 座標。デフォルトは 0。
        safety_margin: Fraction of available memory to use for batch
            processing (0.0 to 1.0). Default is read from config.
            バッチ処理に使用する利用可能メモリの割合（0.0〜1.0）。
            デフォルトは設定から読み込む。
        min_batch_size: Minimum number of pixels per batch. Default is
            read from config.
            バッチあたりの最小ピクセル数。デフォルトは設定から読み込む。
        use_gpu: Whether to use GPU for computation. Default is read
            from config.
            計算に GPU を使用するかどうか。デフォルトは設定から読み込む。

    Returns:
        Color mixing matrix of shape ``(H, W, 4, 3)`` as a NumPy array.
        形状 ``(H, W, 4, 3)`` の NumPy 配列としてのカラーミキシング行列。
    """

    cfg = get_config().compensation
    if safety_margin is None:
        safety_margin = cfg.safety_margin
    if min_batch_size is None:
        min_batch_size = cfg.min_batch_size
    if use_gpu is None:
        use_gpu = cfg.use_gpu

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
        """Estimate a safe batch size (in pixels) from available memory.

        利用可能メモリから安全なバッチサイズ（ピクセル数）を見積もる。

        Args:
            num_patterns: Number of projection patterns (N).
                投影パターン数 (N)。
            safety: Fraction of free memory to use (0.0 to 1.0).
                使用する空きメモリの割合（0.0〜1.0）。
            min_bs: Minimum batch size to return.
                返す最小バッチサイズ。
            use_cuda: Whether to estimate from GPU memory.
                GPU メモリから見積もるかどうか。

        Returns:
            Estimated safe batch size in pixels.
            安全なバッチサイズの推定値（ピクセル数）。
        """

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
