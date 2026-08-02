"""Projector-camera geometric warping.

カメラ座標系とプロジェクタ座標系の間で画像・行列をワーピングする。

GrayCode (``graycode``) は git submodule であり、インストール済みパッケージ
ではない。``external/GrayCode/src`` を ``sys.path`` へ挿入した文脈でしか
import できないため、このモジュールでは GrayCode の import を各関数の内部に
閉じ込め、``import yoshidacomp.geometry.warp`` 自体は GrayCode 非依存に保つ。

同じ理由で ``torch`` も各関数の内部で import する。``torch`` はインストール
済みだが、import 自体に非trivialなコストがかかり（計測: 単体で ``import
yoshidacomp.compensation.pipeline`` が 9.68 秒）、web層が座標変換を一切
行わずに ``pipeline`` を import できることを妨げていた
（``.superpowers/sdd/2026-08-02-target-and-lut/task-1-report.md`` 参照）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Tuple

import numpy as np

if TYPE_CHECKING:
    # torch itself has no type-annotation use in this file (only
    # torch.from_numpy() calls inside function bodies, imported locally
    # there), so it does not need a TYPE_CHECKING import here.
    from graycode.warp_image import PixelMapWarperTorch


WarpMethod = Literal["c2p", "p2c"]


def center_rect(
    image_width: int, image_height: int, proj_width: int, proj_height: int
) -> Tuple[int, int, int, int]:
    """Calculate the rectangle coordinates to center an image within the projector frame.

    プロジェクタフレーム内に画像を中央配置するための矩形座標を計算する。

    Args:
        image_width: Width of the image to be centered.
            中央配置する画像の幅。
        image_height: Height of the image to be centered.
            中央配置する画像の高さ。
        proj_width: Width of the projector display.
            プロジェクタディスプレイの幅。
        proj_height: Height of the projector display.
            プロジェクタディスプレイの高さ。

    Returns:
        A tuple ``(x_start, y_start, width, height)`` representing the
        centered rectangle. When the image is larger than the projector,
        the returned width/height are clamped to the projector frame.
        中央配置された矩形を表すタプル ``(x_start, y_start, width, height)``。
    """
    crop_width = min(image_width, proj_width)
    crop_height = min(image_height, proj_height)
    x_start = max((proj_width - crop_width) // 2, 0)
    y_start = max((proj_height - crop_height) // 2, 0)
    return x_start, y_start, crop_width, crop_height


def create_warper(
    pixel_map_path: str,
    warp_method: WarpMethod,
) -> "PixelMapWarperTorch":
    """Load the configured correspondence map and return a warper."""
    from graycode.warp_image import PixelMapWarperTorch

    if warp_method == "c2p":
        from graycode.interpolate_c2p import load_c2p_numpy

        pixel_map = load_c2p_numpy(pixel_map_path)
    elif warp_method == "p2c":
        from graycode.interpolate_p2c import load_p2c_numpy_array

        pixel_map = load_p2c_numpy_array(pixel_map_path)
    else:
        raise ValueError("warp_method must be 'c2p' or 'p2c'.")

    return PixelMapWarperTorch(pixel_map)


def warp_camera_array_to_projector(
    src_array: np.ndarray,
    pixel_map_path: str,
    proj_width: int,
    proj_height: int,
    image_width: int,
    image_height: int,
    warp_method: WarpMethod,
) -> np.ndarray:
    """Warp an array from camera coordinates into projector coordinates."""
    import torch
    from graycode.warp_image import AggregationMethod, InpaintMethod, SplatMethod

    if src_array.ndim < 3:
        raise ValueError("src_array must have shape (H, W, ...).")

    crop_rect = center_rect(image_width, image_height, proj_width, proj_height)
    src_shape = src_array.shape
    src_tensor = torch.from_numpy(src_array.reshape(src_shape[0], src_shape[1], -1))
    src_tensor = src_tensor.permute(2, 0, 1)
    warper = create_warper(pixel_map_path, warp_method)

    if warp_method == "c2p":
        warped_tensor = warper.forward_warp(
            src_tensor,
            dst_size=(proj_width, proj_height),
            splat_method=SplatMethod.BILINEAR,
            crop_rect=crop_rect,
            # GrayCode maps may use projector blocks larger than one pixel.
            # Bilinear splatting alone then leaves a regular hole grid between
            # block centers; convolution inpainting reconstructs those cells.
            inpaint=InpaintMethod.CONV,
            inpaint_iter=3,
            aggregation=AggregationMethod.MEAN,
        )
    else:
        full_warped = warper.backward_warp(
            src_tensor,
            dst_size=(proj_width, proj_height),
            inpaint=InpaintMethod.NONE,
        )
        cx, cy, cw, ch = crop_rect
        warped_tensor = full_warped[:, cy : cy + ch, cx : cx + cw]

    warped_array = warped_tensor.permute(1, 2, 0).numpy()
    warped_height, warped_width = warped_array.shape[:2]
    return warped_array.reshape(warped_height, warped_width, *src_shape[2:])


def warp_image(
    src_image: np.ndarray,
    pixel_map_path: str,
    proj_width: int,
    proj_height: int,
    image_width: int,
    image_height: int,
    warp_method: WarpMethod = "c2p",
) -> np.ndarray:
    """Warp an image from camera view to projector view using a pixel correspondence map.

    Perform forward warping to transform an image captured by the camera
    into the projector's coordinate system using pre-computed
    camera-to-projector mapping.

    ピクセル対応マップを使用してカメラビューからプロジェクタビューに画像をワーピングする。

    事前計算されたカメラ-プロジェクタ間マッピングを使用して、カメラで
    キャプチャされた画像をプロジェクタの座標系に順方向ワーピングで変換する。

    Args:
        src_image: Source image in camera coordinates with shape
            ``(H, W, 3)``.
            形状 ``(H, W, 3)`` のカメラ座標系のソース画像。
        pixel_map_path: Path to the correspondence map (``.npy`` file).
            対応マップ（``.npy`` ファイル）のパス。
        proj_width: Width of the projector display.
            プロジェクタディスプレイの幅。
        proj_height: Height of the projector display.
            プロジェクタディスプレイの高さ。
        image_width: Width of the target image region.
            目標画像領域の幅。
        image_height: Height of the target image region.
            目標画像領域の高さ。
        warp_method: Correspondence direction (``"c2p"`` or ``"p2c"``).
            対応の向き（``"c2p"`` または ``"p2c"``）。

    Returns:
        Warped image in projector coordinates with shape
        ``(proj_height, proj_width, 3)``.
        形状 ``(proj_height, proj_width, 3)`` のプロジェクタ座標系の
        ワーピング済み画像。
    """
    return warp_camera_array_to_projector(
        src_image,
        pixel_map_path,
        proj_width,
        proj_height,
        image_width,
        image_height,
        warp_method,
    )


def warp_color_mixing_matrices_to_projector(
    color_mixing_matrices: np.ndarray,
    pixel_map_path: str,
    proj_width: int,
    proj_height: int,
    image_width: int,
    image_height: int,
    warp_method: WarpMethod = "p2c",
) -> np.ndarray:
    """Warp camera-space color mixing matrices into projector coordinates."""
    if color_mixing_matrices.ndim != 4 or color_mixing_matrices.shape[2:] != (4, 3):
        raise ValueError(
            "color_mixing_matrices must have shape (H, W, 4, 3)."
        )

    warped = warp_camera_array_to_projector(
        color_mixing_matrices,
        pixel_map_path,
        proj_width,
        proj_height,
        image_width,
        image_height,
        warp_method,
    )
    return warped.astype(color_mixing_matrices.dtype, copy=False)


def invwarp_image(
    src_image: np.ndarray,
    pixel_map_path: str,
    proj_width: int,
    proj_height: int,
    cam_width: int,
    cam_height: int,
    warp_method: WarpMethod = "c2p",
) -> np.ndarray:
    """Inverse warp an image from projector view to camera view.

    Perform backward warping to transform an image from projector
    coordinates into the camera's coordinate system using pre-computed
    mapping.

    プロジェクタビューからカメラビューに画像を逆ワーピングする。

    事前計算されたマッピングを使用して、プロジェクタ座標系の画像を
    カメラの座標系に逆方向ワーピングで変換する。

    Args:
        src_image: Source image in projector coordinates with shape
            ``(H, W, 3)``.
            形状 ``(H, W, 3)`` のプロジェクタ座標系のソース画像。
        pixel_map_path: Path to the correspondence map (``.npy`` file).
            対応マップ（``.npy`` ファイル）のパス。
        proj_width: Width of the projector display.
            プロジェクタディスプレイの幅。
        proj_height: Height of the projector display.
            プロジェクタディスプレイの高さ。
        cam_width: Width of the camera image.
            カメラ画像の幅。
        cam_height: Height of the camera image.
            カメラ画像の高さ。
        warp_method: Correspondence direction (``"c2p"`` or ``"p2c"``).
            対応の向き（``"c2p"`` または ``"p2c"``）。

    Returns:
        Inverse warped image in camera coordinates with shape
        ``(cam_height, cam_width, 3)``.
        形状 ``(cam_height, cam_width, 3)`` のカメラ座標系の
        逆ワーピング済み画像。
    """
    import torch
    from graycode.warp_image import (
        AggregationMethod,
        InpaintMethod,
        PixelMapWarperTorch,
        SplatMethod,
    )
    from graycode.interpolate_c2p import load_c2p_numpy
    from graycode.interpolate_p2c import load_p2c_numpy_array

    src_rect = center_rect(
        src_image.shape[1], src_image.shape[0], proj_width, proj_height
    )
    src_image_tensor = torch.from_numpy(src_image).permute(2, 0, 1)
    if warp_method == "c2p":
        pixel_map = load_c2p_numpy(pixel_map_path)
        warper = PixelMapWarperTorch(pixel_map)
        invwarped_image_tensor = warper.backward_warp(
            src_image_tensor,
            dst_size=(cam_width, cam_height),
            src_rect=src_rect,
            inpaint=InpaintMethod.NONE,
        )
    elif warp_method == "p2c":
        pixel_map = load_p2c_numpy_array(pixel_map_path)
        warper = PixelMapWarperTorch(pixel_map)
        invwarped_image_tensor = warper.forward_warp(
            src_image_tensor,
            dst_size=(cam_width, cam_height),
            src_offset=(src_rect[0], src_rect[1]),
            splat_method=SplatMethod.BILINEAR,
            inpaint=InpaintMethod.NONE,
            aggregation=AggregationMethod.MEAN,
        )
    else:
        raise ValueError("warp_method must be 'c2p' or 'p2c'.")

    invwarped_image = invwarped_image_tensor.permute(1, 2, 0).numpy()
    return invwarped_image
