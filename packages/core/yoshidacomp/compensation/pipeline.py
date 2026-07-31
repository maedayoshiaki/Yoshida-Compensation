"""Hardware-free compensation generation pipeline.

Everything needed to turn a target image plus a color mixing matrix into
a projector code-value image, with no camera or projector attached. The
CLI entry points (``examples/python/sample.py``,
``examples/python/rebuild_compensation_with_map.py``) are thin consumers
of this module.

ハードウェア非依存の補償画像生成パイプライン。

ターゲット画像とカラーミキシング行列からプロジェクタ投影用コード値まで
を、カメラ・プロジェクタ無しで計算するための処理をまとめる。CLI
（``examples/python/sample.py``、
``examples/python/rebuild_compensation_with_map.py``）はこのモジュールの
薄いラッパとして動く。
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Protocol, Tuple, cast

import cv2
import numpy as np

from yoshidacomp.compensation.color_mixing_matrix import (
    apply_inverse_gamma_correction,
)
from yoshidacomp.compensation.inverse_lut import InverseLut, apply_inverse_lut
from yoshidacomp.compensation.photometric_compensation import (
    calc_compensation_image,
)
from yoshidacomp.compensation.projection_region import (
    embed_in_active_region,
    mask_to_active_region,
)
from yoshidacomp.config import PathsConfig
from yoshidacomp.geometry.warp import WarpMethod, invwarp_image, warp_image


TargetImageSpace = Literal["camera", "projector"]
TargetDecodeMode = Literal["none", "srgb", "gamma"]
LinearizerName = Literal["none", "gamma", "inverse_lut"]
SUPPORTED_TARGET_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


class ProjectorLike(Protocol):
    """Minimal projector description consumed by this module.

    このモジュールが参照するプロジェクタ設定の最小インターフェース。
    """

    width: int
    height: int
    gamma: float


def resolve_warp_settings(paths: PathsConfig) -> tuple[WarpMethod, str]:
    """Resolve warp method and map path from configuration."""
    warp_method_raw = paths.warp_method.strip().lower()
    if warp_method_raw not in ("c2p", "p2c"):
        raise ValueError("Invalid `paths.warp_method`. Use \"c2p\" or \"p2c\".")

    warp_method = cast(WarpMethod, warp_method_raw)
    map_path = paths.c2p_map if warp_method == "c2p" else paths.p2c_map
    if map_path.strip() == "":
        key = "paths.c2p_map" if warp_method == "c2p" else "paths.p2c_map"
        raise ValueError(f"`{key}` is empty. Set a valid .npy path.")
    return warp_method, map_path


def resolve_target_image_space(paths: PathsConfig) -> TargetImageSpace:
    """Resolve the configured target-image coordinate system."""
    target_image_space_raw = paths.target_image_space.strip().lower()
    if target_image_space_raw not in ("camera", "projector"):
        raise ValueError(
            "Invalid `paths.target_image_space`. Use \"camera\" or \"projector\"."
        )
    return cast(TargetImageSpace, target_image_space_raw)


def compensation_space_from_warp_method(
    warp_method: WarpMethod,
) -> TargetImageSpace:
    """Return the coordinate system where compensation is computed."""
    return "camera" if warp_method == "c2p" else "projector"


def load_target_images(target_dir: str) -> list[tuple[str, np.ndarray]]:
    """Load target images in a deterministic order."""
    target_images: list[tuple[str, np.ndarray]] = []
    for img_name in sorted(os.listdir(target_dir), key=str.casefold):
        if not img_name.lower().endswith(SUPPORTED_TARGET_IMAGE_SUFFIXES):
            continue
        img_path = os.path.join(target_dir, img_name)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        target_img_array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        target_images.append((img_name, target_img_array))
    return target_images


def linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB values in [0, 1] to sRGB values in [0, 1].

    Apply the standard piecewise sRGB transfer function.

    範囲 [0, 1] のリニア RGB を sRGB に変換する。

    標準の区分的 sRGB 伝達関数を適用する。

    Args:
        linear_rgb: Linear RGB image array.
            リニア RGB 画像配列。

    Returns:
        sRGB image array in range [0, 1].
            範囲 [0, 1] の sRGB 画像配列。
    """
    linear_rgb = np.clip(linear_rgb, 0.0, 1.0)
    return np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * np.power(linear_rgb, 1 / 2.4) - 0.055,
    )


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 1] to linear RGB values in [0, 1].

    Inverse of :func:`linear_to_srgb`. The compensation pipeline operates
    on linear light, so sRGB-encoded inputs (e.g. target images loaded
    from PNG/JPEG) must pass through this before any computation.

    範囲 [0, 1] の sRGB をリニア RGB に変換する。

    :func:`linear_to_srgb` の逆変換。補償パイプラインはリニア光量で
    計算するため、sRGB エンコードされた入力（PNG/JPEG のターゲット画像
    など）は計算前に必ずこの変換を通す。

    Args:
        srgb: sRGB image array in range [0, 1].
            範囲 [0, 1] の sRGB 画像配列。

    Returns:
        Linear RGB image array in range [0, 1].
            範囲 [0, 1] のリニア RGB 画像配列。
    """
    srgb = np.clip(srgb, 0.0, 1.0)
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    )


def capture_to_srgb_uint8(image: np.ndarray) -> np.ndarray:
    """Encode a linear capture as sRGB uint8 for saving/viewing only.

    The pipeline itself stays linear; this exists solely so captured
    frames written to disk are viewable. uint8 input passes through.

    リニア撮影値を保存・表示用に sRGB uint8 へエンコードする。
    パイプライン本体はリニアのまま。uint8 入力はそのまま返す。
    """
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    return (
        (linear_to_srgb(arr.astype(np.float32)) * 255.0).round().astype(np.uint8)
    )


def decode_target(
    target_uint8: np.ndarray,
    *,
    mode: TargetDecodeMode,
    gamma: float = 2.2,
) -> np.ndarray:
    """Convert uint8 target values into the compensation target domain."""
    normalized = target_uint8.astype(np.float32) / 255.0
    if mode == "none":
        return normalized
    if mode == "srgb":
        return srgb_to_linear(normalized).astype(np.float32)
    if mode == "gamma":
        if not np.isfinite(gamma) or gamma <= 0:
            # 引数名で通知する。ここはライブラリなので、呼び出し側が CLI とは
            # 限らず、"--target-gamma" というフラグ名は文脈を持たない。
            raise ValueError(
                f"gamma must be a finite positive number, got {gamma!r}."
            )
        return np.power(normalized, gamma).astype(np.float32)
    raise ValueError(f"Unsupported target decode mode: {mode!r}")


def prepare_target_image_for_compensation(
    target_image: np.ndarray,
    target_image_space: TargetImageSpace,
    pixel_map_path: str,
    proj_width: int,
    proj_height: int,
    cam_width: int,
    cam_height: int,
    warp_method: WarpMethod,
) -> np.ndarray:
    """Convert the target image into the coordinate system used for compensation."""
    compensation_space = compensation_space_from_warp_method(warp_method)
    if target_image_space == compensation_space:
        return target_image

    if target_image_space == "projector":
        return invwarp_image(
            target_image,
            pixel_map_path,
            proj_width,
            proj_height,
            cam_width,
            cam_height,
            warp_method=warp_method,
        )

    return warp_image(
        target_image,
        pixel_map_path,
        proj_width,
        proj_height,
        target_image.shape[1],
        target_image.shape[0],
        warp_method=warp_method,
    )


class Linearizer(Protocol):
    """Map linear compensation codes to projector input codes.

    線形補償コード値をプロジェクタ入力コード値へ写す。
    """

    def __call__(
        self,
        image: np.ndarray,
        *,
        projector: ProjectorLike,
        lut: Optional[InverseLut] = None,
        dtype: np.typing.DTypeLike = np.uint8,
    ) -> np.ndarray:
        ...


def _linearize_none(
    image: np.ndarray,
    *,
    projector: ProjectorLike,
    lut: Optional[InverseLut] = None,
    dtype: np.typing.DTypeLike = np.uint8,
) -> np.ndarray:
    """Pass the compensation codes through unchanged.

    補償コード値をそのまま通す。

    Returns the *same object* it was given -- callers must not mutate the
    result in place, and an identity check against the input is not a
    meaningful assertion about this function.
    受け取ったオブジェクトを**そのまま**返す。呼び出し側は戻り値を破壊的に
    書き換えてはならず、入力との同一性チェックは意味のある検証にならない。
    """
    return image


def _linearize_gamma(
    image: np.ndarray,
    *,
    projector: ProjectorLike,
    lut: Optional[InverseLut] = None,
    dtype: np.typing.DTypeLike = np.uint8,
) -> np.ndarray:
    """Apply the analytic inverse gamma of the projector.

    プロジェクタの解析逆ガンマを適用する。

    Note: apply_inverse_gamma_correction() uses the given exponent directly,
    so brightening requires 1 / projector.gamma. It preserves the input's
    dtype, so *dtype* is accepted only for protocol conformance.
    注: apply_inverse_gamma_correction() は与えた指数をそのまま使うため、
    明るくするには 1 / projector.gamma を渡す必要がある。入力の dtype を
    保つため、*dtype* はプロトコル整合のためだけに受け取る。
    """
    return apply_inverse_gamma_correction(image, gamma=1.0 / projector.gamma)


def _linearize_inverse_lut(
    image: np.ndarray,
    *,
    projector: ProjectorLike,
    lut: Optional[InverseLut] = None,
    dtype: np.typing.DTypeLike = np.uint8,
) -> np.ndarray:
    """Apply a measured per-channel inverse LUT.

    実測のチャンネル別逆LUTを適用する。
    """
    if lut is None:
        raise ValueError("linearizer='inverse_lut' requires an InverseLut")
    return apply_inverse_lut(image, lut, dtype=dtype)


LINEARIZERS: dict[str, Linearizer] = {
    "none": _linearize_none,
    "gamma": _linearize_gamma,
    "inverse_lut": _linearize_inverse_lut,
}


def generate_projection_image(
    target_uint8: np.ndarray,
    warped_matrix: np.ndarray,
    *,
    projector: ProjectorLike,
    active_region: Tuple[int, int, int, int],
    linearizer: LinearizerName,
    lut: Optional[InverseLut] = None,
    target_decode: TargetDecodeMode = "srgb",
    target_gamma: float = 2.2,
    dtype: np.typing.DTypeLike = np.uint8,
    clip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one target image into linear and projector-ready code values.

    The matrix must already be expressed in the same coordinate system as
    the projector frame (warp it beforehand with
    :func:`yoshidacomp.geometry.warp.warp_color_mixing_matrices_to_projector`).

    ターゲット画像1枚から、線形補償画像と投影用コード値を生成する。
    行列は事前にプロジェクタ座標へワープしておくこと。

    Args:
        target_uint8: Target image, shape ``(H, W, 3)`` uint8 RGB.
            ターゲット画像（``(H, W, 3)`` uint8 RGB）。
        warped_matrix: Color mixing matrices in projector space,
            shape ``(H, W, 4, 3)``.
            プロジェクタ座標のカラーミキシング行列（``(H, W, 4, 3)``）。
        projector: Projector description providing width/height/gamma.
            width/height/gamma を持つプロジェクタ設定。
        active_region: ``(x, y, width, height)`` of the projection area.
            投影領域 ``(x, y, width, height)``。
        linearizer: Key into :data:`LINEARIZERS`.
            :data:`LINEARIZERS` のキー。
        lut: Measured inverse LUT, required for ``"inverse_lut"``.
            実測逆LUT。``"inverse_lut"`` のとき必須。
        target_decode: Transfer function used to decode the target.
            ターゲットの復号方式。
        target_gamma: Exponent used when ``target_decode="gamma"``.
            ``target_decode="gamma"`` のときの指数。
        dtype: Output data type of both returned arrays. ``np.uint8``
            (default) reproduces the CLI's 8-bit quantization; a float
            dtype keeps the compensation in normalized units, which is
            what a viewer needs.
            返す2配列の出力データ型。既定の ``np.uint8`` は CLI と同じ
            8bit 量子化。float を指定すると補償値を正規化のまま保持でき、
            ビューアはこちらを必要とする。
        clip: Clamp the compensation to [0, 1] before quantization.
            Defaults to ``True``. ``clip=False`` keeps out-of-gamut
            values so a caller can classify which pixels the matrix
            drives below 0 or above 1; it requires ``linearizer="none"``.
            量子化前に補償値を [0, 1] にクランプするか。既定は ``True``。
            ``clip=False`` は色域外の値を保持し、行列が 0 未満／1 超へ
            振る画素を呼び出し側が判定できるようにする。この場合
            ``linearizer="none"`` と float の *dtype* が必須。

    Returns:
        ``(linear_compensation, projector_codes)``, both of *dtype*.
        ``(線形補償画像, 投影用コード値)``。いずれも *dtype*。

    Raises:
        ValueError: If *linearizer* is unknown, or if ``clip=False`` is
            combined with a linearizer other than ``"none"`` or with an
            integer *dtype* -- both would discard or corrupt the very
            out-of-range values ``clip=False`` preserves.
            *linearizer* が未知の場合、または ``clip=False`` を
            ``"none"`` 以外の linearizer や整数 *dtype* と併用した場合
            （いずれも ``clip=False`` が保持する範囲外の値を捨てる／壊す）。
    """
    try:
        linearize = LINEARIZERS[linearizer]
    except KeyError:
        raise ValueError(f"Unsupported linearizer: {linearizer!r}") from None

    if not clip and linearizer != "none":
        raise ValueError(
            f"clip=False requires linearizer='none', got {linearizer!r}. "
            "apply_inverse_gamma_correction() and apply_inverse_lut() both "
            "assume input in [0, 1] (they normalize/index against that range) "
            "and would return garbage for the out-of-range values clip=False "
            "is meant to preserve."
        )

    if not clip and np.issubdtype(np.dtype(dtype), np.integer):
        raise ValueError(
            f"clip=False requires a float dtype, got {np.dtype(dtype)!s}. "
            "Integer output scales by 255/65535 and casts, which is "
            "undefined for values outside [0, 1] -- exactly the values "
            "clip=False exists to preserve."
        )

    target_linear = decode_target(
        target_uint8,
        mode=target_decode,
        gamma=target_gamma,
    )
    target_linear = embed_in_active_region(
        target_linear,
        (projector.width, projector.height),
        active_region,
    )
    compensation = calc_compensation_image(
        target_image=target_linear,
        color_mixing_matrices=warped_matrix,
        dtype=dtype,
        clip=clip,
    )
    compensation = mask_to_active_region(compensation, active_region)

    projector_image = linearize(
        compensation, projector=projector, lut=lut, dtype=dtype
    )
    if hasattr(projector_image, "cpu"):
        projector_image = projector_image.cpu().numpy()
    return compensation, np.asarray(projector_image, dtype=dtype)
