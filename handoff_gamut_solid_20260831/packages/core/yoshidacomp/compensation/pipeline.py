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

from yoshidacomp.compensation.inverse_lut import InverseLut, apply_inverse_lut
from yoshidacomp.compensation.photometric_compensation import (
    calc_compensation_image,
    quantize_unit_image,
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
    """Convert the target image into the coordinate system used for compensation.

    **This function still costs ~7s and initializes CUDA on first call, and
    that is not fixed by Task 1's numpy switches.** When
    ``target_image_space`` differs from the compensation space, this calls
    ``invwarp_image``/``warp_image`` (``yoshidacomp.geometry.warp``), which
    goes through GrayCode's ``PixelMapWarperTorch`` for a dense scattered
    warp over ~2M points -- genuine GPU-scale work, unlike the twelve
    multiply-adds per pixel :func:`generate_projection_image`'s matmul step
    used to do. Measured in a fresh process: ``'torch' in sys.modules`` is
    ``False`` after ``import yoshidacomp.compensation.pipeline`` and
    ``True`` after the first call to *this* function -- torch imports
    before the call even reaches a missing-pixel-map failure. This is
    intentionally not rewritten to numpy or otherwise guarded: unlike
    ``calc_compensation_image`` (Addendum 4 of the task report), the
    warping workload plausibly justifies GPU batching, and this function
    was not measured as part of Task 1's decision. Do not generalize
    :func:`generate_projection_image`'s "torch-free end to end" claim (see
    :func:`_linearize_gamma`'s docstring) to this function or to
    ``pipeline.py`` as a whole -- they are not the same claim. If a future
    task (plan B4: wiring the P2C warp into the web layer) calls this
    function or ``geometry.warp`` directly from an interactive path, the
    decision is pre-warming torch at process startup versus accepting a
    ~7s stall on first use -- not "already solved" by this task.

    このモジュールを import しただけでは torch を引かないが、この関数を
    実際に呼ぶと約7秒かかりCUDAを初期化する。これはTask 1のnumpy化では
    解決していない。``target_image_space`` が補償を行う座標系と異なる
    場合、GrayCodeの ``PixelMapWarperTorch`` による約200万点の散在
    ワープ（``invwarp_image``/``warp_image``、
    ``yoshidacomp.geometry.warp``）を経由する -- これは本物のGPU規模の
    処理であり、:func:`generate_projection_image` の行列積ステップ
    （1画素12回の積和）とは異なる。実測: 新しいプロセスで
    ``import yoshidacomp.compensation.pipeline`` の直後は
    ``'torch' in sys.modules`` が ``False``、この関数を一度呼ぶと
    ``True`` になる（pixel_map欠落エラーに到達する前にtorchが
    importされる）。意図的にnumpy化・保護のいずれもしていない:
    ``calc_compensation_image``（レポートAddendum 4）と異なり、ワープの
    計算量はGPUバッチ処理を正当化しうる規模であり、この関数はTask 1の
    判断の対象として測定していない。:func:`generate_projection_image`
    の「torch-free end to end」という主張（:func:`_linearize_gamma` の
    docstring参照）をこの関数や ``pipeline.py`` 全体へ一般化しないこと
    -- 同じ主張ではない。将来のタスク（計画B4: P2Cワープをweb層へ配線）
    がこの関数や ``geometry.warp`` を対話的な経路から直接呼ぶ場合、
    判断すべきはプロセス起動時のtorch事前ウォームアップか初回利用時の
    約7秒停止を許容するかであり、「Task 1で既に解決済み」ではない。
    """
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


def _apply_inverse_gamma_numpy(
    image: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Raise a normalized RGB image to *gamma*, clamped to [0, 1].

    ``color_mixing_matrix.apply_inverse_gamma_correction`` の numpy 版。
    *gamma* はそのまま指数として使うので、明るくするには 1/projector.gamma を
    渡す。入力の dtype を保つ。

    torch 版は 600 万画素の冪乗のために CUDA を初期化する。対話的GUIでは
    初回に数秒止まり、プロジェクタを駆動しているのと同じGPUのメモリを取るため
    割に合わない。加えて torch は module scope で psutil ごと web 層へ漏れる。

    **Wired into ``LINEARIZERS["gamma"]``** (via :func:`_linearize_gamma`).

    **2026-08-06:** integer outputs now round half-up (``floor(x*scale+0.5)``)
    instead of truncating, as part of the quantization-order fix (see
    ``tests/test_projection_quantization_order.py``). The torch
    implementation was changed identically in the same commit. The
    measurements quoted below were taken against the *truncating* version:
    the uint8 numpy-vs-torch byte parity is re-verified for the rounding
    version by ``tests/test_pipeline_gamma_numpy.py``, while the uint16
    mismatch counts are historical and have not been re-measured.

    Measured against the torch implementation (see
    ``.superpowers/sdd/2026-08-02-target-and-lut/task-1-report.md`` for the
    full table, corrected after an earlier draft attributed these numbers to
    the wrong device/denominator):

    - **uint8** (every real caller's dtype -- the CLI default and the only
      dtype any current call site passes) matches **byte-for-byte over all
      256 input values in both directions, exhaustively, against both the
      CPU and CUDA torch backends** (0/768 mismatches every time). This is
      the decisive result: for every input this codebase actually produces,
      this function and the torch implementation are bit-identical, so
      switching changes nothing about delivered output.
    - **float32** does not match bit-for-bit, but the disagreement is
      smaller than torch's own cross-device disagreement with itself: on a
      4001-point [0, 1] grid (12003 elements) at gamma=1/2.2, this function
      disagrees with torch-CPU on 222/12003 elements and with torch-CUDA on
      774/12003, always by exactly 1 ULP (5.96e-8 at this magnitude) --
      while torch-CUDA disagrees with torch-CPU, on that same grid, on
      **762/12003** elements, by the same 1 ULP. ``_linearize_gamma`` calls
      torch with the CUDA default, so "the torch answer" was never a single
      fixed value to begin with -- it already depended on which device the
      process happened to run on. There is no bit-exact float32 target this
      function could preserve that torch itself preserves across its own
      backends.
    - An exhaustive uint16 sweep (all 65536 values x 3 channels = 196608,
      no current caller uses this dtype) found 1-LSB disagreements against
      torch-CUDA: 45/196608 brightening (gamma=1/2.2), 21/196608 darkening
      (gamma=2.2); against torch-CPU: 6/196608 brightening, 9/196608
      darkening. Both devices' counts are real, if very small (~0.005-0.02%),
      effects of the same underlying 1-ULP float32 gap landing on an integer
      rounding boundary.

    Because the dtype that decides real behavior (uint8) is exhaustively
    identical, and the dtype that isn't (float32) is off by less than torch
    disagrees with itself, ``_linearize_gamma`` calls this function rather
    than torch: delivery output is unchanged, the float32 preview the web
    layer wants is within torch's own noise floor, and the gamma step's own
    marginal contribution to the ~8s CUDA-initialization stall (and the
    projector GPU's memory contention) is gone. ``calc_compensation_image``
    (``photometric_compensation.py``) was, for a while, the reason
    ``generate_projection_image`` as a whole still paid that stall
    regardless of linearizer; it has since been switched to numpy too (same
    measure-then-decide method, see the task report's Addendum 4), so this
    module now has no torch dependency anywhere in ``generate_projection_image``'s
    call graph.

    ``LINEARIZERS["gamma"]`` に（:func:`_linearize_gamma` 経由で）配線
    している。torch 版との実測比較（詳細は上記レポートの表。旧版はこの
    数値をCPU/CUDAの取り違え・分母の誤りで誤記していたため訂正済み）:

    - **uint8**（実際の呼び出しが渡す唯一のdtype。CLIの既定でもある）は
      256値×両方向×CPU/CUDAいずれでも**バイト単位で全数一致**した
      （0/768、常に）。これが決め手になった結果である -- このコードベースが
      実際に生成するあらゆる入力について、この関数とtorch実装はビット単位で
      同一であり、切り替えても配信物は何も変わらない。
    - **float32** はビット単位で一致しないが、その差は torch 自身の
      CPU/CUDA間の差より小さい: [0, 1] を4001点で刻んだグリッド
      （12003要素）、gamma=1/2.2 で、この関数は torch-CPU と 222/12003、
      torch-CUDA と 774/12003 で不一致（常に1ULP、このレンジで5.96e-8）。
      一方 torch-CUDA と torch-CPU は同じグリッドで **762/12003** 不一致
      （同じ1ULP幅）。``_linearize_gamma`` は既定でCUDAのtorchを呼んでいた
      ため、「torchの答え」はそもそも単一の固定値ではなく、どのデバイスで
      実行したかに依存していた。torch自身が自分のバックエンド間で保てない
      ビット精度を、この関数が保つべき基準は存在しない。
    - uint16 を全数検査（65536値×3ch=196608、現行の呼び出し元は誰も
      このdtypeを使わない）すると、torch-CUDA との相違は1LSBが
      明方向（gamma=1/2.2）45/196608・暗方向（gamma=2.2）21/196608件、
      torch-CPU との相違は明方向6/196608・暗方向9/196608件見つかった。
      いずれのデバイスでも小さい（約0.005〜0.02%）が、同じ1ULPの
      float32ギャップが整数丸め境界に落ちた実在の効果である。

    実際の挙動を決めるdtype（uint8）が全数一致し、一致しないdtype
    （float32）もtorch自身の自己不一致より小さいずれしかないため、
    ``_linearize_gamma`` は torch ではなくこの関数を呼ぶ: 配信物は不変、
    web層が求める float プレビューは torch自身のノイズ幅に収まり、
    ガンマ処理自身が約8秒のCUDA初期化停止（とプロジェクタと同じGPUの
    メモリ競合）に寄与していた分は無くなる。``calc_compensation_image``
    （``photometric_compensation.py``）は、しばらくの間
    ``generate_projection_image`` 全体が linearizer に関わらず
    その停止を払い続けていた原因だったが、同じ「測ってから決める」
    手法で numpy へ切り替え済み（詳細はレポートの Addendum 4）。
    そのため現在は ``generate_projection_image`` の呼び出しグラフ全体で
    torch への依存が無い。

    Args:
        image: RGB image, shape ``(H, W, 3)``. uint8 in [0, 255], uint16 in
            [0, 65535], or float already normalized to roughly [0, 1]
            (values outside that range are accepted; the result is NaN for
            a negative base with a non-integer *gamma*, matching torch, and
            clamps to 1.0 above 1.0).
            形状 ``(H, W, 3)`` の RGB 画像。uint8 は [0, 255]、uint16 は
            [0, 65535]、float はおおむね [0, 1] に正規化済み（範囲外も
            受け付ける。負の底に非整数 *gamma* を与えると torch と同様
            NaN になり、1超は 1.0 にクランプされる）。
        gamma: Exponent applied directly to the normalized image -- **not**
            inverted for you. Brightening a projector whose response is
            ``t ** projector.gamma`` requires passing ``1 / projector.gamma``.
            正規化画像に直接適用される指数。呼び出し側で反転しない。
            応答が ``t ** projector.gamma`` のプロジェクタを明るくするには
            ``1 / projector.gamma`` を渡す。

    Returns:
        Array of the same dtype as *image*, with one deliberate exception:
        for float dtypes other than float32 (float64, float16, ...), torch's
        ``apply_inverse_gamma_correction`` calls ``.float()`` unconditionally
        and returns float32 regardless of the input's float dtype -- this
        function does not replicate that, and instead genuinely preserves
        the input dtype (float64 in, float64 out). Nothing in this codebase
        passes anything but uint8, uint16, or float32 through this path, so
        it is untested and arguably the more correct behavior, but it is a
        real, intentional structural difference from what this function
        otherwise ports faithfully.
        *image* と同じ dtype の配列。ただし1つ意図的な例外がある:
        float32以外のfloat dtype（float64、float16 等）について、torch版の
        ``apply_inverse_gamma_correction`` は無条件に ``.float()`` を
        呼ぶため入力のfloat dtypeに関わらずfloat32を返すが、この関数は
        それを再現せず、入力dtypeを本当に保つ（float64入力→float64出力）。
        このコードベースはuint8・uint16・float32以外をこの経路に渡さない
        ため未検証だが、おそらくこちらの挙動の方が妥当である。とはいえ
        他の点では忠実に移植したこの関数における、意図的で本物の構造差
        である。
    """
    dtype = image.dtype
    if dtype == np.uint8:
        normalized = image.astype(np.float32) / 255.0
    elif dtype == np.uint16:
        normalized = image.astype(np.float32) / 65535.0
    else:
        normalized = image.astype(np.float32)

    # A negative base with a non-integer exponent is undefined for reals;
    # matching torch's behaviour means propagating NaN rather than raising.
    with np.errstate(invalid="ignore"):
        corrected = np.power(normalized, np.float32(gamma), dtype=np.float32)
    # The lower bound (0.0) is currently unreachable in production: the only
    # caller, _linearize_gamma, only ever receives calc_compensation_image's
    # already-clipped [0, 1] output (generate_projection_image rejects
    # clip=False for any linearizer but "none"). Kept anyway because this
    # function's own contract (see Args above) accepts negative input
    # directly, and torch's version clips it too -- dead in this codebase
    # today, not dead by this function's own documented interface.
    corrected = np.clip(corrected, 0.0, 1.0)

    # Round half-up, matching calc_compensation_image / quantize_unit_image.
    # A bare ``.astype`` here would truncate and bias every code down by up
    # to 1 count (2026-08-06 fix; the torch implementation was changed the
    # same way, and the exhaustive parity tests re-verify the two agree).
    if dtype == np.uint8:
        return np.floor(corrected * 255.0 + 0.5).astype(np.uint8)
    if dtype == np.uint16:
        return np.floor(corrected * 65535.0 + 0.5).astype(np.uint16)
    return corrected.astype(dtype)


def _linearize_gamma(
    image: np.ndarray,
    *,
    projector: ProjectorLike,
    lut: Optional[InverseLut] = None,
    dtype: np.typing.DTypeLike = np.uint8,
) -> np.ndarray:
    """Apply the analytic inverse gamma of the projector.

    プロジェクタの解析逆ガンマを適用する。

    Note: the exponent is used directly, so brightening requires
    1 / projector.gamma. This preserves the input's dtype, so *dtype* is
    accepted only for protocol conformance.
    注: 指数をそのまま使うため、明るくするには 1 / projector.gamma を
    渡す必要がある。入力の dtype を保つため、*dtype* はプロトコル整合の
    ためだけに受け取る。

    Calls :func:`_apply_inverse_gamma_numpy`, **not** the torch
    implementation in ``color_mixing_matrix``. *This function's own*
    contribution to torch/psutil/scipy dependency is now zero, on both
    import and first call. See :func:`_apply_inverse_gamma_numpy`'s
    docstring for the measurement that justifies this: byte-for-byte
    identical to torch for every dtype any real caller uses (uint8), and
    for the one dtype where it isn't (float32) the gap is smaller than the
    gap between torch's own CPU and CUDA backends. An earlier version of
    this function called torch directly (lazily imported, to at least keep
    it out of module-scope import); that was reversed after measuring that
    the *first call* -- not just the import -- cost ~7.9s and initialized
    CUDA, which is exactly the interactive-GUI stall the numpy version
    exists to avoid.

    This function's switch alone did **not** make ``generate_projection_image``
    as a whole torch-free on first call: ``calc_compensation_image`` (the
    compensation matrix-multiply step, called unconditionally regardless of
    linearizer) still needed torch and still cost roughly the same
    ~7.6-7.7s on first use -- confirmed by measuring ``linearizer="none"``
    alone, which pulled torch and cost about the same. That has since also
    been switched to numpy (Addendum 4 of the task report), by the same
    measure-then-decide method: torch's own CPU and CUDA backends disagree
    with each other on real data too (by up to 1 count in uint8, never
    exceeded by numpy's own divergence from either backend), so the same
    "no single bit-exact torch answer to preserve" reasoning applied.
    ``generate_projection_image``'s call graph is now torch-free end to end.
    :func:`_apply_inverse_gamma_numpy` を呼ぶ。``color_mixing_matrix`` の
    torch 実装は呼ばない。*この関数自身*が寄与する torch/psutil/scipy
    依存は、import・初回呼び出しのいずれにおいてもゼロになった。
    この判断の根拠となる実測は :func:`_apply_inverse_gamma_numpy` の
    docstring を参照: 実際の呼び出し元が使う唯一の dtype（uint8）では
    torch とバイト単位で完全一致し、一致しない float32 でも、その差は
    torch 自身の CPU/CUDA バックエンド間の差より小さい。以前のこの関数は
    torch を直接呼んでいた（遅延importでmodule scopeからは外していた）が、
    import だけでなく*初回呼び出し*自体が約7.9秒かかりCUDAを初期化する
    ことを実測し、これはまさに numpy 版が避けようとしている対話的GUIの
    停止そのものだったため、この判断を覆した。

    この関数だけの切り替えでは ``generate_projection_image`` 全体が
    初回呼び出しで torch-free になったわけではなかった:
    ``calc_compensation_image``（補償の行列積ステップ。linearizer に
    関わらず無条件に呼ばれる）は依然として torch を必要とし、初回利用時に
    同程度の約7.6〜7.7秒かかっていた -- ``linearizer="none"`` 単体でも
    torch を引き、同程度のコストがかかることを実測して確認済み。これも
    同じ「測ってから決める」手法で numpy へ切り替え済み（レポートの
    Addendum 4）: torch 自身の CPU/CUDA バックエンドは実データでも
    （uint8で最大1カウント）互いにずれており、numpy がどちらかと
    ずれる幅もそれを超えないため、「単一のビット精度なtorchの答えは
    そもそも存在しない」という同じ理屈が成り立った。これで
    ``generate_projection_image`` の呼び出しグラフ全体が torch-free に
    なっている。
    """
    return _apply_inverse_gamma_numpy(image, gamma=1.0 / projector.gamma)


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

# The filename prefix each linearizer's projector images are saved under, e.g.
# ``no_gamma_comp_image_00.png``. Keyed by the same names as ``LINEARIZERS``
# directly above, so a linearizer added there without a prefix here fails
# ``test_web_generate.py::test_every_linearizer_has_a_filename_prefix`` rather
# than raising ``KeyError`` inside whichever writer reached it first.
#
# This dict lives in the core, not in a writer, because there are two writers:
# ``examples/python/rebuild_compensation_with_map.py`` (the CLI) and
# ``packages/web/yoshidacomp_web/generate.py`` (``POST /api/generate``). It was
# the CLI's private literal until the web endpoint needed the same names; a
# second copy there would have been two dicts that must agree and nothing
# making them, so the GUI and the CLI could come to write different filenames
# for the same run.
#
# **The readers import it too**, and that took a second pass: the four scripts
# that glob or clear these files -- ``evaluate_compensation_conditions.py``,
# ``evaluate_recalibrated_compensation.py``, ``sample.py`` and
# ``sample_inverse_lut.py`` -- went on spelling the names out after the writers
# were unified, so "one place" was true of the two files that write and false of
# the four that read. A glob that misses returns nothing *silently*, which in an
# evaluation script means a condition quietly missing from the comparison rather
# than an error. Pinned by
# ``test_web_generate.py::test_no_downstream_tool_spells_a_compensation_filename_itself``;
# the *values* are pinned by
# ``test_the_filename_prefixes_are_the_names_downstream_tools_glob``.
#
# 投影画像のファイル名接頭辞。書き出し側が CLI と Web の2つあるため、
# 片方に literal を置くと2つの辞書が食い違いうる。値は下流のツールが
# glob しているのでテストで固定してある。
LINEARIZER_FILENAME_PREFIXES: dict[str, str] = {
    "none": "no_gamma",
    "gamma": "inv_gamma",
    "inverse_lut": "inv_lut",
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
        dtype: Output data type of both returned arrays. For integer
            dtypes (``np.uint8`` default) the pipeline computes matmul,
            mask and linearizer entirely in float32 and quantizes each
            returned array exactly once at the end with
            ``floor(x * scale + 0.5)`` — the linearizer never sees
            quantized input, so dark projector codes stay reachable. A
            float dtype keeps the compensation in normalized units, which
            is what a viewer needs.
            返す2配列の出力データ型。整数 dtype(既定 ``np.uint8``)では
            行列積・マスク・線形化まで float32 のまま計算し、最後に
            ``floor(x * scale + 0.5)`` で 1 回だけ量子化する(線形化は
            量子化済みの入力を受け取らないため、暗部の投影コードが
            到達可能なまま保たれる)。float を指定すると補償値を正規化の
            まま保持でき、ビューアはこちらを必要とする。
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
            "_apply_inverse_gamma_numpy() and apply_inverse_lut() both "
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

    # The whole chain (matmul -> mask -> linearizer) runs in float; integer
    # output is quantized ONCE at the very end. Quantizing the compensation
    # to 8 bits in linear space *before* the linearizer made projector codes
    # 1-19 unreachable (linear 1/255 maps to code ~21 under 1/2.2), crushing
    # shadow detail. Pinned by tests/test_projection_quantization_order.py.
    out_dtype = np.dtype(dtype)
    integer_output = np.issubdtype(out_dtype, np.integer)
    compute_dtype: np.typing.DTypeLike = np.float32 if integer_output else dtype

    compensation = calc_compensation_image(
        target_image=target_linear,
        color_mixing_matrices=warped_matrix,
        dtype=compute_dtype,
        clip=clip,
    )
    compensation = mask_to_active_region(compensation, active_region)

    projector_image = linearize(
        compensation, projector=projector, lut=lut, dtype=compute_dtype
    )
    # Every entry in LINEARIZERS returns a plain numpy array now that
    # _linearize_gamma calls _apply_inverse_gamma_numpy instead of torch
    # (see that function's docstring); a torch-tensor conversion shim used
    # to live here for the old torch-backed gamma path and has been removed
    # as dead code rather than left to imply a torch tensor could still
    # arrive.
    if integer_output:
        return (
            quantize_unit_image(compensation, out_dtype),
            quantize_unit_image(np.asarray(projector_image), out_dtype),
        )
    return compensation, np.asarray(projector_image, dtype=dtype)
