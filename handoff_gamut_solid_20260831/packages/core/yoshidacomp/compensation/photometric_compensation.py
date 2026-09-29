"""Photometric compensation image calculation module.

Provides functionality to compute compensation images using per-pixel
color mixing matrices.

色補償画像計算モジュール。

ピクセルごとのカラーミキシング行列を用いた補償画像の計算機能を提供する。
"""

import numpy as np
from numpy import ndarray


def calc_compensation_image(
    target_image: ndarray,
    color_mixing_matrices: ndarray,
    dtype: np.typing.DTypeLike = np.float32,
    clip: bool = True,
) -> ndarray:
    """Calculate a compensation image using color mixing matrices.

    Apply photometric compensation by transforming the target image
    through pixel-wise color mixing matrices: for each pixel,
    ``[R, G, B, 1] @ M`` where ``M`` is that pixel's 4x3 matrix.

    カラーミキシング行列を用いて補償画像を計算する。

    ピクセルごとのカラーミキシング行列を通じて目標画像に色補償を適用する:
    各画素で ``[R, G, B, 1] @ M``（``M`` はその画素の 4x3 行列）を計算する。

    This used to run the matmul as a batched ``torch.bmm`` (GPU-accelerated
    when CUDA is available). Measured against that implementation (see
    ``.superpowers/sdd/2026-08-02-target-and-lut/task-1-report.md``,
    Addendum 4, for the full table): this is 12 multiply-adds per pixel
    (~24 MFLOP for a 1920x1080 frame) -- not a workload that needs GPU
    batching -- and torch's own CPU and CUDA backends already disagree
    with each other by up to 1 count (uint8) / ~1e-5 (float32) on real
    data, so there was no single bit-exact "torch answer" to begin with.
    This implementation matches torch-CPU byte-for-byte, exhaustively, on
    every input tested including the real 1920x1080 matrix in
    ``data/linear_pipeline_1080/``; it differs from torch-CUDA only at the
    exact pixels where torch-CUDA already differs from torch-CPU, by the
    same 1 count -- i.e. never worse than torch's own cross-device noise
    floor. **The count of differing pixels is a property of the target
    image, not of the matrix** -- on that same real matrix, three
    different random uint8 targets measured 9, 2, and 3 differing pixels
    out of 6,220,800 respectively (a mismatch occurs only where a float32
    result lands within one ULP of a `.5` quantization boundary, which
    depends on the target). What reproduces across every target tested is
    the *set equality* -- numpy's divergence from torch-CUDA is always
    exactly torch-CUDA's divergence from torch-CPU, same positions, same
    magnitude -- not any particular count. Per-frame numpy
    is slower than torch-CUDA once torch is already loaded and warm
    (measured ~0.22s vs ~0.08s for 1920x1080), but avoids torch's own
    ~7.1s first-import cost entirely, which is the actual bottleneck for
    an interactive first use.

    以前はこの行列積を（CUDAが使えればGPUで加速される）バッチ ``torch.bmm``
    で計算していた。その実装との実測比較（詳細は上記レポートの
    Addendum 4）: 1画素あたり12回の積和（1920x1080フレームで約24MFLOP）
    しかなく、GPUバッチ処理を要する規模ではない。加えて torch 自身の
    CPU/CUDAバックエンドは実データで既に最大1カウント（uint8）・約1e-5
    （float32）ずれており、そもそも単一のビット精度な「torchの答え」は
    存在しなかった。この実装は torch-CPU と、``data/linear_pipeline_1080/``
    の実データを含むあらゆる検証入力でバイト単位・全数一致する。
    torch-CUDA との相違は、torch-CUDA が torch-CPU と既にずれている
    画素に限られ、そのずれ幅も同じ1カウントを超えない（＝torch自身の
    機種間ノイズ床を超えない）。**相違する画素数は行列ではなくターゲット
    画像に依存する** -- 同じ実データの行列に対し、3種の異なるランダム
    uint8ターゲットで6,220,800画素中それぞれ9・2・3画素が相違した
    （相違は float32 の結果が `.5` 量子化境界の1ULP以内に落ちた場合のみ
    起こり、これはターゲット次第）。どのターゲットでも再現するのは
    *集合としての一致* -- numpyのtorch-CUDAからの相違は常に
    torch-CUDAのtorch-CPUからの相違と完全に同じ位置・同じ大きさになる
    ことであり、特定の画素数ではない。
    1フレームあたりの実行速度は、torch が既にロード・ウォームアップ済みの
    場合は numpy の方が torch-CUDA より遅い（1920x1080で実測約0.22秒 対
    約0.08秒）が、torch自身の初回import コスト（実測約7.1秒）を完全に
    回避できる。これは対話的な初回利用における実際のボトルネックである。

    Args:
        target_image: Input target image with shape ``(H, W, 3)``.
            Supports uint8 (0-255), uint16 (0-65535), or float (0.0-1.0)
            formats.
            形状 ``(H, W, 3)`` の入力目標画像。uint8 (0-255)、
            uint16 (0-65535)、または float (0.0-1.0) 形式に対応。
        color_mixing_matrices: Color mixing matrices with shape
            ``(H, W, 4, 3)``. Each pixel has a 4x3 matrix that transforms
            ``[R, G, B, 1]`` to ``[R', G', B']``.
            形状 ``(H, W, 4, 3)`` のカラーミキシング行列。各ピクセルは
            ``[R, G, B, 1]`` を ``[R', G', B']`` に変換する 4x3 行列を持つ。
        dtype: Output data type. Defaults to ``np.float32``.
            If uint8, output is scaled to 0-255.
            If uint16, output is scaled to 0-65535.
            出力データ型。デフォルトは ``np.float32``。
            uint8 の場合は 0-255、uint16 の場合は 0-65535 にスケーリング。
        clip: Clamp the result to [0, 1] before scaling. Defaults to
            ``True``. Set ``False`` (with a float *dtype*) to keep the raw
            values a gamut analysis needs -- which pixels the matrix drives
            below 0 or above 1 is exactly the information clamping erases.
            スケーリング前に結果を [0, 1] にクランプするか。デフォルトは
            ``True``。色域解析に必要な生値を保つには（float の *dtype* と
            併せて）``False`` を指定する。どの画素が 0 未満／1 超に振れる
            かは、クランプが消してしまう情報そのものである。

    Returns:
        Compensation image with shape ``(H, W, 3)`` and the specified
        *dtype*. Values are clamped to the valid range [0, 1] before
        scaling unless ``clip=False``.
        形状 ``(H, W, 3)`` で指定された *dtype* の補償画像。
        ``clip=False`` でない限り、スケーリング前に値は有効範囲 [0, 1]
        にクランプされる。

    Raises:
        ValueError: If *target_image* is not shape ``(H, W, 3)``, or
            *color_mixing_matrices* is not shape ``(H, W, 4, 3)`` with the
            same ``(H, W)`` as *target_image*. The torch implementation this
            replaced validated shape implicitly (a mismatch raised inside
            ``.view()``/``torch.bmm``); the numpy broadcast below does not
            fail on its own for every mismatch (e.g. a degenerate ``(1, 1,
            4, 3)`` matrix silently broadcasts over any target), so this
            check restores that guarantee explicitly.
            *target_image* が ``(H, W, 3)`` でない場合、または
            *color_mixing_matrices* が *target_image* と同じ ``(H, W)`` の
            ``(H, W, 4, 3)`` でない場合。これが置き換えたtorch実装は
            形状不一致を暗黙に検証していた（``.view()``/``torch.bmm`` 内で
            例外になる）。下の numpy ブロードキャストは全ての不一致で
            自然に失敗するわけではない（例えば縮退した ``(1, 1, 4, 3)``
            行列は任意のターゲットへ黙って broadcast される）ため、この
            チェックで明示的に復元する。
    """
    if target_image.ndim != 3 or target_image.shape[-1] != 3:
        raise ValueError(
            f"target_image must have shape (H, W, 3), got {target_image.shape}."
        )
    if (
        color_mixing_matrices.shape[:2] != target_image.shape[:2]
        or color_mixing_matrices.shape[2:] != (4, 3)
    ):
        raise ValueError(
            "color_mixing_matrices must have shape (H, W, 4, 3) matching "
            f"target_image's (H, W); got {color_mixing_matrices.shape} for "
            f"target_image shape {target_image.shape}."
        )

    # Normalize target image to [0, 1] range
    target_dtype = target_image.dtype
    if target_dtype == np.uint8:
        norm_target_image = target_image.astype(np.float32) / 255.0
    elif target_dtype == np.uint16:
        norm_target_image = target_image.astype(np.float32) / 65535.0
    else:
        norm_target_image = target_image.astype(np.float32)

    matrices = np.asarray(color_mixing_matrices, dtype=np.float32)

    # [R, G, B, 1] @ M, per pixel, without ever materializing the augmented
    # (H, W, 4) array: M's four rows (R-weight, G-weight, B-weight, bias)
    # contracted against the target's three channels plus an implicit 1 for
    # the bias row. Matches torch's element order (and therefore its
    # float32 rounding) exactly -- see the docstring above for the measured
    # agreement this achieves.
    compensation_image = (
        norm_target_image[..., 0:1] * matrices[..., 0, :]
        + norm_target_image[..., 1:2] * matrices[..., 1, :]
        + norm_target_image[..., 2:3] * matrices[..., 2, :]
        + matrices[..., 3, :]
    )

    # Clamp to valid range
    if clip:
        compensation_image = np.clip(compensation_image, 0.0, 1.0)

    # Scale to output dtype range
    if dtype == np.uint8:
        compensation_image = compensation_image * 255.0
        compensation_image = np.floor(compensation_image + 0.5).astype(dtype)
    elif dtype == np.uint16:
        compensation_image = compensation_image * 65535.0
        compensation_image = np.floor(compensation_image + 0.5).astype(dtype)
    else:
        # Float dtypes stay in the already-clamped [0, 1] range; rounding to
        # the nearest integer here would silently binarize the output.
        compensation_image = compensation_image.astype(dtype)

    return compensation_image


def quantize_unit_image(
    image: ndarray,
    dtype: np.typing.DTypeLike,
) -> ndarray:
    """Quantize a [0, 1] float image to *dtype* in one rounding step.

    The single, final quantization of the projection pipeline: linear
    compensation values are kept in float through the linearizer (inverse
    gamma or inverse LUT) and converted to integer code values exactly
    once, here, with ``floor(x * scale + 0.5)`` — the same rounding
    :func:`calc_compensation_image` uses. Quantizing earlier (in linear
    space, before the linearizer) makes dark projector codes unreachable:
    linear 1/255 maps to code ~21 under a 1/2.2 inverse gamma, so codes
    1-19 could never be emitted.

    投影パイプラインの最終量子化。線形補償値は線形化(逆ガンマ・逆LUT)
    まで float のまま保ち、整数コード値への変換はここで 1 回だけ
    ``floor(x * scale + 0.5)`` で行う(:func:`calc_compensation_image`
    と同じ丸め)。線形化より前のリニア空間で量子化すると、リニア 1/255 が
    逆ガンマ 1/2.2 でコード約 21 に写るため、コード 1〜19 が到達不能に
    なり暗部が潰れる。

    Args:
        image: Image with values in [0, 1]. Values outside are clipped
            before scaling (integer *dtype* only).
            値域 [0, 1] の画像。範囲外は(整数 *dtype* のときのみ)
            スケーリング前にクリップされる。
        dtype: ``np.uint8`` scales by 255, ``np.uint16`` by 65535; any
            float dtype is a plain cast with no clip and no rounding.
            ``np.uint8`` は 255 倍、``np.uint16`` は 65535 倍。float は
            クリップも丸めもせず型変換のみ。

    Returns:
        Array of the requested *dtype* with the same shape as *image*.
        *image* と同形状で要求 *dtype* の配列。
    """
    array = np.asarray(image, dtype=np.float32)
    out_dtype = np.dtype(dtype)
    if out_dtype == np.uint8:
        scale = 255.0
    elif out_dtype == np.uint16:
        scale = 65535.0
    else:
        return array.astype(out_dtype)
    return np.floor(np.clip(array, 0.0, 1.0) * scale + 0.5).astype(out_dtype)
