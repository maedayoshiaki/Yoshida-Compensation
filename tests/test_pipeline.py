"""Hardware-free generation pipeline tests."""

from __future__ import annotations

import numpy as np
import pytest


def test_srgb_roundtrip() -> None:
    from yoshidacomp.compensation.pipeline import linear_to_srgb, srgb_to_linear

    values = np.array([0.0, 0.02, 0.5, 1.0], dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(values)), values, atol=1e-5)


def test_decode_target_modes() -> None:
    from yoshidacomp.compensation.pipeline import decode_target

    codes = np.full((1, 1, 3), 128, dtype=np.uint8)
    assert np.allclose(decode_target(codes, mode="none"), 128 / 255.0, atol=1e-6)
    assert decode_target(codes, mode="srgb")[0, 0, 0] < 128 / 255.0
    assert np.allclose(
        decode_target(codes, mode="gamma", gamma=2.0), (128 / 255.0) ** 2.0, atol=1e-6
    )
    with pytest.raises(ValueError):
        decode_target(codes, mode="gamma", gamma=0.0)


def test_linearizers_registered() -> None:
    from yoshidacomp.compensation.pipeline import LINEARIZERS

    assert set(LINEARIZERS) == {"none", "gamma", "inverse_lut"}


def test_generate_projection_image_identity_matrix() -> None:
    """恒等行列 + linearizer=none なら、投影画像は線形化したターゲットに一致する。"""
    from dataclasses import dataclass

    from yoshidacomp.compensation.pipeline import generate_projection_image

    @dataclass(frozen=True)
    class Projector:
        width: int = 4
        height: int = 4
        gamma: float = 2.2

    matrices = np.zeros((4, 4, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 1.0

    target = np.full((4, 4, 3), 255, dtype=np.uint8)
    linear, projected = generate_projection_image(
        target,
        matrices,
        projector=Projector(),
        active_region=(0, 0, 4, 4),
        linearizer="none",
        target_decode="srgb",
    )
    assert linear.shape == (4, 4, 3)
    assert projected.shape == (4, 4, 3)
    # 白のsRGBは線形1.0 → 恒等行列 → 255
    assert linear[0, 0].tolist() == [255, 255, 255]

    # linearizer="none" の契約は「補償画像をそのまま投影コード値にする」こと。
    # これを独立に計算した期待値と突き合わせる。
    #
    # NOTE: `assert np.array_equal(linear, projected)` では検証にならない。
    # 整数 dtype では同じ float 補償値から同じ量子化で2配列を作るため
    # 常に等しく、float dtype では _linearize_none() が受け取ったオブジェクト
    # をそのまま返すため `linear is projected` が真になる(2026-08-06 の
    # 量子化順序修正より前は整数 dtype でも同一オブジェクトだった)。
    # いずれにせよ相互比較は決して失敗しないので、独立に計算した期待値と
    # 比較する。
    #
    # NOTE: comparing the two return values against each other cannot fail --
    # for integer dtypes they are quantized from the same float array, and
    # for float dtypes they are the same object -- so the expectation is
    # computed independently instead.
    from yoshidacomp.compensation.pipeline import decode_target
    from yoshidacomp.compensation.photometric_compensation import (
        calc_compensation_image,
    )

    expected = calc_compensation_image(
        target_image=decode_target(target, mode="srgb"),
        color_mixing_matrices=matrices,
        dtype=np.uint8,
    )
    assert np.array_equal(projected, expected)


def _out_of_gamut_setup() -> tuple[np.ndarray, np.ndarray]:
    """A target/matrix pair whose compensation lands outside [0, 1].

    R は 1 超、G は 0 未満、B は範囲内になるよう bias を仕込む。
    """
    matrices = np.zeros((4, 4, 4, 3), dtype=np.float32)
    for c in range(3):
        matrices[:, :, c, c] = 1.0
    matrices[:, :, 3, 0] = 0.5  # R: 1.0 + 0.5  = 1.5  (1超クリップ)
    matrices[:, :, 3, 1] = -1.5  # G: 1.0 - 1.5 = -0.5 (負クリップ)
    target = np.full((4, 4, 3), 255, dtype=np.uint8)
    return target, matrices


def test_generate_projection_image_unclipped_float_keeps_out_of_gamut_values() -> None:
    """``clip=False`` + float dtype で [0, 1] 外の値がそのまま残ること。

    The ``gamut`` view has to tell 負クリップ from 1超クリップ, which is
    exactly the distinction ``torch.clamp(0, 1)`` destroys.
    """
    from dataclasses import dataclass

    from yoshidacomp.compensation.pipeline import generate_projection_image

    @dataclass(frozen=True)
    class Projector:
        width: int = 4
        height: int = 4
        gamma: float = 2.2

    target, matrices = _out_of_gamut_setup()
    linear, projected = generate_projection_image(
        target,
        matrices,
        projector=Projector(),
        active_region=(0, 0, 4, 4),
        linearizer="none",
        target_decode="none",
        dtype=np.float32,
        clip=False,
    )

    assert linear.dtype == np.float32
    assert projected.dtype == np.float32
    assert np.allclose(linear[0, 0], [1.5, -0.5, 1.0], atol=1e-5)
    assert np.allclose(projected[0, 0], [1.5, -0.5, 1.0], atol=1e-5)


def test_generate_projection_image_defaults_are_unchanged() -> None:
    """既定 (dtype=uint8, clip=True) は従来どおりクランプ済み uint8 を返すこと。"""
    from dataclasses import dataclass

    from yoshidacomp.compensation.pipeline import generate_projection_image

    @dataclass(frozen=True)
    class Projector:
        width: int = 4
        height: int = 4
        gamma: float = 2.2

    target, matrices = _out_of_gamut_setup()
    kwargs = dict(
        projector=Projector(),
        active_region=(0, 0, 4, 4),
        linearizer="none",
        target_decode="none",
    )

    linear, projected = generate_projection_image(target, matrices, **kwargs)
    assert linear.dtype == np.uint8
    assert projected.dtype == np.uint8
    # 1.5 -> クランプ後 1.0 -> 255、-0.5 -> クランプ後 0.0 -> 0
    assert linear[0, 0].tolist() == [255, 0, 255]

    explicit_linear, explicit_projected = generate_projection_image(
        target, matrices, dtype=np.uint8, clip=True, **kwargs
    )
    assert np.array_equal(linear, explicit_linear)
    assert np.array_equal(projected, explicit_projected)


@pytest.mark.parametrize("linearizer", ["gamma", "inverse_lut"])
def test_generate_projection_image_rejects_unclipped_linearization(
    linearizer: str,
) -> None:
    """``clip=False`` は ``linearizer='none'`` 以外を拒否すること。

    Both other linearizers assume [0, 1] input, so feeding them the
    out-of-range values ``clip=False`` exists to preserve is meaningless.
    """
    from dataclasses import dataclass

    from yoshidacomp.compensation.pipeline import generate_projection_image

    @dataclass(frozen=True)
    class Projector:
        width: int = 4
        height: int = 4
        gamma: float = 2.2

    target, matrices = _out_of_gamut_setup()
    with pytest.raises(ValueError, match="clip=False requires linearizer='none'"):
        generate_projection_image(
            target,
            matrices,
            projector=Projector(),
            active_region=(0, 0, 4, 4),
            linearizer=linearizer,
            target_decode="none",
            dtype=np.float32,
            clip=False,
        )


def test_generate_projection_image_rejects_unclipped_integer_dtype() -> None:
    """``clip=False`` は整数 dtype を拒否すること。

    Integer output scales by 255 and casts, which is undefined for the
    out-of-range values ``clip=False`` exists to preserve -- silently
    producing wrapped garbage instead of the raw gamut information.
    """
    from dataclasses import dataclass

    from yoshidacomp.compensation.pipeline import generate_projection_image

    @dataclass(frozen=True)
    class Projector:
        width: int = 4
        height: int = 4
        gamma: float = 2.2

    target, matrices = _out_of_gamut_setup()
    with pytest.raises(ValueError, match="clip=False requires a float dtype"):
        generate_projection_image(
            target,
            matrices,
            projector=Projector(),
            active_region=(0, 0, 4, 4),
            linearizer="none",
            target_decode="none",
            clip=False,
        )


def test_gamma_linearizer_brightens_midtones() -> None:
    """逆ガンマは中間調を明るくする（指数は 1/gamma でなければならない）。

    apply_inverse_gamma_correction は与えた指数をそのまま pow に使うため、
    誤って projector.gamma を渡すと逆に暗くなる。
    """
    from dataclasses import dataclass

    import numpy as np

    from yoshidacomp.compensation.pipeline import LINEARIZERS

    @dataclass(frozen=True)
    class Projector:
        gamma: float = 2.2

    mid = np.full((1, 1, 3), 64, dtype=np.uint8)
    brightened = LINEARIZERS["gamma"](mid, projector=Projector(), lut=None)
    passed_through = LINEARIZERS["none"](mid, projector=Projector(), lut=None)

    assert int(brightened[0, 0, 0]) > int(passed_through[0, 0, 0])


def test_inverse_lut_linearizer_requires_a_lut() -> None:
    import numpy as np
    import pytest

    from yoshidacomp.compensation.pipeline import LINEARIZERS

    image = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        LINEARIZERS["inverse_lut"](image, projector=None, lut=None)
