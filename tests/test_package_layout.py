"""Package layout guards for the yoshidacomp core distribution."""

from __future__ import annotations


def test_core_package_is_importable() -> None:
    import yoshidacomp

    assert yoshidacomp.__version__ == "0.1.0"


def test_config_lives_in_core() -> None:
    from pathlib import Path

    from yoshidacomp.config import (
        AppConfig,
        get_config,
        reload_config,
        split_cli_config_path,
    )

    assert AppConfig().projector.gamma == 2.2
    assert callable(get_config)
    assert callable(reload_config)
    assert split_cli_config_path(["prog", "--config", "x.toml"]) == (
        ["prog"],
        Path("x.toml"),
    )


def test_config_still_reads_the_repository_config_toml() -> None:
    """_PROJECT_ROOT の階層修正が正しいことの確認。

    誤ると config.toml を見つけられず全既定値へ静かにフォールバックする。
    """
    from yoshidacomp.config import reload_config

    cfg = reload_config()
    assert cfg.projector.active_width == 722
    assert cfg.projector.active_height == 722
    assert cfg.compensation.num_divisions == 4


def test_projection_region_lives_in_core() -> None:
    import numpy as np
    from yoshidacomp.compensation.projection_region import (
        embed_in_active_region,
        mask_to_active_region,
    )

    image = np.ones((4, 4, 3), dtype=np.uint8)
    masked = mask_to_active_region(image, (1, 1, 2, 2))
    assert masked[0, 0].tolist() == [0, 0, 0]
    assert masked[1, 1].tolist() == [1, 1, 1]

    embedded = embed_in_active_region(image, (8, 6), (2, 1, 4, 4))
    assert embedded.shape == (6, 8, 3)


def test_compensation_math_lives_in_core() -> None:
    import numpy as np
    from yoshidacomp.compensation.color_mixing_matrix import (
        apply_inverse_gamma_correction,
        generate_projection_patterns,
    )
    from yoshidacomp.compensation.photometric_compensation import calc_compensation_image

    patterns = generate_projection_patterns(2, 2, num_divisions=2, dtype=np.uint8)
    assert len(patterns) == 8
    assert patterns[0].shape == (2, 2, 3)

    # gamma=1/2.2 で明るくなること（この関数は指数をそのまま pow に使う）
    mid = np.full((1, 1, 3), 64, dtype=np.uint8)
    brightened = apply_inverse_gamma_correction(mid, gamma=1 / 2.2, device="cpu")
    assert int(brightened[0, 0, 0]) > 64

    identity = np.zeros((1, 1, 4, 3), dtype=np.float32)
    identity[0, 0, 0, 0] = 1.0
    identity[0, 0, 1, 1] = 1.0
    identity[0, 0, 2, 2] = 1.0
    target = np.full((1, 1, 3), 0.5, dtype=np.float32)
    out = calc_compensation_image(target, identity, dtype=np.float32)
    assert np.allclose(out, 0.5, atol=1e-5)


def test_calc_compensation_image_float_dtype_is_not_binarized() -> None:
    """Regression test: calc_compensation_image used to apply floor(x + 0.5)
    (integer-style rounding) unconditionally, even for float dtype output,
    silently collapsing any value into {0.0, 1.0}. Float output must keep
    its actual value; uint8 output must still scale to 0-255 and round
    half-up.
    """
    import numpy as np
    from yoshidacomp.compensation.photometric_compensation import calc_compensation_image

    identity = np.zeros((1, 1, 4, 3), dtype=np.float32)
    identity[0, 0, 0, 0] = 1.0
    identity[0, 0, 1, 1] = 1.0
    identity[0, 0, 2, 2] = 1.0

    float_target = np.full((1, 1, 3), 0.25, dtype=np.float32)
    float_out = calc_compensation_image(float_target, identity, dtype=np.float32)
    assert np.allclose(float_out, 0.25, atol=1e-5)

    uint8_target = np.full((1, 1, 3), 0.5, dtype=np.float32)
    uint8_out = calc_compensation_image(uint8_target, identity, dtype=np.uint8)
    assert uint8_out.dtype == np.uint8
    assert int(uint8_out[0, 0, 0]) == 128


def test_capture_package_exposes_backend_factory() -> None:
    """`yoshidacomp.capture` が import でき、バックエンド生成口を公開すること。

    SDK非搭載環境でも import できることの保証は Task 10 の
    `tests/test_package_boundaries.py::test_core_imports_without_camera_sdk`
    が担う（このvenvには rawpy と edsdk が実在するため、ここでは検証できない）。
    """
    import yoshidacomp.capture as capture

    assert hasattr(capture, "create_camera_backend")


def test_inverse_lut_lives_in_core(tmp_path) -> None:
    import numpy as np
    from yoshidacomp.compensation.inverse_lut import (
        InverseLut,
        apply_inverse_lut,
        load_inverse_lut,
        save_inverse_lut,
    )

    t_grid = np.linspace(0.0, 1.0, 256).astype(np.float32)
    lut = InverseLut(
        lut=np.stack([t_grid, t_grid, t_grid]).astype(np.float32),
        t_grid=t_grid,
        knots=[],
        meta={"source": "channel_ramp"},
    )
    path = save_inverse_lut(tmp_path / "identity.npz", lut)
    reloaded = load_inverse_lut(path)
    assert reloaded.lut.shape == (3, 256)

    # identity LUT なので入力がそのまま返る
    image = np.full((1, 1, 3), 0.25, dtype=np.float32)
    out = apply_inverse_lut(image, reloaded, dtype=np.float32)
    assert np.allclose(out, 0.25, atol=1e-3)
