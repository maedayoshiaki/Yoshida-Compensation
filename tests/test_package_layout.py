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
