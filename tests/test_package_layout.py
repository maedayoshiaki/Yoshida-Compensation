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
