"""Package layout guards for the yoshidacomp core distribution."""

from __future__ import annotations


def test_core_package_is_importable() -> None:
    import yoshidacomp

    assert yoshidacomp.__version__ == "0.1.0"
