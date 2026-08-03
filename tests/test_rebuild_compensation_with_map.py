"""Tests for target transfer decoding used by the offline compensation rebuild.

``_decode_target`` moved out of ``examples/python/rebuild_compensation_with_map.py``
into ``yoshidacomp.compensation.pipeline.decode_target``; the CLI now calls it
through :func:`~yoshidacomp.compensation.pipeline.generate_projection_image`.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

# Import smoke for the CLI itself. Once ``_decode_target`` moved into
# yoshidacomp.compensation.pipeline (see module docstring above), this file
# stopped referencing rebuild_compensation_with_map.py entirely, so its
# module-level code -- including the sys.path inserts for the repo root and
# external/GrayCode/src, and its yoshidacomp/GrayCode imports -- had zero
# test coverage. Importing it here, even unused, re-exercises that code at
# collection time so a broken import in the CLI fails the suite. Do not
# remove as "unused".
#
# No longer unused: the two ``_save_rgb`` tests at the bottom call it. That is
# still the only part of ``main()`` this file can reach -- see their header.
from examples.python import rebuild_compensation_with_map as rebuild
from yoshidacomp.compensation.pipeline import decode_target, srgb_to_linear


def test_decode_target_supports_direct_exact_srgb_and_simple_gamma() -> None:
    target = np.array([[[0, 128, 255]]], dtype=np.uint8)
    normalized = target.astype(np.float32) / 255.0

    direct = decode_target(target, mode="none", gamma=2.2)
    exact = decode_target(target, mode="srgb", gamma=2.2)
    gamma = decode_target(target, mode="gamma", gamma=2.2)

    assert np.allclose(direct, normalized, atol=1e-7)
    assert np.allclose(exact, srgb_to_linear(normalized), atol=1e-7)
    assert np.allclose(gamma, np.power(normalized, 2.2), atol=1e-7)
    assert not np.isclose(exact[0, 0, 1], gamma[0, 0, 1])
    assert gamma[0, 0, 0] == 0.0
    assert gamma[0, 0, 2] == 1.0


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_decode_target_rejects_invalid_gamma(value: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        decode_target(
            np.zeros((1, 1, 3), dtype=np.uint8),
            mode="gamma",
            gamma=value,
        )


# --------------------------------------------------------------------------
# ``_save_rgb``: the one part of ``main()`` that can be exercised on its own.
#
# ``main()`` itself needs a ``p2c_map``, the GrayCode submodule and a torch
# warp over real data, so it has never had a test. ``_save_rgb`` needs none of
# that, and it is the function that writes into the same 99 GB tree that
# ``POST /api/generate`` writes into.
# --------------------------------------------------------------------------


def test_cli_save_rgb_writes_rgb_not_bgr(tmp_path) -> None:
    """R と B を取り違えずに書くこと。

    ``cv2.imwrite`` はチャンネル0を青として書くので、``COLOR_RGB2BGR`` を
    落とすと「赤と青が入れ替わった、完全に正当な PNG」になる。R と B が
    違う値を持つ画像でしか検出できない。
    """
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:, :, 0] = 200  # R
    image[:, :, 1] = 100  # G
    image[:, :, 2] = 10  # B
    path = tmp_path / "out" / "img.png"

    rebuild._save_rgb(path, image)

    written = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert written is not None
    np.testing.assert_array_equal(cv2.cvtColor(written, cv2.COLOR_BGR2RGB), image)
    assert image[0, 0, 0] != image[0, 0, 2], (
        "R equals B in this fixture, so it cannot detect a dropped RGB->BGR "
        "conversion"
    )


def test_cli_save_rgb_raises_when_imwrite_refuses(tmp_path, monkeypatch) -> None:
    """``cv2.imwrite`` が ``False`` を返したら例外にすること。

    戻り値を捨てていたため、書けなかった実行でも ``main()`` は最後まで進み、
    存在しないファイルを記述したメタデータJSONを「成功」として stdout に
    表示していた。``generate.py::_save_rgb`` と同じ形の欠陥が、同じツリーへ
    書く経路として CLI 側に残っていた。
    """
    monkeypatch.setattr(cv2, "imwrite", lambda *args, **kwargs: False)
    path = tmp_path / "out" / "img.png"

    with pytest.raises(OSError, match="could not write"):
        rebuild._save_rgb(path, np.zeros((2, 3, 3), dtype=np.uint8))

    assert not path.exists()
