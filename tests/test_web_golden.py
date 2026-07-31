"""Golden test: backend float32 output through a numpy shader mirror.

`/api/matrix/{id}/view/coef_tiles` returns raw ``float32`` coefficients plus
a symmetric ``vmin``/``vmax`` in ``X-Vmin``/``X-Vmax`` headers. The browser
feeds those two numbers straight into GL uniforms and the fragment shader in
``packages/web/yoshidacomp_web/static/js/shaders.js`` does the rest: pick a
point on the diverging colormap, apply exposure and gamma (in that order),
then the sRGB OETF, then dither. This test reimplements everything except
dither in numpy and checks the result against RGB triples computed by hand,
pixel by pixel, through ``TestClient`` alone -- no browser involved.

A numpy mirror of a shader is only worth anything while the shader still
agrees with it, and it cannot notice on its own when it stops: every constant
below (the ``cold``/``warm`` tips, the ``0.0031308`` sRGB knee, the order of
exposure and gamma) is a *copy*. So the copies are pinned against the shipped
file: ``test_numpy_mirror_still_matches_the_shipped_shader`` reads
``static/js/shaders.js`` and fails if any of them, or the ANGLE/D3D11 NaN
workaround the mask depends on (``isNaNBits``), is edited or deleted.

That last one is why this matters concretely. ``isnan(v)`` and ``v != v`` were
both folded away by ANGLE/D3D11 on this machine, so the NaN mask did nothing
and pixels outside the active projection region rendered as a *plausible*
strong negative coefficient rather than as mask colour. Deleting the integer
bit-pattern test that fixed it must fail a test, not go unnoticed.

``coef_tiles`` is a signed, zero-centred coefficient map (``vmin == -vmax``),
which is exactly what the diverging map (``u_colormap == 0``) is for, so
only that branch is exercised. Dither is excluded (hash12-based: random, not
reproducible in numpy).

シェーダ相当の処理をnumpyで再現し、既知の係数から手計算したRGBと画素単位で
一致することを確認する。ミラーは写しであり、本体が変わっても自力では
気付けないため、実ファイル ``static/js/shaders.js`` を読んで、依存している
定数とANGLE/D3D11向けNaN判定（``isNaNBits``）の存在を固定する。
ディザ（hash12依存の乱数）は対象外。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store  # noqa: E402

# A stand-in projector frame that the 1x4 matrices below cannot match, so they
# classify as "camera" space, active_region stays None and no NaN masking runs
# here -- that branch has its own dedicated test in test_web_views.py. Taking
# this from the live config.toml instead would make the golden values depend on
# an experiment file that is edited constantly.
# 下の1x4行列と一致しない代用プロジェクタ枠。"camera" 空間に分類され
# active_region は None のままなのでNaNマスクは走らない（専用テストは
# test_web_views.py 側）。実 config.toml に依存させない。
_STAND_IN_FRAME = {"width": 64, "height": 48}

_SHADERS_JS = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "web"
    / "yoshidacomp_web"
    / "static"
    / "js"
    / "shaders.js"
)

_COLD = np.array([0.23, 0.30, 0.75])
_WARM = np.array([0.71, 0.02, 0.15])
_WHITE = np.array([1.0, 1.0, 1.0])
_SRGB_KNEE = 0.0031308


def _fragment_source() -> str:
    """Return the ``FRAG_SRC`` template literal out of ``static/js/shaders.js``."""
    text = _SHADERS_JS.read_text(encoding="utf-8")
    match = re.search(r"export const FRAG_SRC = `(.*?)`;", text, re.DOTALL)
    assert match is not None, f"FRAG_SRC template literal not found in {_SHADERS_JS}"
    return match.group(1)


def _vec3(source: str, name: str) -> np.ndarray:
    """Extract ``vec3 <name> = vec3(a, b, c);`` from GLSL source."""
    match = re.search(
        rf"vec3\s+{name}\s*=\s*vec3\(([^)]*)\)\s*;",
        source,
    )
    assert match is not None, f"vec3 {name} not found in the shader"
    return np.array([float(part) for part in match.group(1).split(",")])


def _diverging(t: float) -> np.ndarray:
    """Mirror the GLSL ``diverging()`` for a single scalar ``t``."""
    a = float(np.clip(abs(t), 0.0, 1.0))
    tip = _COLD if t < 0.0 else _WARM
    return _WHITE * (1.0 - a) + tip * a


def _srgb_encode(c: np.ndarray) -> np.ndarray:
    """Mirror the GLSL ``srgbEncode()`` (vectorised over channels)."""
    c = np.clip(c, 0.0, 1.0)
    linear_branch = 12.92 * c
    power_branch = 1.055 * np.power(c, 1.0 / 2.4) - 0.055
    return np.where(c >= _SRGB_KNEE, power_branch, linear_branch)


def _shader_rgb(value: float, vmin: float, vmax: float, exposure: float, gamma: float) -> np.ndarray:
    """Mirror the fragment shader's ``main()``, minus dither, diverging map only."""
    span = max(vmax - vmin, 1e-9)
    t = float(np.clip((2.0 * (value - vmin) / span) - 1.0, -1.0, 1.0))
    rgb = _diverging(t)
    rgb = rgb * (2.0**exposure)
    rgb = np.power(np.maximum(rgb, 0.0), 1.0 / max(gamma, 1e-6))
    return _srgb_encode(rgb)


def test_numpy_mirror_still_matches_the_shipped_shader() -> None:
    """Everything the mirror copies must still be in ``static/js/shaders.js``.

    The mirror above is numpy, and comparing numpy to numpy proves nothing:
    if the shader changes and the mirror does not, the golden test below keeps
    passing while the browser draws something else. This test is the link
    between them -- it reads the file the browser actually loads.

    Pinned here, and why each one:

    - ``isNaNBits`` and its ``highp uint`` / ``0x7F800000u`` bit test: the
      ANGLE/D3D11 workaround for the NaN mask. Measured on this machine
      (Chrome 150 / RTX 3060 Ti / D3D11), both ``isnan(v)`` and ``v != v``
      were folded away, so the mask did nothing and pixels outside the active
      projection region rendered as a *plausible* saturated negative
      coefficient. Removing the workaround restores a wrong-but-believable
      picture, which is the worst kind of regression, so it must fail loudly.
      ``highp`` is part of it: ESSL 3.00 defaults int/uint to ``mediump``, and
      a 16-bit ``mediump`` truncates the mask so every NaN reads as finite.
    - ``cold`` / ``warm`` and the ``0.0031308`` knee: the mirror hard-codes
      these numbers, so a change on either side must be a change on both.
    - The exposure-then-gamma order: swapping the two lines changes every
      expected RGB triple in the golden test, and nothing else would notice.

    numpy同士の比較には意味がないので、ブラウザが実際に読む
    ``static/js/shaders.js`` を読んで、ミラーが写している定数と
    ANGLE/D3D11向けNaN判定の存在を固定する。特に ``isNaNBits`` を消すと、
    有効投影領域外が「それらしい強い負の値」に見える不具合が静かに戻る。
    """
    source = _fragment_source()

    # The ANGLE/D3D11 NaN workaround, both declared and actually wired in.
    assert "bool isNaNBits(float x)" in source
    assert "highp uint" in source
    assert "0x7F800000u" in source
    assert "isNaNBits(v)" in source, "isNaNBits is defined but not called from main()"

    # Constants the numpy mirror hard-codes.
    assert np.allclose(_vec3(source, "cold"), _COLD)
    assert np.allclose(_vec3(source, "warm"), _WARM)
    assert f"step(vec3({_SRGB_KNEE})" in source
    assert "12.92 * c" in source
    assert "1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055" in source

    # Exposure before gamma, gamma before the OETF -- the order the mirror
    # applies and the golden triples below were computed with.
    exposure_at = source.index("rgb *= exp2(u_exposure);")
    gamma_at = source.index("pow(max(rgb, vec3(0.0)), vec3(1.0 / max(u_gamma, 1e-6)))")
    srgb_at = source.index("rgb = srgbEncode(rgb);")
    assert exposure_at < gamma_at < srgb_at


def test_srgb_oetf_matches_hand_computed_values_on_both_branches() -> None:
    """The sRGB OETF is piecewise; exercise both sides of the 0.0031308 knee.

    Hand-computed (not derived from the same code under test):
    - c=0.001 is below the knee: encoded = 12.92 * 0.001 = 0.01292 exactly.
    - c=0.5 is above the knee: encoded = 1.055 * 0.5**(1/2.4) - 0.055.
    """
    below_knee = _srgb_encode(np.array([0.001]))
    assert below_knee == pytest.approx([0.01292], abs=1e-9)

    above_knee = _srgb_encode(np.array([0.5]))
    assert above_knee == pytest.approx([0.7353569830524495], abs=1e-9)


def test_coef_tiles_golden_diverging_colormap_exposure_gamma_and_srgb(
    tmp_path, stand_in_projector_config
) -> None:
    """Backend float + headers, through the numpy shader mirror, match hand math.

    Layout: a 1x4 matrix (camera space, no active-region mask). Channel 0
    (``in_R -> out_R``) carries four probe values chosen so the 99th
    percentile used for ``vmax`` lands exactly on 2.0 (45 of 48 entries sit
    at +/-2.0: 44 untouched +2.0 entries plus the one column driven to
    -2.0), giving clean ``t`` values without any percentile-interpolation
    arithmetic to work through by hand:

    - column 0: -2.0  -> t = -1   (cold end, full saturation)
    - column 1:  0.0  -> t =  0   (exact white point)
    - column 2:  1.0  -> t =  0.5 (warm side, midway mix)
    - column 3: -1.0  -> t = -0.5 (cold side, midway mix)

    Every other entry (44 of 48) is left at +2.0 so ``vmax`` -- and hence
    ``vmin = -vmax`` -- comes out to exactly 2.0 via the 99th-percentile
    calculation in ``coef_tiles._render``, which this test also verifies via
    the response headers before using them. Exposure and gamma (-1.0 EV,
    2.2) are applied in that order -- the order the plan specifies and the
    shader's ``main()`` implements -- so a regression that swapped them
    would change every expected triple below.

    以下の4値それぞれについて発散カラーマップ→露出→γ→sRGB OETFの順で
    手計算したRGBと一致することを確認する。ディザは対象外。
    """
    stand_in_projector_config(**_STAND_IN_FRAME)
    height, width = 1, 4
    matrices = np.full((height, width, 4, 3), 2.0, dtype=np.float32)
    matrices[0, 0, 0, 0] = -2.0  # t = -1
    matrices[0, 1, 0, 0] = 0.0  # t = 0
    matrices[0, 2, 0, 0] = 1.0  # t = 0.5
    matrices[0, 3, 0, 0] = -1.0  # t = -0.5
    path = tmp_path / "golden.npy"
    np.save(path, matrices)

    store.clear()
    client = TestClient(create_app())
    matrix_id = client.post("/api/session/matrix", json={"path": str(path)}).json()["id"]

    response = client.get(f"/api/matrix/{matrix_id}/view/coef_tiles")
    assert response.status_code == 200

    vmin = float(response.headers["X-Vmin"])
    vmax = float(response.headers["X-Vmax"])
    # Sanity check on the inputs to the "hand calculation" below: if this
    # drifts from 2.0/-2.0, every expected triple further down would be
    # computed against the wrong span and the test would be silently wrong
    # instead of failing loudly, so pin it down explicitly first.
    assert vmax == pytest.approx(2.0)
    assert vmin == pytest.approx(-2.0)

    data = np.frombuffer(response.content, dtype="<f4").reshape(height, width, 12)
    channel0 = data[0, :, 0]  # in_R -> out_R, one value per probe column
    assert channel0.tolist() == pytest.approx([-2.0, 0.0, 1.0, -1.0])

    exposure, gamma = -1.0, 2.2

    # Hand-computed expected sRGB-encoded RGB per column (see module
    # docstring and this test's docstring for the derivation: diverging(t)
    # -> *2**exposure -> **(1/gamma) -> srgbEncode). Values below are
    # independent literals, not produced by calling _shader_rgb.
    expected = {
        0: [0.6454132018171781, 0.6815617084455862, 0.8211461630453089],  # t=-1 (cold)
        1: [0.8702076240928223, 0.8702076240928223, 0.8702076240928223],  # t=0 (white)
        2: [0.8431605949845185, 0.7594318823446173, 0.7781472194141827],  # t=0.5 (warm mix)
        3: [0.7888270231765544, 0.7977193663618051, 0.847102485883956],  # t=-0.5 (cold mix)
    }

    for column, expected_rgb in expected.items():
        actual_rgb = _shader_rgb(float(channel0[column]), vmin, vmax, exposure, gamma)
        assert actual_rgb == pytest.approx(expected_rgb, abs=1e-6), f"column {column}"
