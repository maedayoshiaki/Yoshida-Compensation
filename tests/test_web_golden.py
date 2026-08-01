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
which is exactly what the diverging map (``u_colormap == 0``) is for, so of the
two scalar colormaps only that branch is exercised. The shader's other mode --
``u_mode == 1``, the rgb branch ``surface`` renders through -- has its own
mirror and its own goldens further down: it skips the colormap entirely, so no
scalar assertion here can see a regression in it. Dither is excluded
(hash12-based: random, not reproducible in numpy).

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


def _rgb_body(source: str) -> str:
    """Return ``rgbColor()``'s definition, which the shader places after ``main()``."""
    at = source.find("vec3 rgbColor() {")
    assert at != -1, "the shader has no rgbColor() definition"
    return source[at:]


def _scalar_region(source: str) -> str:
    """Return everything before ``rgbColor()``'s definition: the scalar path.

    Assertions about the scalar path must be made against *this*, not against
    the whole source, because the rgb branch repeats the scalar tone chain line
    for line. ``str.index`` returns the first occurrence, so once a second copy
    exists, deleting the first one stops being visible: the index silently
    relocates into ``rgbColor()`` and every ordering assertion still passes
    while ``main()`` no longer applies exposure, gamma or the OETF at all.
    Verified -- deleting those three lines from ``main()`` alone left this
    module green until the searches were scoped.

    スカラー経路の検証はここに対して行う。rgb経路が同じ行を繰り返しているため、
    全体を対象にすると ``str.index`` が2つ目の写しへ滑り、``main()`` から
    3行を消しても検出できなくなる（実際に緑のまま通ることを確認済み）。
    """
    at = source.find("vec3 rgbColor() {")
    assert at != -1, "the shader has no rgbColor() definition"
    return source[:at]


def _shader_rgb_mode(
    values: list[float], vmin: float, vmax: float, exposure: float, gamma: float
) -> np.ndarray:
    """Mirror the fragment shader's ``rgbColor()``, minus dither.

    The rgb branch has no colormap to pick a point on: the three channels are
    one colour already, so the range maps straight onto linear RGB and the same
    exposure -> gamma -> OETF chain runs on it.
    rgb経路にカラーマップは無い。3チャンネルは既に色なので、レンジをそのまま
    線形RGBへ写し、同じ 露出→γ→OETF を掛ける。
    """
    span = max(vmax - vmin, 1e-9)
    rgb = np.clip((np.asarray(values, dtype=np.float64) - vmin) / span, 0.0, 1.0)
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

    Every assertion below is made against the *scalar region* of the shader
    rather than the whole source. The rgb branch repeats the scalar tone chain
    verbatim, so an unscoped ``str.index`` would find ``rgbColor()``'s copy and
    a deletion from ``main()`` would go unnoticed. See ``_scalar_region``.

    numpy同士の比較には意味がないので、ブラウザが実際に読む
    ``static/js/shaders.js`` を読んで、ミラーが写している定数と
    ANGLE/D3D11向けNaN判定の存在を固定する。特に ``isNaNBits`` を消すと、
    有効投影領域外が「それらしい強い負の値」に見える不具合が静かに戻る。
    検証対象はスカラー領域に限定する（``_scalar_region`` 参照）。
    """
    scalar = _scalar_region(_fragment_source())

    # The ANGLE/D3D11 NaN workaround, both declared and actually wired in.
    assert "bool isNaNBits(float x)" in scalar
    assert "highp uint" in scalar
    assert "0x7F800000u" in scalar
    assert "isNaNBits(v)" in scalar, "isNaNBits is defined but not called from main()"

    # Constants the numpy mirror hard-codes.
    assert np.allclose(_vec3(scalar, "cold"), _COLD)
    assert np.allclose(_vec3(scalar, "warm"), _WARM)
    assert f"step(vec3({_SRGB_KNEE})" in scalar
    assert "12.92 * c" in scalar
    assert "1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055" in scalar

    # Exposure before gamma, gamma before the OETF -- the order the mirror
    # applies and the golden triples below were computed with. Scoped, so that
    # deleting any of the three from main() fails here instead of resolving to
    # the rgb branch's identical copy further down the file.
    exposure_at = scalar.index("rgb *= exp2(u_exposure);")
    gamma_at = scalar.index("pow(max(rgb, vec3(0.0)), vec3(1.0 / max(u_gamma, 1e-6)))")
    srgb_at = scalar.index("rgb = srgbEncode(rgb);")
    assert exposure_at < gamma_at < srgb_at


def test_rgb_branch_is_guarded_and_sits_after_the_scalar_one() -> None:
    """The rgb branch must be NaN-guarded, colormap-free, and defined last.

    Three things are pinned here, each for its own reason:

    - **``isNaNBits`` on every component.** The rgb path draws whatever the
      texture holds, so the ANGLE/D3D11 fold-away that made ``isnan()`` useless
      would show masked pixels as a plausible colour -- the same silent
      misreading the scalar mask exists to prevent, and now in a view whose
      output *is* a colour, where nothing looks obviously wrong. A single NaN
      component masks the whole pixel: ``surface`` writes NaN to all three at
      once, so a partial NaN is not a state the backend produces, and painting
      2/3 real data and 1/3 invented would be exactly the plausible-wrong pixel.
    - **No ``u_colormap``.** Running rgb data through the diverging map would
      recolour a colour.
    - **``rgbColor()`` defined *after* ``main()``.** The scalar assertions in
      ``test_numpy_mirror_still_matches_the_shipped_shader`` are scoped to
      everything preceding ``rgbColor()``'s definition, because the rgb branch
      repeats the scalar tone chain line for line and an unscoped ``str.index``
      would happily resolve into the second copy. That scoping is only correct
      while ``main()`` comes first: define ``rgbColor()`` above it and the
      "scalar region" would contain no ``main()`` at all. This assertion is what
      turns that into an immediate, named failure instead of a confusing one in
      another test -- and it is why the shader carries a forward declaration.

    rgb経路の3成分すべてに ``isNaNBits`` が掛かること、カラーマップを通さない
    こと、そして ``rgbColor()`` が ``main()`` より後ろで定義されることを固定する。
    最後の1つはスカラー経路の順序検証が「最初の出現位置」で行われているため。
    """
    source = _fragment_source()

    assert "uniform int   u_mode;" in source, "the shader has no scalar/rgb mode switch"
    assert "if (u_mode == 1) { fragColor = vec4(rgbColor(), 1.0); return; }" in source

    # Defined after main(), so the scalar chain keeps owning the first occurrence.
    assert source.index("void main()") < source.index("vec3 rgbColor() {")

    body = _rgb_body(source)
    assert "texture(u_tex, v_uv).rgb" in body
    for component in ("v.r", "v.g", "v.b"):
        assert f"isNaNBits({component})" in body, (
            f"the rgb path does not bit-test {component} for NaN"
        )
    assert "u_nanColor" in body, "the rgb path never paints the mask colour"
    assert "u_colormap" not in body, "the rgb path must not go through a colormap"

    # Same chain, same order as the scalar path: exposure, gamma, OETF, dither.
    exposure_at = body.index("rgb *= exp2(u_exposure);")
    gamma_at = body.index("pow(max(rgb, vec3(0.0)), vec3(1.0 / max(u_gamma, 1e-6)))")
    srgb_at = body.index("rgb = srgbEncode(rgb);")
    dither_at = body.index("u_dither / 255.0")
    assert exposure_at < gamma_at < srgb_at < dither_at


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

    # The body is planar (C, H, W); the shader mirror below is unaffected,
    # because the shader only ever sees one channel at a time.
    # 本体は planar (C, H, W)。シェーダは1チャンネルしか見ないのでミラー側は
    # 変わらない。
    data = np.frombuffer(response.content, dtype="<f4").reshape(12, height, width)
    channel0 = data[0, 0, :]  # in_R -> out_R, one value per probe column
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


def _surface_matrices() -> np.ndarray:
    """A = 2I, b = (0.5, 0.25, 0.0) on a 4x5 frame -- exact in float32.

    ``P = (1,1,1)`` gives ``((1-0.5)/2, (1-0.25)/2, (1-0)/2) = (0.25, 0.375,
    0.5)`` and ``P = (0,0,0)`` gives ``(-0.25, -0.125, 0.0)``. Every one of
    those is a dyadic rational, so nothing below depends on float32 rounding.
    すべて2進小数なので float32 の丸めに依存しない。
    """
    matrices = np.zeros((4, 5, 4, 3), dtype=np.float32)
    for channel in range(3):
        matrices[:, :, channel, channel] = 2.0
    matrices[:, :, 3, :] = np.array([0.5, 0.25, 0.0], dtype=np.float32)
    return matrices


def test_surface_rgb_golden_linear_range_exposure_gamma_and_srgb(
    tmp_path, stand_in_projector_config
) -> None:
    """Backend rgb float + headers, through the rgb mirror, match hand math.

    The scalar golden above cannot cover this: the rgb branch skips the
    colormap entirely, so its chain is range -> exposure -> gamma -> OETF with
    nothing in front, and a regression that fed rgb data through
    ``diverging()`` would leave every scalar assertion green.

    Hand-computed from the response ``(0.25, 0.375, 0.5)`` with the headers'
    own ``vmin = 0`` / ``vmax = 0.5``, at -1 EV and gamma 2.2:

        t   = v / 0.5              = (0.5,   0.75,   1.0)
        exp = t * 2**-1            = (0.25,  0.375,  0.5)
        gam = exp**(1/2.2)         = (0.5334…, 0.6402…, 0.7309…)
        out = srgbEncode(gam)      = the literals below

    レンジ→露出→γ→sRGB OETF の順で手計算した値と一致することを確認する。
    """
    stand_in_projector_config(width=5, height=4)
    path = tmp_path / "surface_golden.npy"
    np.save(path, _surface_matrices())

    store.clear()
    client = TestClient(create_app())
    matrix_id = client.post("/api/session/matrix", json={"path": str(path)}).json()["id"]

    response = client.get(f"/api/matrix/{matrix_id}/view/surface")
    assert response.status_code == 200
    assert response.headers["X-Kind"] == "rgb"
    assert response.headers["X-Layout"] == "planar"

    vmin = float(response.headers["X-Vmin"])
    vmax = float(response.headers["X-Vmax"])
    # Pin the inputs to the hand calculation before using them, exactly as the
    # scalar golden does: a drifted range would make every triple below wrong
    # rather than make this test fail.
    assert vmin == 0.0
    assert vmax == pytest.approx(0.5)

    planar = np.frombuffer(response.content, dtype="<f4").reshape(3, 4, 5)
    pixel = [float(planar[c, 1, 2]) for c in range(3)]
    assert pixel == pytest.approx([0.25, 0.375, 0.5])

    # Independent literals: computed from the arithmetic in the docstring, not
    # by calling _shader_rgb_mode.
    expected = [0.7563830783691802, 0.8211461630453089, 0.8702076240928223]
    actual = _shader_rgb_mode(pixel, vmin, vmax, exposure=-1.0, gamma=2.2)
    assert actual == pytest.approx(expected, abs=1e-6)


def test_surface_rgb_golden_ambient_only_response_is_visible_not_black(
    tmp_path, stand_in_projector_config
) -> None:
    """``P = (0,0,0)`` is all-negative and must still map to distinguishable pixels.

    This is the end-to-end consequence of ``surface``'s ``vmin`` choice: the
    range floor drops to the data's 1st percentile when that is negative, so
    ``(-0.25, -0.125, 0.0)`` spreads across the whole tone chain instead of
    clipping to one flat black frame. Pinned with hand-computed literals so a
    silent return to a fixed ``vmin = 0`` -- which renders all three as 0.0 --
    fails here rather than being noticed by eye, or not at all.

    ``vmin`` を負まで下げる決定の帰結をここで固定する。0固定に戻すと3成分とも
    0.0 になるため、目視ではなくテストで気付ける。
    """
    stand_in_projector_config(width=5, height=4)
    path = tmp_path / "surface_ambient.npy"
    np.save(path, _surface_matrices())

    store.clear()
    client = TestClient(create_app())
    matrix_id = client.post("/api/session/matrix", json={"path": str(path)}).json()["id"]

    response = client.get(
        f"/api/matrix/{matrix_id}/view/surface?p_r=0&p_g=0&p_b=0"
    )
    assert response.status_code == 200
    vmin = float(response.headers["X-Vmin"])
    vmax = float(response.headers["X-Vmax"])
    assert vmin == pytest.approx(-0.25)
    assert vmax == pytest.approx(1e-6)

    planar = np.frombuffer(response.content, dtype="<f4").reshape(3, 4, 5)
    pixel = [float(planar[c, 1, 2]) for c in range(3)]
    assert pixel == pytest.approx([-0.25, -0.125, 0.0])

    # At 0 EV and gamma 1 the chain is just the range map plus the OETF:
    #   t = (v + 0.25) / 0.250001 = (0.0, 0.49999…, 0.99999…)
    expected = [0.0, 0.7353556657945433, 0.9999982416716485]
    actual = _shader_rgb_mode(pixel, vmin, vmax, exposure=0.0, gamma=1.0)
    assert actual == pytest.approx(expected, abs=1e-6)

    # The point of the whole exercise: three separable pixels, not three zeros.
    assert len(set(np.round(actual, 6))) == 3
