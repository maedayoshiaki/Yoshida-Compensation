"""Golden test: backend float32 output through a numpy shader mirror.

`/api/matrix/{id}/view/coef_tiles` returns raw ``float32`` coefficients plus
a symmetric ``vmin``/``vmax`` in ``X-Vmin``/``X-Vmax`` headers. The browser
feeds those two numbers straight into GL uniforms and the fragment shader in
``packages/web/yoshidacomp_web/static/js/shaders.js`` does the rest: pick a
point on the diverging colormap, apply exposure and gamma (in that order),
then the sRGB OETF, then dither. This test reimplements everything except
dither in numpy and checks the result against RGB triples computed by hand,
pixel by pixel, through ``TestClient`` alone -- no browser involved.

That other file is being written concurrently by a separate task in this
same plan, so the GLSL below is transcribed from
``docs/superpowers/plans/2026-07-31-web-viewer-slice.md`` (Task 4, Step 2)
rather than read from ``static/``, which may be mid-edit:

    vec3 diverging(float t) {          // t in [-1, 1]
        float a = clamp(abs(t), 0.0, 1.0);
        vec3 cold = vec3(0.23, 0.30, 0.75);
        vec3 warm = vec3(0.71, 0.02, 0.15);
        vec3 tip  = (t < 0.0) ? cold : warm;
        return mix(vec3(1.0), tip, a);
    }

    vec3 srgbEncode(vec3 c) {
        c = clamp(c, 0.0, 1.0);
        return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055,
                   step(vec3(0.0031308), c));
    }

    void main() {
        float v = texture(u_tex, v_uv).r;
        if (isnan(v)) { fragColor = vec4(u_nanColor, 1.0); return; }
        float span = max(u_vmax - u_vmin, 1e-9);
        float t = (u_colormap == 0)
            ? clamp((2.0 * (v - u_vmin) / span) - 1.0, -1.0, 1.0)
            : clamp((v - u_vmin) / span, 0.0, 1.0);
        vec3 rgb = (u_colormap == 0) ? diverging(t) : vec3(t);
        rgb *= exp2(u_exposure);
        rgb = pow(max(rgb, vec3(0.0)), vec3(1.0 / max(u_gamma, 1e-6)));
        rgb = srgbEncode(rgb);
        // dither omitted here: hash12-based, not reproducible in numpy.
    }

``coef_tiles`` is a signed, zero-centred coefficient map (``vmin == -vmax``),
which is exactly what the diverging map (``u_colormap == 0``) is for, so
only that branch is exercised. Dither is excluded per the plan (random,
not reproducible).

シェーダ相当の処理をnumpyで再現し、既知の係数から手計算したRGBと画素単位で
一致することを確認する。ディザ（hash12依存の乱数）は対象外。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yoshidacomp_web.app import create_app  # noqa: E402
from yoshidacomp_web.store import store  # noqa: E402

# The real config.toml (1920x1080 projector) is restored before/after every
# test by the autouse fixture in tests/conftest.py. The matrices built below
# are 1x4, which classify as "camera" space (no match to the configured
# projector frame), so active_region stays None and no NaN masking happens
# here -- that branch has its own dedicated test in test_web_views.py.
# 実 config.toml は tests/conftest.py の autouse フィクスチャが前後で復元
# する。本ファイルの行列は1x4で "camera" 空間に分類され（プロジェクタの
# フレームと一致しないため）、active_region は None のままNaNマスクは
# 発生しない。マスクの専用テストは test_web_views.py 側にある。

_COLD = np.array([0.23, 0.30, 0.75])
_WARM = np.array([0.71, 0.02, 0.15])
_WHITE = np.array([1.0, 1.0, 1.0])


def _diverging(t: float) -> np.ndarray:
    """Mirror the GLSL ``diverging()`` above for a single scalar ``t``."""
    a = float(np.clip(abs(t), 0.0, 1.0))
    tip = _COLD if t < 0.0 else _WARM
    return _WHITE * (1.0 - a) + tip * a


def _srgb_encode(c: np.ndarray) -> np.ndarray:
    """Mirror the GLSL ``srgbEncode()`` above (vectorised over channels)."""
    c = np.clip(c, 0.0, 1.0)
    linear_branch = 12.92 * c
    power_branch = 1.055 * np.power(c, 1.0 / 2.4) - 0.055
    return np.where(c >= 0.0031308, power_branch, linear_branch)


def _shader_rgb(value: float, vmin: float, vmax: float, exposure: float, gamma: float) -> np.ndarray:
    """Mirror the fragment shader body above, minus dither, diverging map only."""
    span = max(vmax - vmin, 1e-9)
    t = float(np.clip((2.0 * (value - vmin) / span) - 1.0, -1.0, 1.0))
    rgb = _diverging(t)
    rgb = rgb * (2.0**exposure)
    rgb = np.power(np.maximum(rgb, 0.0), 1.0 / max(gamma, 1e-6))
    return _srgb_encode(rgb)


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


def test_coef_tiles_golden_diverging_colormap_exposure_gamma_and_srgb(tmp_path) -> None:
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

    以下の4値ぞれぞれについて発散カラーマップ→露出→γ→sRGB OETFの順で
    手計算したRGBと一致することを確認する。ディザは対象外。
    """
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
