"""Hardware-independent tests for examples/python/validate_inverse_lut.py.

Build an inverse LUT from a synthetic projector response, then check that
``measure_modes`` reports the LUT-corrected ramp as linear while the raw
(uncorrected) ramp is clearly nonlinear, and that a measured LUT beats the
analytic gamma reference when the projector is not a clean single gamma.

examples/python/validate_inverse_lut.py のハードウェア非依存テスト。

合成プロジェクタ応答から逆LUTを構築し、``measure_modes`` が LUT 補正後の
ランプを線形、無補正ランプを明確に非線形と報告すること、また純ガンマでない
プロジェクタでは測定LUTが解析ガンマ参照を上回ることを検証する。
"""

from pathlib import Path
import importlib.util
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.python import inverse_lut as ilut

_WHITE = 0.9  # < trust_high (245/255) so the top level is not "saturated"


def _load_validate_module():
    path = _REPO_ROOT / "examples" / "python" / "validate_inverse_lut.py"
    spec = importlib.util.spec_from_file_location("validate_inverse_lut_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


val = _load_validate_module()


def _synth_measure(projector_pixel, gammas, black):
    """Synthetic projector: measured_c = black_c + (WHITE-black_c) * p_c**gamma_c."""
    p = np.asarray(projector_pixel, dtype=np.float64) / 255.0
    return np.array(
        [black[c] + (_WHITE - black[c]) * p[c] ** gammas[c] for c in range(3)],
        dtype=np.float64,
    )


def _build_synth_lut(gammas, black, num_levels=17):
    """Build an InverseLut from the synthetic projector via the real path."""
    samples = ilut.generate_calibration_patterns(
        6, 6, num_levels=num_levels, dtype=np.uint8
    )

    def capture_fn(sample):
        frame = np.zeros((6, 6, 3), dtype=np.float32)
        for c in range(3):
            frame[:, :, c] = (
                black[c] + (_WHITE - black[c]) * sample.input_rgb[c] ** gammas[c]
            )
        return frame

    data = ilut.measure_calibration_samples(samples, capture_fn)
    return ilut.build_inverse_lut(data, source="channel_ramp", lut_resolution=256)


def test_measure_modes_inverse_lut_and_gamma_are_linear_raw_is_not():
    gammas = (2.2, 2.2, 2.2)
    black = (0.02, 0.02, 0.02)
    lut = _build_synth_lut(gammas, black)
    levels = val.ramp_levels(17)

    def capture_fn(projector_pixel):
        return _synth_measure(projector_pixel, gammas, black)

    results = val.measure_modes(levels, capture_fn, lut=lut, ref_gamma=2.2)
    assert set(results) == {"raw", "gamma", "inverse_lut"}

    metrics = {
        mode: val.linearity_metrics(levels, val.normalize_response(arr))
        for mode, arr in results.items()
    }
    # LUT-corrected and analytic-gamma (matching gamma) ramps are linear.
    assert np.all(metrics["inverse_lut"]["max_dev"] < 0.05)
    assert np.all(metrics["gamma"]["max_dev"] < 0.05)
    # The uncorrected ramp bends clearly away from the diagonal.
    assert np.all(metrics["raw"]["max_dev"] > 0.1)


def test_measured_lut_beats_analytic_gamma_when_projector_is_not_pure_gamma():
    # Per-channel gammas differ from the single reference gamma (2.2), so the
    # analytic correction cannot linearize G and B; the measured LUT can.
    gammas = (2.2, 2.0, 1.7)
    black = (0.02, 0.02, 0.02)
    lut = _build_synth_lut(gammas, black)
    levels = val.ramp_levels(17)

    def capture_fn(projector_pixel):
        return _synth_measure(projector_pixel, gammas, black)

    results = val.measure_modes(levels, capture_fn, lut=lut, ref_gamma=2.2)
    lut_dev = val.linearity_metrics(levels, val.normalize_response(results["inverse_lut"]))["max_dev"]
    gamma_dev = val.linearity_metrics(levels, val.normalize_response(results["gamma"]))["max_dev"]

    # LUT stays linear on every channel.
    assert np.all(lut_dev < 0.05)
    # On the mismatched channels (G, B) the LUT is clearly better than gamma.
    assert lut_dev[1] < gamma_dev[1]
    assert lut_dev[2] < gamma_dev[2]


def test_corrected_projector_pixel_modes():
    # raw passes the input through unchanged.
    raw = val.corrected_projector_pixel((128, 128, 128), "raw")
    assert np.array_equal(raw, np.array([128, 128, 128], dtype=np.uint8))

    # gamma applies exponent 1/ref_gamma: (128/255)**(1/2.2) > 128/255 -> brighter.
    g = val.corrected_projector_pixel((128, 128, 128), "gamma", ref_gamma=2.2)
    assert np.all(g > 128)

    # inverse_lut requires a LUT.
    try:
        val.corrected_projector_pixel((128, 128, 128), "inverse_lut", lut=None)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when inverse_lut mode lacks a LUT.")


def test_normalize_and_linearity_metrics_on_known_curves():
    levels = val.ramp_levels(5)  # [0, .25, .5, .75, 1]

    # A perfectly linear (already normalized) curve -> zero deviation.
    linear = np.stack([levels, levels, levels], axis=-1)
    # add an offset+scale to prove black-subtract/white-normalize recovers it
    scaled = 0.1 + 0.8 * linear
    norm = val.normalize_response(scaled)
    assert np.allclose(norm, linear, atol=1e-9)
    met = val.linearity_metrics(levels, norm)
    assert np.all(met["max_dev"] < 1e-9)
    assert np.all(met["rmse"] < 1e-9)


def test_plot_results_writes_png(tmp_path):
    levels = val.ramp_levels(9)
    # cheap synthetic curves: raw convex, lut linear
    raw = np.stack([levels**2.2] * 3, axis=-1)
    lut = np.stack([levels] * 3, axis=-1)
    results = {"raw": raw, "inverse_lut": lut}
    out = tmp_path / "plot.png"
    metrics = val.plot_results(levels, results, out, ref_gamma=2.2)
    assert out.exists() and out.stat().st_size > 0
    assert np.all(metrics["inverse_lut"]["max_dev"] < 1e-6)
    assert np.all(metrics["raw"]["max_dev"] > 0.1)
