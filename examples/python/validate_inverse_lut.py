"""Validate that the measured inverse LUT linearizes the projector response.

Project a grayscale ramp under three conditions, capture the camera's
linear RGB response for each input level, and plot input level vs measured
luminance so the linearization can be judged by eye and by metric:

1. ``raw``          - project the level unchanged (the projector's native,
                      nonlinear gamma response).
2. ``gamma``        - project the traditional analytic inverse-gamma
                      correction ``level ** (1/gamma)`` (reference).
3. ``inverse_lut``  - project ``apply_inverse_lut(level)`` (the measured LUT).

If the LUT works, the ``inverse_lut`` curve lies on the ideal diagonal
(measured luminance is linear in input level); ``raw`` bends away from it,
and ``gamma`` is only as straight as the projector matches a pure gamma.

測定ベース逆LUTがプロジェクタ応答を線形化できているかを検証する。

グレーランプを 3 条件で投影し、各入力レベルでカメラのリニア RGB 応答を
撮影して「入力レベル → 計測輝度」をプロットする。逆LUT が機能していれば
``inverse_lut`` 曲線は理想の対角線（計測輝度が入力に対して線形）に乗り、
``raw`` は湾曲し、``gamma`` はプロジェクタが純ガンマにどれだけ近いか次第で
曲がる。比較用に従来の解析的逆ガンマ補正も併せてプロットする。

Usage:
    uv run python examples/python/validate_inverse_lut.py [--config config.toml]
        [--lut data/inverse_lut.npz] [--num-levels 17] [--ref-gamma 2.2]
        [--out-prefix data/inverse_lut_validation] [--no-plot]

Prerequisite: build the inverse LUT first with build_inverse_lut.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.python.color_mixing_matrix import apply_inverse_gamma_correction
from src.python.config import get_config, reload_config, split_cli_config_path
from src.python.inverse_lut import (
    InverseLut,
    apply_inverse_lut,
    load_inverse_lut,
    reduce_frame,
)

# Correction conditions to compare, in plot order.
# 比較する補正条件（プロット順）。
MODES: Tuple[str, ...] = ("raw", "gamma", "inverse_lut")
_MODE_LABELS: Dict[str, str] = {
    "raw": "no correction (raw)",
    "gamma": "inverse gamma",
    "inverse_lut": "measured inverse LUT",
}
_MODE_COLORS: Dict[str, str] = {
    "raw": "tab:red",
    "gamma": "tab:orange",
    "inverse_lut": "tab:green",
}


def ramp_levels(num_levels: int) -> np.ndarray:
    """Return ``num_levels`` equally spaced input levels in [0, 1].

    [0, 1] を等間隔に分割した ``num_levels`` 個の入力レベルを返す。
    """
    return np.linspace(0.0, 1.0, max(2, int(num_levels)))


def corrected_projector_pixel(
    level_rgb: Sequence[int],
    mode: str,
    lut: Optional[InverseLut] = None,
    ref_gamma: float = 2.2,
) -> np.ndarray:
    """Map an 8-bit input color to the projector-input color for one mode.

    1 つの補正条件について、8bit 入力色をプロジェクタ入力色へ変換する。

    Args:
        level_rgb: Desired 8-bit RGB level (e.g. ``(v, v, v)``).
            目標 8bit RGB レベル（例 ``(v, v, v)``）。
        mode: One of ``"raw"``, ``"gamma"``, ``"inverse_lut"``.
            ``"raw"`` / ``"gamma"`` / ``"inverse_lut"`` のいずれか。
        lut: The measured inverse LUT (required for ``"inverse_lut"``).
            測定済み逆LUT（``"inverse_lut"`` で必須）。
        ref_gamma: Projector gamma for the analytic reference; the
            correction applies exponent ``1 / ref_gamma``.
            解析的参照のプロジェクタガンマ。指数 ``1 / ref_gamma`` を適用。

    Returns:
        Projector-input color as an 8-bit array of shape (3,).
            プロジェクタ入力色（8bit, 形状 (3,)）。
    """
    pixel = np.array([[list(level_rgb)]], dtype=np.uint8)  # (1, 1, 3)
    if mode == "raw":
        out = pixel
    elif mode == "gamma":
        out = apply_inverse_gamma_correction(
            pixel, gamma=1.0 / float(ref_gamma), device="cpu"
        )
    elif mode == "inverse_lut":
        if lut is None:
            raise ValueError("mode 'inverse_lut' requires a loaded InverseLut.")
        out = apply_inverse_lut(pixel, lut, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    return np.asarray(out, dtype=np.uint8).reshape(3)


def measure_modes(
    levels: np.ndarray,
    capture_fn: Callable[[np.ndarray], np.ndarray],
    lut: Optional[InverseLut] = None,
    ref_gamma: float = 2.2,
    modes: Sequence[str] = MODES,
) -> Dict[str, np.ndarray]:
    """Project and measure a gray ramp under each correction mode.

    各補正条件でグレーランプを投影・計測する。

    Args:
        levels: Input levels in [0, 1], shape (N,).
            入力レベル（[0, 1]）、形状 (N,)。
        capture_fn: Callable that projects a solid projector-input color
            (8-bit, shape (3,)) and returns the measured linear RGB
            (shape (3,)).
            プロジェクタ入力色（8bit, (3,)）の単色を投影し、計測リニア RGB
            （(3,)）を返す callable。
        lut: The measured inverse LUT (required for the ``inverse_lut`` mode).
            測定済み逆LUT（``inverse_lut`` モードで必須）。
        ref_gamma: Projector gamma for the analytic reference.
            解析的参照のプロジェクタガンマ。
        modes: Modes to measure. Missing prerequisites are skipped.
            計測する条件。前提が欠ける条件はスキップ。

    Returns:
        Mapping ``mode -> measured array (N, 3)`` of linear RGB.
            ``条件 -> 計測配列 (N, 3)``（リニア RGB）の辞書。
    """
    results: Dict[str, np.ndarray] = {}
    for mode in modes:
        if mode == "inverse_lut" and lut is None:
            continue
        measured = np.zeros((len(levels), 3), dtype=np.float64)
        for i, level in enumerate(levels):
            value = int(round(float(level) * 255.0))
            projector_pixel = corrected_projector_pixel(
                (value, value, value), mode, lut, ref_gamma
            )
            measured[i] = np.asarray(capture_fn(projector_pixel), dtype=np.float64).reshape(3)
        results[mode] = measured
    return results


def normalize_response(measured: np.ndarray) -> np.ndarray:
    """Black-subtract and white-normalize a response curve to [0, 1].

    応答曲線を黒引き・白正規化して [0, 1] にそろえる。

    Uses the first level as black and the last as white per channel, so a
    perfectly linear response maps input level ``t`` to ``t``.

    チャンネルごとに最初のレベルを黒、最後を白とする。完全に線形なら入力
    レベル ``t`` は ``t`` に写る。
    """
    measured = np.asarray(measured, dtype=np.float64)
    black = measured[0]
    white = measured[-1]
    denom = white - black
    denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
    return (measured - black) / denom


def linearity_metrics(
    levels: np.ndarray, normalized: np.ndarray
) -> Dict[str, np.ndarray]:
    """Per-channel deviation of a normalized curve from the ideal diagonal.

    正規化曲線の理想対角線からのチャンネル別ずれ。

    Returns ``max_dev`` and ``rmse`` (each shape (3,)); smaller is more
    linear. ``0`` means the curve lies exactly on ``y = x``.

    ``max_dev`` と ``rmse``（各 (3,)）を返す。小さいほど線形。``0`` は
    ``y = x`` に完全一致。
    """
    levels = np.asarray(levels, dtype=np.float64)
    normalized = np.asarray(normalized, dtype=np.float64)
    deviation = normalized - levels[:, None]
    return {
        "max_dev": np.max(np.abs(deviation), axis=0),
        "rmse": np.sqrt(np.mean(deviation**2, axis=0)),
    }


def plot_results(
    levels: np.ndarray,
    results: Dict[str, np.ndarray],
    out_path,
    ref_gamma: float = 2.2,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Plot input level vs normalized measured luminance per channel.

    入力レベル対 正規化計測輝度を、チャンネル別にプロットする。

    Draws one subplot per channel (R, G, B), each overlaying every mode plus
    the ideal linear diagonal, and saves a PNG. Returns per-mode metrics.

    チャンネル (R, G, B) ごとに 1 サブプロットを描き、各条件と理想の線形
    対角線を重ねて PNG 保存する。条件ごとの指標を返す。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalized = {mode: normalize_response(arr) for mode, arr in results.items()}
    metrics = {
        mode: linearity_metrics(levels, normalized[mode]) for mode in results
    }
    x = levels * 255.0
    channel_names = ("R", "G", "B")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for channel in range(3):
        ax = axes[channel]
        ax.plot(
            [x[0], x[-1]], [0.0, 1.0], "k--", lw=1.0, label="ideal (linear)"
        )
        for mode in results:
            label = _MODE_LABELS.get(mode, mode)
            if mode == "gamma":
                label = f"inverse gamma (g={ref_gamma:g})"
            ax.plot(
                x,
                normalized[mode][:, channel],
                marker="o",
                ms=3.5,
                lw=1.4,
                color=_MODE_COLORS.get(mode),
                label=label,
            )
        ax.set_title(f"{channel_names[channel]} channel")
        ax.set_xlabel("projector input level (0-255)")
        if channel == 0:
            ax.set_ylabel("measured luminance (normalized)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Inverse-LUT linearization check: input level vs measured luminance"
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return metrics


def save_data(out_path, levels: np.ndarray, results: Dict[str, np.ndarray], ref_gamma: float):
    """Save raw measured curves to an .npz for later analysis.

    後日の解析用に、計測した生曲線を .npz へ保存する。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"levels": np.asarray(levels, dtype=np.float64), "ref_gamma": float(ref_gamma)}
    for mode, arr in results.items():
        payload[f"measured_{mode}"] = np.asarray(arr, dtype=np.float64)
    np.savez(out_path, **payload)
    return out_path


def _print_metrics(metrics: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Print a per-mode, per-channel linearity summary (lower is better)."""
    print("\nLinearity vs ideal diagonal (lower = more linear):")
    print(f"{'mode':<20} {'metric':<8} {'R':>8} {'G':>8} {'B':>8}")
    print("-" * 56)
    for mode, met in metrics.items():
        for name in ("max_dev", "rmse"):
            r, g, b = met[name]
            print(f"{mode:<20} {name:<8} {r:>8.4f} {g:>8.4f} {b:>8.4f}")


def _split_flag(argv: List[str], flag: str) -> Tuple[List[str], Optional[str]]:
    """Extract ``--flag <value>`` from argv. Returns (cleaned, value)."""
    cleaned: List[str] = []
    value: Optional[str] = None
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            if i + 1 >= len(argv):
                raise ValueError(f"`{flag}` requires a value.")
            value = argv[i + 1]
            i += 2
            continue
        cleaned.append(argv[i])
        i += 1
    return cleaned, value


def _make_real_capture_fn(cfg, camera_backend, window_name: str):
    """Build a capture_fn that projects a solid color and measures it.

    単色を投影して計測する capture_fn を生成する。
    """
    import cv2

    proj = cfg.projector
    roi = (
        cfg.inverse_lut.roi_x,
        cfg.inverse_lut.roi_y,
        cfg.inverse_lut.roi_width,
        cfg.inverse_lut.roi_height,
    )

    def capture_fn(projector_pixel: np.ndarray) -> np.ndarray:
        image = np.empty((proj.height, proj.width, 3), dtype=np.uint8)
        image[:, :] = np.asarray(projector_pixel, dtype=np.uint8).reshape(3)
        cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(cfg.camera.wait_key_ms)
        frame = camera_backend.capture_linear_rgb()
        return reduce_frame(frame, roi)

    return capture_fn


def main(argv: Optional[List[str]] = None) -> None:
    """Run the inverse-LUT linearity validation on real hardware.

    実機で逆LUTの線形性検証を実行する。
    """
    import cv2

    from src.python.camera import create_camera_backend

    if argv is None:
        argv = sys.argv

    try:
        clean_argv, config_path = split_cli_config_path(argv)
        clean_argv, lut_arg = _split_flag(clean_argv, "--lut")
        clean_argv, levels_arg = _split_flag(clean_argv, "--num-levels")
        clean_argv, gamma_arg = _split_flag(clean_argv, "--ref-gamma")
        clean_argv, out_prefix_arg = _split_flag(clean_argv, "--out-prefix")
    except ValueError as exc:
        print(f"Invalid CLI arguments: {exc}")
        raise SystemExit(1)

    no_plot = "--no-plot" in clean_argv
    clean_argv = [a for a in clean_argv if a != "--no-plot"]
    if len(clean_argv) != 1:
        print(
            "Usage: python examples/python/validate_inverse_lut.py "
            "[--config <config.toml>] [--lut <lut.npz>] [--num-levels <n>] "
            "[--ref-gamma <g>] [--out-prefix <path>] [--no-plot]"
        )
        raise SystemExit(1)

    default_config_path = _REPO_ROOT / "config.toml"
    reload_config(config_path if config_path is not None else default_config_path)
    cfg = get_config()

    lut_path = lut_arg if lut_arg is not None else cfg.inverse_lut.inverse_lut_path
    if not Path(lut_path).exists():
        print(
            f"Inverse LUT not found: {lut_path}\n"
            "Build it first, e.g.:\n"
            "  uv run python examples/python/build_inverse_lut.py --config config.toml",
            file=sys.stderr,
        )
        raise SystemExit(1)
    lut = load_inverse_lut(lut_path)

    num_levels = int(levels_arg) if levels_arg is not None else cfg.inverse_lut.num_levels
    ref_gamma = float(gamma_arg) if gamma_arg is not None else cfg.projector.gamma
    out_prefix = out_prefix_arg or "data/inverse_lut_validation"
    levels = ramp_levels(num_levels)

    print(
        f"Validating inverse LUT (source={lut.meta.get('source')}) with "
        f"{num_levels} levels; ref gamma={ref_gamma:g}."
    )

    camera_backend = create_camera_backend(cfg.camera)
    window_name = "inverse_lut_validation"
    proj = cfg.projector

    # Reuse ONE camera session across all captures (~3 modes x num_levels
    # shots); opening a session per shot destabilizes Canon EDSDK.
    with camera_backend:
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.moveWindow(window_name, proj.pos_x, proj.pos_y)
        try:
            capture_fn = _make_real_capture_fn(cfg, camera_backend, window_name)
            results = measure_modes(levels, capture_fn, lut=lut, ref_gamma=ref_gamma)
        finally:
            cv2.destroyWindow(window_name)

    npz_path = save_data(f"{out_prefix}.npz", levels, results, ref_gamma)
    print(f"Saved raw curves to: {npz_path}")

    metrics = {
        mode: linearity_metrics(levels, normalize_response(arr))
        for mode, arr in results.items()
    }
    if not no_plot:
        png_path = f"{out_prefix}.png"
        metrics = plot_results(levels, results, png_path, ref_gamma=ref_gamma)
        print(f"Saved plot to: {png_path}")

    _print_metrics(metrics)


if __name__ == "__main__":
    main()
