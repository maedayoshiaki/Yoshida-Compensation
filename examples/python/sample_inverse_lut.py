"""Photometric compensation sample using a MEASURED inverse LUT.

A variant of ``examples/python/sample.py`` that performs the same
compensation pipeline but replaces the analytic inverse-gamma correction
(``color_mixing_matrix.apply_inverse_gamma_correction``) with the
measured per-channel inverse LUT (``inverse_lut.apply_inverse_lut``)
built by ``examples/python/build_inverse_lut.py``.

Everything except the projector-linearization step is reused from
``sample.py`` unchanged, so the two only differ in *how* the
projector's gamma is cancelled: analytic ``t**(1/gamma)`` vs. a measured
1D LUT ``L_c(t)``.

測定ベース逆LUTを用いた色補償サンプル。

``examples/python/sample.py`` と同じ補償パイプラインを実行するが、
解析的な逆ガンマ補正（``apply_inverse_gamma_correction``）の代わりに、
``build_inverse_lut.py`` で構築した測定済みチャンネル別逆LUT
（``apply_inverse_lut``）を用いる。

プロジェクタ線形化のステップ以外はすべて ``sample.py`` をそのまま再利用
するため、両者はガンマの打ち消し方（解析的 ``t**(1/gamma)`` か、測定した
1D LUT ``L_c(t)`` か）だけが異なる。

Usage:
    uv run python examples/python/sample_inverse_lut.py [--config config.toml]
                                                        [--lut data/inverse_lut.npz]

Prerequisite: build the inverse LUT first, e.g.
    uv run python examples/python/build_inverse_lut.py --config config.toml
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.python import inverse_lut as ilut
from src.python.config import get_config, reload_config, split_cli_config_path

# Output directories for the LUT-linearized artifacts. Kept separate from
# sample.py's ``inv_gamma_*`` outputs so the two corrections can be compared
# side by side without overwriting each other.
# LUT 線形化成果物の出力先。sample.py の ``inv_gamma_*`` 出力を上書きせず
# 並べて比較できるよう別ディレクトリにする。
LUT_PATTERN_DIR = os.path.join("data", "inv_lut_proj_patterns")
LUT_COMP_DIR = os.path.join("data", "inv_lut_comp_images")


def _load_sample_module():
    """Load ``sample.py`` as a module to reuse its pipeline helpers.

    ``sample.py`` をモジュールとして読み込み、パイプライン用ヘルパーを再利用する。

    Loading the sibling script (rather than duplicating ~1000 lines of warp
    and I/O plumbing) keeps this sample focused on the single difference:
    the projector-linearization step. ``sample.py`` itself is unchanged.

    隣接スクリプトを読み込む（ワープ・I/O 配線 約1000行を複製しない）ことで、
    本サンプルは唯一の相違点＝プロジェクタ線形化のステップに集中できる。
    ``sample.py`` 自体は無変更。
    """
    sample_path = _REPO_ROOT / "examples" / "python" / "sample.py"
    spec = importlib.util.spec_from_file_location("yoshida_sample", sample_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {sample_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _split_flag(argv: list[str], flag: str) -> tuple[list[str], str | None]:
    """Extract ``--flag <value>`` from argv. Returns (cleaned, value).

    argv から ``--flag <value>`` を分離する。(整形後, 値) を返す。
    """
    cleaned: list[str] = []
    value: str | None = None
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


def _load_inverse_lut_or_exit(lut_path: str) -> ilut.InverseLut:
    """Load the measured inverse LUT, exiting with guidance if it is missing.

    測定済み逆LUTを読み込む。存在しない場合は案内を出して終了する。
    """
    path = Path(lut_path)
    if not path.exists():
        print(
            f"Inverse LUT not found: {path}\n"
            "Build it first, for example:\n"
            "  uv run python examples/python/build_inverse_lut.py "
            "--config config.toml",
            file=sys.stderr,
        )
        raise SystemExit(1)
    lut = ilut.load_inverse_lut(path)
    print(f"Loaded inverse LUT from: {path} (source={lut.meta.get('source')})")
    return lut


def main(argv: list[str] | None = None) -> None:
    """Run the compensation pipeline with LUT-based projector linearization.

    LUT ベースのプロジェクタ線形化で色補償パイプラインを実行する。

    Mirrors ``sample.py``'s flow (generate patterns -> project & capture ->
    color mixing matrices -> per-target compensation), but every place that
    ``sample.py`` calls ``apply_inverse_gamma_correction`` this sample calls
    ``apply_inverse_lut`` instead.

    ``sample.py`` の流れ（パターン生成 → 投影・撮影 → カラーミキシング行列 →
    ターゲットごとの補償）を踏襲するが、``sample.py`` が
    ``apply_inverse_gamma_correction`` を呼ぶ箇所すべてで、本サンプルは
    代わりに ``apply_inverse_lut`` を呼ぶ。
    """
    if argv is None:
        argv = sys.argv

    try:
        clean_argv, config_path = split_cli_config_path(argv)
        clean_argv, lut_path_arg = _split_flag(clean_argv, "--lut")
    except ValueError as exc:
        print(f"Invalid CLI arguments: {exc}")
        return

    if len(clean_argv) != 1:
        print(
            "Usage: python examples/python/sample_inverse_lut.py "
            "[--config <config.toml>] [--lut <lut.npz>]"
        )
        return

    default_config_path = _REPO_ROOT / "config.toml"
    effective_config_path = (
        config_path if config_path is not None else default_config_path
    )
    # Yoshida config is the single source of truth. Mirror the same path to GrayCode.
    reload_config(effective_config_path)

    sample = _load_sample_module()
    sample.reload_graycode_config(effective_config_path)

    cfg = get_config()
    proj = cfg.projector
    paths = cfg.paths

    # Load the measured inverse LUT (built by build_inverse_lut.py). This is
    # the drop-in replacement for analytic inverse-gamma correction.
    lut_path = (
        lut_path_arg if lut_path_arg is not None else cfg.inverse_lut.inverse_lut_path
    )
    lut = _load_inverse_lut_or_exit(lut_path)

    try:
        warp_method, pixel_map_path = sample.resolve_warp_settings(paths)
        target_image_space = sample.resolve_target_image_space(paths)
    except ValueError as exc:
        print(f"Invalid warp config: {exc}")
        return

    # 1. Generate linear projection patterns and linearize them with the LUT.
    #    リニア投影パターンを生成し、LUT で線形化する。
    linear_proj_patterns = sample.generate_projection_patterns(
        proj.width, proj.height, dtype=np.dtype(np.uint8)
    )
    os.makedirs(paths.linear_pattern_dir, exist_ok=True)
    os.makedirs(LUT_PATTERN_DIR, exist_ok=True)

    lut_patterns = []
    for i, pattern in enumerate(linear_proj_patterns):
        cv2.imwrite(
            os.path.join(paths.linear_pattern_dir, f"pattern_{i:02d}.png"),
            cv2.cvtColor(pattern, cv2.COLOR_RGB2BGR),
        )
        # === LUT-based projector linearization (replaces inverse gamma) ===
        # sample.py: apply_inverse_gamma_correction(pattern, gamma=1/proj.gamma)
        lut_pattern = ilut.apply_inverse_lut(pattern, lut, dtype=np.uint8)
        lut_patterns.append(lut_pattern)
        cv2.imwrite(
            os.path.join(LUT_PATTERN_DIR, f"inv_lut_pattern_{i:02d}.png"),
            cv2.cvtColor(lut_pattern, cv2.COLOR_RGB2BGR),
        )

    # 2. Display each LUT-linearized pattern and capture the camera response.
    #    LUT 線形化パターンを表示し、カメラ応答を撮影する。
    window_name = "projection_window"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(window_name, proj.pos_x, proj.pos_y)

    os.makedirs(paths.captured_image_dir, exist_ok=True)
    captured_images = []
    for pattern in lut_patterns:
        cv2.imshow(window_name, cv2.cvtColor(pattern, cv2.COLOR_RGB2BGR))
        cv2.waitKey(cfg.camera.wait_key_ms)
        # Settle: let the projector display fully switch to the new pattern
        # before capturing (same buffer as calibration/validation).
        time.sleep(ilut._PROJECTION_SETTLE_SECONDS)

        captured_image = sample.capture_image()
        if captured_image is None:
            print("Aborting pipeline because image capture failed.", file=sys.stderr)
            cv2.destroyWindow(window_name)
            return

        captured_images.append(captured_image)
        # Saved PNGs are sRGB-encoded for viewing only; the pipeline keeps
        # using the linear array in `captured_images`.
        cv2.imwrite(
            os.path.join(
                paths.captured_image_dir,
                f"captured_image_{len(captured_images) - 1:02d}.png",
            ),
            cv2.cvtColor(
                sample.capture_to_srgb_uint8(captured_image), cv2.COLOR_RGB2BGR
            ),
        )

    cv2.destroyWindow(window_name)

    # 3. Calculate color mixing matrices from linear patterns and captures.
    #    リニアパターンと撮影画像からカラーミキシング行列を計算する。
    color_mixing_matrices = sample.calc_color_mixing_matrices(
        proj_images=linear_proj_patterns, captured_images=captured_images
    )
    np.save("data/color_mixing_matrices.npy", color_mixing_matrices)

    # 4. Load targets, compute compensation, linearize with the LUT, and save.
    #    ターゲットを読み込み、補償を計算し、LUT で線形化して保存する。
    target_images = sample._load_target_images(paths.target_image_dir)

    compensation_output_dir = Path(paths.compensation_image_dir)
    lut_comp_output_dir = Path(LUT_COMP_DIR)
    sample._clear_generated_pngs(compensation_output_dir, "compensation_image_*.png")
    sample._clear_generated_pngs(lut_comp_output_dir, "inv_lut_comp_image_*.png")

    cam_width = captured_images[0].shape[1]
    cam_height = captured_images[0].shape[0]
    compensation_space = sample.compensation_space_from_warp_method(warp_method)

    for idx, (target_name, target_uint8) in enumerate(target_images):
        # Targets on disk are sRGB-encoded; the matrices map LINEAR camera
        # values, so decode to linear light before any computation.
        target_image = sample.srgb_to_linear(
            target_uint8.astype(np.float32) / 255.0
        ).astype(np.float32)
        compensation_target_image = sample.prepare_target_image_for_compensation(
            target_image,
            target_image_space,
            pixel_map_path,
            proj.width,
            proj.height,
            cam_width,
            cam_height,
            warp_method,
        )

        if compensation_space == "camera":
            sample._ensure_matching_spatial_shape(
                compensation_target_image,
                color_mixing_matrices,
                "c2p compensation",
            )
            before_warped_compensation_image = sample.calc_compensation_image(
                target_image=compensation_target_image,
                color_mixing_matrices=color_mixing_matrices,
                dtype=np.uint8,
            )
            compensation_image = sample.warp_image(
                before_warped_compensation_image,
                pixel_map_path,
                proj.width,
                proj.height,
                target_image.shape[1],
                target_image.shape[0],
                warp_method=warp_method,
            )
        else:
            warped_color_mixing_matrices = (
                sample.warp_color_mixing_matrices_to_projector(
                    color_mixing_matrices,
                    pixel_map_path,
                    proj.width,
                    proj.height,
                    target_image.shape[1],
                    target_image.shape[0],
                    warp_method=warp_method,
                )
            )
            sample._ensure_matching_spatial_shape(
                compensation_target_image,
                warped_color_mixing_matrices,
                "p2c compensation",
            )
            compensation_image = sample.calc_compensation_image(
                target_image=compensation_target_image,
                color_mixing_matrices=warped_color_mixing_matrices,
                dtype=np.uint8,
            )

        if compensation_image is None or compensation_image.size == 0:
            return

        # === LUT-based projector linearization (replaces inverse gamma) ===
        # sample.py: apply_inverse_gamma_correction(compensation_image,
        #                                            gamma=1/proj.gamma)
        lut_comp_image = ilut.apply_inverse_lut(
            compensation_image, lut, dtype=np.uint8
        )

        cv2.imwrite(
            str(compensation_output_dir / f"compensation_image_{idx:02d}.png"),
            cv2.cvtColor(compensation_image, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(lut_comp_output_dir / f"inv_lut_comp_image_{idx:02d}.png"),
            cv2.cvtColor(lut_comp_image, cv2.COLOR_RGB2BGR),
        )

    print(
        "Done. LUT-linearized patterns -> "
        f"{LUT_PATTERN_DIR}, compensation -> {LUT_COMP_DIR}"
    )


if __name__ == "__main__":
    main()
