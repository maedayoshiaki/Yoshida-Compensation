"""Build a measured inverse LUT (projector linearization) from capture.

Generate calibration ramps for the configured sampling method
(``inverse_lut.source``: "channel_ramp" or "gray"), project the raw
patterns and capture their linear response, build a per-channel inverse
LUT, and save it. Optionally apply the LUT to an image for a quick demo.

撮影から測定ベース逆LUT（プロジェクタ線形化）を構築する。

設定のサンプリング系統（``inverse_lut.source``: "channel_ramp" または
"gray"）のキャリブレーションランプを生成し、生パターンを投影してリニア応答を
撮影、チャンネル別逆LUTを構築して保存する。任意で画像に適用するデモも行う。

Usage:
    uv run python examples/python/build_inverse_lut.py [--config config.toml]
                                                       [--load lut.npz]
                                                       [--apply image.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.python import inverse_lut as ilut
from src.python.config import get_config, reload_config, split_cli_config_path
from src.python.projection_region import mask_to_active_region, resolve_active_region


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


def main(argv: list[str] | None = None) -> None:
    """Run the inverse-LUT calibration/build/save workflow.

    逆LUTのキャリブレーション・構築・保存ワークフローを実行する。
    """
    if argv is None:
        argv = sys.argv

    try:
        clean_argv, config_path = split_cli_config_path(argv)
        clean_argv, load_path = _split_flag(clean_argv, "--load")
        clean_argv, apply_path = _split_flag(clean_argv, "--apply")
        clean_argv, debug_dir = _split_flag(clean_argv, "--debug-dir")
    except ValueError as exc:
        print(f"Invalid CLI arguments: {exc}")
        raise SystemExit(1)

    if len(clean_argv) != 1:
        print(
            "Usage: python examples/python/build_inverse_lut.py "
            "[--config <config.toml>] [--load <lut.npz>] [--apply <image>] "
            "[--debug-dir <dir>]"
        )
        raise SystemExit(1)

    default_config_path = _REPO_ROOT / "config.toml"
    reload_config(config_path if config_path is not None else default_config_path)
    cfg = get_config()
    ilut_cfg = cfg.inverse_lut

    if load_path is not None:
        lut = ilut.load_inverse_lut(load_path)
        print(f"Loaded inverse LUT from: {load_path}")
    else:
        samples = ilut.generate_calibration_patterns(
            cfg.projector.width,
            cfg.projector.height,
            num_levels=ilut_cfg.num_levels,
            source=ilut_cfg.source,
            dtype=np.uint8,
        )
        active_region = resolve_active_region(cfg.projector)
        samples = [
            ilut.CalibrationSample(
                input_rgb=sample.input_rgb,
                image=mask_to_active_region(sample.image, active_region),
                kind=sample.kind,
                recorded_channels=sample.recorded_channels,
            )
            for sample in samples
        ]
        print(
            f"Projecting {len(samples)} calibration patterns "
            f"(source={ilut_cfg.source})..."
        )
        data = ilut.run_calibration_capture(samples, debug_dir=debug_dir)
        lut = ilut.build_inverse_lut(data, source=ilut_cfg.source)
        out_path = ilut.save_inverse_lut(ilut_cfg.inverse_lut_path, lut)
        print(f"Saved inverse LUT to: {out_path}")

    if apply_path is not None:
        bgr = cv2.imread(str(apply_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Failed to load image: {apply_path}")
            raise SystemExit(1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        projector_input = ilut.apply_inverse_lut(rgb, lut, dtype=np.uint8)
        out_image_path = Path(apply_path).with_name(
            Path(apply_path).stem + "_inv_lut.png"
        )
        cv2.imwrite(
            str(out_image_path),
            cv2.cvtColor(projector_input, cv2.COLOR_RGB2BGR),
        )
        print(f"Applied inverse LUT, saved to: {out_image_path}")


if __name__ == "__main__":
    main()
