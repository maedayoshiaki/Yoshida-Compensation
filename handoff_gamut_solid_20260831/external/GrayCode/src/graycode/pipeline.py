from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from . import gen_graycode
from . import cap_graycode
from . import decode
from . import interpolate_c2p
from . import warp_image  # まだ自動では呼ばないが import だけしておく
from .config import get_config, reload_config, split_cli_config_path


@dataclass
class GraycodePipelineConfig:
    """グレイコード一連処理の設定"""

    # プロジェクタの解像度
    proj_height: int
    proj_width: int

    # GrayCode のブロックサイズ（gen_graycode / decode で共通利用）
    width_step: int = 1
    height_step: int = 1

    # プロジェクタウィンドウを出す位置
    window_pos_x: int = 0
    window_pos_y: int = 0

    # 将来の拡張用オプション
    run_capture: bool = True
    run_decode: bool = True
    run_interpolate: bool = True
    # run_warp: bool = False  # warp_image まで自動で回したくなったら使う


def run_graycode_pipeline(cfg: GraycodePipelineConfig) -> None:
    """
    グレイコードの
        1. パターン生成
        2. 投影＆撮影
        3. デコード
        4. 対応点補間
    までを順番に実行する高レベルパイプライン関数。
    """

    # 1. GrayCode パターン生成
    gen_argv = [
        "gen_graycode.py",
        str(cfg.proj_height),
        str(cfg.proj_width),
        str(cfg.height_step),  # height_step (gen/decode 共通で height, width 順)
        str(cfg.width_step),  # width_step
    ]
    print("[1/4] Generating graycode patterns...")
    gen_graycode.main(gen_argv)

    # 2. 投影＆撮影
    if cfg.run_capture:
        cap_argv = [
            "cap_graycode.py",
            str(cfg.window_pos_x),
            str(cfg.window_pos_y),
        ]
        print("[2/4] Capturing projected patterns...")
        cap_graycode.main(cap_argv)
    else:
        print("[2/4] Skipped capture (run_capture=False)")

    # 3. デコード（result_c2p.npy / .csv を生成）
    cam_height = 0
    cam_width = 0
    if cfg.run_decode:
        dec_argv = [
            "decode.py",
            str(cfg.proj_height),
            str(cfg.proj_width),
            str(cfg.height_step),
            str(cfg.width_step),
        ]
        print("[3/4] Decoding captured images...")
        cam_size = decode.main(dec_argv)
        if cam_size is not None:
            cam_height, cam_width = cam_size
    else:
        print("[3/4] Skipped decode (run_decode=False)")

    # 4. 対応点補間（result_c2p_compensated.npy / .csv を生成）
    if cfg.run_interpolate and cam_height > 0 and cam_width > 0:
        # decode.py が出力する既定ファイル名をそのまま使う (output_dir 配下, M6)
        app_cfg = get_config().pipeline
        out_dir = get_config().paths.output_dir
        input_path = str(Path(out_dir) / app_cfg.default_input_file)
        # 補間手法は interpolate_c2p.default_method を単一の真実源とする (M3)
        method = get_config().interpolate_c2p.default_method
        interp_argv = [
            "interpolate_c2p.py",
            input_path,
            str(cam_height),
            str(cam_width),
            method,
        ]
        print("[4/4] Interpolating c2p correspondences...")
        interpolate_c2p.main(interp_argv)
    else:
        print("[4/4] Skipped interpolate (run_interpolate=False or invalid cam size)")

    print("Graycode pipeline finished.")


def main(argv: list[str] | None = None) -> None:
    """
    CLI エントリポイント。
    例:
        python -m graycode.pipeline 1080 1920 1 1 0 0
    """
    if argv is None:
        argv = sys.argv

    try:
        argv, config_path = split_cli_config_path(argv)
    except ValueError as e:
        print(e)
        return

    if config_path is not None:
        reload_config(config_path)

    # 引数なし（argv[0] のみ）の場合は config.toml の値をそのまま使用。
    # 引数ありの場合は CLI 値を優先し、省略分は config から補完。
    if len(argv) > 7:
        pcfg = get_config().pipeline
        print(
            "Usage: python -m graycode.pipeline "
            "[proj_height] [proj_width] [height_step] [width_step] "
            "[window_pos_x] [window_pos_y] [--config <config.toml>]"
        )
        print(
            f"  全引数省略時は config.toml の値を使用します "
            f"(現在: {pcfg.proj_height}x{pcfg.proj_width}, "
            f"step={pcfg.height_step}x{pcfg.width_step}, "
            f"window=({pcfg.window_pos_x},{pcfg.window_pos_y}))"
        )
        return

    pcfg = get_config().pipeline

    try:
        proj_height = int(argv[1]) if len(argv) >= 2 else pcfg.proj_height
        proj_width = int(argv[2]) if len(argv) >= 3 else pcfg.proj_width
        height_step = int(argv[3]) if len(argv) >= 4 else pcfg.height_step
        width_step = int(argv[4]) if len(argv) >= 5 else pcfg.width_step
        window_pos_x = int(argv[5]) if len(argv) >= 6 else pcfg.window_pos_x
        window_pos_y = int(argv[6]) if len(argv) >= 7 else pcfg.window_pos_y
    except ValueError:
        print(
            "proj_height, proj_width, height_step, width_step, "
            "window_pos_x, window_pos_y は整数で指定してください。"
        )
        return

    cfg = GraycodePipelineConfig(
        proj_height=proj_height,
        proj_width=proj_width,
        height_step=height_step,
        width_step=width_step,
        window_pos_x=window_pos_x,
        window_pos_y=window_pos_y,
    )
    run_graycode_pipeline(cfg)


if __name__ == "__main__":
    main()
