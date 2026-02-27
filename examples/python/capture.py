"""Simple manual capture and warp example script.

Captures an image from the camera, applies geometric warping to
projector coordinates, and saves both the raw and warped results.

手動キャプチャおよびワーピングのサンプルスクリプト。

カメラから画像をキャプチャし、プロジェクタ座標系への幾何学的ワーピングを
適用して、元画像とワーピング済み画像の両方を保存する。
"""

import os
import sys

import cv2

from examples.python.sample import capture_image, resolve_warp_settings, warp_image
from src.python.config import get_config, reload_config, split_cli_config_path


def main(argv: list[str] | None = None) -> None:
    """Run one-shot capture and warp workflow.

    Optional CLI:
        --config <path> / -c <path>

    1 回の撮影とワーピング処理を実行する。

    任意の CLI オプション:
        --config <path> / -c <path>
    """
    if argv is None:
        argv = sys.argv

    try:
        clean_argv, config_path = split_cli_config_path(argv)
    except ValueError as e:
        print(f"Invalid CLI arguments: {e}")
        return

    if len(clean_argv) != 1:
        print("Usage: python examples/python/capture.py [--config <config.toml>]")
        return

    if config_path is not None:
        reload_config(config_path)

    cfg = get_config()
    try:
        warp_method, pixel_map_path = resolve_warp_settings(cfg.paths)
    except ValueError as e:
        print(f"Invalid warp config: {e}")
        raise SystemExit(1)

    captured_image = capture_image()
    if captured_image is None:
        print("Failed to capture image")
        raise SystemExit(1)

    os.makedirs("data/manual_capture/wo_warp", exist_ok=True)
    cv2.imwrite(
        "data/manual_capture/wo_warp/captured_image.png",
        cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR),
    )

    warped_image = warp_image(
        captured_image,
        pixel_map_path,
        cfg.projector.width,
        cfg.projector.height,
        512,
        512,
        warp_method=warp_method,
    )

    os.makedirs("data/manual_capture/with_warp", exist_ok=True)
    cv2.imwrite(
        "data/manual_capture/with_warp/warped_image.png",
        cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR),
    )
    print("Captured and warped images saved.")


if __name__ == "__main__":
    main()
