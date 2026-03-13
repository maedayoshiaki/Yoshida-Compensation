"""Project a fixed image and capture camera frames on demand.

Set ``INPUT_IMAGE_PATH`` at the top of this file.

Controls:
    s: Capture one image and save it to ``data/perceived``
    q: Quit the program
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from examples.python.sample import capture_image
from src.python.config import get_config, reload_config, split_cli_config_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# INPUT_IMAGE_PATH = Path("data/inv_gamma_comp_images/inv_gamma_comp_image_03.png")
INPUT_IMAGE_PATH = Path("data/sample_warp_debug_DEBUG_ONLY/03_metal/01_prepare_target_camera_to_projector_output.png")
OUTPUT_DIR = PROJECT_ROOT / "data" / "perceived"
WINDOW_NAME = "Projection"


def _resolve_input_image_path() -> Path:
    if INPUT_IMAGE_PATH.is_absolute():
        return INPUT_IMAGE_PATH
    return PROJECT_ROOT / INPUT_IMAGE_PATH


def _fit_image_to_projector(
    image_bgr: np.ndarray,
    projector_width: int,
    projector_height: int,
) -> np.ndarray:
    """Resize image with aspect ratio preserved and center it on black."""
    src_height, src_width = image_bgr.shape[:2]
    scale = min(projector_width / src_width, projector_height / src_height)
    resized_width = max(1, int(round(src_width * scale)))
    resized_height = max(1, int(round(src_height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    canvas = np.zeros((projector_height, projector_width, 3), dtype=np.uint8)
    offset_x = (projector_width - resized_width) // 2
    offset_y = (projector_height - resized_height) // 2
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return canvas


def _load_projection_image(image_path: Path, width: int, height: int) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to load input image: {image_path}")
    return _fit_image_to_projector(image_bgr, width, height)


def _build_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return OUTPUT_DIR / f"perceived_{timestamp}.png"


def _save_captured_image(captured_rgb: np.ndarray) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _build_output_path()
    captured_bgr = cv2.cvtColor(captured_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), captured_bgr):
        raise OSError(f"Failed to save captured image: {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv

    try:
        clean_argv, config_path = split_cli_config_path(argv)
    except ValueError as exc:
        print(f"Invalid CLI arguments: {exc}")
        raise SystemExit(1)

    if len(clean_argv) != 1:
        print("Usage: python src/python/project_and_capture.py [--config <config.toml>]")
        raise SystemExit(1)

    default_config_path = PROJECT_ROOT / "config.toml"
    reload_config(config_path if config_path is not None else default_config_path)
    cfg = get_config()

    image_path = _resolve_input_image_path()
    try:
        projection_image = _load_projection_image(
            image_path,
            cfg.projector.width,
            cfg.projector.height,
        )
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1)

    print(f"Projecting image: {image_path}")
    print(f"Saving captures to: {OUTPUT_DIR.resolve()}")
    print("Press 's' to capture, 'q' to quit.")

    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(WINDOW_NAME, cfg.projector.pos_x, cfg.projector.pos_y)

    try:
        while True:
            cv2.imshow(WINDOW_NAME, projection_image)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                print("Quit requested.")
                break

            if key == ord("s"):
                print("Capturing image...")
                captured_rgb = capture_image()
                if captured_rgb is None:
                    print("Capture failed.")
                    continue

                try:
                    output_path = _save_captured_image(captured_rgb)
                except OSError as exc:
                    print(exc)
                    continue

                print(f"Saved: {output_path}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
