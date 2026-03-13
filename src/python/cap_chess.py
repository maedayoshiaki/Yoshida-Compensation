import time
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

from edsdk.camera_controller import CameraController
from src.python.camera.canon_edsdk import capture_srgb_uint8
from src.python.config import get_config, reload_config, split_cli_config_path


@dataclass(frozen=True)
class CaptureConfig:
    capture_dir: Path = Path("data/chess_captured")
    num_images: int = 20
    interval_sec: float = 5.0
    preview_scale: float = 0.6
    show_overlay: bool = True
    window_name: str = "Canon LiveView"


CONFIG = CaptureConfig()


def to_bgr_from_liveview(live_view_rgb: np.ndarray) -> np.ndarray:
    """Convert the Live View frame from RGB/gray into BGR for OpenCV display."""
    if live_view_rgb.ndim == 2:
        return cv2.cvtColor(live_view_rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(live_view_rgb, cv2.COLOR_RGB2BGR)


def to_gray_from_capture(captured_rgb: np.ndarray) -> np.ndarray:
    """Convert the captured RGB image into grayscale for calibration use."""
    return cv2.cvtColor(captured_rgb, cv2.COLOR_RGB2GRAY)


def draw_overlay(frame: np.ndarray, captured_count: int, next_capture_at: float, config: CaptureConfig) -> None:
    """Draw capture progress on the preview frame in place."""
    remaining_sec = max(0.0, next_capture_at - time.time())
    cv2.putText(
        frame,
        f"Captured: {captured_count}/{config.num_images}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Next capture in: {remaining_sec:0.1f}s   (q: quit)",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def grab_preview_frame(camera: CameraController, config: CaptureConfig) -> np.ndarray:
    """Fetch one Live View frame and apply preview resizing."""
    frame = to_bgr_from_liveview(camera.grab_live_view_numpy())
    if config.preview_scale != 1.0:
        frame = cv2.resize(
            frame,
            None,
            fx=config.preview_scale,
            fy=config.preview_scale,
            interpolation=cv2.INTER_AREA,
        )
    return frame


def capture_and_save(
    camera: CameraController,
    image_index: int,
    config: CaptureConfig,
    image_quality: str,
) -> bool:
    """Capture a still image, convert it to grayscale, and save it."""
    rgb_image = capture_srgb_uint8(camera, image_quality)
    gray_image = to_gray_from_capture(rgb_image)

    output_path = config.capture_dir / f"chess_{image_index:02d}.png"
    if cv2.imwrite(str(output_path), gray_image):
        print(f"Saved: {output_path.name}")
        return True

    print(f"[warn] failed to save: {output_path}")
    return False


def load_camera_settings(argv: list[str]) -> tuple[str, str, str, str]:
    """Load camera capture settings from config.toml."""
    clean_argv, config_path = split_cli_config_path(argv)
    if len(clean_argv) != 1:
        print("Usage: python src/python/cap_chess.py [--config <config.toml>]")
        raise SystemExit(1)

    default_config_path = Path(__file__).resolve().parents[2] / "config.toml"
    reload_config(config_path if config_path is not None else default_config_path)
    camera_config = get_config().camera

    if camera_config.backend.strip().lower() != "canon_edsdk":
        raise RuntimeError(
            "cap_chess.py requires camera.backend = \"canon_edsdk\" in config.toml."
        )

    return (
        camera_config.av,
        camera_config.tv,
        camera_config.iso,
        camera_config.image_quality,
    )


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv

    config = CONFIG
    av, tv, iso, image_quality = load_camera_settings(argv)
    config.capture_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {config.capture_dir.resolve()}")
    print(f"Will capture {config.num_images} images every {config.interval_sec} seconds.")
    print("Press 'q' to quit.")

    with CameraController(register_property_events=False) as camera:
        camera.set_properties(
            av=av,
            tv=tv,
            iso=iso,
            image_quality=image_quality,
        )
        camera.start_live_view()
        cv2.namedWindow(config.window_name, cv2.WINDOW_AUTOSIZE)

        next_capture_at = time.time() + config.interval_sec
        captured_count = 0

        try:
            while True:
                try:
                    frame = grab_preview_frame(camera, config)
                    if config.show_overlay:
                        draw_overlay(frame, captured_count, next_capture_at, config)
                    cv2.imshow(config.window_name, frame)
                except Exception as exc:
                    print(f"[warn] liveview grab failed: {exc}")

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quit requested.")
                    break

                if time.time() >= next_capture_at and captured_count < config.num_images:
                    print(f"[{captured_count + 1}/{config.num_images}] Capturing & saving...")
                    if capture_and_save(camera, captured_count, config, image_quality):
                        captured_count += 1
                    next_capture_at = time.time() + config.interval_sec

                    if captured_count >= config.num_images:
                        print("All images captured.")
                        break

                time.sleep(0.002)
        finally:
            try:
                camera.stop_live_view()
            except Exception:
                pass
            cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()
