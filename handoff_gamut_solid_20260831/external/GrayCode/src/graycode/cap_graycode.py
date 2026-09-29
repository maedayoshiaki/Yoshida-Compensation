import cv2
import numpy as np
import glob
import sys
from pathlib import Path
from typing import Callable, List, Optional

from .config import get_config, reload_config, split_cli_config_path
from .decode import decode_c2p
from .gen_graycode import generate_expanded_patterns

# A capture provider: returns one camera frame (grayscale (H,W) or color
# (H,W,3)). Injectable so callers can drive any camera (Canon EDSDK, SR-5100,
# a fake in tests) instead of the hardwired default.
CaptureFn = Callable[[], np.ndarray]
# A display provider: shows one projector pattern (index, pattern image).
DisplayFn = Callable[[int, np.ndarray], None]


def open_cam() -> None:
    pass


def close_cam() -> None:
    pass


def capture() -> np.ndarray:
    """Default capture provider — Canon EOS via EDSDK.

    The EDSDK import is lazy so ``import graycode`` works without the Canon SDK
    installed (e.g. when reusing only the warp / decode / calibration API with
    a different camera).
    """
    from edsdk.camera_controller import CameraController

    cam_cfg = get_config().camera
    with CameraController(register_property_events=False) as camera:
        camera.set_properties(
            av=cam_cfg.av,
            tv=cam_cfg.tv,
            iso=cam_cfg.iso,
            image_quality=cam_cfg.image_quality,
        )
        imgs = camera.capture_numpy()
        img = imgs[0]
    return img


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Coerce a captured frame to single-channel grayscale uint8/float."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _default_display(window_pos: tuple[int, int], wait_ms: int) -> DisplayFn:
    """Fullscreen OpenCV projector display (same sequence as the CLI)."""
    cv2.namedWindow("Pattern", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Pattern", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("Pattern", window_pos[0], window_pos[1])

    def _show(index: int, pattern: np.ndarray) -> None:
        cv2.imshow("Pattern", pattern)
        cv2.waitKey(wait_ms)

    return _show


def run_capture(
    patterns: List[np.ndarray],
    *,
    capture_fn: CaptureFn,
    display_fn: Optional[DisplayFn] = None,
    window_pos: tuple[int, int] = (0, 0),
    wait_ms: int = 500,
) -> List[np.ndarray]:
    """Display each pattern and capture a grayscale frame per pattern.

    Returns the captured grayscale frames in the SAME order as ``patterns``
    (so the trailing two are the white/black references produced by
    :func:`generate_expanded_patterns`). ``display_fn`` defaults to the
    fullscreen OpenCV window; pass a no-op to run headless (tests). Captured
    frames are coerced to grayscale.
    """
    owns_window = display_fn is None
    show = display_fn if display_fn is not None else _default_display(window_pos, wait_ms)
    frames: List[np.ndarray] = []
    try:
        for i, pat in enumerate(patterns):
            show(i, pat)
            frames.append(_to_gray(capture_fn()))
    finally:
        if owns_window:
            cv2.destroyAllWindows()
    return frames


def run_calibration(
    *,
    capture_fn: CaptureFn,
    proj_height: int,
    proj_width: int,
    height_step: int = 1,
    width_step: int = 1,
    window_pos: tuple[int, int] = (0, 0),
    wait_ms: int = 500,
    black_threshold: Optional[int] = None,
    white_threshold: Optional[int] = None,
    display_fn: Optional[DisplayFn] = None,
    debug_dir: Optional[str] = None,
) -> np.ndarray:
    """Run a full in-memory gray-code calibration and return the p2c map.

    Generates patterns, displays + captures each via the injected
    ``capture_fn``, decodes, and returns the projector→camera correspondence as
    an ``(N, 4)`` float32 array of ``[proj_x, proj_y, cam_x, cam_y]`` rows
    (the format :func:`graycode.interpolate_p2c.load_p2c_numpy_array` and
    :class:`graycode.warp_image.PixelMapWarperTorch` accept).

    No files are written unless ``debug_dir`` is given, in which case the raw
    gray-code captures / white / black / valid mask are saved there (for later
    visualization with ``scripts/visualize_graycode.py``). Thresholds default to
    the active config's decode section.
    """
    decode_cfg = get_config().decode
    bthr = decode_cfg.black_threshold if black_threshold is None else black_threshold
    wthr = decode_cfg.white_threshold if white_threshold is None else white_threshold

    patterns = generate_expanded_patterns(proj_height, proj_width, height_step, width_step)
    frames = run_capture(
        patterns,
        capture_fn=capture_fn,
        display_fn=display_fn,
        window_pos=window_pos,
        wait_ms=wait_ms,
    )
    black = frames.pop()
    white = frames.pop()
    c2p_list, _cam_hw = decode_c2p(
        frames,
        white,
        black,
        proj_height=proj_height,
        proj_width=proj_width,
        height_step=height_step,
        width_step=width_step,
        black_threshold=bthr,
        white_threshold=wthr,
        debug_dir=debug_dir,
    )
    return np.array(
        [[px, py, cx, cy] for (cx, cy), (px, py) in c2p_list], dtype=np.float32
    ).reshape(-1, 4)


def print_usage() -> None:
    print(
        "Usage : python cap_graycode.py <window position x> <window position y> "
        "[--config <config.toml>]"
    )
    print()


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv
    try:
        argv, config_path = split_cli_config_path(argv)
    except ValueError as e:
        print(e)
        print_usage()
        return

    if config_path is not None:
        reload_config(config_path)

    if len(argv) != 3:
        print_usage()
        return

    try:
        window_pos_x = int(argv[1])
        window_pos_y = int(argv[2])
    except ValueError:
        print("height, width は整数で指定してください。")
        print_usage()
        return
    cfg = get_config()
    target_dir = Path(cfg.paths.pattern_dir)
    capture_dir = Path(cfg.paths.captured_dir)
    wait_ms = cfg.camera.wait_key_ms

    cam_height = 0
    cam_width = 0
    graycode_imgs: List[np.ndarray] = []
    # グレイコードをファイルから参照
    for idx, fname in enumerate(sorted(glob.glob(str(target_dir / "pattern_*.png")))):
        print(f"Loading pattern image: {fname}")
        pat_img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if cam_height == 0 and cam_width == 0:
            cam_height, cam_width = pat_img.shape
        graycode_imgs.append(pat_img)

    cv2.namedWindow("Pattern", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Pattern", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("Pattern", window_pos_x, window_pos_y)

    open_cam()

    # キャプチャディレクトリ作成
    capture_dir.mkdir(parents=True, exist_ok=True)

    for i, pat in enumerate(graycode_imgs):
        print(f"Displaying pattern image {i:02d}...")
        cv2.imshow("Pattern", pat)
        cv2.waitKey(wait_ms)
        captured_img = capture()
        captured_img_gray = cv2.cvtColor(captured_img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(f"{capture_dir}/capture_{i:02d}.png", captured_img_gray)
        print(f"Captured and saved image: capture_{i:02d}.png")

    cv2.destroyAllWindows()
    close_cam()

    print("All patterns have been captured and saved.")

    print()
    print("=== Next step ===")
    print(
        "Run 'python decode.py <projector image height> <projector image width>' to decode the captured images."
    )
    print()


if __name__ == "__main__":
    main()
