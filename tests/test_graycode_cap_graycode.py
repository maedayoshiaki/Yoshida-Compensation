import cv2
import numpy as np

from graycode import cap_graycode


def test_run_capture_uses_injected_camera_and_preserves_order() -> None:
    patterns = [
        np.full((2, 3), 10, dtype=np.uint8),
        np.full((2, 3), 20, dtype=np.uint8),
    ]
    captured = [
        np.full((2, 3, 3), (255, 0, 0), dtype=np.uint8),
        np.full((2, 3, 3), (0, 255, 0), dtype=np.uint8),
    ]
    displayed: list[tuple[int, int]] = []

    def capture_fn() -> np.ndarray:
        return captured.pop(0)

    frames = cap_graycode.run_capture(
        patterns,
        capture_fn=capture_fn,
        display_fn=lambda index, pattern: displayed.append(
            (index, int(pattern[0, 0]))
        ),
    )

    assert displayed == [(0, 10), (1, 20)]
    assert len(frames) == 2
    assert frames[0].ndim == 2
    assert int(frames[0][0, 0]) == int(
        cv2.cvtColor(
            np.array([[[255, 0, 0]]], dtype=np.uint8),
            cv2.COLOR_RGB2GRAY,
        )[0, 0]
    )
    assert int(frames[1][0, 0]) == int(
        cv2.cvtColor(
            np.array([[[0, 255, 0]]], dtype=np.uint8),
            cv2.COLOR_RGB2GRAY,
        )[0, 0]
    )
