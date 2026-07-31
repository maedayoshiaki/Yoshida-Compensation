"""Helpers for limiting projector output to a configured active rectangle."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import cv2


class ProjectorRegionConfig(Protocol):
    width: int
    height: int
    active_x: int
    active_y: int
    active_width: int
    active_height: int


def resolve_active_region(projector: ProjectorRegionConfig) -> tuple[int, int, int, int]:
    """Return and validate ``(x, y, width, height)`` in projector pixels.

    Non-positive active width or height keeps the backwards-compatible
    full-frame behavior.
    """
    frame_width = int(projector.width)
    frame_height = int(projector.height)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Projector width and height must be positive.")

    width = int(projector.active_width)
    height = int(projector.active_height)
    if width <= 0 or height <= 0:
        return 0, 0, frame_width, frame_height

    x = int(projector.active_x)
    y = int(projector.active_y)
    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError(
            "Projector active region must fit inside the projector frame: "
            f"frame=({frame_width}, {frame_height}), "
            f"region=({x}, {y}, {width}, {height})."
        )
    return x, y, width, height


def mask_to_active_region(
    image: np.ndarray,
    region: tuple[int, int, int, int],
) -> np.ndarray:
    """Return a copy with all pixels outside *region* set to black."""
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("Projection image must have at least two dimensions.")

    x, y, width, height = region
    image_height, image_width = array.shape[:2]
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image_width
        or y + height > image_height
    ):
        raise ValueError(
            f"Region {region} does not fit image size "
            f"({image_width}, {image_height})."
        )

    masked = np.zeros_like(array)
    masked[y : y + height, x : x + width, ...] = array[
        y : y + height, x : x + width, ...
    ]
    return masked


def embed_in_active_region(
    image: np.ndarray,
    frame_size: tuple[int, int],
    region: tuple[int, int, int, int],
) -> np.ndarray:
    """Resize *image* into *region* on an otherwise black projector frame."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("Projection image must have shape (H, W, C).")

    frame_width, frame_height = frame_size
    x, y, width, height = region
    if (
        frame_width <= 0
        or frame_height <= 0
        or x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > frame_width
        or y + height > frame_height
    ):
        raise ValueError(
            f"Region {region} does not fit frame size "
            f"({frame_width}, {frame_height})."
        )

    interpolation = (
        cv2.INTER_AREA
        if array.shape[1] > width or array.shape[0] > height
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(array, (width, height), interpolation=interpolation)
    frame = np.zeros((frame_height, frame_width, array.shape[2]), dtype=array.dtype)
    frame[y : y + height, x : x + width] = resized
    return frame
