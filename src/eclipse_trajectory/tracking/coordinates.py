from __future__ import annotations

from eclipse_trajectory.schemas import NormalizedPoint


def pixel_to_normalized(x: float, y: float, width: int, height: int) -> NormalizedPoint:
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    if not 0 <= x <= width or not 0 <= y <= height:
        raise ValueError("Pixel coordinate lies outside the frame")
    return (x / width, y / height)
