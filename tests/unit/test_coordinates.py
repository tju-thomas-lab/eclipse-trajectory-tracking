from __future__ import annotations

import pytest

from eclipse_trajectory.tracking.coordinates import pixel_to_normalized


def test_pixel_coordinate_conversion() -> None:
    assert pixel_to_normalized(960, 540, 1920, 1080) == (0.5, 0.5)
    assert pixel_to_normalized(0, 0, 1920, 1080) == (0.0, 0.0)
    with pytest.raises(ValueError):
        pixel_to_normalized(1921, 540, 1920, 1080)
