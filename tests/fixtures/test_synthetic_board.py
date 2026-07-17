from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.synthetic_board import render_ring_target


def test_ring_target_contains_five_grayscale_rings() -> None:
    target = render_ring_target(image_size=(720, 960), ring_gray=92)

    assert target.center_px == pytest.approx((480.0, 360.0), abs=1.0)
    assert np.asarray(target.radii_px) / target.radii_px[0] == pytest.approx(
        [1, 2, 3, 4, 5], rel=0.03
    )


def test_ring_target_supports_partial_visibility_and_shadow() -> None:
    target = render_ring_target(
        image_size=(480, 640),
        board_center=(-20.0, 240.0),
        perspective=0.10,
        shadow_strength=0.45,
        blur_sigma=1.2,
    )

    assert target.image.shape == (480, 640, 3)
    assert target.partially_outside is True
    assert target.center_px[0] < 0.0
