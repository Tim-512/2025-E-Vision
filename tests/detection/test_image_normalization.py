from __future__ import annotations

import cv2
import numpy as np

from ev_vision.config import ImageNormalizationConfig
from ev_vision.detection.image_normalization import normalize_frame
from tests.fixtures.synthetic_board import render_ring_target


def test_normalization_recovers_white_paper_under_shadow() -> None:
    target = render_ring_target(shadow_strength=0.55, ring_gray=105)
    result = normalize_frame(target.image, ImageNormalizationConfig())
    mask = np.zeros(result.white_mask.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.asarray(target.corners_px, np.int32), 255)

    occupancy = (
        np.count_nonzero(result.white_mask & mask) / np.count_nonzero(mask)
    )

    assert occupancy > 0.58


def test_saturated_spot_is_removed_from_ring_edges() -> None:
    image = render_ring_target().image.copy()
    cv2.circle(image, (480, 360), 10, (255, 255, 255), -1)

    result = normalize_frame(
        image, ImageNormalizationConfig(), mask_radius_px=12
    )

    assert np.count_nonzero(result.saturated_mask[348:373, 468:493]) > 0
    assert np.count_nonzero(result.ring_edge_mask[348:373, 468:493]) == 0
