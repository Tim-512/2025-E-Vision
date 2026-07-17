from __future__ import annotations

import cv2
import numpy as np
import pytest

from ev_vision.config import ImageNormalizationConfig, WhiteBoardConfig
from ev_vision.detection.image_normalization import normalize_frame
from ev_vision.detection.white_board import find_white_board_candidates
from tests.fixtures.synthetic_board import render_ring_target


def candidates_for(image: np.ndarray):
    frame = normalize_frame(image, ImageNormalizationConfig())
    return find_white_board_candidates(frame, WhiteBoardConfig())


def test_complete_a4_board_is_a_candidate():
    target = render_ring_target(perspective=0.12, shadow_strength=0.35)
    best = max(candidates_for(target.image), key=lambda item: item.geometry_score)
    assert best.white_occupancy >= 0.58
    assert best.aspect_ratio_error <= 0.24
    assert best.center_px == pytest.approx(target.center_px, abs=10.0)


def test_plain_white_wall_is_not_a_candidate():
    assert candidates_for(np.full((720, 960, 3), 220, np.uint8)) == ()


def test_small_white_distractor_is_rejected():
    image = np.full((720, 960, 3), 35, np.uint8)
    cv2.rectangle(image, (20, 20), (55, 70), (230, 230, 230), -1)
    assert candidates_for(image) == ()


def test_roi_coordinates_are_reported_in_full_image_space():
    target = render_ring_target(board_center=(650.0, 360.0))
    normalized = normalize_frame(target.image, ImageNormalizationConfig())
    candidates = find_white_board_candidates(
        normalized,
        WhiteBoardConfig(),
        roi_xyxy=(350, 40, 950, 700),
    )
    best = max(candidates, key=lambda item: item.geometry_score)
    assert best.center_px == pytest.approx(target.center_px, abs=10.0)
