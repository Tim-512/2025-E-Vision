from __future__ import annotations

import numpy as np
import pytest

from ev_vision.detection.board_geometry import BoardGeometryDetector
from tests.fixtures.synthetic_board import render_board


def corner_errors(actual, expected) -> np.ndarray:
    return np.linalg.norm(np.asarray(actual) - np.asarray(expected), axis=1)


def test_clean_perspective_board_has_subpixel_median_corner_error() -> None:
    sample = render_board()
    observation = BoardGeometryDetector().detect(sample.image, captured_ns=123)

    assert observation is not None
    assert observation.captured_ns == 123
    assert np.median(corner_errors(observation.corners_px, sample.corners)) <= 1.0
    assert observation.confidence >= 0.8
    assert observation.homography_valid


@pytest.mark.parametrize(
    ("kwargs", "maximum_error"),
    [
        ({"blur_sigma": 2.0}, 3.0),
        ({"gradient_strength": 0.35}, 3.0),
        ({"distractors": True}, 2.0),
        ({"blur_sigma": 1.5, "gradient_strength": 0.25, "distractors": True}, 4.0),
    ],
)
def test_detector_tolerates_image_degradation(kwargs, maximum_error: float) -> None:
    sample = render_board(**kwargs)
    observation = BoardGeometryDetector().detect(sample.image, captured_ns=0)

    assert observation is not None
    assert max(corner_errors(observation.corners_px, sample.corners)) <= maximum_error


def test_corners_are_ordered_clockwise_from_top_left() -> None:
    corners = np.array([[190, 120], [1090, 260], [910, 900], [250, 790]], np.float32)
    sample = render_board(corners)
    observation = BoardGeometryDetector().detect(sample.image, captured_ns=0)

    assert observation is not None
    assert corner_errors(observation.corners_px, corners).max() <= 2.0


@pytest.mark.parametrize(
    "corners",
    [
        np.array([[-20, 100], [900, 150], [850, 800], [-30, 760]], np.float32),
        np.array([[300, 300], [700, 302], [705, 320], [295, 318]], np.float32),
    ],
)
def test_clipped_or_implausibly_thin_board_is_rejected(corners) -> None:
    sample = render_board(corners)
    assert BoardGeometryDetector().detect(sample.image, captured_ns=0) is None


def test_blank_image_is_rejected_deterministically() -> None:
    sample = render_board()
    blank = np.full_like(sample.image, 180)
    detector = BoardGeometryDetector()

    assert detector.detect(blank, captured_ns=0) is None
    assert detector.detect(blank, captured_ns=1) is None
