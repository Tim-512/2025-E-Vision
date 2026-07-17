from __future__ import annotations

import pytest

from ev_vision.detection.contracts import ObservationSource
from ev_vision.detection.failures import DetectionFailure
from ev_vision.detection.partial_board import (
    BoardHistory,
    fuse_partial_observation,
    predicted_roi,
)


def history():
    return BoardHistory(
        center_px=(320.0, 240.0),
        velocity_px_s=(100.0, 0.0),
        scale_px_per_mm=2.0,
        corners_px=(
            (110.0, -57.0),
            (530.0, -57.0),
            (530.0, 537.0),
            (110.0, 537.0),
        ),
        timestamp_ns=1_000_000_000,
    )


def test_multiple_arcs_and_white_region_are_fused():
    result = fuse_partial_observation(
        history=history(),
        timestamp_ns=1_020_000_000,
        image_size=(640, 480),
        arc_center_px=(323.0, 241.0),
        arc_count=3,
        arc_confidence=0.84,
        white_center_px=(328.0, 239.0),
        white_confidence=0.70,
    )
    assert result.source == ObservationSource.FUSED_PARTIAL
    assert result.center_px == pytest.approx((324.9, 240.6), abs=2.0)
    assert result.partially_outside is True


def test_partial_requires_confirmed_history():
    result = fuse_partial_observation(
        history=None,
        timestamp_ns=1,
        image_size=(640, 480),
        arc_center_px=(320.0, 240.0),
        arc_count=3,
        arc_confidence=0.9,
        white_center_px=None,
        white_confidence=0.0,
    )
    assert result.valid is False
    assert result.failure_reason == DetectionFailure.PARTIAL_HISTORY_REQUIRED


def test_single_arc_is_strictly_gated():
    result = fuse_partial_observation(
        history=history(),
        timestamp_ns=1_020_000_000,
        image_size=(640, 480),
        arc_center_px=(520.0, 420.0),
        arc_count=1,
        arc_confidence=0.95,
        observed_scale_px_per_mm=4.0,
        white_center_px=None,
        white_confidence=0.0,
    )
    assert result.valid is False


def test_multi_arc_without_white_region_keeps_concentric_source():
    result = fuse_partial_observation(
        history=history(),
        timestamp_ns=1_020_000_000,
        image_size=(640, 480),
        arc_center_px=(324.0, 240.0),
        arc_count=3,
        arc_confidence=0.9,
        white_center_px=None,
        white_confidence=0.0,
    )
    assert result.valid is True
    assert result.source == ObservationSource.CONCENTRIC_ARCS


@pytest.mark.parametrize(
    "corners, expected",
    [
        (((-100.0, 100.0), (200.0, 100.0), (200.0, 350.0), (-100.0, 350.0)), (0, 50, 260, 400)),
        (((440.0, 100.0), (740.0, 100.0), (740.0, 350.0), (440.0, 350.0)), (380, 50, 640, 400)),
        (((100.0, -80.0), (400.0, -80.0), (400.0, 170.0), (100.0, 170.0)), (40, 0, 460, 220)),
        (((100.0, 310.0), (400.0, 310.0), (400.0, 560.0), (100.0, 560.0)), (40, 260, 460, 480)),
    ],
)
def test_predicted_roi_clips_all_four_sides(corners, expected):
    item = BoardHistory(
        center_px=(250.0, 225.0),
        velocity_px_s=(0.0, 0.0),
        scale_px_per_mm=1.0,
        corners_px=corners,
        timestamp_ns=1,
    )
    assert predicted_roi(item, timestamp_ns=1, image_size=(640, 480)) == expected

