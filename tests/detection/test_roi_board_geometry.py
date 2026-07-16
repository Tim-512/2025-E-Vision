from __future__ import annotations

import numpy as np
import pytest

from ev_vision.detection.roi_board_geometry import (
    GeometryFailure,
    RoiBoardGeometry,
    expand_roi,
    order_corners,
)
from tests.fixtures.synthetic_board import (
    geometry_failure_fixture,
    synthetic_cluttered_board,
)


def test_expand_roi_clips_to_image_and_maps_points_back() -> None:
    roi = expand_roi((5.0, 10.0, 95.0, 90.0), (100, 80), padding_fraction=0.20)

    assert roi.xyxy_px == (0, 0, 100, 80)
    assert roi.to_source(((1.0, 2.0), (50.0, 40.0))) == (
        (1.0, 2.0),
        (50.0, 40.0),
    )


def test_expand_roi_uses_floor_start_ceil_end_and_half_open_bounds() -> None:
    roi = expand_roi((20.5, 30.5, 80.5, 70.5), (120, 100), padding_fraction=0.10)

    assert roi.xyxy_px == (14, 26, 87, 75)
    assert roi.to_source(((0.0, 0.0), (72.0, 48.0))) == (
        (14.0, 26.0),
        (86.0, 74.0),
    )


@pytest.mark.parametrize(
    "box",
    [
        (10.0, 10.0, 10.0, 30.0),
        (20.0, 20.0, 10.0, 30.0),
        (float("nan"), 0.0, 5.0, 5.0),
    ],
)
def test_expand_roi_rejects_invalid_boxes(box) -> None:
    with pytest.raises(ValueError, match="ROI"):
        expand_roi(box, (100, 80), padding_fraction=0.10)


def test_order_corners_is_tl_tr_br_bl() -> None:
    points = np.asarray([[80, 70], [20, 10], [15, 75], [85, 15]], np.float32)

    assert np.asarray(order_corners(points)) == pytest.approx(
        np.asarray(((20.0, 10.0), (85.0, 15.0), (80.0, 70.0), (15.0, 75.0)))
    )


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [1, 0], [1, 0], [0, 1]],
        [[0, 0], [1, 0], [2, 0], [3, 0]],
        [[0, 0], [1, 0], [1, 1], [float("inf"), 1]],
    ],
)
def test_order_corners_rejects_invalid_quadrilaterals(points) -> None:
    with pytest.raises(ValueError):
        order_corners(np.asarray(points, np.float32))


def test_refine_accepts_synthetic_board_inside_cluttered_roi() -> None:
    expected = synthetic_cluttered_board()

    result = RoiBoardGeometry().refine(expected.image, model_box=expected.model_box)

    assert result.accepted is True
    assert result.failure_reason is None
    assert result.geometry_score >= 0.55
    assert result.edge_support_score >= 0.45
    assert result.structure_score >= 0.35
    assert result.center_px == pytest.approx(tuple(expected.corners.mean(axis=0)), abs=6.0)
    assert np.allclose(result.corners_px, expected.corners, atol=8.0)
    assert result.debug is None


@pytest.mark.parametrize(
    ("fixture_name", "reason"),
    [
        ("non_convex", GeometryFailure.NO_VALID_QUADRILATERAL),
        ("truncated", GeometryFailure.TRUNCATED_QUADRILATERAL),
        ("undersized", GeometryFailure.UNDERSIZED_QUADRILATERAL),
        ("weak_edges", GeometryFailure.LOW_EDGE_SUPPORT),
        ("no_internal_frame", GeometryFailure.LOW_INTERNAL_STRUCTURE),
    ],
)
def test_refine_rejects_hard_geometry_failures(
    fixture_name: str,
    reason: GeometryFailure,
) -> None:
    image, model_box = geometry_failure_fixture(fixture_name)

    result = RoiBoardGeometry().refine(image, model_box=model_box)

    assert result.accepted is False
    assert result.failure_reason is reason
    assert result.corners_px is None
    assert result.center_px is None


def test_refine_rejects_invalid_roi_without_raising() -> None:
    expected = synthetic_cluttered_board()

    result = RoiBoardGeometry().refine(
        expected.image,
        model_box=(40.0, 50.0, 40.0, 80.0),
    )

    assert result.accepted is False
    assert result.failure_reason is GeometryFailure.INVALID_ROI
    assert result.roi_xyxy_px == (0, 0, 0, 0)


def test_refine_returns_requested_debug_images() -> None:
    expected = synthetic_cluttered_board()

    result = RoiBoardGeometry().refine(
        expected.image,
        model_box=expected.model_box,
        include_debug=True,
    )

    assert result.accepted is True
    assert result.debug is not None
    x0, y0, x1, y1 = result.roi_xyxy_px
    assert result.debug.roi_bgr.shape == (y1 - y0, x1 - x0, 3)
    assert result.debug.edges.shape == (y1 - y0, x1 - x0)
    assert result.debug.candidates_bgr.shape == result.debug.roi_bgr.shape
    assert result.debug.roi_bgr.dtype == np.uint8
    assert result.debug.edges.dtype == np.uint8
    assert np.count_nonzero(result.debug.edges) > 0
