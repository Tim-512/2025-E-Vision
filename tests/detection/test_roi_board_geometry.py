from __future__ import annotations

import cv2
import numpy as np
import pytest

import ev_vision.detection.roi_board_geometry as roi_geometry
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


def test_order_corners_handles_rolled_perspective_quadrilateral() -> None:
    points = np.asarray(
        [(100, 30), (300, 80), (260, 300), (0, 100)],
        np.float32,
    )

    assert np.asarray(order_corners(points)) == pytest.approx(
        np.asarray(((100.0, 30.0), (300.0, 80.0), (260.0, 300.0), (0.0, 100.0)))
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


def test_refine_accepts_board_when_model_box_touches_image_boundary() -> None:
    expected = synthetic_cluttered_board()

    result = RoiBoardGeometry().refine(
        expected.image,
        model_box=(0.0, 55.0, 455.0, 420.0),
    )

    assert result.accepted is True
    assert result.failure_reason is None
    assert np.allclose(result.corners_px, expected.corners, atol=8.0)


def test_candidate_touching_internal_roi_boundary_is_not_truncated() -> None:
    detector = RoiBoardGeometry()
    expected = synthetic_cluttered_board()
    corners = expected.corners - np.asarray((198.0, 70.0), np.float32)
    gray = detector._as_gray(expected.image[70:406, 198:445])
    primary_edges = np.full_like(gray, 255)

    result = detector._evaluate_candidate(
        corners,
        gray=gray,
        primary_edges=primary_edges,
        model_area=247.0 * 336.0,
        roi_size=(247, 336),
        roi_touches_image_boundary=(False, False, False, False),
    )

    assert getattr(result, "reason", None) is not GeometryFailure.TRUNCATED_QUADRILATERAL

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


@pytest.mark.parametrize("thickness", (18, 20, 22))
def test_refine_rejects_plain_dark_hollow_rectangle(thickness: int) -> None:
    image = np.full((570, 760, 3), 180, np.uint8)
    cv2.rectangle(image, (160, 100), (600, 480), (20, 20, 20), thickness)

    result = RoiBoardGeometry().refine(
        image,
        model_box=(140.0, 80.0, 620.0, 500.0),
    )

    assert result.accepted is False
    assert result.failure_reason is GeometryFailure.LOW_INTERNAL_STRUCTURE


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


def test_representative_rejection_keeps_reason_and_scores_together() -> None:
    low_structure = roi_geometry._RejectedCandidate(
        np.zeros((4, 2), np.float32),
        GeometryFailure.LOW_INTERNAL_STRUCTURE,
        geometry_score=0.70,
        edge_support_score=0.80,
        structure_score=0.20,
    )
    stronger_early_rejection = roi_geometry._RejectedCandidate(
        np.ones((4, 2), np.float32),
        GeometryFailure.LOW_EDGE_SUPPORT,
        geometry_score=0.99,
        edge_support_score=0.44,
        structure_score=0.0,
    )

    representative = RoiBoardGeometry._representative_rejected(
        (stronger_early_rejection, low_structure),
        saw_contour=True,
        saw_four_points=True,
    )

    assert representative is low_structure
    assert representative.reason is GeometryFailure.LOW_INTERNAL_STRUCTURE
    assert representative.geometry_score == pytest.approx(0.70)
    assert representative.edge_support_score == pytest.approx(0.80)
    assert representative.structure_score == pytest.approx(0.20)


def test_refine_rejects_invalid_roi_without_raising() -> None:
    expected = synthetic_cluttered_board()

    result = RoiBoardGeometry().refine(
        expected.image,
        model_box=(40.0, 50.0, 40.0, 80.0),
    )

    assert result.accepted is False
    assert result.failure_reason is GeometryFailure.INVALID_ROI
    assert result.roi_xyxy_px == (0, 0, 0, 0)


def test_refine_skips_bgr_debug_copy_when_debug_is_disabled(monkeypatch) -> None:
    expected = synthetic_cluttered_board()

    def fail_if_called(image: np.ndarray) -> np.ndarray:
        pytest.fail("_as_bgr should only be called for requested debug images")

    monkeypatch.setattr(RoiBoardGeometry, "_as_bgr", staticmethod(fail_if_called))

    result = RoiBoardGeometry().refine(
        expected.image,
        model_box=expected.model_box,
        include_debug=False,
    )

    assert result.accepted is True
    assert result.debug is None


class _ControlledRefinementGeometry(RoiBoardGeometry):
    refinement_offset = np.asarray(
        ((3.0, 3.0), (-3.0, 3.0), (-3.0, -3.0), (3.0, -3.0)),
        np.float32,
    )

    def _subpixel_refine(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        return corners + self.refinement_offset


def test_refine_rechecks_scores_after_subpixel_refinement() -> None:
    expected = synthetic_cluttered_board()
    detector = _ControlledRefinementGeometry()

    result = detector.refine(expected.image, model_box=expected.model_box)

    assert result.accepted is True
    assert result.corners_px is not None
    x0, y0, x1, y1 = result.roi_xyxy_px
    refined_local = np.asarray(result.corners_px, np.float32) - np.asarray(
        (x0, y0), np.float32
    )
    gray = detector._as_gray(expected.image[y0:y1, x0:x1])
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    primary_edges = cv2.Canny(blurred, detector.canny_low, detector.canny_high)
    model_width = expected.model_box[2] - expected.model_box[0]
    model_height = expected.model_box[3] - expected.model_box[1]
    reevaluated = detector._evaluate_candidate(
        refined_local,
        gray=gray,
        primary_edges=primary_edges,
        model_area=model_width * model_height,
        roi_size=(x1 - x0, y1 - y0),
        roi_touches_image_boundary=(
            x0 == 0,
            y0 == 0,
            x1 == expected.image.shape[1],
            y1 == expected.image.shape[0],
        ),
    )

    assert isinstance(reevaluated, roi_geometry._Candidate)
    assert result.geometry_score == pytest.approx(reevaluated.geometry_score)
    assert result.edge_support_score == pytest.approx(reevaluated.edge_support_score)
    assert result.structure_score == pytest.approx(reevaluated.structure_score)


@pytest.mark.parametrize(
    "refined_corners",
    [
        np.asarray(
            ((-500.0, -500.0), (500.0, -500.0), (500.0, 500.0), (-500.0, 500.0)),
            np.float32,
        ),
        np.zeros((4, 2), np.float32),
    ],
    ids=("out_of_bounds", "degenerate"),
)
def test_refine_rejects_unsafe_subpixel_result(refined_corners: np.ndarray) -> None:
    expected = synthetic_cluttered_board()

    class UnsafeRefinementGeometry(RoiBoardGeometry):
        def _subpixel_refine(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
            return refined_corners.copy()

    result = UnsafeRefinementGeometry().refine(
        expected.image,
        model_box=expected.model_box,
    )

    assert result.accepted is False
    assert result.corners_px is None
    assert result.failure_reason is GeometryFailure.NO_VALID_QUADRILATERAL


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
