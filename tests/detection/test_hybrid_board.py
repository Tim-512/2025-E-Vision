from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from ev_vision.config import CandidateScoringConfig, DetectionConfig
from ev_vision.detection.failures import DetectionFailure
from ev_vision.detection.hybrid_board import HybridBoardDetector
from ev_vision.detection.roi_board_geometry import GeometryFailure, GeometryResult
from ev_vision.detection.yolo_board import BoardSearchResult


class FakeModel:
    def __init__(
        self,
        candidates: tuple[BoardSearchResult, ...] | list[BoardSearchResult],
        *,
        backend: object | None = object(),
    ) -> None:
        self.candidates = tuple(candidates)
        self.backend = backend
        self.calls = 0

    def detect_candidates(self, image: np.ndarray) -> tuple[BoardSearchResult, ...]:
        self.calls += 1
        return self.candidates


class RaisingModel:
    backend = object()

    def detect_candidates(self, image: np.ndarray) -> tuple[BoardSearchResult, ...]:
        raise RuntimeError("model inference failed")


class FakeGeometry:
    def __init__(self, results: dict[int, GeometryResult]) -> None:
        self.results = results
        self.calls: list[tuple[tuple[float, float, float, float], bool]] = []

    def refine(
        self,
        image: np.ndarray,
        model_box: tuple[float, float, float, float],
        *,
        include_debug: bool = False,
    ) -> GeometryResult:
        self.calls.append((model_box, include_debug))
        return self.results[len(self.calls) - 1]


@dataclass
class FakeTracker:
    scores: dict[tuple[float, float], float]
    temporal_calls: list[tuple[float, float]]
    update_calls: list[object]

    def __init__(self, scores: dict[tuple[float, float], float] | None = None) -> None:
        self.scores = scores or {}
        self.temporal_calls = []
        self.update_calls = []

    def temporal_score(self, center_px: tuple[float, float]) -> float:
        self.temporal_calls.append(center_px)
        return self.scores.get(center_px, 0.0)

    def update(self, observation: object) -> None:
        self.update_calls.append(observation)


class IncrementingClock:
    def __init__(self, *values: int) -> None:
        self.values = iter(values)

    def __call__(self) -> int:
        return next(self.values)


def frame_image() -> np.ndarray:
    return np.zeros((240, 320, 3), np.uint8)


def candidate(confidence: float, *, x: float = 10.0) -> BoardSearchResult:
    return BoardSearchResult((x, 20.0, x + 100.0, 180.0), confidence)


def accepted_geometry(
    geometry_score: float,
    edge_support_score: float,
    structure_score: float,
    *,
    x: float = 10.0,
    debug: object | None = None,
) -> GeometryResult:
    corners = (
        (x, 20.0),
        (x + 100.0, 20.0),
        (x + 100.0, 180.0),
        (x, 180.0),
    )
    return GeometryResult(
        accepted=True,
        corners_px=corners,
        center_px=(x + 50.0, 100.0),
        geometry_score=geometry_score,
        edge_support_score=edge_support_score,
        structure_score=structure_score,
        roi_xyxy_px=(int(x), 20, int(x + 100), 180),
        failure_reason=None,
        debug=debug,
    )


def rejected_geometry(reason: GeometryFailure) -> GeometryResult:
    return GeometryResult(
        accepted=False,
        corners_px=None,
        center_px=None,
        geometry_score=0.0,
        edge_support_score=0.0,
        structure_score=0.0,
        roi_xyxy_px=(0, 0, 0, 0),
        failure_reason=reason,
    )


def test_no_model_candidate_returns_safe_invalid_result() -> None:
    detector = HybridBoardDetector(model=FakeModel([]), geometry=FakeGeometry({}))

    result = detector.detect(frame_image(), captured_ns=1_000, source_sequence=5)

    assert result.detected is False
    assert result.target_valid is False
    assert result.tracking_state == "SEARCHING"
    assert result.candidate_count == 0
    assert result.failure_reason is DetectionFailure.NO_MODEL_CANDIDATE
    assert result.corners_px is None


@pytest.mark.parametrize(
    ("geometry_failure", "detection_failure"),
    [
        (GeometryFailure.INVALID_ROI, DetectionFailure.NO_VALID_QUADRILATERAL),
        (
            GeometryFailure.NO_VALID_QUADRILATERAL,
            DetectionFailure.NO_VALID_QUADRILATERAL,
        ),
        (
            GeometryFailure.TRUNCATED_QUADRILATERAL,
            DetectionFailure.NO_VALID_QUADRILATERAL,
        ),
        (
            GeometryFailure.UNDERSIZED_QUADRILATERAL,
            DetectionFailure.NO_VALID_QUADRILATERAL,
        ),
        (GeometryFailure.INVALID_ASPECT_RATIO, DetectionFailure.INVALID_ASPECT_RATIO),
        (GeometryFailure.LOW_EDGE_SUPPORT, DetectionFailure.LOW_EDGE_SUPPORT),
        (GeometryFailure.LOW_INTERNAL_STRUCTURE, DetectionFailure.LOW_INTERNAL_STRUCTURE),
        (GeometryFailure.AMBIGUOUS_GEOMETRY, DetectionFailure.AMBIGUOUS_CANDIDATES),
        (GeometryFailure.CORNER_ORDER_FAILED, DetectionFailure.NO_VALID_QUADRILATERAL),
    ],
)
def test_geometry_rejection_maps_to_stable_detection_failure(
    geometry_failure: GeometryFailure,
    detection_failure: DetectionFailure,
) -> None:
    detector = HybridBoardDetector(
        model=FakeModel([candidate(0.9)]),
        geometry=FakeGeometry({0: rejected_geometry(geometry_failure)}),
    )

    result = detector.detect(frame_image(), captured_ns=1_000, source_sequence=6)

    assert result.detected is False
    assert result.candidate_count == 1
    assert len(result.candidates) == 1
    assert result.failure_reason is detection_failure


def test_unique_candidate_combines_scores_and_returns_corners() -> None:
    model = FakeModel([candidate(0.80), candidate(0.60, x=190.0)])
    geometry = FakeGeometry(
        {
            0: accepted_geometry(0.90, 0.80, 0.70),
            1: rejected_geometry(GeometryFailure.NO_VALID_QUADRILATERAL),
        }
    )
    detector = HybridBoardDetector(model=model, geometry=geometry)

    result = detector.detect(frame_image(), captured_ns=1_000, source_sequence=7)

    assert result.detected is True
    assert result.target_valid is False
    assert result.tracking_state == "SEARCHING"
    assert result.candidate_count == 2
    assert result.model_confidence == pytest.approx(0.80)
    assert result.geometry_score == pytest.approx(0.90)
    assert result.edge_support_score == pytest.approx(0.80)
    assert result.structure_score == pytest.approx(0.70)
    assert result.combined_score == pytest.approx(
        0.45 * 0.80 + 0.30 * 0.90 + 0.15 * 0.70
    )
    assert result.corners_px == geometry.results[0].corners_px
    assert result.failure_reason is None


def test_accepted_candidates_are_ranked_by_combined_score_not_model_confidence() -> None:
    detector = HybridBoardDetector(
        model=FakeModel([candidate(0.95), candidate(0.70, x=190.0)]),
        geometry=FakeGeometry(
            {
                0: accepted_geometry(0.40, 0.80, 0.20),
                1: accepted_geometry(0.95, 0.80, 0.90, x=190.0),
            }
        ),
    )

    result = detector.detect(frame_image(), captured_ns=1_000, source_sequence=8)

    assert result.detected is True
    assert result.model_confidence == pytest.approx(0.70)
    assert [item.model.confidence for item in result.candidates] == [0.70, 0.95]
    assert result.combined_score > result.candidates[1].combined_score


def test_close_candidate_scores_are_ambiguous() -> None:
    scoring = CandidateScoringConfig(
        model_weight=1.0,
        geometry_weight=0.0,
        structure_weight=0.0,
        temporal_weight=0.0,
        ambiguity_margin=0.08,
    )
    detector = HybridBoardDetector(
        model=FakeModel([candidate(0.81), candidate(0.75, x=190.0)]),
        geometry=FakeGeometry(
            {
                0: accepted_geometry(1.0, 1.0, 1.0),
                1: accepted_geometry(1.0, 1.0, 1.0, x=190.0),
            }
        ),
        config=DetectionConfig(candidate_scoring=scoring),
    )

    result = detector.detect(frame_image(), captured_ns=1_000, source_sequence=9)

    assert result.detected is False
    assert result.failure_reason is DetectionFailure.AMBIGUOUS_CANDIDATES
    assert result.corners_px is None
    assert result.center_px is None
    assert [item.combined_score for item in result.candidates] == pytest.approx(
        [0.81, 0.75]
    )


def test_model_exception_returns_safe_invalid_result() -> None:
    detector = HybridBoardDetector(model=RaisingModel(), geometry=FakeGeometry({}))

    result = detector.detect(frame_image(), captured_ns=1_000, source_sequence=10)

    assert result.detected is False
    assert result.failure_reason is DetectionFailure.MODEL_ERROR
    assert result.target_valid is False
    assert result.candidate_count == 0


def test_missing_model_backend_returns_model_unavailable_without_geometry() -> None:
    geometry = FakeGeometry({})
    detector = HybridBoardDetector(model=FakeModel([], backend=None), geometry=geometry)

    result = detector.detect(frame_image(), captured_ns=1_000, source_sequence=11)

    assert result.detected is False
    assert result.failure_reason is DetectionFailure.MODEL_UNAVAILABLE
    assert result.model_state == "UNAVAILABLE"
    assert result.candidate_count == 0
    assert geometry.calls == []


def test_include_debug_is_lazy_and_forwards_only_when_requested() -> None:
    debug = type(
        "Debug",
        (),
        {
            "roi_bgr": np.zeros((2, 2, 3), np.uint8),
            "edges": np.zeros((2, 2), np.uint8),
            "candidates_bgr": np.ones((2, 2, 3), np.uint8),
        },
    )()
    geometry = FakeGeometry(
        {
            0: accepted_geometry(0.9, 0.8, 0.7),
            1: accepted_geometry(0.9, 0.8, 0.7, debug=debug),
        }
    )
    detector = HybridBoardDetector(model=FakeModel([candidate(0.9)]), geometry=geometry)

    without_debug = detector.detect(
        frame_image(), captured_ns=1_000, source_sequence=12, include_debug=False
    )
    with_debug = detector.detect(
        frame_image(), captured_ns=2_000, source_sequence=13, include_debug=True
    )

    assert geometry.calls[0][1] is False
    assert without_debug.debug_images == {}
    assert geometry.calls[1][1] is True
    assert set(with_debug.debug_images) == {
        "candidate_0_roi",
        "candidate_0_edges",
        "candidate_0_geometry",
    }


def test_update_tracker_false_has_no_tracker_side_effects_but_scores_candidate() -> None:
    tracker = FakeTracker({(60.0, 100.0): 0.75})
    detector = HybridBoardDetector(
        model=FakeModel([candidate(0.80)]),
        geometry=FakeGeometry({0: accepted_geometry(0.90, 0.80, 0.70)}),
        tracker=tracker,
    )

    result = detector.detect(
        frame_image(),
        captured_ns=1_000,
        source_sequence=14,
        update_tracker=False,
    )

    assert result.detected is True
    assert result.temporal_score == pytest.approx(0.75)
    assert tracker.temporal_calls == [(60.0, 100.0)]
    assert tracker.update_calls == []


def test_injectable_clock_separates_model_geometry_and_total_timings() -> None:
    detector = HybridBoardDetector(
        model=FakeModel([candidate(0.8)]),
        geometry=FakeGeometry({0: accepted_geometry(0.9, 0.8, 0.7)}),
        clock_ns=IncrementingClock(100, 1_100, 1_600, 3_600),
    )

    result = detector.detect(frame_image(), captured_ns=99_000, source_sequence=15)

    assert result.timestamp_ns == 99_000
    assert result.inference_ms == pytest.approx(0.001)
    assert result.geometry_ms == pytest.approx(0.002)
    assert result.total_ms == pytest.approx(0.0035)
