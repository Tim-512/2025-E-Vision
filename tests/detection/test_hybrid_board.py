from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from ev_vision.config import (
    BoardTrackingConfig,
    CandidateScoringConfig,
    DetectionConfig,
)
from ev_vision.detection.failures import DetectionFailure
from ev_vision.detection.hybrid_board import HybridBoardDetector
from ev_vision.detection.roi_board_geometry import GeometryFailure, GeometryResult
from ev_vision.detection.yolo_board import (
    BoardSearchResult,
    RawDetection,
    YoloBoardDetector,
)
from ev_vision.tracking.board_tracker import (
    BoardTracker,
    TrackObservation,
    TrackedBoardResult,
    TrackingState,
)


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
    now_calls: list[int | None]
    latest: TrackedBoardResult

    def __init__(
        self,
        scores: dict[tuple[float, float], float] | None = None,
        *,
        state: TrackingState = TrackingState.SEARCHING,
        target_valid: bool = False,
    ) -> None:
        self.scores = scores or {}
        self.temporal_calls = []
        self.update_calls = []
        self.now_calls = []
        self.latest = TrackedBoardResult(
            state=state,
            target_valid=target_valid,
            observation=None,
            predicted_center_px=None,
            confirmation_count=3 if state is TrackingState.TRACKING else 0,
            miss_count=0,
            failure_reason=None,
        )

    def temporal_score(self, center_px: tuple[float, float]) -> float:
        self.temporal_calls.append(center_px)
        return self.scores.get(center_px, 0.0)

    def update(
        self,
        observation: object,
        *,
        now_ns: int | None = None,
    ) -> TrackedBoardResult:
        self.update_calls.append(observation)
        self.now_calls.append(now_ns)
        return self.latest


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



def test_temporal_score_participates_before_final_candidate_ranking() -> None:
    scoring = CandidateScoringConfig(
        model_weight=0.35,
        geometry_weight=0.25,
        structure_weight=0.10,
        temporal_weight=0.30,
        ambiguity_margin=0.01,
    )
    tracker = FakeTracker(
        {
            (60.0, 100.0): 0.0,
            (240.0, 100.0): 1.0,
        }
    )
    detector = HybridBoardDetector(
        model=FakeModel([candidate(0.90), candidate(0.80, x=190.0)]),
        geometry=FakeGeometry(
            {
                0: accepted_geometry(0.90, 0.80, 0.90),
                1: accepted_geometry(0.90, 0.80, 0.90, x=190.0),
            }
        ),
        config=DetectionConfig(candidate_scoring=scoring),
        tracker=tracker,
    )

    result = detector.detect(
        frame_image(),
        captured_ns=1_000,
        source_sequence=81,
        update_tracker=False,
    )

    assert tracker.temporal_calls == [(60.0, 100.0), (240.0, 100.0)]
    assert result.center_px == (240.0, 100.0)
    assert result.model_confidence == pytest.approx(0.80)
    assert result.temporal_score == pytest.approx(1.0)

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
        clock_ns=IncrementingClock(100, 1_100, 1_600, 3_600, 4_100),
    )

    result = detector.detect(frame_image(), captured_ns=99_000, source_sequence=15)

    assert result.timestamp_ns == 99_000
    assert result.inference_ms == pytest.approx(0.001)
    assert result.geometry_ms == pytest.approx(0.002)
    assert result.total_ms == pytest.approx(0.004)


class NoBackendModel:
    def __init__(self, candidates: list[BoardSearchResult]) -> None:
        self.candidates = tuple(candidates)
        self.calls = 0

    def detect_candidates(self, image: np.ndarray) -> tuple[BoardSearchResult, ...]:
        self.calls += 1
        return self.candidates


class RaisingGeometry:
    def __init__(self) -> None:
        self.calls = 0

    def refine(
        self,
        image: np.ndarray,
        model_box: tuple[float, float, float, float],
        *,
        include_debug: bool = False,
    ) -> GeometryResult:
        self.calls += 1
        raise RuntimeError("geometry refinement failed")


class RaisingTracker(FakeTracker):
    def update(
        self,
        observation: object,
        *,
        now_ns: int | None = None,
    ) -> TrackedBoardResult:
        self.update_calls.append(observation)
        self.now_calls.append(now_ns)
        raise RuntimeError("tracker failed")


def tracking_config() -> DetectionConfig:
    return DetectionConfig(
        tracking=BoardTrackingConfig(
            confirm_frames=3,
            predict_frames=2,
            lost_frames=3,
            max_center_jump_px=160.0,
            max_result_age_ms=100.0,
        )
    )


def tracker_hit(sequence: int, timestamp_ns: int, x: float) -> TrackObservation:
    geometry = accepted_geometry(0.9, 0.8, 0.7, x=x)
    return TrackObservation(
        timestamp_ns=timestamp_ns,
        source_sequence=sequence,
        detected=True,
        center_px=geometry.center_px,
        corners_px=geometry.corners_px,
        failure_reason=None,
    )


def prime_tracker(tracker: BoardTracker) -> None:
    tracker.update(tracker_hit(1, 0, 10.0))
    tracker.update(tracker_hit(2, 20_000_000, 12.0))
    tracker.update(tracker_hit(3, 40_000_000, 14.0))
    assert tracker.latest.state is TrackingState.TRACKING


def test_three_unique_hybrid_hits_confirm_tracking_and_authorize_target() -> None:
    tracker = BoardTracker(tracking_config().tracking)
    detector = HybridBoardDetector(
        model=FakeModel([candidate(0.9)]),
        geometry=FakeGeometry(
            {
                0: accepted_geometry(0.9, 0.8, 0.7, x=10.0),
                1: accepted_geometry(0.9, 0.8, 0.7, x=12.0),
                2: accepted_geometry(0.9, 0.8, 0.7, x=14.0),
            }
        ),
        tracker=tracker,
        clock_ns=IncrementingClock(
            0, 1, 2, 3, 4, 5,
            20_000_000, 20_000_001, 20_000_002, 20_000_003, 20_000_004, 20_000_005,
            40_000_000, 40_000_001, 40_000_002, 40_000_003, 40_000_004, 40_000_005,
        ),
    )

    results = [
        detector.detect(frame_image(), captured_ns=0, source_sequence=1),
        detector.detect(frame_image(), captured_ns=20_000_000, source_sequence=2),
        detector.detect(frame_image(), captured_ns=40_000_000, source_sequence=3),
    ]

    assert [result.tracking_state for result in results] == [
        "CONFIRMING",
        "CONFIRMING",
        "TRACKING",
    ]
    assert [result.target_valid for result in results] == [False, False, True]


def test_hybrid_miss_enters_prediction_but_never_authorizes_target() -> None:
    tracker = BoardTracker(tracking_config().tracking)
    prime_tracker(tracker)
    detector = HybridBoardDetector(
        model=NoBackendModel([]),
        geometry=FakeGeometry({}),
        tracker=tracker,
        clock_ns=IncrementingClock(
            60_000_000, 60_000_001, 60_000_002, 60_000_003
        ),
    )

    result = detector.detect(
        frame_image(), captured_ns=60_000_000, source_sequence=4
    )

    assert result.tracking_state == "PREDICTING"
    assert result.target_valid is False
    assert result.failure_reason is DetectionFailure.NO_MODEL_CANDIDATE


def test_hybrid_jump_hit_fails_safe_without_authorized_coordinates() -> None:
    tracker = BoardTracker(tracking_config().tracking)
    prime_tracker(tracker)
    detector = HybridBoardDetector(
        model=FakeModel([candidate(0.9, x=350.0)]),
        geometry=FakeGeometry(
            {0: accepted_geometry(0.9, 0.8, 0.7, x=350.0)}
        ),
        tracker=tracker,
        clock_ns=IncrementingClock(
            60_000_000,
            60_000_001,
            60_000_002,
            60_000_003,
            60_000_004,
            60_000_005,
        ),
    )

    result = detector.detect(
        frame_image(), captured_ns=60_000_000, source_sequence=4
    )

    assert result.failure_reason is DetectionFailure.EXCESSIVE_POSITION_JUMP
    assert result.target_valid is False
    assert result.center_px is None
    assert result.corners_px is None


def test_hybrid_stale_hit_fails_safe_without_authorized_coordinates() -> None:
    tracker = BoardTracker(tracking_config().tracking)
    prime_tracker(tracker)
    detector = HybridBoardDetector(
        model=FakeModel([candidate(0.9, x=16.0)]),
        geometry=FakeGeometry({0: accepted_geometry(0.9, 0.8, 0.7, x=16.0)}),
        tracker=tracker,
        clock_ns=IncrementingClock(
            200_000_001,
            200_000_002,
            200_000_003,
            200_000_004,
            200_000_005,
            200_000_006,
        ),
    )

    result = detector.detect(
        frame_image(), captured_ns=60_000_000, source_sequence=4
    )

    assert result.failure_reason is DetectionFailure.STALE_FRAME
    assert result.target_valid is False
    assert result.center_px is None
    assert result.corners_px is None


def test_update_tracker_false_reports_state_but_never_authorizes_or_mutates() -> None:
    tracker = BoardTracker(tracking_config().tracking)
    prime_tracker(tracker)
    before = tracker.latest
    detector = HybridBoardDetector(
        model=FakeModel([candidate(0.9, x=16.0)]),
        geometry=FakeGeometry({0: accepted_geometry(0.9, 0.8, 0.7, x=16.0)}),
        tracker=tracker,
    )

    result = detector.detect(
        frame_image(),
        captured_ns=60_000_000,
        source_sequence=4,
        update_tracker=False,
    )

    assert result.tracking_state == "TRACKING"
    assert result.target_valid is False
    assert tracker.latest == before


def test_geometry_exception_returns_stable_safe_invalid_result() -> None:
    detector = HybridBoardDetector(
        model=FakeModel([candidate(0.9)]),
        geometry=RaisingGeometry(),
        clock_ns=IncrementingClock(0, 1_000, 2_000, 5_000, 8_000),
    )

    result = detector.detect(frame_image(), captured_ns=0, source_sequence=20)

    assert result.detected is False
    assert result.target_valid is False
    assert result.failure_reason is DetectionFailure.MODEL_ERROR
    assert result.center_px is None
    assert result.corners_px is None
    assert result.geometry_ms == pytest.approx(0.003)
    assert result.total_ms == pytest.approx(0.008)


def test_tracker_exception_cannot_authorize_or_publish_selected_coordinates() -> None:
    tracker = RaisingTracker(state=TrackingState.TRACKING, target_valid=True)
    detector = HybridBoardDetector(
        model=FakeModel([candidate(0.9)]),
        geometry=FakeGeometry({0: accepted_geometry(0.9, 0.8, 0.7)}),
        tracker=tracker,
        clock_ns=IncrementingClock(0, 1_000, 2_000, 3_000, 4_000, 8_000),
    )

    result = detector.detect(frame_image(), captured_ns=0, source_sequence=21)

    assert result.detected is False
    assert result.target_valid is False
    assert result.failure_reason is DetectionFailure.MODEL_ERROR
    assert result.center_px is None
    assert result.corners_px is None
    assert len(tracker.update_calls) == 1


def test_model_without_backend_attribute_is_available_by_contract() -> None:
    model = NoBackendModel([candidate(0.9)])
    detector = HybridBoardDetector(
        model=model,
        geometry=FakeGeometry({0: accepted_geometry(0.9, 0.8, 0.7)}),
    )

    result = detector.detect(frame_image(), captured_ns=1_000, source_sequence=22)

    assert model.calls == 1
    assert result.detected is True
    assert result.failure_reason is None


def test_explicit_unavailable_state_short_circuits_model_without_backend_attribute() -> None:
    model = NoBackendModel([candidate(0.9)])
    geometry = FakeGeometry({})
    detector = HybridBoardDetector(
        model=model,
        geometry=geometry,
        model_state="UNAVAILABLE",
        clock_ns=IncrementingClock(0, 5_000),
    )

    result = detector.detect(frame_image(), captured_ns=0, source_sequence=23)

    assert model.calls == 0
    assert geometry.calls == []
    assert result.failure_reason is DetectionFailure.MODEL_UNAVAILABLE
    assert result.total_ms == pytest.approx(0.005)


def test_total_timing_includes_tracker_and_dto_postprocessing() -> None:
    tracker = FakeTracker()
    detector = HybridBoardDetector(
        model=FakeModel([candidate(0.8)]),
        geometry=FakeGeometry({0: accepted_geometry(0.9, 0.8, 0.7)}),
        tracker=tracker,
        clock_ns=IncrementingClock(0, 1_000, 2_000, 5_000, 7_000, 11_000),
    )

    result = detector.detect(frame_image(), captured_ns=0, source_sequence=24)

    assert result.inference_ms == pytest.approx(0.001)
    assert result.geometry_ms == pytest.approx(0.003)
    assert tracker.now_calls == [7_000]
    assert result.total_ms == pytest.approx(0.011)


class HybridInferenceBackend:
    input_size = (640, 640)

    def infer(self, tensor: np.ndarray) -> tuple[RawDetection, ...]:
        return (RawDetection((20.0, 20.0, 220.0, 220.0), 0.9, 0),)


def test_yolo_board_detector_backend_contract_remains_available() -> None:
    backend = HybridInferenceBackend()
    model = YoloBoardDetector(backend, confidence_threshold=0.5)
    detector = HybridBoardDetector(
        model=model,
        geometry=FakeGeometry({0: accepted_geometry(0.9, 0.8, 0.7)}),
    )

    result = detector.detect(frame_image(), captured_ns=1_000, source_sequence=25)

    assert model.backend is backend
    assert result.model_state == "READY"
    assert result.model_backend == "HybridInferenceBackend"
    assert result.detected is True
