from __future__ import annotations

import math

import pytest

from ev_vision.config import BoardTrackingConfig
from ev_vision.detection.contracts import ObservationSource
from ev_vision.detection.failures import DetectionFailure
from ev_vision.tracking.board_tracker import BoardTracker, TrackObservation, TrackingState
from ev_vision.tracking.predictor import MotionPredictor


def config(**overrides: object) -> BoardTrackingConfig:
    values = {
        "confirm_frames": 3,
        "predict_frames": 2,
        "predict_max_frames": 3,
        "predict_max_ms": 150.0,
        "lost_frames": 4,
        "max_single_arc_frames": 2,
        "max_center_jump_px": 160.0,
        "max_scale_jump_fraction": 0.30,
        "max_velocity_px_s": 5000.0,
        "max_acceleration_px_s2": 30000.0,
        "max_result_age_ms": 100.0,
    }
    values.update(overrides)
    return BoardTrackingConfig(**values)


def full(
    sequence: int,
    timestamp_ns: int,
    *,
    center: tuple[float, float] = (100.0, 100.0),
    scale: float = 1.0,
) -> TrackObservation:
    x, y = center
    return TrackObservation(
        timestamp_ns=timestamp_ns,
        source_sequence=sequence,
        detected=True,
        source=ObservationSource.FULL_BOARD,
        center_px=center,
        corners_px=(
            (x - 50.0, y - 80.0),
            (x + 50.0, y - 80.0),
            (x + 50.0, y + 80.0),
            (x - 50.0, y + 80.0),
        ),
        scale_px_per_mm=scale,
        confidence=0.9,
    )


def partial(
    sequence: int,
    timestamp_ns: int,
    source: ObservationSource,
    *,
    center: tuple[float, float] = (104.0, 100.0),
    scale: float = 1.0,
) -> TrackObservation:
    return TrackObservation(
        timestamp_ns=timestamp_ns,
        source_sequence=sequence,
        detected=True,
        source=source,
        center_px=center,
        scale_px_per_mm=scale,
        confidence=0.7,
    )


def miss(
    sequence: int,
    timestamp_ns: int,
    reason: DetectionFailure = DetectionFailure.NO_WHITE_CANDIDATE,
) -> TrackObservation:
    return TrackObservation(
        timestamp_ns=timestamp_ns,
        source_sequence=sequence,
        detected=False,
        failure_reason=reason,
    )


def confirmed_tracker(cfg: BoardTrackingConfig | None = None) -> BoardTracker:
    tracker = BoardTracker(cfg or config(confirm_frames=2))
    assert tracker.update(full(1, 1_000_000_000, center=(100.0, 100.0))).state == TrackingState.CONFIRMING
    assert tracker.update(full(2, 1_010_000_000, center=(102.0, 100.0))).state == TrackingState.TRACKING
    return tracker


def lost_tracker() -> BoardTracker:
    tracker = confirmed_tracker(config(confirm_frames=2, predict_max_frames=1, predict_max_ms=20.0))
    assert tracker.update(miss(3, 1_020_000_000)).state == TrackingState.PREDICTING
    assert tracker.update(miss(4, 1_040_000_001)).state == TrackingState.LOST
    return tracker


def test_initial_state_is_safe_and_empty() -> None:
    tracker = BoardTracker(config())
    assert tracker.latest.state == TrackingState.SEARCHING
    assert tracker.latest.target_valid is False
    assert tracker.latest.observation_source == ObservationSource.NONE
    assert tracker.latest.predicted_frames == 0


def test_partial_is_valid_only_after_full_confirmation() -> None:
    tracker = BoardTracker(config(confirm_frames=2))
    before = tracker.update(partial(1, 990_000_000, ObservationSource.CONCENTRIC_ARCS))
    assert before.state == TrackingState.SEARCHING
    assert before.target_valid is False
    assert tracker.update(full(2, 1_000_000_000)).state == TrackingState.CONFIRMING
    assert tracker.update(full(3, 1_010_000_000, center=(102.0, 100.0))).state == TrackingState.TRACKING
    result = tracker.update(partial(4, 1_020_000_000, ObservationSource.CONCENTRIC_ARCS))
    assert result.state == TrackingState.TRACKING
    assert result.target_valid is True
    assert result.observation_source == ObservationSource.CONCENTRIC_ARCS


def test_prediction_expires_on_frame_limit() -> None:
    tracker = confirmed_tracker(config(confirm_frames=2, predict_max_frames=3, predict_max_ms=150.0))
    for sequence in (3, 4, 5):
        result = tracker.update(miss(sequence, 1_000_000_000 + sequence * 10_000_000))
        assert result.state == TrackingState.PREDICTING
        assert result.target_valid is True
        assert result.observation_source == ObservationSource.PREDICTED
    expired = tracker.update(miss(6, 1_060_000_000))
    assert expired.state == TrackingState.LOST
    assert expired.target_valid is False
    assert expired.predicted_center_px is None


def test_prediction_expires_on_time_limit_first() -> None:
    tracker = confirmed_tracker(config(confirm_frames=2, predict_max_frames=10, predict_max_ms=150.0))
    result = tracker.update(miss(3, 1_200_000_001))
    assert result.state == TrackingState.LOST
    assert result.target_valid is False
    assert result.failure_reason == DetectionFailure.PREDICTION_EXPIRED


def test_lost_cannot_reacquire_from_partial_arcs() -> None:
    tracker = lost_tracker()
    partial_result = tracker.update(partial(20, 2_000_000_000, ObservationSource.CONCENTRIC_ARCS))
    assert partial_result.state == TrackingState.LOST
    assert partial_result.target_valid is False
    assert tracker.update(full(21, 2_010_000_000)).state == TrackingState.CONFIRMING


def test_single_arc_limit_and_scale_jump_are_rejected() -> None:
    tracker = confirmed_tracker(config(confirm_frames=2, max_single_arc_frames=2))
    assert tracker.update(partial(3, 1_030_000_000, ObservationSource.SINGLE_ARC)).target_valid
    assert tracker.update(partial(4, 1_040_000_000, ObservationSource.SINGLE_ARC)).target_valid
    rejected = tracker.update(partial(5, 1_050_000_000, ObservationSource.SINGLE_ARC))
    assert rejected.target_valid is False
    assert rejected.failure_reason == DetectionFailure.SINGLE_ARC_LIMIT

    tracker = confirmed_tracker(config(confirm_frames=2))
    scale_jump = tracker.update(full(6, 1_060_000_000, scale=3.0))
    assert scale_jump.target_valid is False
    assert scale_jump.failure_reason == DetectionFailure.EXCESSIVE_SCALE_JUMP


def test_non_finite_and_stale_observations_fail_closed() -> None:
    tracker = confirmed_tracker()
    bad = partial(3, 1_020_000_000, ObservationSource.CONCENTRIC_ARCS, center=(math.nan, 1.0))
    result = tracker.update(bad)
    assert result.target_valid is False
    assert result.failure_reason == DetectionFailure.NO_VALID_QUADRILATERAL

    tracker = confirmed_tracker()
    stale = tracker.update(full(3, 1_020_000_000), now_ns=1_200_000_001)
    assert stale.target_valid is False
    assert stale.failure_reason == DetectionFailure.STALE_FRAME


def test_velocity_jump_is_rejected_before_history_is_remembered() -> None:
    tracker = confirmed_tracker(config(confirm_frames=2, max_center_jump_px=1000.0, max_velocity_px_s=250.0))
    rejected = tracker.update(partial(3, 1_020_000_000, ObservationSource.CONCENTRIC_ARCS, center=(120.0, 100.0)))
    assert rejected.target_valid is False
    assert rejected.failure_reason == DetectionFailure.EXCESSIVE_VELOCITY


def test_full_refresh_resets_prediction_and_single_arc_counts() -> None:
    tracker = confirmed_tracker(config(confirm_frames=2, max_single_arc_frames=2))
    tracker.update(partial(3, 1_020_000_000, ObservationSource.SINGLE_ARC))
    tracker.update(miss(4, 1_030_000_000))
    refreshed = tracker.update(full(5, 1_040_000_000, center=(106.0, 100.0)))
    assert refreshed.state == TrackingState.TRACKING
    assert refreshed.predicted_frames == 0
    assert refreshed.miss_count == 0
    assert tracker.update(partial(6, 1_050_000_000, ObservationSource.SINGLE_ARC)).target_valid


def test_preview_does_not_mutate_tracker_and_config_can_update_atomically() -> None:
    tracker = confirmed_tracker()
    before = tracker.latest
    preview = tracker.preview(miss(3, 1_020_000_000))
    assert preview.state == TrackingState.PREDICTING
    assert tracker.latest == before
    updated = config(confirm_frames=2, predict_max_frames=1)
    tracker.apply_config(updated)
    assert tracker.config == updated
    assert tracker.latest == before


def test_temporal_score_prefers_constant_velocity_prediction() -> None:
    tracker = confirmed_tracker(config(confirm_frames=2, max_center_jump_px=100.0))
    assert tracker.temporal_score((104.0, 100.0)) == pytest.approx(1.0)
    assert tracker.temporal_score((154.0, 100.0)) == pytest.approx(0.5)


def test_motion_predictor_enforces_maximum_horizon() -> None:
    predictor = MotionPredictor()
    predictor.observe((100.0, 80.0), 1_000_000_000)
    predictor.observe((104.0, 82.0), 1_020_000_000)
    assert predictor.predict(1_040_000_000, max_horizon_ns=150_000_000) == pytest.approx((108.0, 84.0))
    assert predictor.predict(1_200_000_001, max_horizon_ns=150_000_000) is None
