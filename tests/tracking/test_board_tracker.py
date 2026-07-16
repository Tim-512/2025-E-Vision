from __future__ import annotations

import pytest

from ev_vision.config import BoardTrackingConfig
from ev_vision.detection.failures import DetectionFailure
from ev_vision.tracking.board_tracker import BoardTracker, TrackObservation, TrackingState


def config(**overrides: object) -> BoardTrackingConfig:
    values = {
        "confirm_frames": 3,
        "predict_frames": 2,
        "lost_frames": 3,
        "max_center_jump_px": 160.0,
        "max_result_age_ms": 100.0,
    }
    values.update(overrides)
    return BoardTrackingConfig(**values)


def hit(
    sequence: int,
    timestamp_ns: int,
    *,
    center: tuple[float, float] = (100.0, 100.0),
) -> TrackObservation:
    x, y = center
    return TrackObservation(
        timestamp_ns=timestamp_ns,
        source_sequence=sequence,
        detected=True,
        center_px=center,
        corners_px=(
            (x - 50.0, y - 80.0),
            (x + 50.0, y - 80.0),
            (x + 50.0, y + 80.0),
            (x - 50.0, y + 80.0),
        ),
        failure_reason=None,
    )


def miss(
    sequence: int,
    timestamp_ns: int,
    reason: DetectionFailure = DetectionFailure.NO_MODEL_CANDIDATE,
) -> TrackObservation:
    return TrackObservation(
        timestamp_ns=timestamp_ns,
        source_sequence=sequence,
        detected=False,
        center_px=None,
        corners_px=None,
        failure_reason=reason,
    )


def tracking_tracker(
    *, center: tuple[float, float] = (100.0, 100.0)
) -> BoardTracker:
    tracker = BoardTracker(config())
    tracker.update(hit(1, 0, center=center))
    tracker.update(hit(2, 20_000_000, center=(center[0] + 2.0, center[1])))
    tracked = tracker.update(hit(3, 40_000_000, center=(center[0] + 4.0, center[1])))
    assert tracked.state is TrackingState.TRACKING
    return tracker


def test_initial_state_is_safe_and_empty() -> None:
    tracker = BoardTracker(config())

    assert tracker.latest.state is TrackingState.SEARCHING
    assert tracker.latest.target_valid is False
    assert tracker.latest.observation is None
    assert tracker.latest.predicted_center_px is None
    assert tracker.latest.confirmation_count == 0
    assert tracker.latest.miss_count == 0


def test_three_hits_track_two_predictions_then_loss() -> None:
    tracker = BoardTracker(config())

    results = [
        tracker.update(hit(1, 0)),
        tracker.update(hit(2, 20_000_000)),
        tracker.update(hit(3, 40_000_000)),
        tracker.update(miss(4, 60_000_000)),
        tracker.update(miss(5, 80_000_000)),
        tracker.update(miss(6, 100_000_000)),
        tracker.update(miss(7, 120_000_000)),
    ]

    assert [result.state for result in results] == [
        TrackingState.CONFIRMING,
        TrackingState.CONFIRMING,
        TrackingState.TRACKING,
        TrackingState.PREDICTING,
        TrackingState.PREDICTING,
        TrackingState.LOST,
        TrackingState.LOST,
    ]
    assert [result.target_valid for result in results] == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert tracker.latest.miss_count == 4


def test_confirmation_failure_returns_to_searching_and_clears_history() -> None:
    tracker = BoardTracker(config())
    tracker.update(hit(1, 0))
    tracker.update(hit(2, 20_000_000))

    failed = tracker.update(miss(3, 40_000_000))
    restarted = tracker.update(hit(4, 60_000_000))

    assert failed.state is TrackingState.SEARCHING
    assert failed.confirmation_count == 0
    assert restarted.state is TrackingState.CONFIRMING
    assert restarted.confirmation_count == 1


def test_prediction_uses_constant_velocity_and_never_grants_target_valid() -> None:
    tracker = BoardTracker(config())
    tracker.update(hit(1, 0, center=(100.0, 80.0)))
    tracker.update(hit(2, 20_000_000, center=(104.0, 82.0)))
    tracker.update(hit(3, 40_000_000, center=(108.0, 84.0)))

    predicted = tracker.update(miss(4, 60_000_000))

    assert predicted.state is TrackingState.PREDICTING
    assert predicted.target_valid is False
    assert predicted.predicted_center_px == pytest.approx((112.0, 86.0))


def test_lost_next_accepted_hit_restarts_confirmation() -> None:
    tracker = tracking_tracker()
    tracker.update(miss(4, 60_000_000))
    tracker.update(miss(5, 80_000_000))
    lost = tracker.update(miss(6, 100_000_000))

    recovered = tracker.update(hit(7, 120_000_000, center=(108.0, 100.0)))

    assert lost.state is TrackingState.LOST
    assert recovered.state is TrackingState.CONFIRMING
    assert recovered.confirmation_count == 1
    assert recovered.target_valid is False


def test_jump_fails_safe_clears_history_and_next_hit_starts_confirming() -> None:
    tracker = tracking_tracker(center=(100.0, 100.0))

    jumped = tracker.update(hit(4, 60_000_000, center=(400.0, 100.0)))
    next_hit = tracker.update(hit(5, 80_000_000, center=(110.0, 100.0)))

    assert jumped.state is TrackingState.SEARCHING
    assert jumped.failure_reason is DetectionFailure.EXCESSIVE_POSITION_JUMP
    assert jumped.target_valid is False
    assert jumped.confirmation_count == 0
    assert next_hit.state is TrackingState.CONFIRMING
    assert next_hit.confirmation_count == 1
    assert next_hit.target_valid is False


def test_stale_hit_fails_safe_and_requires_reconfirmation() -> None:
    tracker = tracking_tracker()

    stale = tracker.update(hit(4, 60_000_000), now_ns=200_000_001)
    next_hit = tracker.update(
        hit(5, 220_000_000, center=(110.0, 100.0)), now_ns=220_000_000
    )

    assert stale.state is TrackingState.SEARCHING
    assert stale.failure_reason is DetectionFailure.STALE_FRAME
    assert stale.target_valid is False
    assert stale.confirmation_count == 0
    assert next_hit.state is TrackingState.CONFIRMING
    assert next_hit.target_valid is False


def test_target_valid_requires_complete_detected_geometry() -> None:
    tracker = tracking_tracker()
    incomplete = TrackObservation(
        timestamp_ns=60_000_000,
        source_sequence=4,
        detected=True,
        center_px=(106.0, 100.0),
        corners_px=None,
        failure_reason=None,
    )

    result = tracker.update(incomplete)

    assert result.state is TrackingState.SEARCHING
    assert result.target_valid is False
    assert result.confirmation_count == 0


def test_temporal_score_is_bounded_and_prefers_predicted_motion() -> None:
    tracker = BoardTracker(config(max_center_jump_px=100.0))

    assert tracker.temporal_score((999.0, 999.0)) == pytest.approx(0.0)
    tracker.update(hit(1, 0, center=(100.0, 100.0)))
    assert tracker.temporal_score((100.0, 100.0)) == pytest.approx(1.0)
    assert tracker.temporal_score((150.0, 100.0)) == pytest.approx(0.5)
    assert tracker.temporal_score((250.0, 100.0)) == pytest.approx(0.0)

    tracker.update(hit(2, 20_000_000, center=(110.0, 100.0)))
    assert tracker.temporal_score((120.0, 100.0)) == pytest.approx(1.0)
    assert 0.0 <= tracker.temporal_score((-1_000.0, -1_000.0)) <= 1.0


def test_reset_returns_to_searching_and_clears_motion_history() -> None:
    tracker = tracking_tracker()

    tracker.reset()

    assert tracker.latest.state is TrackingState.SEARCHING
    assert tracker.latest.observation is None
    assert tracker.latest.confirmation_count == 0
    assert tracker.latest.miss_count == 0
    assert tracker.temporal_score((104.0, 100.0)) == pytest.approx(0.0)
