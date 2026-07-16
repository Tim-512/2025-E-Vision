from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from ev_vision.config import BoardTrackingConfig
from ev_vision.detection.failures import DetectionFailure


class TrackingState(str, Enum):
    SEARCHING = "SEARCHING"
    CONFIRMING = "CONFIRMING"
    TRACKING = "TRACKING"
    PREDICTING = "PREDICTING"
    LOST = "LOST"


@dataclass(frozen=True)
class TrackObservation:
    timestamp_ns: int
    source_sequence: int
    detected: bool
    center_px: tuple[float, float] | None
    corners_px: tuple[tuple[float, float], ...] | None
    failure_reason: DetectionFailure | None


@dataclass(frozen=True)
class TrackedBoardResult:
    state: TrackingState
    target_valid: bool
    observation: TrackObservation | None
    predicted_center_px: tuple[float, float] | None
    confirmation_count: int
    miss_count: int
    failure_reason: DetectionFailure | None


class BoardTracker:
    """Validate board detections over time before authorizing coordinates."""

    def __init__(self, config: BoardTrackingConfig) -> None:
        self.config = config
        self._accepted: list[tuple[tuple[float, float], int]] = []
        self.latest = self._safe_result(TrackingState.SEARCHING)

    def update(
        self,
        result: TrackObservation,
        *,
        now_ns: int | None = None,
    ) -> TrackedBoardResult:
        """Advance one observation while enforcing age, jump, and miss gates."""
        effective_now_ns = result.timestamp_ns if now_ns is None else now_ns
        max_age_ns = self.config.max_result_age_ms * 1_000_000.0
        if effective_now_ns - result.timestamp_ns > max_age_ns:
            return self._invalidate(result, DetectionFailure.STALE_FRAME)

        if not self._is_complete_hit(result):
            if result.detected:
                return self._invalidate(
                    result,
                    result.failure_reason
                    or DetectionFailure.NO_VALID_QUADRILATERAL,
                )
            return self._handle_miss(result)

        assert result.center_px is not None
        if self._accepted:
            latest_center, latest_timestamp_ns = self._accepted[-1]
            if result.timestamp_ns <= latest_timestamp_ns:
                return self._invalidate(result, DetectionFailure.STALE_FRAME)
            if _distance(result.center_px, latest_center) > self.config.max_center_jump_px:
                return self._invalidate(
                    result, DetectionFailure.EXCESSIVE_POSITION_JUMP
                )

        previous_state = self.latest.state
        if previous_state in {TrackingState.SEARCHING, TrackingState.LOST}:
            self._accepted.clear()
            confirmation_count = 1
        elif previous_state is TrackingState.CONFIRMING:
            confirmation_count = self.latest.confirmation_count + 1
        else:
            confirmation_count = self.config.confirm_frames

        self._remember(result.center_px, result.timestamp_ns)
        state = (
            TrackingState.TRACKING
            if confirmation_count >= self.config.confirm_frames
            else TrackingState.CONFIRMING
        )
        self.latest = TrackedBoardResult(
            state=state,
            target_valid=state is TrackingState.TRACKING,
            observation=result,
            predicted_center_px=None,
            confirmation_count=min(confirmation_count, self.config.confirm_frames),
            miss_count=0,
            failure_reason=None,
        )
        return self.latest

    def reset(self) -> None:
        """Return to SEARCHING and clear accepted history."""
        self._accepted.clear()
        self.latest = self._safe_result(TrackingState.SEARCHING)

    def temporal_score(self, center_px: tuple[float, float]) -> float:
        """Return [0, 1] consistency with the latest accepted motion."""
        expected = self._next_expected_center()
        if expected is None:
            return 0.0
        distance = _distance(center_px, expected)
        return _clamp01(1.0 - distance / self.config.max_center_jump_px)

    def _handle_miss(self, result: TrackObservation) -> TrackedBoardResult:
        previous = self.latest
        miss_count = previous.miss_count + 1
        failure_reason = result.failure_reason

        if previous.state is TrackingState.CONFIRMING:
            self._accepted.clear()
            state = TrackingState.SEARCHING
            confirmation_count = 0
            predicted_center = None
        elif previous.state in {TrackingState.TRACKING, TrackingState.PREDICTING}:
            if miss_count >= self.config.lost_frames:
                self._accepted.clear()
                state = TrackingState.LOST
                confirmation_count = 0
                predicted_center = None
            else:
                state = TrackingState.PREDICTING
                confirmation_count = previous.confirmation_count
                predicted_center = (
                    self._predict_center(result.timestamp_ns)
                    if miss_count <= self.config.predict_frames
                    else None
                )
        elif previous.state is TrackingState.LOST:
            state = TrackingState.LOST
            confirmation_count = 0
            predicted_center = None
        else:
            state = TrackingState.SEARCHING
            confirmation_count = 0
            predicted_center = None

        self.latest = TrackedBoardResult(
            state=state,
            target_valid=False,
            observation=result,
            predicted_center_px=predicted_center,
            confirmation_count=confirmation_count,
            miss_count=miss_count,
            failure_reason=failure_reason,
        )
        return self.latest

    def _invalidate(
        self,
        result: TrackObservation,
        failure_reason: DetectionFailure,
    ) -> TrackedBoardResult:
        self._accepted.clear()
        self.latest = TrackedBoardResult(
            state=TrackingState.SEARCHING,
            target_valid=False,
            observation=result,
            predicted_center_px=None,
            confirmation_count=0,
            miss_count=0,
            failure_reason=failure_reason,
        )
        return self.latest

    def _remember(self, center_px: tuple[float, float], timestamp_ns: int) -> None:
        self._accepted.append((center_px, timestamp_ns))
        if len(self._accepted) > 2:
            del self._accepted[:-2]

    def _predict_center(self, target_ns: int) -> tuple[float, float] | None:
        if not self._accepted:
            return None
        latest_center, latest_ns = self._accepted[-1]
        if len(self._accepted) < 2:
            return latest_center
        previous_center, previous_ns = self._accepted[-2]
        elapsed_ns = latest_ns - previous_ns
        if elapsed_ns <= 0:
            return latest_center
        horizon = (target_ns - latest_ns) / elapsed_ns
        return (
            latest_center[0] + (latest_center[0] - previous_center[0]) * horizon,
            latest_center[1] + (latest_center[1] - previous_center[1]) * horizon,
        )

    def _next_expected_center(self) -> tuple[float, float] | None:
        if not self._accepted:
            return None
        latest_center, latest_ns = self._accepted[-1]
        if len(self._accepted) < 2:
            return latest_center
        _, previous_ns = self._accepted[-2]
        interval_ns = latest_ns - previous_ns
        return self._predict_center(latest_ns + max(0, interval_ns))

    @staticmethod
    def _is_complete_hit(result: TrackObservation) -> bool:
        return (
            result.detected
            and result.center_px is not None
            and result.corners_px is not None
            and len(result.corners_px) == 4
            and result.failure_reason is None
            and all(math.isfinite(value) for value in result.center_px)
            and all(
                math.isfinite(value)
                for corner in result.corners_px
                for value in corner
            )
        )

    @staticmethod
    def _safe_result(state: TrackingState) -> TrackedBoardResult:
        return TrackedBoardResult(
            state=state,
            target_valid=False,
            observation=None,
            predicted_center_px=None,
            confirmation_count=0,
            miss_count=0,
            failure_reason=None,
        )


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
