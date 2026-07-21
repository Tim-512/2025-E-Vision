from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
import math

from ev_vision.config import BoardTrackingConfig
from ev_vision.detection.contracts import ObservationSource
from ev_vision.detection.failures import DetectionFailure
from ev_vision.tracking.predictor import MotionPredictor


class TrackingState(str, Enum):
    SEARCHING = "SEARCHING"
    CONFIRMING = "CONFIRMING"
    TRACKING = "TRACKING"
    PREDICTING = "PREDICTING"
    LOST = "LOST"


_PARTIAL_SOURCES = {
    ObservationSource.CONCENTRIC_ARCS,
    ObservationSource.FUSED_PARTIAL,
    ObservationSource.SINGLE_ARC,
    ObservationSource.WHITE_REGION,
}


@dataclass(frozen=True)
class TrackObservation:
    timestamp_ns: int
    source_sequence: int
    detected: bool
    source: ObservationSource = ObservationSource.NONE
    center_px: tuple[float, float] | None = None
    corners_px: tuple[tuple[float, float], ...] | None = ()
    scale_px_per_mm: float | None = None
    confidence: float = 0.0
    failure_reason: DetectionFailure | None = None
    acquisition_eligible: bool = False


@dataclass(frozen=True)
class TrackedBoardResult:
    state: TrackingState
    target_valid: bool
    observation: TrackObservation | None
    predicted_center_px: tuple[float, float] | None
    confirmation_count: int
    miss_count: int
    failure_reason: DetectionFailure | None
    observation_source: ObservationSource = ObservationSource.NONE
    predicted_frames: int = 0
    source_age_us: int = 0
    velocity_px_s: tuple[float, float] | None = None
    scale_px_per_mm: float | None = None


class BoardTracker:
    """Single fail-closed state machine for full, partial and predicted targets."""

    def __init__(self, config: BoardTrackingConfig) -> None:
        self.config = config
        self._predictor = MotionPredictor()
        self._latest_corners: tuple[tuple[float, float], ...] = ()
        self._latest_scale: float | None = None
        self._latest_velocity: tuple[float, float] | None = None
        self._last_real_timestamp_ns: int | None = None
        self._single_arc_frames = 0
        self.latest = self._safe_result(TrackingState.SEARCHING)

    def update(
        self,
        result: TrackObservation,
        *,
        now_ns: int | None = None,
    ) -> TrackedBoardResult:
        effective_now_ns = result.timestamp_ns if now_ns is None else now_ns
        # Legacy hybrid observations predate explicit sources; complete geometry is
        # unambiguously a full-board observation and remains backward compatible.
        if (
            result.source is ObservationSource.NONE
            and result.corners_px is not None
            and len(result.corners_px) == 4
        ):
            result = replace(result, source=ObservationSource.FULL_BOARD)
        if effective_now_ns < result.timestamp_ns:
            return self._reject(result, DetectionFailure.STALE_FRAME)
        max_age_ns = int(self.config.max_result_age_ms * 1_000_000.0)
        if effective_now_ns - result.timestamp_ns > max_age_ns:
            return self._reject(result, DetectionFailure.STALE_FRAME)

        if not result.detected:
            return self._handle_miss(result, effective_now_ns)
        if not self._valid_center(result.center_px):
            return self._reject(result, DetectionFailure.NO_VALID_QUADRILATERAL)

        acquisition = self.latest.state in {
            TrackingState.SEARCHING,
            TrackingState.CONFIRMING,
            TrackingState.LOST,
        }
        if acquisition:
            if self._is_complete_full_board(result):
                return self._accept_full(result, confirming=True)
            if (
                result.source is ObservationSource.CONCENTRIC_ARCS
                and result.acquisition_eligible
            ):
                return self._accept_confirming_rings(result)
            return self._ignore_partial_during_acquisition(result)

        if result.source is ObservationSource.FULL_BOARD:
            if not self._is_complete_full_board(result):
                return self._reject(result, DetectionFailure.NO_VALID_QUADRILATERAL)
            return self._accept_full(result, confirming=False)
        if result.source not in _PARTIAL_SOURCES:
            return self._reject(result, DetectionFailure.NO_VALID_QUADRILATERAL)
        return self._accept_partial(result)

    def preview(
        self,
        result: TrackObservation,
        *,
        now_ns: int | None = None,
    ) -> TrackedBoardResult:
        clone = deepcopy(self)
        return clone.update(result, now_ns=now_ns)

    def reset(self) -> None:
        self._clear_history()
        self.latest = self._safe_result(TrackingState.SEARCHING)

    def apply_config(self, config: BoardTrackingConfig) -> None:
        self.config = config

    def temporal_score(self, center_px: tuple[float, float]) -> float:
        if not self._valid_center(center_px):
            return 0.0
        expected = self._next_expected_center()
        if expected is None:
            return 0.0
        return _clamp01(
            1.0 - _distance(center_px, expected) / self.config.max_center_jump_px
        )

    def _accept_full(
        self,
        result: TrackObservation,
        *,
        confirming: bool,
    ) -> TrackedBoardResult:
        rejection = self._motion_rejection(result, require_corners=True)
        if rejection is not None:
            return self._reject(result, rejection)

        previous_state = self.latest.state
        if confirming:
            confirmation_count = (
                self.latest.confirmation_count + 1
                if previous_state is TrackingState.CONFIRMING
                else 1
            )
            state = (
                TrackingState.TRACKING
                if confirmation_count >= self.config.confirm_frames
                else TrackingState.CONFIRMING
            )
        else:
            confirmation_count = self.config.confirm_frames
            state = TrackingState.TRACKING

        self._remember(result)
        self._single_arc_frames = 0
        self.latest = TrackedBoardResult(
            state=state,
            target_valid=state is TrackingState.TRACKING,
            observation=result,
            observation_source=ObservationSource.FULL_BOARD,
            predicted_center_px=None,
            confirmation_count=min(confirmation_count, self.config.confirm_frames),
            miss_count=0,
            predicted_frames=0,
            source_age_us=0,
            velocity_px_s=self._predictor.velocity_px_s,
            scale_px_per_mm=self._latest_scale,
            failure_reason=None,
        )
        return self.latest

    def _accept_confirming_rings(self, result: TrackObservation) -> TrackedBoardResult:
        rejection = self._motion_rejection(result, require_corners=False)
        if rejection is not None:
            return self._reject(result, rejection)

        confirmation_count = (
            self.latest.confirmation_count + 1
            if self.latest.state is TrackingState.CONFIRMING
            else 1
        )
        state = (
            TrackingState.TRACKING
            if confirmation_count >= self.config.confirm_frames
            else TrackingState.CONFIRMING
        )
        self._remember(result)
        self._single_arc_frames = 0
        self.latest = TrackedBoardResult(
            state=state,
            target_valid=state is TrackingState.TRACKING,
            observation=result,
            observation_source=result.source,
            predicted_center_px=None,
            confirmation_count=min(confirmation_count, self.config.confirm_frames),
            miss_count=0,
            predicted_frames=0,
            source_age_us=0,
            velocity_px_s=self._predictor.velocity_px_s,
            scale_px_per_mm=self._latest_scale,
            failure_reason=None,
        )
        return self.latest

    def _accept_partial(self, result: TrackObservation) -> TrackedBoardResult:
        rejection = self._motion_rejection(result, require_corners=False)
        if rejection is not None:
            return self._reject(result, rejection)
        if result.source is ObservationSource.SINGLE_ARC:
            if self._single_arc_frames >= self.config.max_single_arc_frames:
                return self._reject(result, DetectionFailure.SINGLE_ARC_LIMIT)
            self._single_arc_frames += 1
        else:
            self._single_arc_frames = 0

        self._remember(result)
        self.latest = TrackedBoardResult(
            state=TrackingState.TRACKING,
            target_valid=True,
            observation=result,
            observation_source=result.source,
            predicted_center_px=None,
            confirmation_count=self.config.confirm_frames,
            miss_count=0,
            predicted_frames=0,
            source_age_us=0,
            velocity_px_s=self._predictor.velocity_px_s,
            scale_px_per_mm=self._latest_scale,
            failure_reason=None,
        )
        return self.latest

    def _handle_miss(
        self,
        result: TrackObservation,
        now_ns: int,
    ) -> TrackedBoardResult:
        if self.latest.state is TrackingState.CONFIRMING:
            self._clear_history()
            self.latest = self._safe_result(
                TrackingState.SEARCHING,
                observation=result,
                failure_reason=result.failure_reason,
            )
            return self.latest
        if self.latest.state in {TrackingState.SEARCHING, TrackingState.LOST}:
            state = self.latest.state
            self.latest = self._safe_result(
                state,
                observation=result,
                miss_count=self.latest.miss_count + 1,
                failure_reason=result.failure_reason,
            )
            return self.latest

        predicted_frames = self.latest.predicted_frames + 1
        max_horizon_ns = int(self.config.predict_max_ms * 1_000_000.0)
        predicted = self._predictor.predict(
            now_ns,
            max_horizon_ns=max_horizon_ns,
        )
        if predicted_frames > self.config.predict_max_frames or predicted is None:
            self._clear_history()
            self.latest = self._safe_result(
                TrackingState.LOST,
                observation=result,
                miss_count=self.latest.miss_count + 1,
                failure_reason=DetectionFailure.PREDICTION_EXPIRED,
            )
            return self.latest

        source_age_us = 0
        if self._last_real_timestamp_ns is not None:
            source_age_us = max(0, (now_ns - self._last_real_timestamp_ns) // 1_000)
        self.latest = TrackedBoardResult(
            state=TrackingState.PREDICTING,
            target_valid=True,
            observation=result,
            observation_source=ObservationSource.PREDICTED,
            predicted_center_px=predicted,
            confirmation_count=self.config.confirm_frames,
            miss_count=self.latest.miss_count + 1,
            predicted_frames=predicted_frames,
            source_age_us=int(source_age_us),
            velocity_px_s=self._predictor.velocity_px_s,
            scale_px_per_mm=self._latest_scale,
            failure_reason=result.failure_reason,
        )
        return self.latest

    def _ignore_partial_during_acquisition(
        self,
        result: TrackObservation,
    ) -> TrackedBoardResult:
        state = (
            TrackingState.LOST
            if self.latest.state is TrackingState.LOST
            else TrackingState.SEARCHING
        )
        if self.latest.state is TrackingState.CONFIRMING:
            self._clear_history()
        self.latest = self._safe_result(
            state,
            observation=result,
            failure_reason=result.failure_reason or DetectionFailure.PARTIAL_HISTORY_REQUIRED,
        )
        return self.latest

    def _motion_rejection(
        self,
        result: TrackObservation,
        *,
        require_corners: bool,
    ) -> DetectionFailure | None:
        assert result.center_px is not None
        latest_ns = self._predictor.latest_timestamp_ns
        latest_center = self._predictor.latest_point
        if latest_ns is not None:
            if result.timestamp_ns <= latest_ns:
                return DetectionFailure.STALE_FRAME
            if latest_center is not None and _distance(result.center_px, latest_center) > self.config.max_center_jump_px:
                return DetectionFailure.EXCESSIVE_POSITION_JUMP

        if require_corners:
            if result.corners_px is None or len(result.corners_px) != 4 or not _finite_corners(result.corners_px):
                return DetectionFailure.NO_VALID_QUADRILATERAL
            if self._latest_corners and any(
                _distance(current, previous) > self.config.max_center_jump_px
                for current, previous in zip(result.corners_px, self._latest_corners, strict=True)
            ):
                return DetectionFailure.EXCESSIVE_POSITION_JUMP

        if result.scale_px_per_mm is not None:
            if not math.isfinite(result.scale_px_per_mm) or result.scale_px_per_mm <= 0.0:
                return DetectionFailure.EXCESSIVE_SCALE_JUMP
            if self._latest_scale is not None:
                fraction = abs(result.scale_px_per_mm - self._latest_scale) / self._latest_scale
                if fraction > self.config.max_scale_jump_fraction:
                    return DetectionFailure.EXCESSIVE_SCALE_JUMP

        if latest_center is not None and latest_ns is not None:
            dt_s = (result.timestamp_ns - latest_ns) / 1e9
            velocity = (
                (result.center_px[0] - latest_center[0]) / dt_s,
                (result.center_px[1] - latest_center[1]) / dt_s,
            )
            if math.hypot(*velocity) > self.config.max_velocity_px_s:
                return DetectionFailure.EXCESSIVE_VELOCITY
            if self._latest_velocity is not None:
                acceleration = (
                    (velocity[0] - self._latest_velocity[0]) / dt_s,
                    (velocity[1] - self._latest_velocity[1]) / dt_s,
                )
                if math.hypot(*acceleration) > self.config.max_acceleration_px_s2:
                    return DetectionFailure.EXCESSIVE_VELOCITY
        return None

    def _remember(self, result: TrackObservation) -> None:
        assert result.center_px is not None
        previous_velocity = self._predictor.velocity_px_s
        self._predictor.observe(result.center_px, result.timestamp_ns)
        current_velocity = self._predictor.velocity_px_s
        self._latest_velocity = current_velocity or previous_velocity
        if result.corners_px:
            self._latest_corners = result.corners_px
        if result.scale_px_per_mm is not None:
            self._latest_scale = result.scale_px_per_mm
        self._last_real_timestamp_ns = result.timestamp_ns

    def _reject(
        self,
        result: TrackObservation,
        failure_reason: DetectionFailure,
    ) -> TrackedBoardResult:
        state = (
            TrackingState.LOST
            if self.latest.state in {TrackingState.TRACKING, TrackingState.PREDICTING, TrackingState.LOST}
            else TrackingState.SEARCHING
        )
        self._clear_history()
        self.latest = self._safe_result(
            state,
            observation=result,
            failure_reason=failure_reason,
        )
        return self.latest

    def _next_expected_center(self) -> tuple[float, float] | None:
        latest_ns = self._predictor.latest_timestamp_ns
        if latest_ns is None:
            return None
        velocity = self._predictor.velocity_px_s
        if velocity is None:
            return self._predictor.latest_point
        interval_ns = 0
        if self._last_real_timestamp_ns is not None:
            interval_ns = max(0, latest_ns - self._last_real_timestamp_ns)
        # The predictor's latest timestamp and last real timestamp coincide;
        # use a short 10 ms look-ahead when velocity is known.
        return self._predictor.predict(
            latest_ns + max(interval_ns, 10_000_000),
            max_horizon_ns=int(self.config.predict_max_ms * 1_000_000.0),
        )

    def _clear_history(self) -> None:
        self._predictor.reset()
        self._latest_corners = ()
        self._latest_scale = None
        self._latest_velocity = None
        self._last_real_timestamp_ns = None
        self._single_arc_frames = 0

    @staticmethod
    def _valid_center(center_px: tuple[float, float] | None) -> bool:
        return center_px is not None and all(math.isfinite(value) for value in center_px)

    @classmethod
    def _is_complete_full_board(cls, result: TrackObservation) -> bool:
        return (
            result.source is ObservationSource.FULL_BOARD
            and cls._valid_center(result.center_px)
            and result.corners_px is not None
            and len(result.corners_px) == 4
            and _finite_corners(result.corners_px)
            and result.failure_reason is None
        )

    @staticmethod
    def _safe_result(
        state: TrackingState,
        *,
        observation: TrackObservation | None = None,
        miss_count: int = 0,
        failure_reason: DetectionFailure | None = None,
    ) -> TrackedBoardResult:
        return TrackedBoardResult(
            state=state,
            target_valid=False,
            observation=observation,
            observation_source=ObservationSource.NONE,
            predicted_center_px=None,
            confirmation_count=0,
            miss_count=miss_count,
            predicted_frames=0,
            source_age_us=0,
            velocity_px_s=None,
            scale_px_per_mm=None,
            failure_reason=failure_reason,
        )


def _finite_corners(corners: tuple[tuple[float, float], ...]) -> bool:
    return all(math.isfinite(value) for corner in corners for value in corner)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
