from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ev_vision.gimbal_usb.protocol import GimbalTargetCommand
from ev_vision.gimbal_usb.target_angles import AngleConverter
from ev_vision.tuning.models import DetectionSnapshot


REAL_OBSERVATION_SOURCES = frozenset(
    {
        "FULL_BOARD",
        "CONCENTRIC_ARCS",
        "SINGLE_ARC",
        "WHITE_REGION",
        "FUSED_PARTIAL",
    }
)


@dataclass(frozen=True)
class GateLimits:
    max_result_age_ms: float = 60.0
    predicted_control_max_frames: int = 3
    predicted_control_max_age_us: int = 60_000
    predicted_max_angle_step_deg: float = 1.5
    image_size: tuple[int, int] = (1280, 1024)

    def __post_init__(self) -> None:
        positive_finite = (
            ("max_result_age_ms", self.max_result_age_ms),
            ("predicted_max_angle_step_deg", self.predicted_max_angle_step_deg),
        )
        for name, value in positive_finite:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite")

        integer_limits = (
            ("predicted_control_max_frames", self.predicted_control_max_frames),
            ("predicted_control_max_age_us", self.predicted_control_max_age_us),
        )
        for name, value in integer_limits:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        try:
            width, height = self.image_size
        except (TypeError, ValueError) as exc:
            raise ValueError("image_size must contain width and height") from exc
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (width, height)
        ):
            raise ValueError("image_size width and height must be positive integers")


@dataclass(frozen=True)
class GateDecision:
    command: GimbalTargetCommand
    reason: str
    observation_source: str
    result_age_ms: float | None
    source_age_us: int


class GimbalControlGate:
    """Fail-closed stateful gate for real and short predicted observations."""

    def __init__(self, converter: AngleConverter, limits: GateLimits) -> None:
        self._converter = converter
        self._limits = limits
        self._previous_valid: GimbalTargetCommand | None = None
        self._prediction_latched_invalid = False

    @staticmethod
    def _source_name(detection: DetectionSnapshot | None) -> str:
        if detection is None:
            return "NONE"
        source = detection.observation_source
        if isinstance(source, Enum):
            source = source.value
        return str(source)

    @staticmethod
    def _source_age(detection: DetectionSnapshot | None) -> int:
        if detection is None:
            return 0
        try:
            return int(detection.source_age_us)
        except (TypeError, ValueError, OverflowError):
            return 0

    def _reject(
        self,
        reason: str,
        detection: DetectionSnapshot | None,
        *,
        latch_prediction: bool = False,
    ) -> GateDecision:
        if latch_prediction:
            self._prediction_latched_invalid = True
        return GateDecision(
            command=GimbalTargetCommand.safe(),
            reason=reason,
            observation_source=self._source_name(detection),
            result_age_ms=None if detection is None else detection.result_age_ms,
            source_age_us=self._source_age(detection),
        )

    def _reject_for_source(
        self,
        reason: str,
        detection: DetectionSnapshot,
        source: str,
    ) -> GateDecision:
        return self._reject(
            reason,
            detection,
            latch_prediction=source == "PREDICTED",
        )

    def _valid_center(
        self, center_px: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if center_px is None:
            return None
        try:
            if len(center_px) != 2:
                return None
            x = float(center_px[0])
            y = float(center_px[1])
        except (TypeError, ValueError, OverflowError):
            return None
        width, height = self._limits.image_size
        if (
            not math.isfinite(x)
            or not math.isfinite(y)
            or not 0.0 <= x < float(width)
            or not 0.0 <= y < float(height)
        ):
            return None
        return x, y

    def evaluate(self, detection: DetectionSnapshot | None) -> GateDecision:
        if detection is None:
            return self._reject("detection_missing", detection)

        source = self._source_name(detection)
        is_prediction = source == "PREDICTED"
        if is_prediction and self._prediction_latched_invalid:
            return self._reject("prediction_latched_invalid", detection)

        if not detection.enabled:
            return self._reject_for_source("detection_disabled", detection, source)
        if detection.error:
            return self._reject_for_source("detection_error", detection, source)
        if not detection.target_valid:
            return self._reject_for_source("target_invalid", detection, source)
        if detection.failure_reason:
            return self._reject_for_source("upstream_failure", detection, source)

        result_age = detection.result_age_ms
        if (
            isinstance(result_age, bool)
            or not isinstance(result_age, (int, float))
            or not math.isfinite(float(result_age))
            or float(result_age) < 0.0
            or float(result_age) > self._limits.max_result_age_ms
        ):
            return self._reject_for_source("result_stale", detection, source)

        center = self._valid_center(detection.center_px)
        if center is None:
            return self._reject_for_source("center_invalid", detection, source)

        if source in REAL_OBSERVATION_SOURCES:
            if detection.tracking_state != "TRACKING":
                return self._reject("tracking_state", detection)
        elif is_prediction:
            if detection.tracking_state != "PREDICTING":
                return self._reject(
                    "tracking_state", detection, latch_prediction=True
                )
            predicted_frames = detection.predicted_frames
            if (
                isinstance(predicted_frames, bool)
                or not isinstance(predicted_frames, int)
                or not 1
                <= predicted_frames
                <= self._limits.predicted_control_max_frames
            ):
                return self._reject(
                    "prediction_frame_limit", detection, latch_prediction=True
                )
            source_age_us = detection.source_age_us
            if (
                isinstance(source_age_us, bool)
                or not isinstance(source_age_us, int)
                or not 0
                <= source_age_us
                <= self._limits.predicted_control_max_age_us
            ):
                return self._reject(
                    "prediction_age_limit", detection, latch_prediction=True
                )
            if self._previous_valid is None:
                return self._reject(
                    "prediction_without_real", detection, latch_prediction=True
                )
        else:
            return self._reject("observation_source", detection)

        try:
            if not bool(self._converter.available):
                raise RuntimeError("angle converter unavailable")
            yaw_deg, pitch_deg = self._converter.convert(center)
            yaw_deg = float(yaw_deg)
            pitch_deg = float(pitch_deg)
            if not math.isfinite(yaw_deg) or not math.isfinite(pitch_deg):
                raise ValueError("converted angles must be finite")
        except Exception:
            return self._reject_for_source(
                "angle_conversion_failed", detection, source
            )

        command = GimbalTargetCommand(yaw_deg, pitch_deg, True)
        if source in REAL_OBSERVATION_SOURCES:
            # A valid real observation immediately recovers control. Deliberately
            # do not compare it with the previous prediction or real command.
            self._previous_valid = command
            self._prediction_latched_invalid = False
            return GateDecision(
                command,
                "real_observation",
                source,
                float(result_age),
                self._source_age(detection),
            )

        previous = self._previous_valid
        assert previous is not None
        if (
            abs(command.yaw_deg - previous.yaw_deg)
            > self._limits.predicted_max_angle_step_deg
            or abs(command.pitch_deg - previous.pitch_deg)
            > self._limits.predicted_max_angle_step_deg
        ):
            return self._reject(
                "prediction_angle_step", detection, latch_prediction=True
            )

        self._previous_valid = command
        return GateDecision(
            command,
            "predicted_observation",
            source,
            float(result_age),
            self._source_age(detection),
        )
