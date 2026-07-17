from __future__ import annotations

from dataclasses import dataclass, field

from ev_vision.config import BoardTrackingConfig
from ev_vision.models import OperatingMode, RateCommand
from ev_vision.protocol import (
    ObservationSourceCode,
    TrackingStateCode,
    VisionControlFlagsV2,
    VisionControlPayloadV2,
)
from ev_vision.vision_result import VisionTargetResult


@dataclass
class AxisController:
    kp: float
    kd: float
    max_rate: float
    max_accel: float
    derivative_alpha: float = 0.25
    deadband_enter_deg: float = 0.0
    deadband_exit_deg: float = 0.0
    _last_error: float | None = field(default=None, init=False)
    _last_time: float | None = field(default=None, init=False)
    _last_rate: float = field(default=0.0, init=False)
    _filtered_derivative: float = field(default=0.0, init=False)
    _in_deadband: bool = field(default=True, init=False)

    def update(self, error_deg: float, now_s: float) -> float:
        dt = 0.0 if self._last_time is None else max(0.0, now_s - self._last_time)
        threshold = self.deadband_exit_deg if self._in_deadband else self.deadband_enter_deg
        if abs(error_deg) <= threshold:
            self._in_deadband = True
            desired = 0.0
        else:
            self._in_deadband = False
            derivative = 0.0 if self._last_error is None or dt <= 0 else (error_deg - self._last_error) / dt
            self._filtered_derivative += self.derivative_alpha * (derivative - self._filtered_derivative)
            desired = self.kp * error_deg + self.kd * self._filtered_derivative
            desired = max(-self.max_rate, min(self.max_rate, desired))
        if dt > 0:
            delta = self.max_accel * dt
            desired = max(self._last_rate - delta, min(self._last_rate + delta, desired))
        self._last_error = error_deg
        self._last_time = now_s
        self._last_rate = desired
        return desired

    def reset(self) -> None:
        self._last_error = self._last_time = None
        self._last_rate = self._filtered_derivative = 0.0
        self._in_deadband = True


@dataclass
class VisualServo:
    yaw: AxisController
    pitch: AxisController
    source_timeout_ms: int = 100
    yaw_sign: int = 1
    pitch_sign: int = 1
    swap_axes: bool = False

    @classmethod
    def simple(cls, kp: float, max_rate: float, max_accel: float, **kwargs: object) -> "VisualServo":
        return cls(
            AxisController(kp, 0.0, max_rate, max_accel),
            AxisController(kp, 0.0, max_rate, max_accel),
            **kwargs,
        )

    def update(self, angular_error_deg: tuple[float, float], now_ns: int, source_ns: int) -> RateCommand:
        age_ns = max(0, now_ns - source_ns)
        if age_ns > self.source_timeout_ms * 1_000_000:
            self.yaw.reset()
            self.pitch.reset()
            return RateCommand(source_age_us=age_ns // 1000)
        yaw_error, pitch_error = angular_error_deg
        if self.swap_axes:
            yaw_error, pitch_error = pitch_error, yaw_error
        now_s = now_ns / 1e9
        return RateCommand(
            yaw_rate_deg_s=self.yaw_sign * self.yaw.update(yaw_error, now_s),
            pitch_rate_deg_s=self.pitch_sign * self.pitch.update(pitch_error, now_s),
            target_valid=True,
            source_age_us=age_ns // 1000,
        )

def _clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def make_vision_control_v2(
    result: VisionTargetResult,
    command: RateCommand,
    *,
    operating_mode: OperatingMode,
    angular_error_deg: tuple[float, float],
    camera_healthy: bool,
    feedback_stale: bool = False,
    gimbal_fault_received: bool = False,
    emergency_stop: bool = False,
) -> VisionControlPayloadV2:
    """Map neutral tracking/control data to the laser-free protocol V2 payload."""

    state = TrackingStateCode[result.tracking_state]
    source = ObservationSourceCode[result.observation_source]
    flags = VisionControlFlagsV2(0)
    if camera_healthy:
        flags |= VisionControlFlagsV2.CAMERA_HEALTHY

    flags |= {
        "FULL_BOARD": VisionControlFlagsV2.FULL_BOARD_VISIBLE,
        "CONCENTRIC_ARCS": VisionControlFlagsV2.MULTIPLE_ARCS_VALID,
        "SINGLE_ARC": VisionControlFlagsV2.SINGLE_ARC_VALID,
        "WHITE_REGION": VisionControlFlagsV2.WHITE_REGION_VALID,
        "FUSED_PARTIAL": (
            VisionControlFlagsV2.MULTIPLE_ARCS_VALID
            | VisionControlFlagsV2.WHITE_REGION_VALID
        ),
        "PREDICTED": VisionControlFlagsV2.USING_PREDICTION,
    }.get(result.observation_source, VisionControlFlagsV2(0))

    if result.homography_valid:
        flags |= VisionControlFlagsV2.HOMOGRAPHY_VALID
    if result.near_image_edge:
        flags |= VisionControlFlagsV2.TARGET_NEAR_IMAGE_EDGE
    if result.partially_outside:
        flags |= VisionControlFlagsV2.TARGET_PARTIALLY_OUTSIDE

    observation_stale = (
        result.frame_age_ms > BoardTrackingConfig().max_result_age_ms
    )
    if observation_stale:
        flags |= VisionControlFlagsV2.OBSERVATION_STALE
    if result.failure_reason == "EXCESSIVE_POSITION_JUMP":
        flags |= VisionControlFlagsV2.CENTER_JUMP_REJECTED
    if result.failure_reason == "EXCESSIVE_SCALE_JUMP":
        flags |= VisionControlFlagsV2.SCALE_JUMP_REJECTED
    if feedback_stale:
        flags |= VisionControlFlagsV2.FEEDBACK_STALE
    if gimbal_fault_received:
        flags |= VisionControlFlagsV2.GIMBAL_FAULT_RECEIVED
    if emergency_stop:
        flags |= VisionControlFlagsV2.EMERGENCY_STOP

    controlling_state = result.tracking_state in {"TRACKING", "PREDICTING"}
    target_valid = bool(
        result.target_valid
        and command.target_valid
        and controlling_state
        and not observation_stale
        and camera_healthy
        and not feedback_stale
        and not gimbal_fault_received
        and not emergency_stop
    )
    yaw_rate = _clamp_int(command.yaw_rate_deg_s * 100.0, -32768, 32767)
    pitch_rate = _clamp_int(command.pitch_rate_deg_s * 100.0, -32768, 32767)

    return VisionControlPayloadV2(
        operating_mode=operating_mode,
        target_valid=target_valid,
        tracking_state=state,
        observation_source=source,
        flags=flags,
        confidence_permille=_clamp_int(result.confidence * 1000.0, 0, 1000),
        yaw_rate_cdeg_s=yaw_rate if target_valid else 0,
        pitch_rate_cdeg_s=pitch_rate if target_valid else 0,
        error_yaw_mdeg=_clamp_int(angular_error_deg[0] * 1000.0, -32768, 32767),
        error_pitch_mdeg=_clamp_int(angular_error_deg[1] * 1000.0, -32768, 32767),
        target_x_px=_clamp_int(result.center_x_px or 0.0, -32768, 32767),
        target_y_px=_clamp_int(result.center_y_px or 0.0, -32768, 32767),
        source_age_us=_clamp_int(result.frame_age_ms * 1000.0, 0, 0xFFFFFFFF),
        source_frame_sequence=_clamp_int(result.frame_sequence, 0, 0xFFFFFFFF),
    )
