from __future__ import annotations

import pytest

from ev_vision.control import make_vision_control_v2
from ev_vision.models import LaserMode, OperatingMode, RateCommand
from ev_vision.protocol import (
    ObservationSourceCode,
    TrackingStateCode,
    VisionControlFlagsV2,
)
from ev_vision.vision_result import VisionTargetResult


def mapped_result(**overrides: object) -> VisionTargetResult:
    values: dict[str, object] = {
        "timestamp_ms": 50,
        "frame_sequence": 17,
        "target_valid": True,
        "tracking_state": "TRACKING",
        "confidence": 0.92,
        "center_x_px": 700.0,
        "center_y_px": 480.0,
        "offset_x_px": 60.0,
        "offset_y_px": -32.0,
        "target_x_mm": 0.0,
        "target_y_mm": 0.0,
        "corners": ((100.0, 80.0), (1180.0, 80.0), (1180.0, 944.0), (100.0, 944.0)),
        "frame_age_ms": 40.0,
        "observation_source": "FULL_BOARD",
        "homography_valid": True,
        "predicted_frames": 0,
        "near_image_edge": False,
        "partially_outside": False,
        "failure_reason": None,
    }
    values.update(overrides)
    return VisionTargetResult(**values)  # type: ignore[arg-type]


def test_full_board_command_maps_units_and_flags() -> None:
    result = mapped_result(
        target_valid=True, tracking_state="TRACKING",
        observation_source="FULL_BOARD", confidence=0.812,
        center_x_px=700.4, center_y_px=480.6,
        homography_valid=True, frame_age_ms=25.0,
    )
    command = RateCommand(
        yaw_rate_deg_s=-1.23, pitch_rate_deg_s=4.56,
        target_valid=True, source_age_us=25_000,
    )

    payload = make_vision_control_v2(
        result, command, operating_mode=OperatingMode.TRACK,
        angular_error_deg=(-0.050, 0.075), camera_healthy=True,
    )

    assert payload.tracking_state == TrackingStateCode.TRACKING
    assert payload.observation_source == ObservationSourceCode.FULL_BOARD
    assert payload.confidence_permille == 812
    assert payload.yaw_rate_cdeg_s == -123
    assert payload.pitch_rate_cdeg_s == 456
    assert payload.error_yaw_mdeg == -50
    assert payload.error_pitch_mdeg == 75
    assert payload.target_x_px == 700
    assert payload.target_y_px == 481
    assert payload.flags & VisionControlFlagsV2.CAMERA_HEALTHY
    assert payload.flags & VisionControlFlagsV2.FULL_BOARD_VISIBLE
    assert payload.flags & VisionControlFlagsV2.HOMOGRAPHY_VALID


def test_invalid_or_stale_result_forces_zero_rates() -> None:
    result = mapped_result(
        target_valid=False, tracking_state="LOST",
        observation_source="NONE", frame_age_ms=200.0,
    )
    command = RateCommand(
        yaw_rate_deg_s=9.0, pitch_rate_deg_s=-8.0,
        target_valid=True, source_age_us=200_000,
    )

    payload = make_vision_control_v2(
        result, command, operating_mode=OperatingMode.SEARCH,
        angular_error_deg=(1.0, -1.0), camera_healthy=True,
    )

    assert payload.target_valid is False
    assert payload.yaw_rate_cdeg_s == payload.pitch_rate_cdeg_s == 0
    assert payload.flags & VisionControlFlagsV2.OBSERVATION_STALE


def test_stale_result_is_fail_closed_even_if_marked_valid() -> None:
    payload = make_vision_control_v2(
        mapped_result(target_valid=True, frame_age_ms=200.0),
        RateCommand(yaw_rate_deg_s=9.0, pitch_rate_deg_s=-8.0, target_valid=True),
        operating_mode=OperatingMode.TRACK,
        angular_error_deg=(1.0, -1.0),
        camera_healthy=True,
    )

    assert payload.target_valid is False
    assert payload.yaw_rate_cdeg_s == payload.pitch_rate_cdeg_s == 0


def test_v2_mapping_ignores_legacy_laser_mode() -> None:
    result = mapped_result(target_valid=True)
    command = RateCommand(
        target_valid=True, laser_mode=LaserMode.ON,
        yaw_rate_deg_s=1.0, pitch_rate_deg_s=2.0,
    )

    payload = make_vision_control_v2(
        result, command, operating_mode=OperatingMode.TRACK,
        angular_error_deg=(0.1, 0.2), camera_healthy=True,
    )

    assert not hasattr(payload, "laser_mode")
    assert not hasattr(payload, "laser_confidence")


@pytest.mark.parametrize(
    ("state", "source"),
    [("SEARCHING", "NONE"), ("CONFIRMING", "FULL_BOARD"),
     ("LOST", "NONE"), ("FAULT", "NONE")],
)
def test_invalid_states_never_emit_motion(state: str, source: str) -> None:
    payload = make_vision_control_v2(
        mapped_result(target_valid=True, tracking_state=state, observation_source=source),
        RateCommand(yaw_rate_deg_s=20.0, pitch_rate_deg_s=20.0, target_valid=True),
        operating_mode=OperatingMode.TRACK,
        angular_error_deg=(2.0, 2.0), camera_healthy=True,
    )

    assert payload.target_valid is False
    assert payload.yaw_rate_cdeg_s == payload.pitch_rate_cdeg_s == 0


@pytest.mark.parametrize(
    "safety_overrides",
    [
        {"camera_healthy": False},
        {"feedback_stale": True},
        {"gimbal_fault_received": True},
        {"emergency_stop": True},
    ],
)
def test_external_safety_conditions_force_zero_motion(safety_overrides: dict[str, bool]) -> None:
    kwargs = {
        "camera_healthy": True,
        "feedback_stale": False,
        "gimbal_fault_received": False,
        "emergency_stop": False,
    }
    kwargs.update(safety_overrides)
    payload = make_vision_control_v2(
        mapped_result(),
        RateCommand(yaw_rate_deg_s=3.0, pitch_rate_deg_s=-2.0, target_valid=True),
        operating_mode=OperatingMode.TRACK,
        angular_error_deg=(0.1, -0.1),
        **kwargs,
    )

    assert payload.target_valid is False
    assert payload.yaw_rate_cdeg_s == payload.pitch_rate_cdeg_s == 0
