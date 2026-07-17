import pytest

from ev_vision.control import AxisController, VisualServo
from ev_vision.tracking.predictor import MotionPredictor


def test_axis_controller_proportional_and_saturation() -> None:
    axis = AxisController(kp=2.0, kd=0.0, max_rate=5.0, max_accel=1000.0)
    assert axis.update(error_deg=10.0, now_s=0.0) == pytest.approx(5.0)


def test_axis_controller_acceleration_limit() -> None:
    axis = AxisController(kp=10.0, kd=0.0, max_rate=100.0, max_accel=20.0)
    axis.update(error_deg=0.0, now_s=0.0)
    assert axis.update(error_deg=10.0, now_s=0.1) == pytest.approx(2.0)


def test_hysteresis_deadband() -> None:
    axis = AxisController(kp=1.0, kd=0.0, max_rate=10.0, max_accel=1000.0, deadband_enter_deg=0.05, deadband_exit_deg=0.1)
    assert axis.update(0.04, 0.0) == 0.0
    assert axis.update(0.08, 0.1) == 0.0
    assert axis.update(0.11, 0.2) > 0.0
    assert axis.update(0.04, 0.3) == 0.0


def test_visual_servo_axis_mapping_and_stale_source() -> None:
    servo = VisualServo.simple(kp=1.0, max_rate=20.0, max_accel=1000.0, swap_axes=True, yaw_sign=-1, pitch_sign=1, source_timeout_ms=100)
    command = servo.update((1.0, 2.0), now_ns=50_000_000, source_ns=0)
    assert command.target_valid
    assert command.yaw_rate_deg_s == pytest.approx(-2.0)
    assert command.pitch_rate_deg_s == pytest.approx(1.0)
    stale = servo.update((1.0, 2.0), now_ns=200_000_000, source_ns=0)
    assert not stale.target_valid
    assert stale.yaw_rate_deg_s == stale.pitch_rate_deg_s == 0.0


def test_motion_predictor_projects_constant_velocity() -> None:
    predictor = MotionPredictor()
    predictor.observe((0.0, 0.0), 0)
    predictor.observe((1.0, -2.0), 1_000_000_000)
    assert predictor.predict(1_500_000_000) == pytest.approx((1.5, -3.0))


def test_visual_servo_stale_update_explicitly_returns_zero_motion() -> None:
    servo = VisualServo.simple(
        kp=1.0, max_rate=20.0, max_accel=1000.0, source_timeout_ms=100
    )

    stale = servo.update((5.0, -4.0), now_ns=100_000_001, source_ns=0)

    assert stale.target_valid is False
    assert stale.yaw_rate_deg_s == 0.0
    assert stale.pitch_rate_deg_s == 0.0
