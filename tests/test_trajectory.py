import math

import pytest

from ev_vision.models import ChassisProgress
from ev_vision.trajectory import CenterTarget, CircleTrajectory


def test_center_target() -> None:
    assert CenterTarget().point_cm() == (0.0, 0.0)


@pytest.mark.parametrize("progress, expected", [(0, (6, 0)), (250, (0, 6)), (500, (-6, 0)), (750, (0, -6)), (1000, (6, 0))])
def test_circle_cardinal_positions(progress: int, expected: tuple[float, float]) -> None:
    trajectory = CircleTrajectory(radius_cm=6.0, start_phase_deg=0.0, phase_direction=1)
    sample = ChassisProgress(True, 0, progress, 0, received_ns=1_000_000_000)
    point = trajectory.point_cm(1_000_000_000, sample)
    assert point == pytest.approx(expected, abs=1e-6)
    assert math.hypot(*point) == pytest.approx(6.0)


def test_stale_progress_uses_continuous_local_phase() -> None:
    trajectory = CircleTrajectory(radius_cm=6.0, progress_timeout_ms=100, nominal_lap_s=4.0)
    progress = ChassisProgress(True, 0, 0, 0, received_ns=0)
    first = trajectory.point_cm(0, progress)
    second = trajectory.point_cm(1_000_000_000, progress)
    assert first == pytest.approx((6.0, 0.0))
    assert second == pytest.approx((0.0, 6.0), abs=1e-6)
    assert not trajectory.sync_valid


def test_phase_correction_is_rate_limited() -> None:
    trajectory = CircleTrajectory(radius_cm=6.0, max_phase_correction_deg_s=10.0, nominal_lap_s=1000.0)
    trajectory.point_cm(0, None)
    progress = ChassisProgress(True, 0, 500, 0, received_ns=1_000_000_000)
    point = trajectory.point_cm(1_000_000_000, progress)
    angle_deg = math.degrees(math.atan2(point[1], point[0]))
    assert angle_deg == pytest.approx(10.36, abs=0.1)
