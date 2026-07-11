from ev_vision.control import VisualServo
from ev_vision.models import BoardObservation, GimbalFeedback, LaserMode
from ev_vision.runtime import CycleInput, VisionRuntime
from ev_vision.state_machine import Event, State, VisionStateMachine


def tracked_runtime() -> VisionRuntime:
    machine = VisionStateMachine()
    for event in (Event.BOOT_COMPLETE, Event.START, Event.BOARD_FOUND, Event.BOARD_STABLE, Event.LASER_CALIBRATED, Event.CENTERED, Event.TRACK_REQUESTED):
        machine.handle(event)
    return VisionRuntime(machine, VisualServo.simple(kp=1.0, max_rate=20.0, max_accel=1000.0))


def board(now_ns: int, homography: bool = True) -> BoardObservation:
    return BoardObservation(now_ns, ((0, 0), (1, 0), (1, 1), (0, 1)), (640, 512), 0.9, homography)


def test_valid_track_cycle_allows_laser_and_motion() -> None:
    runtime = tracked_runtime()
    result = runtime.step(CycleInput(50_000_000, board(0), (1.0, -2.0), GimbalFeedback(received_ns=50_000_000)))
    assert result.command.target_valid
    assert result.command.yaw_rate_deg_s == 1.0
    assert result.command.laser_mode is LaserMode.ON


def test_stale_frame_is_fail_closed() -> None:
    runtime = tracked_runtime()
    result = runtime.step(CycleInput(200_000_000, board(0), (1.0, 1.0), GimbalFeedback(received_ns=200_000_000)))
    assert not result.command.target_valid
    assert result.command.yaw_rate_deg_s == result.command.pitch_rate_deg_s == 0.0
    assert result.command.laser_mode is LaserMode.OFF


def test_gimbal_fault_latches_fault_state() -> None:
    runtime = tracked_runtime()
    result = runtime.step(CycleInput(1, board(1), (1.0, 1.0), GimbalFeedback(fault_flags=1, received_ns=1)))
    assert runtime.state_machine.state is State.FAULT
    assert result.command.laser_mode is LaserMode.OFF
    assert result.command.yaw_rate_deg_s == 0.0


def test_circle_without_homography_is_rejected() -> None:
    runtime = tracked_runtime()
    runtime.state_machine.handle(Event.CIRCLE_REQUESTED, homography_valid=True, circle_sync_valid=True)
    result = runtime.step(CycleInput(1, board(1, homography=False), (1.0, 1.0), GimbalFeedback(received_ns=1)))
    assert runtime.state_machine.state is State.FAULT
    assert result.command.laser_mode is LaserMode.OFF
