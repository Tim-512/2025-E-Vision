from __future__ import annotations

from dataclasses import dataclass, replace

from ev_vision.control import VisualServo
from ev_vision.models import BoardObservation, GimbalFeedback, LaserMode, RateCommand
from ev_vision.state_machine import Event, State, VisionStateMachine


@dataclass(frozen=True)
class CycleInput:
    now_ns: int
    board: BoardObservation | None
    angular_error_deg: tuple[float, float] | None
    gimbal: GimbalFeedback | None


@dataclass(frozen=True)
class CycleResult:
    command: RateCommand
    reason: str


class VisionRuntime:
    def __init__(self, state_machine: VisionStateMachine, servo: VisualServo) -> None:
        self.state_machine = state_machine
        self.servo = servo

    def _stop(self, reason: str) -> CycleResult:
        return CycleResult(RateCommand(laser_mode=LaserMode.OFF), reason)

    def step(self, cycle: CycleInput) -> CycleResult:
        if cycle.gimbal is None or cycle.gimbal.fault_flags:
            self.state_machine.handle(Event.FAULT)
            return self._stop("gimbal fault or missing feedback")
        if cycle.board is None or cycle.angular_error_deg is None:
            self.state_machine.handle(Event.TARGET_LOST)
            return self._stop("target unavailable")
        if self.state_machine.state is State.CIRCLE and not cycle.board.homography_valid:
            self.state_machine.handle(Event.FAULT)
            return self._stop("circle requires a valid homography")
        policy = self.state_machine.policy
        if not policy.motion_allowed:
            return self._stop("motion not allowed in current state")
        command = self.servo.update(cycle.angular_error_deg, cycle.now_ns, cycle.board.captured_ns)
        if not command.target_valid:
            return self._stop("vision source is stale")
        command = replace(command, laser_mode=LaserMode.ON if policy.laser_allowed else LaserMode.OFF)
        return CycleResult(command, "ok")
