from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class State(Enum):
    BOOT = auto()
    SAFE = auto()
    SEARCH = auto()
    ACQUIRE = auto()
    CENTER = auto()
    LASER_CALIBRATE = auto()
    AIM = auto()
    TRACK = auto()
    CIRCLE = auto()
    RECOVER = auto()
    FAULT = auto()


class Event(Enum):
    BOOT_COMPLETE = auto()
    START = auto()
    BOARD_FOUND = auto()
    BOARD_STABLE = auto()
    LASER_CALIBRATED = auto()
    CENTERED = auto()
    TRACK_REQUESTED = auto()
    CIRCLE_REQUESTED = auto()
    TARGET_LOST = auto()
    RECOVERY_TIMEOUT = auto()
    STOP = auto()
    FAULT = auto()
    RESET = auto()


@dataclass(frozen=True)
class OutputPolicy:
    motion_allowed: bool
    laser_allowed: bool
    target_kind: str | None = None


class VisionStateMachine:
    def __init__(self) -> None:
        self.state = State.BOOT

    @property
    def policy(self) -> OutputPolicy:
        if self.state in (State.AIM, State.TRACK):
            return OutputPolicy(True, True, "center")
        if self.state is State.CIRCLE:
            return OutputPolicy(True, True, "circle")
        if self.state in (State.SEARCH, State.ACQUIRE, State.CENTER, State.LASER_CALIBRATE):
            return OutputPolicy(True, False, "center")
        return OutputPolicy(False, False, None)

    def handle(self, event: Event, *, homography_valid: bool = False, circle_sync_valid: bool = False) -> State:
        if event is Event.FAULT:
            self.state = State.FAULT
            return self.state
        if self.state is State.FAULT:
            if event is Event.RESET:
                self.state = State.SAFE
            return self.state
        if event is Event.STOP:
            self.state = State.SAFE
            return self.state
        transitions = {
            (State.BOOT, Event.BOOT_COMPLETE): State.SAFE,
            (State.SAFE, Event.START): State.SEARCH,
            (State.SEARCH, Event.BOARD_FOUND): State.ACQUIRE,
            (State.ACQUIRE, Event.BOARD_STABLE): State.CENTER,
            (State.CENTER, Event.LASER_CALIBRATED): State.LASER_CALIBRATE,
            (State.LASER_CALIBRATE, Event.CENTERED): State.AIM,
            (State.AIM, Event.TRACK_REQUESTED): State.TRACK,
            (State.RECOVER, Event.BOARD_FOUND): State.ACQUIRE,
            (State.RECOVER, Event.RECOVERY_TIMEOUT): State.SEARCH,
        }
        if event is Event.TARGET_LOST and self.state in (State.ACQUIRE, State.CENTER, State.LASER_CALIBRATE, State.AIM, State.TRACK, State.CIRCLE):
            self.state = State.RECOVER
        elif event is Event.CIRCLE_REQUESTED and self.state in (State.AIM, State.TRACK):
            if homography_valid and circle_sync_valid:
                self.state = State.CIRCLE
        else:
            self.state = transitions.get((self.state, event), self.state)
        return self.state
