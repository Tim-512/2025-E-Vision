from __future__ import annotations

from typing import Protocol

from ev_vision.models import ChassisProgress, Frame, GimbalFeedback, RateCommand


class CameraPort(Protocol):
    def latest(self) -> Frame | None: ...


class GimbalPort(Protocol):
    def send(self, command: RateCommand) -> None: ...
    def feedback(self) -> GimbalFeedback | None: ...


class ChassisPort(Protocol):
    def progress(self) -> ChassisProgress | None: ...


class LaserPort(Protocol):
    def set_enabled(self, enabled: bool, power_permille: int = 1000) -> None: ...
