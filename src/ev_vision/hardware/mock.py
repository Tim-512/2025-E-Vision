from __future__ import annotations

from dataclasses import dataclass, field

from ev_vision.models import GimbalFeedback, RateCommand


@dataclass
class MockLaser:
    enabled: bool = False
    history: list[bool] = field(default_factory=list)

    def set_enabled(self, enabled: bool, power_permille: int = 1000) -> None:
        self.enabled = enabled
        self.history.append(enabled)


@dataclass
class MockGimbal:
    current_feedback: GimbalFeedback = field(default_factory=GimbalFeedback)
    commands: list[RateCommand] = field(default_factory=list)

    def send(self, command: RateCommand) -> None:
        self.commands.append(command)

    def feedback(self) -> GimbalFeedback:
        return self.current_feedback
