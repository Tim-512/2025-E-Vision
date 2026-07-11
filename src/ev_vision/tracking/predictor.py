from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MotionPredictor:
    _previous: tuple[tuple[float, float], int] | None = None
    _latest: tuple[tuple[float, float], int] | None = None

    def observe(self, point: tuple[float, float], captured_ns: int) -> None:
        if self._latest is not None and captured_ns <= self._latest[1]:
            raise ValueError("observation timestamps must increase")
        self._previous = self._latest
        self._latest = (point, captured_ns)

    def predict(self, target_ns: int) -> tuple[float, float] | None:
        if self._latest is None:
            return None
        if self._previous is None:
            return self._latest[0]
        previous_point, previous_ns = self._previous
        latest_point, latest_ns = self._latest
        dt = (latest_ns - previous_ns) / 1e9
        if dt <= 0:
            return latest_point
        horizon = (target_ns - latest_ns) / 1e9
        vx = (latest_point[0] - previous_point[0]) / dt
        vy = (latest_point[1] - previous_point[1]) / dt
        return latest_point[0] + vx * horizon, latest_point[1] + vy * horizon
