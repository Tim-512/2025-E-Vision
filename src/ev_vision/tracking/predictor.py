from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class MotionPredictor:
    _previous: tuple[tuple[float, float], int] | None = None
    _latest: tuple[tuple[float, float], int] | None = None

    def observe(self, point: tuple[float, float], captured_ns: int) -> None:
        if not all(math.isfinite(value) for value in point):
            raise ValueError("observation point must be finite")
        if self._latest is not None and captured_ns <= self._latest[1]:
            raise ValueError("observation timestamps must increase")
        self._previous = self._latest
        self._latest = ((float(point[0]), float(point[1])), int(captured_ns))

    def reset(self) -> None:
        self._previous = None
        self._latest = None

    @property
    def latest_timestamp_ns(self) -> int | None:
        return None if self._latest is None else self._latest[1]

    @property
    def latest_point(self) -> tuple[float, float] | None:
        return None if self._latest is None else self._latest[0]

    @property
    def velocity_px_s(self) -> tuple[float, float] | None:
        if self._previous is None or self._latest is None:
            return None
        previous_point, previous_ns = self._previous
        latest_point, latest_ns = self._latest
        dt_s = (latest_ns - previous_ns) / 1e9
        if dt_s <= 0.0:
            return None
        return (
            (latest_point[0] - previous_point[0]) / dt_s,
            (latest_point[1] - previous_point[1]) / dt_s,
        )

    def predict(
        self,
        target_ns: int,
        *,
        max_horizon_ns: int,
    ) -> tuple[float, float] | None:
        if self._latest is None:
            return None
        latest_point, latest_ns = self._latest
        horizon_ns = target_ns - latest_ns
        if horizon_ns < 0 or horizon_ns > max_horizon_ns:
            return None
        velocity = self.velocity_px_s
        if velocity is None:
            return latest_point
        horizon_s = horizon_ns / 1e9
        return (
            latest_point[0] + velocity[0] * horizon_s,
            latest_point[1] + velocity[1] * horizon_s,
        )
