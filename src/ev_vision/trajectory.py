from __future__ import annotations

from dataclasses import dataclass, field
import math

from ev_vision.models import ChassisProgress


@dataclass(frozen=True)
class CenterTarget:
    def point_cm(self) -> tuple[float, float]:
        return 0.0, 0.0


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class CircleTrajectory:
    radius_cm: float = 6.0
    start_phase_deg: float = 0.0
    phase_direction: int = 1
    progress_timeout_ms: int = 300
    max_phase_correction_deg_s: float = 45.0
    nominal_lap_s: float = 4.0
    sync_valid: bool = field(default=False, init=False)
    _phase: float | None = field(default=None, init=False)
    _last_ns: int | None = field(default=None, init=False)

    def point_cm(self, now_ns: int, progress: ChassisProgress | None) -> tuple[float, float]:
        start_phase = math.radians(self.start_phase_deg)
        if self._phase is None:
            self._phase = start_phase
        dt = 0.0 if self._last_ns is None else max(0.0, (now_ns - self._last_ns) / 1e9)
        self._last_ns = now_ns
        local_rate = self.phase_direction * 2.0 * math.pi / self.nominal_lap_s
        self._phase += local_rate * dt

        fresh = progress is not None and now_ns - progress.received_ns <= self.progress_timeout_ms * 1_000_000
        self.sync_valid = bool(fresh)
        if fresh and progress is not None:
            target_phase = start_phase + self.phase_direction * 2.0 * math.pi * (progress.progress_permille % 1000) / 1000.0
            if dt == 0.0:
                self._phase = target_phase
            else:
                error = _wrap(target_phase - self._phase)
                max_correction = math.radians(self.max_phase_correction_deg_s) * dt
                self._phase += max(-max_correction, min(max_correction, error))

        return self.radius_cm * math.cos(self._phase), self.radius_cm * math.sin(self._phase)
