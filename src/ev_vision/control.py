from __future__ import annotations

from dataclasses import dataclass, field

from ev_vision.models import RateCommand


@dataclass
class AxisController:
    kp: float
    kd: float
    max_rate: float
    max_accel: float
    derivative_alpha: float = 0.25
    deadband_enter_deg: float = 0.0
    deadband_exit_deg: float = 0.0
    _last_error: float | None = field(default=None, init=False)
    _last_time: float | None = field(default=None, init=False)
    _last_rate: float = field(default=0.0, init=False)
    _filtered_derivative: float = field(default=0.0, init=False)
    _in_deadband: bool = field(default=True, init=False)

    def update(self, error_deg: float, now_s: float) -> float:
        dt = 0.0 if self._last_time is None else max(0.0, now_s - self._last_time)
        threshold = self.deadband_exit_deg if self._in_deadband else self.deadband_enter_deg
        if abs(error_deg) <= threshold:
            self._in_deadband = True
            desired = 0.0
        else:
            self._in_deadband = False
            derivative = 0.0 if self._last_error is None or dt <= 0 else (error_deg - self._last_error) / dt
            self._filtered_derivative += self.derivative_alpha * (derivative - self._filtered_derivative)
            desired = self.kp * error_deg + self.kd * self._filtered_derivative
            desired = max(-self.max_rate, min(self.max_rate, desired))
        if dt > 0:
            delta = self.max_accel * dt
            desired = max(self._last_rate - delta, min(self._last_rate + delta, desired))
        self._last_error = error_deg
        self._last_time = now_s
        self._last_rate = desired
        return desired

    def reset(self) -> None:
        self._last_error = self._last_time = None
        self._last_rate = self._filtered_derivative = 0.0
        self._in_deadband = True


@dataclass
class VisualServo:
    yaw: AxisController
    pitch: AxisController
    source_timeout_ms: int = 100
    yaw_sign: int = 1
    pitch_sign: int = 1
    swap_axes: bool = False

    @classmethod
    def simple(cls, kp: float, max_rate: float, max_accel: float, **kwargs: object) -> "VisualServo":
        return cls(
            AxisController(kp, 0.0, max_rate, max_accel),
            AxisController(kp, 0.0, max_rate, max_accel),
            **kwargs,
        )

    def update(self, angular_error_deg: tuple[float, float], now_ns: int, source_ns: int) -> RateCommand:
        age_ns = max(0, now_ns - source_ns)
        if age_ns > self.source_timeout_ms * 1_000_000:
            self.yaw.reset()
            self.pitch.reset()
            return RateCommand(source_age_us=age_ns // 1000)
        yaw_error, pitch_error = angular_error_deg
        if self.swap_axes:
            yaw_error, pitch_error = pitch_error, yaw_error
        now_s = now_ns / 1e9
        return RateCommand(
            yaw_rate_deg_s=self.yaw_sign * self.yaw.update(yaw_error, now_s),
            pitch_rate_deg_s=self.pitch_sign * self.pitch.update(pitch_error, now_s),
            target_valid=True,
            source_age_us=age_ns // 1000,
        )
