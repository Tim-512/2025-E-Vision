from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml


class ConfigError(ValueError):
    """Raised when configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1280
    height: int = 1024
    pixel_format: str = "BayerRG8"
    acquisition_fps: int = 120
    exposure_us: int = 800
    gain_db: float = 6.0
    auto_exposure: bool = False
    auto_gain: bool = False
    auto_white_balance: bool = False
    buffer_size: int = 2


@dataclass(frozen=True)
class BoardConfig:
    width_cm: float = 21.0
    height_cm: float = 29.7
    rectified_px_per_cm: float = 40.0


@dataclass(frozen=True)
class CircleConfig:
    radius_cm: float = 6.0
    start_phase_deg: float = 0.0
    phase_direction: int = 1
    progress_timeout_ms: int = 300
    max_phase_correction_deg_s: float = 45.0


@dataclass(frozen=True)
class ControlConfig:
    command_hz: int = 100
    estimated_latency_ms: float = 25.0
    source_timeout_ms: int = 100
    kp_yaw: float = 1.0
    kd_yaw: float = 0.0
    kp_pitch: float = 1.0
    kd_pitch: float = 0.0
    derivative_alpha: float = 0.25
    deadband_enter_mdeg: int = 50
    deadband_exit_mdeg: int = 100
    max_yaw_rate_deg_s: float | None = None
    max_pitch_rate_deg_s: float | None = None
    max_yaw_accel_deg_s2: float | None = None
    max_pitch_accel_deg_s2: float | None = None
    yaw_sign: int = 1
    pitch_sign: int = 1
    swap_axes: bool = False


@dataclass(frozen=True)
class SerialConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 921600
    feedback_timeout_ms: int = 300


@dataclass(frozen=True)
class LaserConfig:
    backend: str = "mock"
    active_high: bool = True
    gpio_line: int | None = None


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    board: BoardConfig
    circle: CircleConfig
    control: ControlConfig
    serial: SerialConfig
    laser: LaserConfig


T = TypeVar("T")
_SECTIONS: dict[str, type[Any]] = {
    "camera": CameraConfig,
    "board": BoardConfig,
    "circle": CircleConfig,
    "control": ControlConfig,
    "serial": SerialConfig,
    "laser": LaserConfig,
}


def _build(cls: type[T], values: Mapping[str, Any] | None, section: str) -> T:
    values = values or {}
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"unknown configuration keys in {section}: {', '.join(unknown)}")
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"invalid {section} configuration: {exc}") from exc


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive")


def _validate(cfg: AppConfig, hardware_required: bool) -> None:
    _positive("camera.width", cfg.camera.width)
    _positive("camera.height", cfg.camera.height)
    _positive("camera.acquisition_fps", cfg.camera.acquisition_fps)
    _positive("board.width_cm", cfg.board.width_cm)
    _positive("board.height_cm", cfg.board.height_cm)
    _positive("board.rectified_px_per_cm", cfg.board.rectified_px_per_cm)
    _positive("circle.radius_cm", cfg.circle.radius_cm)
    _positive("control.command_hz", cfg.control.command_hz)
    _positive("serial.baudrate", cfg.serial.baudrate)
    if cfg.circle.phase_direction not in (-1, 1):
        raise ConfigError("circle.phase_direction must be -1 or 1")
    if cfg.control.yaw_sign not in (-1, 1) or cfg.control.pitch_sign not in (-1, 1):
        raise ConfigError("control axis signs must be -1 or 1")
    if not 0.0 <= cfg.control.derivative_alpha <= 1.0:
        raise ConfigError("control.derivative_alpha must be within [0, 1]")
    if hardware_required:
        required = (
            "max_yaw_rate_deg_s",
            "max_pitch_rate_deg_s",
            "max_yaw_accel_deg_s2",
            "max_pitch_accel_deg_s2",
        )
        for name in required:
            if getattr(cfg.control, name) is None:
                raise ConfigError(f"control.{name} is required in hardware mode")
        if cfg.laser.backend == "gpio" and cfg.laser.gpio_line is None:
            raise ConfigError("laser.gpio_line is required for the gpio backend")


def load_config(path: str | Path, hardware_required: bool = False) -> AppConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load configuration {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError("configuration root must be a mapping")
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ConfigError(f"unknown configuration keys: {', '.join(unknown)}")
    sections = {name: _build(cls, raw.get(name), name) for name, cls in _SECTIONS.items()}
    cfg = AppConfig(**sections)
    _validate(cfg, hardware_required)
    return cfg
