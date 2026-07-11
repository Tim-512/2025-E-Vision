from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Any


class OperatingMode(IntEnum):
    SAFE = 0
    SEARCH = 1
    CENTER = 2
    AIM = 3
    TRACK = 4
    CIRCLE = 5
    RECOVER = 6
    FAULT = 7


class LaserMode(IntEnum):
    OFF = 0
    ON = 1
    PWM = 2


class ControlFlags(IntFlag):
    BOARD_FULL_VALID = 1 << 0
    HOMOGRAPHY_VALID = 1 << 1
    LASER_SPOT_VALID = 1 << 2
    USING_PREDICTION = 1 << 3
    USING_LASER_MODEL = 1 << 4
    CIRCLE_SYNC_VALID = 1 << 5
    CAMERA_HEALTHY = 1 << 6
    EMERGENCY_STOP = 1 << 7


@dataclass(frozen=True)
class Frame:
    sequence: int
    captured_ns: int
    image: Any


@dataclass(frozen=True)
class BoardObservation:
    captured_ns: int
    corners_px: tuple[tuple[float, float], ...]
    center_px: tuple[float, float]
    confidence: float
    homography_valid: bool


@dataclass(frozen=True)
class LaserObservation:
    captured_ns: int
    position_px: tuple[float, float]
    confidence: float


@dataclass(frozen=True)
class ChassisProgress:
    running: bool
    lap_index: int
    progress_permille: int
    elapsed_ms: int
    received_ns: int = 0


@dataclass(frozen=True)
class GimbalFeedback:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_rate_deg_s: float = 0.0
    pitch_rate_deg_s: float = 0.0
    limit_flags: int = 0
    fault_flags: int = 0
    ack_sequence: int = 0
    received_ns: int = 0


@dataclass(frozen=True)
class RateCommand:
    yaw_rate_deg_s: float = 0.0
    pitch_rate_deg_s: float = 0.0
    target_valid: bool = False
    laser_mode: LaserMode = LaserMode.OFF
    flags: ControlFlags = ControlFlags(0)
    source_age_us: int = 0
