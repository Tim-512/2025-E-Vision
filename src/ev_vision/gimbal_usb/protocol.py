from __future__ import annotations

import math
import struct
from dataclasses import dataclass

MAGIC = b"\xA5\x5A"
VERSION = 1
MESSAGE_TYPE_TARGET = 0x01
PAYLOAD_LENGTH = 16
FLAGS = 0

_HEADER = struct.Struct("<2sBBBBH")
_PAYLOAD = struct.Struct("<fffBBBB")
_CRC = struct.Struct("<H")


@dataclass(frozen=True)
class GimbalTargetCommand:
    """Target angles and tracking validity sent to the gimbal controller."""

    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    tracking: bool = False

    @classmethod
    def safe(cls) -> "GimbalTargetCommand":
        """Return the fail-closed command with an all-zero payload."""

        return cls()


def crc16_modbus(data: bytes) -> int:
    """Calculate CRC16/MODBUS using polynomial 0xA001 and initial 0xFFFF."""

    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 0x0001 else crc >> 1
    return crc & 0xFFFF


def encode_target_frame(command: GimbalTargetCommand, sequence: int) -> bytes:
    """Encode one independent A5 5A target frame for the gimbal USB link."""

    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise ValueError("sequence must be a uint16")
    if not 0 <= sequence <= 0xFFFF:
        raise ValueError("sequence must be a uint16")

    tracking = bool(command.tracking)
    yaw_deg = float(command.yaw_deg) if tracking else 0.0
    pitch_deg = float(command.pitch_deg) if tracking else 0.0
    if tracking and (not math.isfinite(yaw_deg) or not math.isfinite(pitch_deg)):
        raise ValueError("valid target angles must be finite")

    body = _HEADER.pack(
        MAGIC,
        VERSION,
        MESSAGE_TYPE_TARGET,
        PAYLOAD_LENGTH,
        FLAGS,
        sequence,
    )
    body += _PAYLOAD.pack(
        yaw_deg,
        pitch_deg,
        0.0,
        int(tracking),
        0,
        0,
        0,
    )
    return body + _CRC.pack(crc16_modbus(body))
