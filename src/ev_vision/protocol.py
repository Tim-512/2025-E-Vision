from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct

from ev_vision.models import ChassisProgress, ControlFlags, LaserMode, OperatingMode

MAGIC = b"\xAA\x55"
PROTOCOL_VERSION = 1
HEADER = struct.Struct("<2sBBHH")
CRC = struct.Struct("<H")
MAX_PAYLOAD_LENGTH = 1024


class MessageType(IntEnum):
    VISION_CONTROL = 0x01
    GIMBAL_FEEDBACK = 0x02
    HEARTBEAT = 0x03
    PARAMETER = 0x04
    CHASSIS_PROGRESS = 0x05
    SYSTEM_EVENT = 0x06


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class DecodedFrame:
    version: int
    message_type: MessageType
    sequence: int
    payload: bytes


_CONTROL_STRUCT = struct.Struct("<BBBBhhhhHHI")
_CHASSIS_STRUCT = struct.Struct("<BBHI")


@dataclass(frozen=True)
class VisionControlPayload:
    mode: OperatingMode
    target_valid: bool
    laser_mode: LaserMode
    flags: ControlFlags
    yaw_rate_cdeg_s: int
    pitch_rate_cdeg_s: int
    error_yaw_mdeg: int
    error_pitch_mdeg: int
    board_confidence_permille: int
    laser_confidence_permille: int
    source_age_us: int

    def pack(self) -> bytes:
        return _CONTROL_STRUCT.pack(
            int(self.mode), int(self.target_valid), int(self.laser_mode), int(self.flags),
            self.yaw_rate_cdeg_s, self.pitch_rate_cdeg_s,
            self.error_yaw_mdeg, self.error_pitch_mdeg,
            self.board_confidence_permille, self.laser_confidence_permille,
            self.source_age_us,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "VisionControlPayload":
        if len(data) != _CONTROL_STRUCT.size:
            raise ProtocolError(f"control payload must be {_CONTROL_STRUCT.size} bytes")
        values = _CONTROL_STRUCT.unpack(data)
        return cls(
            mode=OperatingMode(values[0]), target_valid=bool(values[1]),
            laser_mode=LaserMode(values[2]), flags=ControlFlags(values[3]),
            yaw_rate_cdeg_s=values[4], pitch_rate_cdeg_s=values[5],
            error_yaw_mdeg=values[6], error_pitch_mdeg=values[7],
            board_confidence_permille=values[8], laser_confidence_permille=values[9],
            source_age_us=values[10],
        )


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode_frame(message_type: MessageType | int, sequence: int, payload: bytes, version: int = PROTOCOL_VERSION) -> bytes:
    if len(payload) > MAX_PAYLOAD_LENGTH:
        raise ProtocolError("payload too large")
    if not 0 <= sequence <= 0xFFFF:
        raise ProtocolError("sequence outside uint16 range")
    header = HEADER.pack(MAGIC, version, int(message_type), len(payload), sequence)
    body = header + payload
    return body + CRC.pack(crc16_ccitt_false(body))


def decode_frame(data: bytes) -> DecodedFrame:
    if len(data) < HEADER.size + CRC.size:
        raise ProtocolError("frame too short")
    magic, version, type_value, length, sequence = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ProtocolError("invalid magic")
    if length > MAX_PAYLOAD_LENGTH:
        raise ProtocolError("payload too large")
    expected_size = HEADER.size + length + CRC.size
    if len(data) != expected_size:
        raise ProtocolError("frame length mismatch")
    expected_crc = CRC.unpack_from(data, expected_size - CRC.size)[0]
    actual_crc = crc16_ccitt_false(data[:-CRC.size])
    if actual_crc != expected_crc:
        raise ProtocolError("CRC mismatch")
    try:
        message_type = MessageType(type_value)
    except ValueError as exc:
        raise ProtocolError(f"unknown message type {type_value}") from exc
    return DecodedFrame(version, message_type, sequence, data[HEADER.size:-CRC.size])


class FrameParser:
    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[DecodedFrame]:
        self._buffer.extend(data)
        frames: list[DecodedFrame] = []
        while True:
            start = self._buffer.find(MAGIC)
            if start < 0:
                self._buffer[:] = self._buffer[-1:] if self._buffer.endswith(MAGIC[:1]) else b""
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < HEADER.size:
                break
            _, _, _, length, _ = HEADER.unpack_from(self._buffer)
            if length > MAX_PAYLOAD_LENGTH:
                del self._buffer[0]
                continue
            frame_size = HEADER.size + length + CRC.size
            if len(self._buffer) < frame_size:
                break
            candidate = bytes(self._buffer[:frame_size])
            try:
                frames.append(decode_frame(candidate))
                del self._buffer[:frame_size]
            except ProtocolError:
                del self._buffer[0]
        return frames


def pack_chassis_progress(progress: ChassisProgress) -> bytes:
    return _CHASSIS_STRUCT.pack(int(progress.running), progress.lap_index, progress.progress_permille, progress.elapsed_ms)


def unpack_chassis_progress(data: bytes) -> ChassisProgress:
    if len(data) != _CHASSIS_STRUCT.size:
        raise ProtocolError(f"chassis payload must be {_CHASSIS_STRUCT.size} bytes")
    running, lap_index, progress, elapsed = _CHASSIS_STRUCT.unpack(data)
    return ChassisProgress(bool(running), lap_index, progress, elapsed)
