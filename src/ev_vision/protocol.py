from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import struct

from ev_vision.models import ChassisProgress, ControlFlags, LaserMode, OperatingMode

MAGIC = b"\xAA\x55"
PROTOCOL_VERSION_V1 = 1
PROTOCOL_VERSION_V2 = 2
PROTOCOL_VERSION = PROTOCOL_VERSION_V2
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
_CONTROL_V2_STRUCT = struct.Struct("<BBBBHHhhhhhhII")  # 28 bytes
_GIMBAL_V2_STRUCT = struct.Struct("<BBHiihhHHI")       # 24 bytes
_CHASSIS_STRUCT = struct.Struct("<BBHI")


class TrackingStateCode(IntEnum):
    SEARCHING = 0
    CONFIRMING = 1
    TRACKING = 2
    PREDICTING = 3
    LOST = 4
    FAULT = 5


class ObservationSourceCode(IntEnum):
    NONE = 0
    FULL_BOARD = 1
    CONCENTRIC_ARCS = 2
    SINGLE_ARC = 3
    WHITE_REGION = 4
    FUSED_PARTIAL = 5
    PREDICTED = 6


class VisionControlFlagsV2(IntFlag):
    CAMERA_HEALTHY = 1 << 0
    FULL_BOARD_VISIBLE = 1 << 1
    HOMOGRAPHY_VALID = 1 << 2
    MULTIPLE_ARCS_VALID = 1 << 3
    SINGLE_ARC_VALID = 1 << 4
    WHITE_REGION_VALID = 1 << 5
    USING_PREDICTION = 1 << 6
    TARGET_NEAR_IMAGE_EDGE = 1 << 7
    TARGET_PARTIALLY_OUTSIDE = 1 << 8
    OBSERVATION_STALE = 1 << 9
    CENTER_JUMP_REJECTED = 1 << 10
    SCALE_JUMP_REJECTED = 1 << 11
    FEEDBACK_STALE = 1 << 12
    GIMBAL_FAULT_RECEIVED = 1 << 13
    RESERVED = 1 << 14
    EMERGENCY_STOP = 1 << 15


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


@dataclass(frozen=True)
class VisionControlPayloadV2:
    operating_mode: OperatingMode
    target_valid: bool
    tracking_state: TrackingStateCode
    observation_source: ObservationSourceCode
    flags: VisionControlFlagsV2
    confidence_permille: int
    yaw_rate_cdeg_s: int
    pitch_rate_cdeg_s: int
    error_yaw_mdeg: int
    error_pitch_mdeg: int
    target_x_px: int
    target_y_px: int
    source_age_us: int
    source_frame_sequence: int

    def pack(self) -> bytes:
        yaw_rate = self.yaw_rate_cdeg_s if self.target_valid else 0
        pitch_rate = self.pitch_rate_cdeg_s if self.target_valid else 0
        return _CONTROL_V2_STRUCT.pack(
            int(self.operating_mode), int(self.target_valid),
            int(self.tracking_state), int(self.observation_source),
            int(self.flags), self.confidence_permille,
            yaw_rate, pitch_rate, self.error_yaw_mdeg,
            self.error_pitch_mdeg, self.target_x_px, self.target_y_px,
            self.source_age_us, self.source_frame_sequence,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "VisionControlPayloadV2":
        if len(data) != _CONTROL_V2_STRUCT.size:
            raise ProtocolError(f"control V2 payload must be {_CONTROL_V2_STRUCT.size} bytes")
        values = _CONTROL_V2_STRUCT.unpack(data)
        return cls(
            operating_mode=OperatingMode(values[0]),
            target_valid=bool(values[1]),
            tracking_state=TrackingStateCode(values[2]),
            observation_source=ObservationSourceCode(values[3]),
            flags=VisionControlFlagsV2(values[4]),
            confidence_permille=values[5],
            yaw_rate_cdeg_s=values[6],
            pitch_rate_cdeg_s=values[7],
            error_yaw_mdeg=values[8],
            error_pitch_mdeg=values[9],
            target_x_px=values[10],
            target_y_px=values[11],
            source_age_us=values[12],
            source_frame_sequence=values[13],
        )


@dataclass(frozen=True)
class GimbalFeedbackPayloadV2:
    state: int = 0
    fault_flags: int = 0
    ack_sequence: int = 0
    yaw_angle_mdeg: int = 0
    pitch_angle_mdeg: int = 0
    yaw_rate_cdeg_s: int = 0
    pitch_rate_cdeg_s: int = 0
    control_latency_us: int = 0
    reserved: int = 0
    controller_time_us: int = 0

    def pack(self) -> bytes:
        return _GIMBAL_V2_STRUCT.pack(
            self.state, self.fault_flags, self.ack_sequence,
            self.yaw_angle_mdeg, self.pitch_angle_mdeg,
            self.yaw_rate_cdeg_s, self.pitch_rate_cdeg_s,
            self.control_latency_us, self.reserved,
            self.controller_time_us,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "GimbalFeedbackPayloadV2":
        if len(data) != _GIMBAL_V2_STRUCT.size:
            raise ProtocolError(f"gimbal feedback V2 payload must be {_GIMBAL_V2_STRUCT.size} bytes")
        return cls(*_GIMBAL_V2_STRUCT.unpack(data))


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
