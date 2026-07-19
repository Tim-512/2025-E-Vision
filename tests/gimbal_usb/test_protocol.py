from __future__ import annotations

import math
import struct

import pytest

from ev_vision.gimbal_usb.protocol import (
    FLAGS,
    MAGIC,
    MESSAGE_TYPE_TARGET,
    PAYLOAD_LENGTH,
    VERSION,
    GimbalTargetCommand,
    crc16_modbus,
    encode_target_frame,
)

_FRAME = struct.Struct("<2sBBBBHfffBBBBH")


def _unpack(frame: bytes) -> tuple[object, ...]:
    assert len(frame) == _FRAME.size == 26
    return _FRAME.unpack(frame)


def test_crc16_modbus_known_vector() -> None:
    assert crc16_modbus(b"123456789") == 0x4B37


def test_valid_target_frame_has_exact_26_byte_layout() -> None:
    frame = encode_target_frame(
        GimbalTargetCommand(yaw_deg=1.25, pitch_deg=-2.5, tracking=True),
        sequence=0x1234,
    )

    unpacked = _unpack(frame)

    assert unpacked[:6] == (
        MAGIC,
        VERSION,
        MESSAGE_TYPE_TARGET,
        PAYLOAD_LENGTH,
        FLAGS,
        0x1234,
    )
    assert unpacked[6] == pytest.approx(1.25)
    assert unpacked[7] == pytest.approx(-2.5)
    assert unpacked[8:13] == (0.0, 1, 0, 0, 0)
    assert unpacked[13] == crc16_modbus(frame[:-2])
    assert frame[-2:] == struct.pack("<H", crc16_modbus(frame[:-2]))


def test_invalid_command_forces_entire_payload_to_zero() -> None:
    frame = encode_target_frame(
        GimbalTargetCommand(yaw_deg=99.0, pitch_deg=-88.0, tracking=False),
        sequence=7,
    )

    assert _unpack(frame)[6:13] == (0.0, 0.0, 0.0, 0, 0, 0, 0)


def test_safe_command_is_all_zero_payload() -> None:
    frame = encode_target_frame(GimbalTargetCommand.safe(), sequence=0)

    assert _unpack(frame)[6:13] == (0.0, 0.0, 0.0, 0, 0, 0, 0)


@pytest.mark.parametrize("sequence", [0, 1, 0xFFFE, 0xFFFF])
def test_sequence_accepts_uint16_boundaries(sequence: int) -> None:
    frame = encode_target_frame(GimbalTargetCommand.safe(), sequence)

    assert _unpack(frame)[5] == sequence


@pytest.mark.parametrize("sequence", [-1, 0x10000, True, False, 1.0])
def test_sequence_rejects_values_that_are_not_uint16(sequence: object) -> None:
    with pytest.raises(ValueError, match="uint16"):
        encode_target_frame(GimbalTargetCommand.safe(), sequence)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("yaw_deg", "pitch_deg"),
    [
        (math.nan, 0.0),
        (math.inf, 0.0),
        (-math.inf, 0.0),
        (0.0, math.nan),
        (0.0, math.inf),
        (0.0, -math.inf),
    ],
)
def test_tracking_command_requires_finite_angles(
    yaw_deg: float,
    pitch_deg: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        encode_target_frame(
            GimbalTargetCommand(
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                tracking=True,
            ),
            sequence=0,
        )


def test_invalid_command_ignores_nonfinite_angles_and_remains_safe() -> None:
    frame = encode_target_frame(
        GimbalTargetCommand(yaw_deg=math.nan, pitch_deg=math.inf, tracking=False),
        sequence=9,
    )

    assert _unpack(frame)[6:13] == (0.0, 0.0, 0.0, 0, 0, 0, 0)
