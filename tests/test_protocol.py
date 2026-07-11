import pytest

from ev_vision.models import ChassisProgress, ControlFlags, LaserMode, OperatingMode
from ev_vision.protocol import (
    FrameParser,
    MessageType,
    ProtocolError,
    VisionControlPayload,
    crc16_ccitt_false,
    decode_frame,
    encode_frame,
)


def test_crc_known_vector() -> None:
    assert crc16_ccitt_false(b"123456789") == 0x29B1


def test_frame_round_trip() -> None:
    frame = encode_frame(MessageType.HEARTBEAT, sequence=513, payload=b"abc")
    decoded = decode_frame(frame)
    assert decoded.message_type == MessageType.HEARTBEAT
    assert decoded.sequence == 513
    assert decoded.payload == b"abc"


def test_corrupted_frame_is_rejected() -> None:
    frame = bytearray(encode_frame(MessageType.HEARTBEAT, 1, b"abc"))
    frame[8] ^= 1
    with pytest.raises(ProtocolError, match="CRC"):
        decode_frame(bytes(frame))


def test_stream_parser_recovers_from_noise_partial_and_concatenated_frames() -> None:
    first = encode_frame(MessageType.HEARTBEAT, 1, b"a")
    second = encode_frame(MessageType.SYSTEM_EVENT, 2, b"bc")
    parser = FrameParser()
    assert parser.feed(b"noise" + first[:5]) == []
    frames = parser.feed(first[5:] + second)
    assert [(f.sequence, f.payload) for f in frames] == [(1, b"a"), (2, b"bc")]


def test_vision_control_payload_round_trip() -> None:
    payload = VisionControlPayload(
        mode=OperatingMode.TRACK, target_valid=True, laser_mode=LaserMode.ON,
        flags=ControlFlags.HOMOGRAPHY_VALID | ControlFlags.CAMERA_HEALTHY,
        yaw_rate_cdeg_s=-123, pitch_rate_cdeg_s=456,
        error_yaw_mdeg=-50, error_pitch_mdeg=75,
        board_confidence_permille=900, laser_confidence_permille=800,
        source_age_us=2500,
    )
    assert VisionControlPayload.unpack(payload.pack()) == payload


def test_chassis_progress_payload_round_trip() -> None:
    payload = ChassisProgress(True, 2, 750, 12345)
    from ev_vision.protocol import pack_chassis_progress, unpack_chassis_progress
    assert unpack_chassis_progress(pack_chassis_progress(payload)) == payload
