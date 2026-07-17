import dataclasses
import pytest

from ev_vision.models import ChassisProgress, ControlFlags, LaserMode, OperatingMode
from ev_vision.protocol import (
    FrameParser,
    MessageType,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_V1,
    PROTOCOL_VERSION_V2,
    GimbalFeedbackPayloadV2,
    ObservationSourceCode,
    TrackingStateCode,
    VisionControlFlagsV2,
    VisionControlPayloadV2,
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


def test_protocol_v2_versions_and_payload_sizes() -> None:
    assert (PROTOCOL_VERSION_V1, PROTOCOL_VERSION_V2, PROTOCOL_VERSION) == (1, 2, 2)
    control = VisionControlPayloadV2(
        operating_mode=OperatingMode.TRACK, target_valid=True,
        tracking_state=TrackingStateCode.TRACKING,
        observation_source=ObservationSourceCode.FULL_BOARD,
        flags=VisionControlFlagsV2.CAMERA_HEALTHY,
        confidence_permille=900, yaw_rate_cdeg_s=100,
        pitch_rate_cdeg_s=-100, error_yaw_mdeg=250,
        error_pitch_mdeg=-125, target_x_px=640, target_y_px=512,
        source_age_us=25_000, source_frame_sequence=17,
    )
    assert len(control.pack()) == 28
    assert len(GimbalFeedbackPayloadV2().pack()) == 24


def test_protocol_v2_code_values_are_frozen() -> None:
    assert [int(value) for value in TrackingStateCode] == [0, 1, 2, 3, 4, 5]
    assert [int(value) for value in ObservationSourceCode] == [0, 1, 2, 3, 4, 5, 6]
    expected_flags = {
        "CAMERA_HEALTHY": 1 << 0,
        "FULL_BOARD_VISIBLE": 1 << 1,
        "HOMOGRAPHY_VALID": 1 << 2,
        "MULTIPLE_ARCS_VALID": 1 << 3,
        "SINGLE_ARC_VALID": 1 << 4,
        "WHITE_REGION_VALID": 1 << 5,
        "USING_PREDICTION": 1 << 6,
        "TARGET_NEAR_IMAGE_EDGE": 1 << 7,
        "TARGET_PARTIALLY_OUTSIDE": 1 << 8,
        "OBSERVATION_STALE": 1 << 9,
        "CENTER_JUMP_REJECTED": 1 << 10,
        "SCALE_JUMP_REJECTED": 1 << 11,
        "FEEDBACK_STALE": 1 << 12,
        "GIMBAL_FAULT_RECEIVED": 1 << 13,
        "RESERVED": 1 << 14,
        "EMERGENCY_STOP": 1 << 15,
    }
    assert {flag.name: int(flag) for flag in VisionControlFlagsV2} == expected_flags


def test_vision_control_v2_round_trip_has_no_laser_fields() -> None:
    payload = VisionControlPayloadV2(
        operating_mode=OperatingMode.TRACK, target_valid=True,
        tracking_state=TrackingStateCode.TRACKING,
        observation_source=ObservationSourceCode.CONCENTRIC_ARCS,
        flags=(VisionControlFlagsV2.CAMERA_HEALTHY |
               VisionControlFlagsV2.MULTIPLE_ARCS_VALID),
        confidence_permille=812, yaw_rate_cdeg_s=-123,
        pitch_rate_cdeg_s=456, error_yaw_mdeg=-50,
        error_pitch_mdeg=75, target_x_px=701, target_y_px=481,
        source_age_us=2_500, source_frame_sequence=0x10203040,
    )
    assert VisionControlPayloadV2.unpack(payload.pack()) == payload
    assert not any("laser" in field.name for field in dataclasses.fields(payload))


def test_invalid_v2_payload_always_serializes_zero_rates() -> None:
    payload = VisionControlPayloadV2(
        operating_mode=OperatingMode.SEARCH, target_valid=False,
        tracking_state=TrackingStateCode.LOST,
        observation_source=ObservationSourceCode.NONE,
        flags=VisionControlFlagsV2.CAMERA_HEALTHY,
        confidence_permille=0, yaw_rate_cdeg_s=900,
        pitch_rate_cdeg_s=-800, error_yaw_mdeg=0,
        error_pitch_mdeg=0, target_x_px=0, target_y_px=0,
        source_age_us=200_000, source_frame_sequence=9,
    )
    decoded = VisionControlPayloadV2.unpack(payload.pack())
    assert decoded.target_valid is False
    assert decoded.yaw_rate_cdeg_s == decoded.pitch_rate_cdeg_s == 0


def test_gimbal_feedback_v2_round_trip() -> None:
    payload = GimbalFeedbackPayloadV2(
        state=2, fault_flags=3, ack_sequence=513,
        yaw_angle_mdeg=-12_345, pitch_angle_mdeg=67_890,
        yaw_rate_cdeg_s=-210, pitch_rate_cdeg_s=345,
        control_latency_us=2_500, reserved=0,
        controller_time_us=0x10203040,
    )
    assert GimbalFeedbackPayloadV2.unpack(payload.pack()) == payload


def test_v1_payload_stays_explicitly_available() -> None:
    payload = VisionControlPayload(
        mode=OperatingMode.TRACK, target_valid=True, laser_mode=LaserMode.ON,
        flags=ControlFlags.CAMERA_HEALTHY, yaw_rate_cdeg_s=1,
        pitch_rate_cdeg_s=2, error_yaw_mdeg=3, error_pitch_mdeg=4,
        board_confidence_permille=5, laser_confidence_permille=6,
        source_age_us=7,
    )
    frame = encode_frame(MessageType.VISION_CONTROL, 8, payload.pack(),
                         version=PROTOCOL_VERSION_V1)
    assert decode_frame(frame).version == PROTOCOL_VERSION_V1
    assert VisionControlPayload.unpack(decode_frame(frame).payload) == payload
