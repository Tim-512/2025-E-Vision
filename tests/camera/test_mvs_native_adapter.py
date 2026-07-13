from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import pytest

from ev_vision_mvs_adapter import MvsSdkError, NativeMvsApi


@dataclass
class FakeDevice:
    serial: str
    raw: object


class FakeFrameInfo:
    def __init__(
        self,
        *,
        width=4,
        height=3,
        pixel_type=0x01080009,
        sequence=7,
        timestamp=123456789,
        frame_len=12,
    ):
        self.nWidth = width
        self.nHeight = height
        self.enPixelType = pixel_type
        self.nFrameNum = sequence
        self.nHostTimeStamp = timestamp
        self.nFrameLen = frame_len


class FakeFrame:
    def __init__(self):
        self.pBufAddr = None
        self.stFrameInfo = FakeFrameInfo()


class FakeCamera:
    def __init__(self, owner):
        self.owner = owner
        self.calls = []
        self.handle = object()

    def MV_CC_CreateHandle(self, raw):
        self.calls.append(("create", raw))
        return self.owner.return_codes.get("create", 0)

    def MV_CC_OpenDevice(self, access, key):
        self.calls.append(("open", access, key))
        return self.owner.return_codes.get("open", 0)

    def MV_CC_SetEnumValueByString(self, name, value):
        self.calls.append(("enum", name, value))
        return self.owner.return_codes.get(("enum", name), 0)

    def MV_CC_SetIntValue(self, name, value):
        self.calls.append(("int", name, value))
        return self.owner.return_codes.get(("int", name), 0)

    def MV_CC_SetImageNodeNum(self, value):
        self.calls.append(("nodes", value))
        return self.owner.return_codes.get("nodes", 0)

    def MV_CC_SetFloatValue(self, name, value):
        self.calls.append(("float", name, value))
        return self.owner.return_codes.get(("float", name), 0)

    def MV_CC_StartGrabbing(self):
        self.calls.append(("start",))
        return self.owner.return_codes.get("start", 0)

    def MV_CC_GetImageBuffer(self, frame, timeout_ms):
        self.calls.append(("get", timeout_ms))
        result = self.owner.frame_results.pop(0)
        if isinstance(result, int):
            return result
        frame.pBufAddr = result.pBufAddr
        frame.stFrameInfo = result.stFrameInfo
        return 0

    def MV_CC_FreeImageBuffer(self, frame):
        self.calls.append(("free", frame))
        self.owner.freed.append(frame)
        return self.owner.return_codes.get("free", 0)

    def MV_CC_StopGrabbing(self):
        self.calls.append(("stop",))
        return self.owner.return_codes.get("stop", 0)

    def MV_CC_CloseDevice(self):
        self.calls.append(("close",))
        return self.owner.return_codes.get("close", 0)

    def MV_CC_DestroyHandle(self):
        self.calls.append(("destroy",))
        return self.owner.return_codes.get("destroy", 0)


class FakeBindings:
    MV_ACCESS_Exclusive = 1
    MV_E_NODATA = 0x80000007
    PIXEL_NAMES = {0x01080009: "BayerRG8"}

    def __init__(self):
        self.devices = [FakeDevice("SERIAL-A", object()), FakeDevice("SERIAL-B", object())]
        self.return_codes = {}
        self.frame_results = []
        self.freed = []
        self.cameras = []
        self.initialize_calls = 0
        self.finalize_calls = 0

    def initialize(self):
        self.initialize_calls += 1

    def finalize(self):
        self.finalize_calls += 1

    def enumerate_devices(self):
        return list(self.devices)

    def create_camera(self):
        camera = FakeCamera(self)
        self.cameras.append(camera)
        return camera

    def create_frame(self):
        return FakeFrame()

    def array_from_frame(self, frame):
        return np.asarray(frame.pBufAddr, dtype=np.uint8)


def make_api(*, clock_ns=None):
    bindings = FakeBindings()
    return NativeMvsApi(bindings, clock_ns=clock_ns), bindings


def test_lists_devices_and_opens_selected_serial_exclusively():
    api, bindings = make_api()
    assert api.list_devices() == ["SERIAL-A", "SERIAL-B"]
    handle = api.open_device("SERIAL-B")
    camera = bindings.cameras[-1]
    assert camera.calls[:2] == [
        ("create", bindings.devices[1].raw),
        ("open", bindings.MV_ACCESS_Exclusive, 0),
    ]
    assert handle.serial == "SERIAL-B"
    assert bindings.initialize_calls == 1


def test_open_failure_destroys_created_handle():
    api, bindings = make_api()
    bindings.return_codes["open"] = 0x8000000B
    with pytest.raises(MvsSdkError, match="MV_CC_OpenDevice"):
        api.open_device("SERIAL-A")
    assert bindings.cameras[-1].calls[-1] == ("destroy",)


def test_parameter_calls_and_stream_buffer_count_mapping():
    api, bindings = make_api()
    handle = api.open_device("SERIAL-A")
    api.set_enum(handle, "PixelFormat", "BayerRG8")
    api.set_int(handle, "Width", 1280)
    api.set_int(handle, "StreamBufferCountManual", 2)
    api.set_float(handle, "ExposureTime", 800.0)
    api.start_grabbing(handle)
    assert bindings.cameras[-1].calls[-5:] == [
        ("enum", "PixelFormat", "BayerRG8"),
        ("int", "Width", 1280),
        ("nodes", 2),
        ("float", "ExposureTime", 800.0),
        ("start",),
    ]


def test_get_frame_exposes_bayerrg8_data_sequence_then_releases_once():
    api, bindings = make_api(clock_ns=lambda: 987_654_321)
    handle = api.open_device("SERIAL-A")
    api.start_grabbing(handle)
    raw = np.arange(12, dtype=np.uint8).reshape(3, 4)
    sdk_frame = FakeFrame()
    sdk_frame.pBufAddr = raw
    sdk_frame.stFrameInfo = FakeFrameInfo(timestamp=42)
    bindings.frame_results.append(sdk_frame)

    packet = api.get_frame(handle, timeout_ms=25)

    assert packet is not None
    assert packet.pixel_format == "BayerRG8"
    assert packet.sequence == 7
    assert packet.timestamp_ns == 987_654_321
    assert np.array_equal(packet.data, raw)
    api.release_frame(handle, packet)
    api.release_frame(handle, packet)
    assert len(bindings.freed) == 1


def test_no_data_is_a_timeout_without_buffer_release():
    api, bindings = make_api()
    handle = api.open_device("SERIAL-A")
    api.start_grabbing(handle)
    bindings.frame_results.append(bindings.MV_E_NODATA)
    assert api.get_frame(handle, timeout_ms=5) is None
    assert bindings.freed == []


def test_invalid_frame_is_released_before_error_is_raised():
    api, bindings = make_api()
    handle = api.open_device("SERIAL-A")
    api.start_grabbing(handle)
    sdk_frame = FakeFrame()
    sdk_frame.pBufAddr = np.zeros((2, 2), dtype=np.uint8)
    sdk_frame.stFrameInfo = FakeFrameInfo(width=4, height=3, frame_len=4)
    bindings.frame_results.append(sdk_frame)
    with pytest.raises(MvsSdkError, match="short frame"):
        api.get_frame(handle, timeout_ms=5)
    assert len(bindings.freed) == 1


def test_sdk_is_finalized_when_api_is_garbage_collected():
    api, bindings = make_api()
    assert bindings.initialize_calls == 1
    del api
    gc.collect()
    assert bindings.finalize_calls == 1


def test_stop_and_close_are_idempotent_and_destroy_handle():
    api, bindings = make_api()
    handle = api.open_device("SERIAL-A")
    api.start_grabbing(handle)
    api.stop_grabbing(handle)
    api.stop_grabbing(handle)
    api.close_device(handle)
    api.close_device(handle)
    assert bindings.cameras[-1].calls[-3:] == [("stop",), ("close",), ("destroy",)]


def test_stream_buffer_count_falls_back_to_genicam_integer_node():
    api, bindings = make_api()
    handle = api.open_device("SERIAL-A")
    camera = bindings.cameras[-1]
    camera.MV_CC_SetImageNodeNum = None

    api.set_int(handle, "StreamBufferCountManual", 3)

    assert camera.calls[-1] == ("int", "StreamBufferCountManual", 3)


def test_non_timeout_sdk_error_is_propagated_without_buffer_release():
    api, bindings = make_api()
    handle = api.open_device("SERIAL-A")
    api.start_grabbing(handle)
    bindings.frame_results.append(0x80000302)

    with pytest.raises(MvsSdkError, match="0x80000302"):
        api.get_frame(handle, timeout_ms=5)

    assert bindings.freed == []


def test_stop_failure_does_not_leave_handle_marked_as_grabbing():
    api, bindings = make_api()
    handle = api.open_device("SERIAL-A")
    api.start_grabbing(handle)
    bindings.return_codes["stop"] = 0x80000302

    with pytest.raises(MvsSdkError, match="MV_CC_StopGrabbing"):
        api.stop_grabbing(handle)

    assert handle.grabbing is False


def test_close_attempts_close_and_destroy_after_stop_failure():
    api, bindings = make_api()
    handle = api.open_device("SERIAL-A")
    api.start_grabbing(handle)
    bindings.return_codes["stop"] = 0x80000302

    with pytest.raises(MvsSdkError, match="MV_CC_StopGrabbing"):
        api.close_device(handle)

    assert handle.closed is True
    assert bindings.cameras[-1].calls[-3:] == [("stop",), ("close",), ("destroy",)]


def test_release_failure_remains_retryable_instead_of_leaking_buffer_state():
    api, bindings = make_api()
    handle = api.open_device("SERIAL-A")
    api.start_grabbing(handle)
    raw = np.arange(12, dtype=np.uint8).reshape(3, 4)
    sdk_frame = FakeFrame()
    sdk_frame.pBufAddr = raw
    bindings.frame_results.append(sdk_frame)
    packet = api.get_frame(handle, timeout_ms=5)
    assert packet is not None
    bindings.return_codes["free"] = 0x80000302

    with pytest.raises(MvsSdkError, match="MV_CC_FreeImageBuffer"):
        api.release_frame(handle, packet)

    assert packet.released is False
    bindings.return_codes["free"] = 0
    api.release_frame(handle, packet)
    assert packet.released is True
    assert len(bindings.freed) == 2
