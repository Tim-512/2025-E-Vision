from __future__ import annotations

import sys

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from ev_vision.camera.hikrobot import CameraDisconnected, HikrobotCamera
from ev_vision.config import CameraConfig


@dataclass
class FakePacket:
    data: np.ndarray
    timestamp_ns: int
    sequence: int
    pixel_format: str = "BayerRG8"


class FakeMvsApi:
    def __init__(self, devices=("SERIAL-A", "SERIAL-B"), packets=()):
        self.devices = list(devices)
        self.packets = list(packets)
        self.calls = []
        self.released = []

    def list_devices(self):
        self.calls.append(("list_devices",))
        return self.devices

    def open_device(self, serial):
        self.calls.append(("open_device", serial))
        if serial not in self.devices:
            raise LookupError(serial)
        return {"serial": serial}

    def set_enum(self, handle, name, value):
        self.calls.append(("set_enum", name, value))

    def set_int(self, handle, name, value):
        self.calls.append(("set_int", name, value))

    def set_float(self, handle, name, value):
        self.calls.append(("set_float", name, value))

    def start_grabbing(self, handle):
        self.calls.append(("start_grabbing",))

    def get_frame(self, handle, timeout_ms):
        self.calls.append(("get_frame", timeout_ms))
        if not self.packets:
            return None
        packet = self.packets.pop(0)
        if isinstance(packet, Exception):
            raise packet
        return packet

    def release_frame(self, handle, packet):
        self.released.append(packet)

    def stop_grabbing(self, handle):
        self.calls.append(("stop_grabbing",))

    def close_device(self, handle):
        self.calls.append(("close_device",))


def config() -> CameraConfig:
    return CameraConfig(width=4, height=4, acquisition_fps=100, exposure_us=700, gain_db=3.5, buffer_size=1)


def test_selects_device_by_serial_and_applies_fixed_parameters() -> None:
    api = FakeMvsApi()
    with HikrobotCamera(api, config(), serial_number="SERIAL-B"):
        pass

    assert ("open_device", "SERIAL-B") in api.calls
    assert ("set_int", "Width", 4) in api.calls
    assert ("set_int", "Height", 4) in api.calls
    assert ("set_enum", "PixelFormat", "BayerRG8") in api.calls
    assert ("set_enum", "ExposureAuto", "Off") in api.calls
    assert ("set_float", "ExposureTime", 700.0) in api.calls
    assert ("set_enum", "GainAuto", "Off") in api.calls
    assert ("set_float", "Gain", 3.5) in api.calls
    assert api.calls[-2:] == [("stop_grabbing",), ("close_device",)]


def test_bayerrg8_conversion_timestamp_and_buffer_release() -> None:
    bayer = np.array(
        [[255, 0, 255, 0], [0, 128, 0, 128], [255, 0, 255, 0], [0, 128, 0, 128]],
        dtype=np.uint8,
    )
    packet = FakePacket(bayer, timestamp_ns=987654, sequence=12)
    api = FakeMvsApi(packets=[packet])

    with HikrobotCamera(api, config(), serial_number="SERIAL-A") as camera:
        frame = camera.read(timeout_ms=20)

    assert frame.sequence == 12
    assert frame.captured_ns == 987654
    assert frame.image.shape == (4, 4, 3)
    assert np.array_equal(frame.image, cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR))
    assert api.released == [packet]


def test_timeout_does_not_release_nonexistent_packet() -> None:
    api = FakeMvsApi()
    with HikrobotCamera(api, config()) as camera:
        with pytest.raises(TimeoutError):
            camera.read(timeout_ms=5)
    assert api.released == []


def test_disconnect_is_translated_and_packet_is_still_released() -> None:
    class DisconnectPacket(FakePacket):
        pass

    packet = DisconnectPacket(np.zeros((4, 4), np.uint8), 1, 1, pixel_format="Unsupported")
    api = FakeMvsApi(packets=[packet])
    with HikrobotCamera(api, config()) as camera:
        with pytest.raises(CameraDisconnected):
            camera.read(timeout_ms=10)
    assert api.released == [packet]


def test_latest_returns_only_newest_published_frame() -> None:
    packets = [
        FakePacket(np.zeros((4, 4), np.uint8), 100, 1),
        FakePacket(np.full((4, 4), 20, np.uint8), 200, 2),
    ]
    api = FakeMvsApi(packets=packets)
    with HikrobotCamera(api, config()) as camera:
        camera.read(timeout_ms=10)
        camera.read(timeout_ms=10)
        latest = camera.latest()
    assert latest is not None
    assert latest.sequence == 2
    assert latest.captured_ns == 200

def test_native_api_factory_is_lazy_and_uses_version_specific_adapter(monkeypatch) -> None:
    import types

    expected = object()
    module = types.SimpleNamespace(create_api=lambda: expected)
    monkeypatch.setitem(sys.modules, "fake_mvs_adapter", module)
    from ev_vision.camera.hikrobot import create_native_api

    assert create_native_api("fake_mvs_adapter") is expected


def test_open_failure_before_handle_assignment_is_translated_without_unbound_local():
    class OpenFailApi(FakeMvsApi):
        def open_device(self, serial):
            self.calls.append(("open_device", serial))
            raise RuntimeError("exclusive open failed")

    api = OpenFailApi()
    with pytest.raises(CameraDisconnected, match="failed to initialize") as caught:
        HikrobotCamera(api, config(), serial_number="SERIAL-A").open()

    assert isinstance(caught.value.__cause__, RuntimeError)
