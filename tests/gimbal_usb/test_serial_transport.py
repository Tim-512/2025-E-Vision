from __future__ import annotations

import builtins
import importlib
import struct
import sys
from collections.abc import Iterable
from typing import Any

import pytest

from ev_vision.gimbal_usb.protocol import GimbalTargetCommand, crc16_modbus

_FRAME = struct.Struct("<2sBBBBHfffBBBBH")


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeSerial:
    def __init__(self, write_results: Iterable[int | BaseException] = ()) -> None:
        self.writes: list[bytes] = []
        self.closed = False
        self._write_results = iter(write_results)

    def write(self, frame: bytes) -> int:
        self.writes.append(bytes(frame))
        try:
            result = next(self._write_results)
        except StopIteration:
            return len(frame)
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self) -> None:
        self.closed = True


class RecordingFactory:
    def __init__(self, outcomes: Iterable[FakeSerial | BaseException]) -> None:
        self._outcomes = iter(outcomes)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeSerial:
        self.calls.append(dict(kwargs))
        outcome = next(self._outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _decode(frame: bytes) -> dict[str, Any]:
    assert len(frame) == _FRAME.size == 26
    unpacked = _FRAME.unpack(frame)
    assert unpacked[0] == b"\xA5\x5A"
    assert unpacked[-1] == crc16_modbus(frame[:-2])
    return {
        "sequence": unpacked[5],
        "yaw_deg": unpacked[6],
        "pitch_deg": unpacked[7],
        "distance": unpacked[8],
        "tracking": unpacked[9],
        "fire": unpacked[10],
        "target_id": unpacked[11],
        "reserved": unpacked[12],
    }


def _transport_type() -> type[Any]:
    from ev_vision.gimbal_usb.serial_transport import GimbalSerialTransport

    return GimbalSerialTransport


def test_module_import_does_not_require_pyserial(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "ev_vision.gimbal_usb.serial_transport"
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "serial" or name.startswith("serial."):
            raise AssertionError("pyserial imported while importing transport module")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop(module_name, None)

    imported = importlib.import_module(module_name)

    assert hasattr(imported, "GimbalSerialTransport")


def test_open_uses_115200_8n1_and_disables_flow_control() -> None:
    serial_handle = FakeSerial()
    factory = RecordingFactory([serial_handle])
    transport = _transport_type()(
        "/dev/ttyACM-test",
        baudrate=115200,
        serial_factory=factory,
        clock=FakeClock(),
        sleep=lambda _seconds: None,
    )

    assert transport.transmit(GimbalTargetCommand.safe()) is False

    assert factory.calls == [
        {
            "port": "/dev/ttyACM-test",
            "baudrate": 115200,
            "bytesize": 8,
            "parity": "N",
            "stopbits": 1,
            "timeout": 0,
            "write_timeout": 0.2,
            "xonxoff": False,
            "rtscts": False,
            "dsrdtr": False,
        }
    ]


def test_first_frame_after_open_is_safe_then_requested_frame_is_sent() -> None:
    serial_handle = FakeSerial()
    transport = _transport_type()(
        "/dev/test",
        serial_factory=RecordingFactory([serial_handle]),
        clock=FakeClock(),
    )
    requested = GimbalTargetCommand(1.25, -2.5, True)

    assert transport.transmit(requested) is False
    assert _decode(serial_handle.writes[0]) == {
        "sequence": 0,
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "distance": 0.0,
        "tracking": 0,
        "fire": 0,
        "target_id": 0,
        "reserved": 0,
    }

    assert transport.transmit(requested) is True
    decoded = _decode(serial_handle.writes[1])
    assert decoded["sequence"] == 1
    assert decoded["yaw_deg"] == pytest.approx(1.25)
    assert decoded["pitch_deg"] == pytest.approx(-2.5)
    assert decoded["tracking"] == 1


def test_open_failure_is_contained_and_retried_only_after_interval() -> None:
    clock = FakeClock(10.0)
    serial_handle = FakeSerial()
    factory = RecordingFactory([OSError("port busy"), serial_handle])
    transport = _transport_type()(
        "/dev/test",
        reconnect_interval_s=1.0,
        serial_factory=factory,
        clock=clock,
    )

    assert transport.transmit(GimbalTargetCommand.safe()) is False
    assert transport.transmit(GimbalTargetCommand.safe()) is False
    assert len(factory.calls) == 1
    assert transport.snapshot().open_errors == 1
    assert "port busy" in (transport.snapshot().last_error or "")

    clock.advance(0.999)
    assert transport.transmit(GimbalTargetCommand.safe()) is False
    assert len(factory.calls) == 1

    clock.advance(0.001)
    assert transport.transmit(GimbalTargetCommand.safe()) is False
    assert len(factory.calls) == 2
    assert transport.snapshot().connected is True


def test_short_write_disconnects_without_advancing_sequence() -> None:
    clock = FakeClock()
    first = FakeSerial(write_results=[26, 7])
    second = FakeSerial()
    factory = RecordingFactory([first, second])
    transport = _transport_type()(
        "/dev/test",
        reconnect_interval_s=1.0,
        serial_factory=factory,
        clock=clock,
    )

    assert transport.transmit(GimbalTargetCommand.safe()) is False
    assert transport.snapshot().sequence == 1
    assert transport.transmit(GimbalTargetCommand(1.0, 1.0, True)) is False
    assert first.closed is True
    assert transport.snapshot().connected is False
    assert transport.snapshot().sequence == 1
    assert transport.snapshot().write_errors == 1
    assert "short serial write" in (transport.snapshot().last_error or "")

    clock.advance(1.0)
    assert transport.transmit(GimbalTargetCommand(9.0, 9.0, True)) is False
    reopened = _decode(second.writes[0])
    assert reopened["sequence"] == 1
    assert reopened["tracking"] == 0
    assert transport.snapshot().sequence == 2


def test_write_exception_is_contained_and_reconnect_starts_with_safe_frame() -> None:
    clock = FakeClock()
    first = FakeSerial(write_results=[26, OSError("USB unplugged")])
    second = FakeSerial()
    transport = _transport_type()(
        "/dev/test",
        reconnect_interval_s=1.0,
        serial_factory=RecordingFactory([first, second]),
        clock=clock,
    )

    assert transport.transmit(GimbalTargetCommand.safe()) is False
    assert transport.transmit(GimbalTargetCommand(3.0, 4.0, True)) is False
    assert first.closed is True
    assert "USB unplugged" in (transport.snapshot().last_error or "")

    clock.advance(1.0)
    assert transport.transmit(GimbalTargetCommand(3.0, 4.0, True)) is False
    assert _decode(second.writes[0])["tracking"] == 0

    assert transport.transmit(GimbalTargetCommand(3.0, 4.0, True)) is True
    assert _decode(second.writes[1])["tracking"] == 1


def test_safe_handshake_write_failure_retries_after_interval() -> None:
    clock = FakeClock()
    first = FakeSerial(write_results=[OSError("handshake failed")])
    second = FakeSerial()
    factory = RecordingFactory([first, second])
    transport = _transport_type()(
        "/dev/test",
        reconnect_interval_s=1.0,
        serial_factory=factory,
        clock=clock,
    )

    assert transport.transmit(GimbalTargetCommand(1.0, 2.0, True)) is False
    assert first.closed is True
    assert transport.snapshot().connected is False
    assert transport.snapshot().write_errors == 1
    assert transport.snapshot().sequence == 0

    assert transport.transmit(GimbalTargetCommand.safe()) is False
    assert len(factory.calls) == 1
    clock.advance(1.0)
    assert transport.transmit(GimbalTargetCommand.safe()) is False
    assert len(factory.calls) == 2
    assert _decode(second.writes[0])["sequence"] == 0


def test_sequence_wraps_after_only_successful_complete_writes() -> None:
    serial_handle = FakeSerial()
    transport = _transport_type()(
        "/dev/test",
        serial_factory=RecordingFactory([serial_handle]),
        clock=FakeClock(),
        initial_sequence=0xFFFF,
    )

    assert transport.transmit(GimbalTargetCommand.safe()) is False
    assert transport.transmit(GimbalTargetCommand.safe()) is True

    assert [_decode(frame)["sequence"] for frame in serial_handle.writes] == [
        0xFFFF,
        0,
    ]
    assert transport.snapshot().sequence == 1


def test_sequence_continues_across_reconnect_and_does_not_reset() -> None:
    clock = FakeClock()
    first = FakeSerial(write_results=[26, 26, OSError("disconnect")])
    second = FakeSerial()
    transport = _transport_type()(
        "/dev/test",
        reconnect_interval_s=1.0,
        serial_factory=RecordingFactory([first, second]),
        clock=clock,
        initial_sequence=42,
    )

    assert transport.transmit(GimbalTargetCommand.safe()) is False
    assert transport.transmit(GimbalTargetCommand(1.0, 2.0, True)) is True
    assert transport.transmit(GimbalTargetCommand(3.0, 4.0, True)) is False
    assert transport.snapshot().sequence == 44

    clock.advance(1.0)
    assert transport.transmit(GimbalTargetCommand(5.0, 6.0, True)) is False

    assert [_decode(frame)["sequence"] for frame in first.writes] == [42, 43, 44]
    assert _decode(second.writes[0])["sequence"] == 44
    assert _decode(second.writes[0])["tracking"] == 0
    assert transport.snapshot().sequence == 45
    assert transport.snapshot().reconnects == 1


def test_snapshot_reports_successful_frames_and_clears_error_after_reconnect() -> None:
    clock = FakeClock()
    serial_handle = FakeSerial()
    factory = RecordingFactory([OSError("missing"), serial_handle])
    transport = _transport_type()(
        "/dev/test",
        reconnect_interval_s=1.0,
        serial_factory=factory,
        clock=clock,
    )

    assert transport.transmit(GimbalTargetCommand.safe()) is False
    failed = transport.snapshot()
    assert failed.connected is False
    assert failed.sent_frames == 0
    assert failed.connect_count == 0
    assert failed.open_errors == 1
    assert failed.last_error is not None

    clock.advance(1.0)
    assert transport.transmit(GimbalTargetCommand.safe()) is False
    recovered = transport.snapshot()
    assert recovered.connected is True
    assert recovered.sent_frames == 1
    assert recovered.connect_count == 1
    assert recovered.sequence == 1
    assert recovered.last_error is None


def test_close_attempts_five_safe_frames_and_then_closes() -> None:
    serial_handle = FakeSerial()
    transport = _transport_type()(
        "/dev/test",
        serial_factory=RecordingFactory([serial_handle]),
        clock=FakeClock(),
    )
    assert transport.transmit(GimbalTargetCommand.safe()) is False
    assert transport.transmit(GimbalTargetCommand(1.0, 2.0, True)) is True

    transport.close()

    shutdown_frames = serial_handle.writes[-5:]
    assert len(shutdown_frames) == 5
    assert [_decode(frame)["tracking"] for frame in shutdown_frames] == [0] * 5
    assert [_decode(frame)["sequence"] for frame in shutdown_frames] == [2, 3, 4, 5, 6]
    assert serial_handle.closed is True
    assert transport.snapshot().connected is False
    assert transport.snapshot().closed is True


def test_close_keeps_attempting_all_five_safe_frames_after_failures() -> None:
    serial_handle = FakeSerial(
        write_results=[26, OSError("shutdown 1"), 3, 26, OSError("shutdown 4"), 26]
    )
    transport = _transport_type()(
        "/dev/test",
        serial_factory=RecordingFactory([serial_handle]),
        clock=FakeClock(),
        sleep=lambda _seconds: None,
    )
    assert transport.transmit(GimbalTargetCommand.safe()) is False

    transport.close(safe_frames=5)

    assert len(serial_handle.writes) == 6
    assert [_decode(frame)["tracking"] for frame in serial_handle.writes[-5:]] == [0] * 5
    assert serial_handle.closed is True
    assert transport.snapshot().write_errors == 3
    assert transport.snapshot().sent_frames == 3
    assert transport.snapshot().sequence == 3


def test_close_is_idempotent_and_transmit_after_close_is_deferred() -> None:
    serial_handle = FakeSerial()
    transport = _transport_type()(
        "/dev/test",
        serial_factory=RecordingFactory([serial_handle]),
        clock=FakeClock(),
    )
    assert transport.transmit(GimbalTargetCommand.safe()) is False

    transport.close(safe_frames=2)
    writes_after_first_close = len(serial_handle.writes)
    transport.close(safe_frames=2)

    assert len(serial_handle.writes) == writes_after_first_close
    assert transport.transmit(GimbalTargetCommand(1.0, 2.0, True)) is False
    assert len(serial_handle.writes) == writes_after_first_close
