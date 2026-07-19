from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .protocol import GimbalTargetCommand, encode_target_frame


@dataclass(frozen=True)
class SerialStats:
    """Bounded diagnostic state for the reconnecting CDC-ACM transport."""

    connected: bool
    sent_frames: int
    connect_count: int
    reconnects: int
    open_errors: int
    write_errors: int
    sequence: int
    last_error: str | None
    closed: bool


class GimbalSerialTransport:
    """Best-effort fail-closed sender for the independent gimbal USB link.

    The class never raises serial open or write errors from :meth:`transmit`.
    Opening is lazy, reconnect attempts are rate-limited, and every successful
    connection starts with one safe-invalid frame before a requested command
    may be transmitted.
    """

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 115200,
        reconnect_interval_s: float = 1.0,
        write_timeout_s: float = 0.2,
        serial_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        initial_sequence: int = 0,
    ) -> None:
        if not isinstance(port, str) or not port.strip():
            raise ValueError("port must be a non-empty string")
        if isinstance(baudrate, bool) or not isinstance(baudrate, int) or baudrate <= 0:
            raise ValueError("baudrate must be a positive integer")
        if (
            isinstance(reconnect_interval_s, bool)
            or not isinstance(reconnect_interval_s, (int, float))
            or not math.isfinite(float(reconnect_interval_s))
            or float(reconnect_interval_s) <= 0.0
        ):
            raise ValueError("reconnect_interval_s must be positive and finite")
        if (
            isinstance(write_timeout_s, bool)
            or not isinstance(write_timeout_s, (int, float))
            or not math.isfinite(float(write_timeout_s))
            or float(write_timeout_s) <= 0.0
        ):
            raise ValueError("write_timeout_s must be positive and finite")
        if isinstance(initial_sequence, bool) or not isinstance(initial_sequence, int):
            raise ValueError("initial_sequence must be a uint16")
        if not 0 <= initial_sequence <= 0xFFFF:
            raise ValueError("initial_sequence must be a uint16")
        if serial_factory is not None and not callable(serial_factory):
            raise ValueError("serial_factory must be callable")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if not callable(sleep):
            raise ValueError("sleep must be callable")

        self._port = port.strip()
        self._baudrate = baudrate
        self._reconnect_interval_s = float(reconnect_interval_s)
        self._write_timeout_s = float(write_timeout_s)
        self._serial_factory = serial_factory
        self._clock = clock
        # Kept injectable with the other time dependency for composition and
        # deterministic tests. Reconnect itself is non-blocking and deadline based.
        self._sleep = sleep

        self._serial: Any | None = None
        self._next_retry_at = 0.0
        self._sequence = initial_sequence
        self._sent_frames = 0
        self._connect_count = 0
        self._reconnects = 0
        self._open_errors = 0
        self._write_errors = 0
        self._last_error: str | None = None
        self._closed = False

    @staticmethod
    def _error_text(operation: str, error: BaseException) -> str:
        detail = str(error).strip()
        name = type(error).__name__
        return f"{operation} failed: {name}: {detail}" if detail else f"{operation} failed: {name}"

    def _factory(self, **kwargs: Any) -> Any:
        if self._serial_factory is not None:
            return self._serial_factory(**kwargs)

        # Deliberately lazy: importing this module must work on development
        # machines and in test environments where pyserial is not installed.
        import serial

        return serial.Serial(**kwargs)

    def _open(self) -> Any:
        return self._factory(
            port=self._port,
            baudrate=self._baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=0,
            write_timeout=self._write_timeout_s,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )

    def _write_to(self, serial_handle: Any, command: GimbalTargetCommand) -> None:
        frame = encode_target_frame(command, self._sequence)
        written = serial_handle.write(frame)
        if isinstance(written, bool) or not isinstance(written, int) or written != len(frame):
            raise OSError(f"short serial write: {written}/{len(frame)}")
        self._sequence = (self._sequence + 1) & 0xFFFF
        self._sent_frames += 1

    @staticmethod
    def _close_handle(serial_handle: Any) -> str | None:
        try:
            serial_handle.close()
        except Exception as exc:  # Serial cleanup must never stop detection.
            detail = str(exc).strip()
            name = type(exc).__name__
            return f"serial close failed: {name}: {detail}" if detail else f"serial close failed: {name}"
        return None

    def _schedule_retry(self, now: float) -> None:
        self._next_retry_at = now + self._reconnect_interval_s

    def _disconnect_after_write_error(self, now: float, error: BaseException) -> None:
        self._write_errors += 1
        self._last_error = self._error_text("serial write", error)
        serial_handle, self._serial = self._serial, None
        if serial_handle is not None:
            close_error = self._close_handle(serial_handle)
            if close_error is not None:
                self._last_error = f"{self._last_error}; {close_error}"
        self._schedule_retry(now)

    def transmit(self, command: GimbalTargetCommand) -> bool:
        """Try to send a requested command without propagating serial errors.

        ``True`` means the requested command itself was completely written.
        ``False`` means it was deferred, rejected by transport state, or a
        connection/write failure occurred. A newly opened connection writes a
        safe frame synchronously and always returns ``False`` for that call.
        """

        if self._closed:
            return False

        try:
            now = float(self._clock())
        except Exception as exc:
            self._last_error = self._error_text("serial clock", exc)
            return False
        if not math.isfinite(now):
            self._last_error = "serial clock failed: non-finite value"
            return False

        if self._serial is None:
            if now < self._next_retry_at:
                return False

            try:
                serial_handle = self._open()
            except Exception as exc:
                self._open_errors += 1
                self._last_error = self._error_text("serial open", exc)
                self._schedule_retry(now)
                return False

            previously_connected = self._connect_count > 0
            self._serial = serial_handle
            self._connect_count += 1
            if previously_connected:
                self._reconnects += 1

            try:
                self._write_to(serial_handle, GimbalTargetCommand.safe())
            except Exception as exc:
                self._disconnect_after_write_error(now, exc)
                return False

            self._last_error = None
            self._next_retry_at = 0.0
            return False

        try:
            self._write_to(self._serial, command)
        except Exception as exc:
            self._disconnect_after_write_error(now, exc)
            return False

        self._last_error = None
        return True

    def snapshot(self) -> SerialStats:
        return SerialStats(
            connected=self._serial is not None and not self._closed,
            sent_frames=self._sent_frames,
            connect_count=self._connect_count,
            reconnects=self._reconnects,
            open_errors=self._open_errors,
            write_errors=self._write_errors,
            sequence=self._sequence,
            last_error=self._last_error,
            closed=self._closed,
        )

    def close(self, *, safe_frames: int = 5) -> None:
        """Best-effort send shutdown safety frames, then close the handle."""

        if self._closed:
            return
        self._closed = True

        serial_handle, self._serial = self._serial, None
        if serial_handle is None:
            return

        try:
            attempts = 0 if isinstance(safe_frames, bool) else max(0, int(safe_frames))
            for _ in range(attempts):
                try:
                    self._write_to(serial_handle, GimbalTargetCommand.safe())
                except Exception as exc:
                    # Keep using the handle for all requested shutdown attempts:
                    # even after a failed write, a later safe frame may get through.
                    self._write_errors += 1
                    self._last_error = self._error_text("shutdown safe write", exc)
        finally:
            close_error = self._close_handle(serial_handle)
            if close_error is not None:
                self._last_error = close_error
