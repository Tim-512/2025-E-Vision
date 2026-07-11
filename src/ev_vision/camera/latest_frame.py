from __future__ import annotations

import threading
import time

from ev_vision.models import Frame


class LatestFrameBuffer:
    """Single-slot frame buffer that never queues stale camera frames."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: Frame | None = None

    def publish(self, frame: Frame) -> None:
        with self._condition:
            if self._frame is not None and frame.sequence <= self._frame.sequence:
                raise ValueError("frame sequence must increase monotonically")
            self._frame = frame
            self._condition.notify_all()

    def wait_next(self, after_sequence: int, timeout_s: float) -> Frame:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._frame is None or self._frame.sequence <= after_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for latest frame")
                self._condition.wait(remaining)
            return self._frame

    def peek(self) -> Frame | None:
        with self._condition:
            return self._frame
