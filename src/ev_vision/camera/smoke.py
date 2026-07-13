from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from ev_vision.models import Frame


class ReadableCamera(Protocol):
    def read(self, *, timeout_ms: int) -> Frame: ...


@dataclass(frozen=True)
class CameraSmokeStats:
    elapsed_s: float
    frames: int
    timeouts: int
    sequence_gaps: int
    non_monotonic_timestamps: int
    first_sequence: int | None
    last_sequence: int | None
    start_rss_bytes: int
    end_rss_bytes: int
    disconnect: str | None

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def attempts(self) -> int:
        return self.frames + self.timeouts

    @property
    def timeout_rate(self) -> float:
        return self.timeouts / self.attempts if self.attempts else 0.0

    @property
    def sequence_gap_rate(self) -> float:
        if self.first_sequence is None or self.last_sequence is None:
            return 0.0
        advance = self.last_sequence - self.first_sequence
        return self.sequence_gaps / advance if advance > 0 else 0.0

    @property
    def rss_growth_bytes(self) -> int:
        return self.end_rss_bytes - self.start_rss_bytes


def process_rss_bytes() -> int:
    """Return Linux VmRSS without adding a psutil dependency."""
    status = Path("/proc/self/status")
    try:
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def run_camera_smoke(
    camera: ReadableCamera,
    *,
    duration_s: float,
    timeout_ms: int,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    rss_bytes: Callable[[], int] = process_rss_bytes,
    on_frame: Callable[[Frame], None] | None = None,
) -> CameraSmokeStats:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")

    start_ns = monotonic_ns()
    deadline_ns = start_ns + int(duration_s * 1_000_000_000)
    start_rss = rss_bytes()
    frames = 0
    timeouts = 0
    sequence_gaps = 0
    non_monotonic_timestamps = 0
    first_sequence: int | None = None
    last_sequence: int | None = None
    last_timestamp_ns: int | None = None
    disconnect: str | None = None
    now_ns = start_ns

    while now_ns < deadline_ns:
        try:
            frame = camera.read(timeout_ms=timeout_ms)
        except TimeoutError:
            timeouts += 1
        except Exception as exc:
            disconnect = str(exc) or type(exc).__name__
            now_ns = monotonic_ns()
            break
        else:
            sequence = int(frame.sequence)
            timestamp_ns = int(frame.captured_ns)
            if first_sequence is None:
                first_sequence = sequence
            if last_sequence is not None and sequence > last_sequence + 1:
                sequence_gaps += sequence - last_sequence - 1
            if last_timestamp_ns is not None and timestamp_ns <= last_timestamp_ns:
                non_monotonic_timestamps += 1
            last_sequence = sequence
            last_timestamp_ns = timestamp_ns
            frames += 1
            if on_frame is not None:
                on_frame(frame)
        now_ns = monotonic_ns()

    end_rss = rss_bytes()
    elapsed_s = max(0.0, (now_ns - start_ns) / 1_000_000_000)
    return CameraSmokeStats(
        elapsed_s=elapsed_s,
        frames=frames,
        timeouts=timeouts,
        sequence_gaps=sequence_gaps,
        non_monotonic_timestamps=non_monotonic_timestamps,
        first_sequence=first_sequence,
        last_sequence=last_sequence,
        start_rss_bytes=start_rss,
        end_rss_bytes=end_rss,
        disconnect=disconnect,
    )
