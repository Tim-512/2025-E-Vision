from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from ev_vision.camera.smoke import CameraSmokeStats, run_camera_smoke
from ev_vision.models import Frame


@dataclass
class FakeCamera:
    outcomes: list[Frame | Exception]

    def read(self, *, timeout_ms: int) -> Frame:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def frame(sequence: int, captured_ns: int) -> Frame:
    return Frame(
        sequence=sequence,
        captured_ns=captured_ns,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def test_smoke_stats_track_fps_timeouts_gaps_timestamps_and_rss_growth():
    camera = FakeCamera(
        [
            frame(10, 100),
            TimeoutError("late"),
            frame(12, 200),
            frame(13, 150),
        ]
    )
    clock_values = iter([0, 250_000_000, 500_000_000, 750_000_000, 1_000_000_000])
    rss_values = iter([1000, 1300])

    stats = run_camera_smoke(
        camera,
        duration_s=1.0,
        timeout_ms=5,
        monotonic_ns=lambda: next(clock_values),
        rss_bytes=lambda: next(rss_values),
    )

    assert stats == CameraSmokeStats(
        elapsed_s=1.0,
        frames=3,
        timeouts=1,
        sequence_gaps=1,
        non_monotonic_timestamps=1,
        first_sequence=10,
        last_sequence=13,
        start_rss_bytes=1000,
        end_rss_bytes=1300,
        disconnect=None,
    )
    assert stats.fps == pytest.approx(3.0)
    assert stats.timeout_rate == pytest.approx(0.25)
    assert stats.rss_growth_bytes == 300


def test_smoke_stops_and_reports_disconnect_without_hiding_completed_metrics():
    camera = FakeCamera([frame(1, 100), RuntimeError("USB device removed")])
    clock_values = iter([0, 100_000_000, 200_000_000])

    stats = run_camera_smoke(
        camera,
        duration_s=10.0,
        timeout_ms=5,
        monotonic_ns=lambda: next(clock_values),
        rss_bytes=lambda: 4096,
    )

    assert stats.frames == 1
    assert stats.elapsed_s == pytest.approx(0.2)
    assert stats.disconnect == "USB device removed"


def test_smoke_rejects_nonpositive_duration_or_timeout():
    camera = FakeCamera([])
    with pytest.raises(ValueError, match="duration_s"):
        run_camera_smoke(camera, duration_s=0, timeout_ms=5)
    with pytest.raises(ValueError, match="timeout_ms"):
        run_camera_smoke(camera, duration_s=1, timeout_ms=0)
