from __future__ import annotations

import json

import cv2
import numpy as np

from ev_vision.models import BoardObservation, LaserObservation
from ev_vision.telemetry import DebugRecord, TelemetryWriter, render_overlay


def test_bounded_queue_is_nonblocking_and_counts_drops(tmp_path) -> None:
    writer = TelemetryWriter(tmp_path / "telemetry.jsonl", capacity=2)

    assert writer.submit(DebugRecord(frame_sequence=1, captured_ns=10, mode="SEARCH"))
    assert writer.submit(DebugRecord(frame_sequence=2, captured_ns=20, mode="TRACK"))
    assert not writer.submit(DebugRecord(frame_sequence=3, captured_ns=30, mode="TRACK"))
    assert writer.dropped_records == 1
    writer.close()


def test_jsonl_schema_and_async_flush(tmp_path) -> None:
    path = tmp_path / "telemetry.jsonl"
    with TelemetryWriter(path, capacity=4) as writer:
        writer.start()
        assert writer.submit(
            DebugRecord(
                frame_sequence=7,
                captured_ns=123,
                mode="TRACK",
                board_confidence=0.91,
                laser_confidence=0.84,
                target_px=(100.5, 80.25),
            )
        )

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "board_confidence": 0.91,
            "captured_ns": 123,
            "frame_sequence": 7,
            "laser_confidence": 0.84,
            "mode": "TRACK",
            "target_px": [100.5, 80.25],
        }
    ]


def test_overlay_returns_annotated_copy() -> None:
    image = np.zeros((160, 240, 3), np.uint8)
    board = BoardObservation(1, ((20, 20), (210, 25), (205, 135), (25, 130)), (115, 77.5), 0.9, True)
    laser = LaserObservation(1, (125.5, 82.5), 0.8)

    overlay = render_overlay(image, board=board, laser=laser, target_px=(140, 90), mode="TRACK")

    assert overlay.shape == image.shape
    assert np.count_nonzero(image) == 0
    assert np.count_nonzero(overlay) > 0
    assert np.any(overlay[80:86, 123:129])
