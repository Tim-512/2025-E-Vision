from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import queue
import threading
from typing import Any

import cv2
import numpy as np

from ev_vision.models import BoardObservation, LaserObservation


@dataclass(frozen=True)
class DebugRecord:
    frame_sequence: int
    captured_ns: int
    mode: str
    board_confidence: float | None = None
    laser_confidence: float | None = None
    target_px: tuple[float, float] | None = None

    def as_json(self) -> str:
        payload = {key: value for key, value in asdict(self).items() if value is not None}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class TelemetryWriter:
    def __init__(self, path: str | Path, *, capacity: int = 128) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.path = Path(path)
        self._queue: queue.Queue[DebugRecord | object] = queue.Queue(maxsize=capacity)
        self._sentinel = object()
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._dropped_records = 0

    @property
    def dropped_records(self) -> int:
        return self._dropped_records

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        if self._started:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="ev-telemetry", daemon=True)
        self._started = True
        self._thread.start()

    def submit(self, record: DebugRecord) -> bool:
        if self._closed:
            return False
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_records += 1
            return False
        return True

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            while True:
                item = self._queue.get()
                try:
                    if item is self._sentinel:
                        return
                    assert isinstance(item, DebugRecord)
                    stream.write(item.as_json() + "\n")
                    stream.flush()
                finally:
                    self._queue.task_done()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._started:
            return
        self._queue.put(self._sentinel)
        assert self._thread is not None
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise TimeoutError("telemetry writer did not stop")

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _point(point: tuple[float, float]) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def render_overlay(
    image: np.ndarray,
    *,
    board: BoardObservation | None = None,
    laser: LaserObservation | None = None,
    target_px: tuple[float, float] | None = None,
    mode: str = "",
) -> np.ndarray:
    output = image.copy()
    if board is not None:
        corners = np.rint(np.asarray(board.corners_px, dtype=np.float32)).astype(np.int32)
        cv2.polylines(output, [corners], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(output, _point(board.center_px), 4, (0, 200, 0), -1, cv2.LINE_AA)
    if laser is not None:
        center = _point(laser.position_px)
        cv2.drawMarker(output, center, (255, 0, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    if target_px is not None:
        center = _point(target_px)
        cv2.drawMarker(output, center, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
    if mode:
        cv2.putText(output, mode, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return output
