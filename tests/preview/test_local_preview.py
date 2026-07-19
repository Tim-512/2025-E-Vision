from __future__ import annotations

from dataclasses import replace
import threading

import numpy as np

from ev_vision.models import Frame
from ev_vision.preview.local import render_local_overlay, resize_preview, run_local_preview
from ev_vision.tuning.models import DetectionSnapshot


def _valid_detection(sequence: int = 7) -> DetectionSnapshot:
    return DetectionSnapshot(
        enabled=True,
        detected=True,
        source_sequence=sequence,
        target_valid=True,
        corners_px=((20.0, 15.0), (90.0, 15.0), (90.0, 75.0), (20.0, 75.0)),
        center_px=(55.0, 45.0),
    )


def test_render_local_overlay_draws_only_valid_sequence_matched_green_geometry() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)

    valid = render_local_overlay(image, source_sequence=7, detection=_valid_detection())
    invalid = render_local_overlay(
        image,
        source_sequence=7,
        detection=replace(_valid_detection(), target_valid=False),
    )
    stale = render_local_overlay(image, source_sequence=8, detection=_valid_detection())

    assert np.any((valid[:, :, 1] > 0) & (valid[:, :, 0] == 0) & (valid[:, :, 2] == 0))
    assert np.array_equal(invalid, image)
    assert np.array_equal(stale, image)
    assert np.array_equal(image, np.zeros_like(image))


def test_resize_preview_preserves_aspect_ratio_and_does_not_upscale() -> None:
    image = np.zeros((1024, 1280, 3), dtype=np.uint8)
    assert resize_preview(image, max_width=640).shape == (512, 640, 3)
    small = np.zeros((100, 200, 3), dtype=np.uint8)
    assert resize_preview(small, max_width=640) is small


class _Service:
    def __init__(self) -> None:
        self.calls = 0

    def detection_frame_for_latest(self):
        self.calls += 1
        frame = Frame(sequence=self.calls, captured_ns=self.calls, image=np.zeros((20, 30, 3), dtype=np.uint8))
        return frame, _valid_detection(self.calls)


class _Cv:
    WINDOW_NORMAL = 0
    WND_PROP_VISIBLE = 1

    def __init__(self) -> None:
        self.images: list[np.ndarray] = []
        self.destroyed = False

    def namedWindow(self, name, flags):
        self.name = name

    def imshow(self, name, image):
        self.images.append(image.copy())

    def waitKey(self, delay):
        return ord("q")

    def getWindowProperty(self, name, prop):
        return 1.0

    def destroyWindow(self, name):
        self.destroyed = True


def test_run_local_preview_uses_latest_detection_frame_and_closes_on_q() -> None:
    service = _Service()
    cv = _Cv()

    result = run_local_preview(service, max_width=640, display_fps=30.0, cv=cv)

    assert result == "key"
    assert service.calls == 1
    assert len(cv.images) == 1
    assert cv.destroyed is True


def test_run_local_preview_exits_when_stop_event_is_set() -> None:
    service = _Service()
    cv = _Cv()
    stop_event = threading.Event()
    stop_event.set()

    result = run_local_preview(
        service,
        max_width=640,
        display_fps=30.0,
        cv=cv,
        stop_event=stop_event,
    )

    assert result == "stop"
    assert service.calls == 0
    assert cv.destroyed is True
