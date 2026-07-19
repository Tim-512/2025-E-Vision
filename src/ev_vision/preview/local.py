from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np


_GREEN = (0, 255, 0)
_WINDOW_NAME = "EV Vision - local preview"


def render_local_overlay(
    image: np.ndarray,
    *,
    source_sequence: int,
    detection: Any,
) -> np.ndarray:
    """Draw only a confirmed target outline and center on a copied frame."""
    output = image.copy()
    if (
        not bool(getattr(detection, "target_valid", False))
        or getattr(detection, "source_sequence", None) != source_sequence
    ):
        return output

    height, width = output.shape[:2]
    corners = np.asarray(getattr(detection, "corners_px", ()) or (), dtype=float)
    if corners.shape == (4, 2) and np.isfinite(corners).all():
        corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
        cv2.polylines(
            output,
            [np.rint(corners).astype(np.int32)],
            True,
            _GREEN,
            3,
            cv2.LINE_8,
        )

    center = np.asarray(getattr(detection, "center_px", ()) or (), dtype=float)
    if center.shape == (2,) and np.isfinite(center).all():
        point = (
            int(np.clip(round(center[0]), 0, width - 1)),
            int(np.clip(round(center[1]), 0, height - 1)),
        )
        cv2.drawMarker(
            output,
            point,
            _GREEN,
            cv2.MARKER_CROSS,
            17,
            3,
            cv2.LINE_8,
        )
        cv2.circle(output, point, 6, _GREEN, 2, cv2.LINE_8)
    return output


def resize_preview(image: np.ndarray, *, max_width: int) -> np.ndarray:
    if isinstance(max_width, bool) or not isinstance(max_width, int) or max_width <= 0:
        raise ValueError("max_width must be a positive integer")
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / float(width)
    return cv2.resize(
        image,
        (max_width, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def run_local_preview(
    service: Any,
    *,
    max_width: int = 640,
    display_fps: float = 30.0,
    window_name: str = _WINDOW_NAME,
    cv: Any = cv2,
    sleep: Callable[[float], None] = time.sleep,
    stop_event: threading.Event | None = None,
) -> str:
    """Run a latest-only OpenCV GUI loop. The caller owns service lifecycle."""
    if isinstance(display_fps, bool) or not isinstance(display_fps, (int, float)):
        raise ValueError("display_fps must be numeric")
    if not math.isfinite(display_fps) or display_fps <= 0:
        raise ValueError("display_fps must be positive and finite")
    if isinstance(max_width, bool) or not isinstance(max_width, int) or max_width <= 0:
        raise ValueError("max_width must be a positive integer")

    delay_ms = max(1, int(round(1000.0 / float(display_fps))))
    last_sequence: int | None = None
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                return "stop"
            matched = service.detection_frame_for_latest()
            if matched is not None:
                frame, detection = matched
                if frame.sequence != last_sequence:
                    image = render_local_overlay(
                        frame.image,
                        source_sequence=frame.sequence,
                        detection=detection,
                    )
                    cv.imshow(window_name, resize_preview(image, max_width=max_width))
                    last_sequence = frame.sequence

            key = int(cv.waitKey(delay_ms)) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return "key"
            try:
                if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                    return "window"
            except (AttributeError, TypeError):
                pass
            if matched is None:
                sleep(min(0.01, 1.0 / float(display_fps)))
    finally:
        cv.destroyWindow(window_name)