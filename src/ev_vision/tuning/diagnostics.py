from __future__ import annotations

import cv2
import numpy as np

from ev_vision.tuning.models import (
    DetectionSnapshot,
    ImageDiagnostics,
    OverlayOptions,
)

_ROI_SCALE = 0.5
_OUTLINE_COLOR = (0, 255, 0)
_CORNER_COLOR = (0, 255, 255)
_CENTER_COLOR = (0, 200, 0)
_GUIDE_COLOR = (255, 255, 0)
_TEXT_COLOR = (255, 255, 255)
_ERROR_COLOR = (0, 0, 255)


def _center_roi(width: int, height: int) -> tuple[int, int, int, int]:
    roi_width = max(1, int(round(width * _ROI_SCALE)))
    roi_height = max(1, int(round(height * _ROI_SCALE)))
    return (
        (width - roi_width) // 2,
        (height - roi_height) // 2,
        roi_width,
        roi_height,
    )


def _histogram(channel: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.bincount(channel.ravel(), minlength=256))


def compute_diagnostics(
    image: np.ndarray,
    *,
    source_sequence: int,
    computed_ns: int,
) -> ImageDiagnostics:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blue, green, red = cv2.split(image)
    x, y, width, height = _center_roi(image.shape[1], image.shape[0])
    roi = gray[y : y + height, x : x + width]
    pixel_count = gray.size
    return ImageDiagnostics(
        source_sequence=source_sequence,
        computed_ns=computed_ns,
        gray_histogram=_histogram(gray),
        blue_histogram=_histogram(blue),
        green_histogram=_histogram(green),
        red_histogram=_histogram(red),
        dark_percent=float(np.count_nonzero(gray <= 5) * 100.0 / pixel_count),
        bright_percent=float(np.count_nonzero(gray >= 250) * 100.0 / pixel_count),
        focus_score=float(cv2.Laplacian(roi, cv2.CV_64F).var()),
        roi_px=(x, y, width, height),
    )


def _point(point: tuple[float, float]) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def render_overlay(
    image: np.ndarray,
    *,
    source_sequence: int,
    detection: DetectionSnapshot,
    options: OverlayOptions = OverlayOptions(),
) -> np.ndarray:
    output = image.copy()
    if not options.enabled:
        return output

    height, width = output.shape[:2]
    if options.show_crosshair:
        cv2.drawMarker(
            output,
            (width // 2, height // 2),
            _GUIDE_COLOR,
            cv2.MARKER_CROSS,
            20,
            2,
            cv2.LINE_AA,
        )
    if options.show_center_roi:
        x, y, roi_width, roi_height = _center_roi(width, height)
        cv2.rectangle(
            output,
            (x, y),
            (x + roi_width - 1, y + roi_height - 1),
            _GUIDE_COLOR,
            1,
            cv2.LINE_AA,
        )

    observation = detection.observation
    geometry_is_current = (
        detection.enabled
        and detection.detected
        and observation is not None
        and detection.source_sequence == source_sequence
    )
    if geometry_is_current:
        corners = np.rint(np.asarray(observation.corners_px, dtype=np.float32)).astype(np.int32)
        if options.show_board_outline:
            cv2.polylines(output, [corners], True, _OUTLINE_COLOR, 2, cv2.LINE_AA)
        if options.show_corners:
            for corner in corners:
                cv2.circle(output, tuple(int(value) for value in corner), 4, _CORNER_COLOR, -1, cv2.LINE_AA)
        if options.show_center:
            cv2.circle(output, _point(observation.center_px), 5, _CENTER_COLOR, -1, cv2.LINE_AA)

    if options.show_detection_text:
        if detection.error is not None:
            text = "error"
            color = _ERROR_COLOR
        elif geometry_is_current:
            text = "detected"
            color = _OUTLINE_COLOR
        else:
            text = "not-detected"
            color = _TEXT_COLOR
        cv2.putText(
            output,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return output
