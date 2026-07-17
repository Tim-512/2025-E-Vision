from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ev_vision.detection.contracts import ObservationSource
from ev_vision.detection.failures import DetectionFailure


@dataclass(frozen=True)
class BoardHistory:
    center_px: tuple[float, float]
    velocity_px_s: tuple[float, float]
    scale_px_per_mm: float
    corners_px: tuple[tuple[float, float], ...]
    timestamp_ns: int


@dataclass(frozen=True)
class PartialObservation:
    valid: bool
    source: ObservationSource
    center_px: tuple[float, float] | None
    confidence: float
    scale_px_per_mm: float | None
    predicted_center_px: tuple[float, float] | None
    roi_xyxy: tuple[int, int, int, int] | None
    near_image_edge: bool
    partially_outside: bool
    failure_reason: DetectionFailure | None = None


def _predicted_geometry(
    history: BoardHistory,
    timestamp_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    elapsed_s = max(0.0, (timestamp_ns - history.timestamp_ns) / 1_000_000_000.0)
    offset = np.asarray(history.velocity_px_s, np.float64) * elapsed_s
    center = np.asarray(history.center_px, np.float64) + offset
    corners = np.asarray(history.corners_px, np.float64) + offset
    return center, corners


def predicted_roi(
    history: BoardHistory,
    *,
    timestamp_ns: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    _, corners = _predicted_geometry(history, timestamp_ns)
    minimum = corners.min(axis=0)
    maximum = corners.max(axis=0)
    expansion = (maximum - minimum) * 0.20
    minimum -= expansion
    maximum += expansion
    x0 = max(0, int(np.floor(minimum[0])))
    y0 = max(0, int(np.floor(minimum[1])))
    x1 = min(image_width, int(np.ceil(maximum[0])))
    y1 = min(image_height, int(np.ceil(maximum[1])))
    return (x0, y0, x1, y1)


def weighted_center(
    items: list[tuple[tuple[float, float], float]],
) -> tuple[float, float]:
    total = sum(weight for _, weight in items)
    if total <= 0.0:
        raise ValueError("weighted center requires a positive total weight")
    return (
        sum(point[0] * weight for point, weight in items) / total,
        sum(point[1] * weight for point, weight in items) / total,
    )


def _invalid(
    failure_reason: DetectionFailure,
    *,
    predicted_center_px: tuple[float, float] | None = None,
    roi_xyxy: tuple[int, int, int, int] | None = None,
    near_image_edge: bool = False,
    partially_outside: bool = False,
) -> PartialObservation:
    return PartialObservation(
        valid=False,
        source=ObservationSource.NONE,
        center_px=None,
        confidence=0.0,
        scale_px_per_mm=None,
        predicted_center_px=predicted_center_px,
        roi_xyxy=roi_xyxy,
        near_image_edge=near_image_edge,
        partially_outside=partially_outside,
        failure_reason=failure_reason,
    )


def fuse_partial_observation(
    *,
    history: BoardHistory | None,
    timestamp_ns: int,
    image_size: tuple[int, int],
    arc_center_px: tuple[float, float] | None,
    arc_count: int,
    arc_confidence: float,
    white_center_px: tuple[float, float] | None,
    white_confidence: float,
    observed_scale_px_per_mm: float | None = None,
) -> PartialObservation:
    if history is None:
        return _invalid(DetectionFailure.PARTIAL_HISTORY_REQUIRED)

    image_width, image_height = image_size
    predicted_array, transformed_corners = _predicted_geometry(history, timestamp_ns)
    predicted = (float(predicted_array[0]), float(predicted_array[1]))
    roi = predicted_roi(history, timestamp_ns=timestamp_ns, image_size=image_size)
    partially_outside = bool(
        np.any(transformed_corners[:, 0] < 0.0)
        or np.any(transformed_corners[:, 0] >= image_width)
        or np.any(transformed_corners[:, 1] < 0.0)
        or np.any(transformed_corners[:, 1] >= image_height)
    )
    edge_x = 0.08 * image_width
    edge_y = 0.08 * image_height
    near_image_edge = bool(
        predicted[0] <= edge_x
        or predicted[0] >= image_width - edge_x
        or predicted[1] <= edge_y
        or predicted[1] >= image_height - edge_y
    )

    scale_jump = 0.0
    if observed_scale_px_per_mm is not None:
        scale_jump = abs(observed_scale_px_per_mm - history.scale_px_per_mm) / max(
            history.scale_px_per_mm, 1e-6
        )
    distance_limit = max(32.0, 45.0 * history.scale_px_per_mm)
    arc_distance = (
        float(np.linalg.norm(np.subtract(arc_center_px, predicted)))
        if arc_center_px is not None
        else float("inf")
    )
    white_distance = (
        float(np.linalg.norm(np.subtract(white_center_px, predicted)))
        if white_center_px is not None
        else float("inf")
    )

    has_multiple_arcs = (
        arc_center_px is not None
        and arc_count >= 2
        and arc_confidence >= 0.45
        and arc_distance <= distance_limit
        and scale_jump <= 0.30
    )
    has_single_arc = (
        arc_center_px is not None
        and arc_count == 1
        and arc_confidence >= 0.75
        and arc_distance <= 0.45 * distance_limit
        and scale_jump <= 0.18
    )
    has_white = (
        white_center_px is not None
        and white_confidence >= 0.60
        and white_distance <= distance_limit
    )
    if not (has_multiple_arcs or has_single_arc or has_white):
        reason = (
            DetectionFailure.EXCESSIVE_SCALE_JUMP
            if scale_jump > 0.30
            else DetectionFailure.EXCESSIVE_POSITION_JUMP
        )
        return _invalid(
            reason,
            predicted_center_px=predicted,
            roi_xyxy=roi,
            near_image_edge=near_image_edge,
            partially_outside=partially_outside,
        )

    sources: list[tuple[tuple[float, float], float]] = [(predicted, 0.35)]
    if has_multiple_arcs or has_single_arc:
        assert arc_center_px is not None
        sources.append((arc_center_px, 0.55 * arc_confidence))
    if has_white:
        assert white_center_px is not None
        sources.append((white_center_px, 0.25 * white_confidence))
    center = weighted_center(sources)

    if (has_multiple_arcs or has_single_arc) and has_white:
        source = ObservationSource.FUSED_PARTIAL
    elif has_multiple_arcs:
        source = ObservationSource.CONCENTRIC_ARCS
    elif has_single_arc:
        source = ObservationSource.SINGLE_ARC
    else:
        source = ObservationSource.WHITE_REGION
    confidence_terms = []
    if has_multiple_arcs or has_single_arc:
        confidence_terms.append(arc_confidence)
    if has_white:
        confidence_terms.append(white_confidence)
    confidence = float(np.mean(confidence_terms))
    return PartialObservation(
        valid=True,
        source=source,
        center_px=center,
        confidence=confidence,
        scale_px_per_mm=(
            observed_scale_px_per_mm
            if observed_scale_px_per_mm is not None
            else history.scale_px_per_mm
        ),
        predicted_center_px=predicted,
        roi_xyxy=roi,
        near_image_edge=near_image_edge,
        partially_outside=partially_outside,
        failure_reason=None,
    )
