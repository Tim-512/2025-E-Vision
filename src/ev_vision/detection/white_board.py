from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ev_vision.config import WhiteBoardConfig
from ev_vision.detection.image_normalization import NormalizedFrame

_RECTIFIED_SIZE = (420, 594)


@dataclass(frozen=True)
class WhiteBoardCandidate:
    contour: np.ndarray = field(compare=False)
    corners_px: tuple[tuple[float, float], ...]
    center_px: tuple[float, float]
    bbox_xyxy: tuple[int, int, int, int]
    area_fraction: float
    white_occupancy: float
    convexity: float
    aspect_ratio_error: float
    texture_std: float
    border_support: float
    geometry_score: float


def _order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _candidate_mask(normalized: NormalizedFrame) -> np.ndarray:
    high = float(np.percentile(normalized.normalized_gray, 90.0))
    threshold = max(130.0, high - 10.0)
    mask = cv2.threshold(
        normalized.normalized_gray, threshold, 255, cv2.THRESH_BINARY
    )[1]
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def _rectified_measurements(
    normalized: NormalizedFrame,
    corners: np.ndarray,
    config: WhiteBoardConfig,
) -> tuple[float, float, float]:
    destination = np.asarray(
        [[0, 0], [_RECTIFIED_SIZE[0] - 1, 0],
         [_RECTIFIED_SIZE[0] - 1, _RECTIFIED_SIZE[1] - 1],
         [0, _RECTIFIED_SIZE[1] - 1]],
        np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    gray = cv2.warpPerspective(normalized.gray, transform, _RECTIFIED_SIZE)
    mask = cv2.warpPerspective(
        _candidate_mask(normalized), transform, _RECTIFIED_SIZE
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    white_occupancy = float(np.mean(mask > 0))
    texture_std = float(np.std(cv2.GaussianBlur(gray, (15, 15), 0)))

    band = max(2, int(round(min(_RECTIFIED_SIZE) * config.border_band_fraction)))
    inner = gray[band:-band, band:-band]
    strips = np.concatenate(
        (gray[:band].ravel(), gray[-band:].ravel(),
         gray[band:-band, :band].ravel(), gray[band:-band, -band:].ravel())
    )
    border_support = float(
        np.clip((float(np.median(inner)) - float(np.median(strips))) / 128.0, 0.0, 1.0)
    )
    return white_occupancy, texture_std, border_support


def _measure_candidate(
    contour: np.ndarray,
    normalized: NormalizedFrame,
    config: WhiteBoardConfig,
) -> WhiteBoardCandidate | None:
    height, width = normalized.gray.shape
    image_area = float(height * width)
    area = float(cv2.contourArea(contour))
    area_fraction = area / image_area
    if not config.min_area_fraction <= area_fraction <= config.max_area_fraction:
        return None

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0.0:
        return None
    convexity = area / hull_area
    if convexity < config.min_convexity:
        return None

    perimeter = cv2.arcLength(hull, True)
    approximate = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
    if len(approximate) == 4:
        corners = _order_corners(approximate.reshape(4, 2))
    else:
        corners = _order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))

    side_lengths = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    if float(np.min(side_lengths)) < config.min_side_px:
        return None
    width_px = 0.5 * (side_lengths[0] + side_lengths[2])
    height_px = 0.5 * (side_lengths[1] + side_lengths[3])
    short, long = sorted((float(width_px), float(height_px)))
    aspect_ratio_error = abs(short / max(long, 1e-6) - config.expected_aspect_ratio)
    if aspect_ratio_error > config.aspect_ratio_tolerance:
        return None

    white_occupancy, texture_std, border_support = _rectified_measurements(
        normalized, corners, config
    )
    if white_occupancy < config.min_white_occupancy:
        return None
    if texture_std > config.max_texture_std:
        return None

    aspect_score = 1.0 - min(1.0, aspect_ratio_error / config.aspect_ratio_tolerance)
    geometry_score = float(np.clip(
        0.40 * aspect_score + 0.25 * convexity + 0.25 * white_occupancy
        + 0.10 * border_support,
        0.0,
        1.0,
    ))
    delta = corners[1] - corners[0]
    direction_a = corners[2] - corners[0]
    direction_b = corners[3] - corners[1]
    cross_ab = float(
        direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    )
    if abs(cross_ab) < 1e-6:
        center = corners.mean(axis=0)
    else:
        cross_delta_b = float(
            delta[0] * direction_b[1] - delta[1] * direction_b[0]
        )
        center = corners[0] + (cross_delta_b / cross_ab) * direction_a
    x, y, w, h = cv2.boundingRect(contour)
    return WhiteBoardCandidate(
        contour=contour.copy(),
        corners_px=tuple((float(px), float(py)) for px, py in corners),
        center_px=(float(center[0]), float(center[1])),
        bbox_xyxy=(x, y, x + w, y + h),
        area_fraction=area_fraction,
        white_occupancy=white_occupancy,
        convexity=convexity,
        aspect_ratio_error=aspect_ratio_error,
        texture_std=texture_std,
        border_support=border_support,
        geometry_score=geometry_score,
    )


def find_white_board_candidates(
    normalized: NormalizedFrame,
    config: WhiteBoardConfig,
    *,
    roi_xyxy: tuple[int, int, int, int] | None = None,
) -> tuple[WhiteBoardCandidate, ...]:
    height, width = normalized.gray.shape
    x0, y0, x1, y1 = (0, 0, width, height) if roi_xyxy is None else roi_xyxy
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(width, int(x1)), min(height, int(y1))
    if x1 <= x0 or y1 <= y0:
        return ()

    mask = _candidate_mask(normalized)
    roi_mask = np.zeros_like(mask)
    roi_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    contours, _ = cv2.findContours(
        roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    accepted = [
        candidate
        for contour in contours
        if (candidate := _measure_candidate(contour, normalized, config)) is not None
    ]
    return tuple(sorted(accepted, key=lambda item: item.geometry_score, reverse=True))


