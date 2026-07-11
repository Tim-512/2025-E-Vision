from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from ev_vision.models import BoardObservation


def _cross2(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return quadrilateral corners as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    if _cross2(ordered[1] - ordered[0], ordered[2] - ordered[1]) < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def _line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    p, r = first[0], first[1] - first[0]
    q, s = second[0], second[1] - second[0]
    denominator = float(_cross2(r, s))
    if abs(denominator) < 1e-7:
        return None
    t = float(_cross2(q - p, s) / denominator)
    return p + t * r


def _fit_edge(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1.0:
        return np.stack((start, end))
    distances = np.abs(_cross2(direction, points - start)) / length
    along = ((points - start) @ direction) / (length * length)
    selected = points[(distances <= 3.0) & (along >= -0.03) & (along <= 1.03)]
    if len(selected) < 6:
        selected = np.stack((start, end))
    vx, vy, x0, y0 = cv2.fitLine(selected.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    span = max(length, 1.0)
    return np.array([[x0 - vx * span, y0 - vy * span], [x0 + vx * span, y0 + vy * span]], np.float32)


def _refine_quad(contour: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    points = contour.reshape(-1, 2).astype(np.float32)
    quad = order_corners(quad)
    lines = [_fit_edge(points, quad[index], quad[(index + 1) % 4]) for index in range(4)]
    intersections = [_line_intersection(lines[index - 1], lines[index]) for index in range(4)]
    if any(point is None for point in intersections):
        return None
    return order_corners(np.asarray(intersections, dtype=np.float32))


@dataclass(frozen=True)
class BoardGeometryDetector:
    min_area_fraction: float = 0.08
    max_area_fraction: float = 0.90
    minimum_side_px: float = 60.0
    border_margin_px: float = 2.0

    def _dark_mask(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError("image must be grayscale or BGR")
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        illumination = cv2.GaussianBlur(gray, (0, 0), 35.0)
        normalized = cv2.divide(gray, illumination, scale=180.0)
        _, mask = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def detect(self, image: np.ndarray, *, captured_ns: int) -> BoardObservation | None:
        if image is None or image.size == 0:
            return None
        height, width = image.shape[:2]
        image_area = float(height * width)
        mask = self._dark_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for contour in contours:
            area = abs(float(cv2.contourArea(contour)))
            fraction = area / image_area
            if not self.min_area_fraction <= fraction <= self.max_area_fraction:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            polygon = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            quad = order_corners(polygon.reshape(4, 2))
            if (
                quad[:, 0].min() <= self.border_margin_px
                or quad[:, 1].min() <= self.border_margin_px
                or quad[:, 0].max() >= width - 1 - self.border_margin_px
                or quad[:, 1].max() >= height - 1 - self.border_margin_px
            ):
                continue
            sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            if sides.min() < self.minimum_side_px or sides.max() / sides.min() > 4.0:
                continue
            rectangularity = area / max(float(cv2.contourArea(quad)), 1.0)
            score = fraction * max(0.0, min(1.0, rectangularity))
            candidates.append((score, contour, quad))

        if not candidates:
            return None
        score, contour, rough_quad = max(candidates, key=lambda candidate: candidate[0])
        corners = _refine_quad(contour, rough_quad)
        if corners is None:
            return None
        sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        if sides.min() < self.minimum_side_px:
            return None

        center = tuple(float(value) for value in corners.mean(axis=0))
        confidence = float(np.clip(0.55 + 1.2 * score, 0.0, 1.0))
        return BoardObservation(
            captured_ns=captured_ns,
            corners_px=tuple((float(x), float(y)) for x, y in corners),
            center_px=center,
            confidence=confidence,
            homography_valid=True,
        )
