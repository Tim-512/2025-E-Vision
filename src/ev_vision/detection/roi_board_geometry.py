"""Model-guided quadrilateral refinement inside a bounded image ROI.

This module deliberately has no model or tracking dependencies.  It receives a
single model bounding box, searches only its expanded ROI, and either returns a
validated board quadrilateral or an explicit geometry failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import cv2
import numpy as np


class GeometryFailure(str, Enum):
    INVALID_ROI = "INVALID_ROI"
    NO_VALID_QUADRILATERAL = "NO_VALID_QUADRILATERAL"
    TRUNCATED_QUADRILATERAL = "TRUNCATED_QUADRILATERAL"
    UNDERSIZED_QUADRILATERAL = "UNDERSIZED_QUADRILATERAL"
    INVALID_ASPECT_RATIO = "INVALID_ASPECT_RATIO"
    LOW_EDGE_SUPPORT = "LOW_EDGE_SUPPORT"
    LOW_INTERNAL_STRUCTURE = "LOW_INTERNAL_STRUCTURE"
    AMBIGUOUS_GEOMETRY = "AMBIGUOUS_GEOMETRY"
    CORNER_ORDER_FAILED = "CORNER_ORDER_FAILED"


@dataclass(frozen=True)
class RoiWindow:
    """A half-open ``(x0, y0, x1, y1)`` image window."""

    xyxy_px: tuple[int, int, int, int]

    def to_source(
        self, points: Sequence[Sequence[float]]
    ) -> tuple[tuple[float, float], ...]:
        x0, y0, _, _ = self.xyxy_px
        return tuple((float(x) + x0, float(y) + y0) for x, y in points)


@dataclass(frozen=True)
class GeometryDebugImages:
    roi_bgr: np.ndarray
    edges: np.ndarray
    candidates_bgr: np.ndarray


@dataclass(frozen=True)
class GeometryResult:
    accepted: bool
    corners_px: tuple[tuple[float, float], ...] | None
    center_px: tuple[float, float] | None
    geometry_score: float
    edge_support_score: float
    structure_score: float
    roi_xyxy_px: tuple[int, int, int, int]
    failure_reason: GeometryFailure | None
    debug: GeometryDebugImages | None = None


@dataclass(frozen=True)
class _Candidate:
    corners: np.ndarray
    geometry_score: float
    edge_support_score: float
    structure_score: float
    total_score: float


@dataclass(frozen=True)
class _RejectedCandidate:
    corners: np.ndarray
    reason: GeometryFailure
    geometry_score: float = 0.0
    edge_support_score: float = 0.0
    structure_score: float = 0.0


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def expand_roi(
    model_box: Sequence[float],
    image_size: tuple[int, int],
    *,
    padding_fraction: float,
) -> RoiWindow:
    """Expand an XYXY model box and clip it to half-open image bounds.

    ``image_size`` is interpreted as ``(width, height)``.  Padding is applied
    independently using the box width and height.
    """

    try:
        x0, y0, x1, y1 = (float(value) for value in model_box)
        image_width, image_height = (int(value) for value in image_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("ROI values must be numeric") from exc

    values = (x0, y0, x1, y1, float(padding_fraction))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("ROI values must be finite")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("ROI image size must be positive")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI model box must have positive area")
    if padding_fraction < 0.0:
        raise ValueError("ROI padding_fraction must be non-negative")

    x_padding = (x1 - x0) * padding_fraction
    y_padding = (y1 - y0) * padding_fraction
    left = max(0, min(image_width, math.floor(x0 - x_padding)))
    top = max(0, min(image_height, math.floor(y0 - y_padding)))
    right = max(0, min(image_width, math.ceil(x1 + x_padding)))
    bottom = max(0, min(image_height, math.ceil(y1 + y_padding)))
    if right <= left or bottom <= top:
        raise ValueError("ROI does not intersect the image")
    return RoiWindow((left, top, right, bottom))


def order_corners(
    points: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[tuple[float, float], ...]:
    """Order four unique finite points as TL, TR, BR, BL.

    Ordering by angle around the centroid works for perspective quadrilaterals
    without relying on the fragile sum/difference heuristic.  A clockwise
    image-coordinate winding is enforced after rotating the top-left point to
    index zero.
    """

    array = np.asarray(points, dtype=np.float64)
    if array.size != 8:
        raise ValueError("exactly four corner points are required")
    array = array.reshape(4, 2)
    if not np.isfinite(array).all():
        raise ValueError("corner points must be finite")
    if np.unique(array, axis=0).shape[0] != 4:
        raise ValueError("corner points must be unique")

    center = array.mean(axis=0)
    angles = np.arctan2(array[:, 1] - center[1], array[:, 0] - center[0])
    ordered = array[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)

    signed_area = 0.5 * float(
        np.sum(
            ordered[:, 0] * np.roll(ordered[:, 1], -1)
            - ordered[:, 1] * np.roll(ordered[:, 0], -1)
        )
    )
    if abs(signed_area) <= 1e-6:
        raise ValueError("corner ordering has zero area")
    if signed_area < 0.0:
        ordered = ordered[[0, 3, 2, 1]]

    contour = ordered.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        raise ValueError("corner ordering is not convex")
    return tuple((float(x), float(y)) for x, y in ordered)


@dataclass(frozen=True)
class RoiBoardGeometry:
    padding_fraction: float = 0.08
    canny_low: int = 60
    canny_high: int = 180
    min_edge_support: float = 0.45
    min_geometry_score: float = 0.55
    expected_aspect_ratio: float = 0.707
    aspect_ratio_tolerance: float = 0.35
    minimum_side_px: float = 40.0
    minimum_area_fraction: float = 0.25
    maximum_area_fraction: float = 1.15
    minimum_structure_score: float = 0.35
    border_margin_px: float = 2.0
    ambiguity_margin: float = 0.05
    refine_corners: bool = True

    def refine(
        self,
        image: np.ndarray,
        model_box: Sequence[float],
        *,
        include_debug: bool = False,
    ) -> GeometryResult:
        """Refine one model box to a validated source-image quadrilateral."""

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return self._failure(GeometryFailure.INVALID_ROI)
        if image.ndim not in (2, 3):
            return self._failure(GeometryFailure.INVALID_ROI)

        image_height, image_width = image.shape[:2]
        try:
            roi = expand_roi(
                model_box,
                (image_width, image_height),
                padding_fraction=self.padding_fraction,
            )
            model_values = tuple(float(value) for value in model_box)
        except (TypeError, ValueError):
            return self._failure(GeometryFailure.INVALID_ROI)

        x0, y0, x1, y1 = roi.xyxy_px
        roi_image = image[y0:y1, x0:x1]
        if roi_image.size == 0:
            return self._failure(GeometryFailure.INVALID_ROI)
        model_touches_image_boundary = (
            model_values[0] <= self.border_margin_px
            or model_values[1] <= self.border_margin_px
            or model_values[2] >= image_width - self.border_margin_px
            or model_values[3] >= image_height - self.border_margin_px
        )
        roi_bgr = self._as_bgr(roi_image)
        gray = self._as_gray(roi_image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        primary_edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        secondary_edges = cv2.Canny(
            blurred,
            max(1, int(round(self.canny_low * 0.75))),
            max(1, int(round(self.canny_high * 0.75))),
        )
        combined_edges = cv2.bitwise_or(primary_edges, secondary_edges)
        edges = cv2.morphologyEx(
            combined_edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = list(contours)
        model_width = model_values[2] - model_values[0]
        model_height = model_values[3] - model_values[1]
        model_area = max(model_width * model_height, 1.0)

        # Real captures can have short breaks where the dark frame meets a
        # similarly dark background. Add a bridged contour pass, but only for
        # contours already large enough to satisfy the area safety gate. This
        # keeps morphology-created small loops out of candidate evaluation.
        bridged_edges = cv2.morphologyEx(
            combined_edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        )
        bridged_contours, _ = cv2.findContours(
            bridged_edges,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        minimum_contour_area = self.minimum_area_fraction * model_area
        contours.extend(
            contour
            for contour in bridged_contours
            if abs(float(cv2.contourArea(contour))) >= minimum_contour_area
        )
        edges = cv2.bitwise_or(edges, bridged_edges)

        accepted: list[_Candidate] = []
        rejected: list[_RejectedCandidate] = []
        seen: list[np.ndarray] = []
        saw_contour = False
        saw_four_points = False

        for contour in contours:
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            saw_contour = True
            for epsilon_fraction in (0.015, 0.025, 0.04):
                approximation = cv2.approxPolyDP(
                    contour,
                    epsilon_fraction * perimeter,
                    True,
                )
                if len(approximation) != 4:
                    continue
                saw_four_points = True
                if not cv2.isContourConvex(approximation):
                    continue
                try:
                    corners = np.asarray(
                        order_corners(approximation.reshape(4, 2)),
                        dtype=np.float32,
                    )
                except ValueError:
                    rejected.append(
                        _RejectedCandidate(
                            approximation.reshape(4, 2).astype(np.float32),
                            GeometryFailure.CORNER_ORDER_FAILED,
                        )
                    )
                    continue
                if self._is_duplicate(corners, seen):
                    continue
                seen.append(corners)

                evaluation = self._evaluate_candidate(
                    corners,
                    gray=gray,
                    primary_edges=primary_edges,
                    model_area=model_area,
                    roi_size=(x1 - x0, y1 - y0),
                    model_touches_image_boundary=model_touches_image_boundary,
                )
                if isinstance(evaluation, _Candidate):
                    accepted.append(evaluation)
                else:
                    rejected.append(evaluation)

        debug = None
        if include_debug:
            debug = self._build_debug(roi_bgr, edges, accepted, rejected, None)

        if not accepted:
            reason = self._best_failure_reason(
                rejected,
                saw_contour=saw_contour,
                saw_four_points=saw_four_points,
            )
            best_rejected = self._best_rejected(rejected)
            return self._failure(
                reason,
                roi=roi,
                geometry_score=(best_rejected.geometry_score if best_rejected else 0.0),
                edge_support_score=(
                    best_rejected.edge_support_score if best_rejected else 0.0
                ),
                structure_score=(best_rejected.structure_score if best_rejected else 0.0),
                debug=debug,
            )

        accepted.sort(key=lambda candidate: candidate.total_score, reverse=True)
        winner = accepted[0]
        if len(accepted) > 1:
            gap = winner.total_score - accepted[1].total_score
            if gap < self.ambiguity_margin:
                if include_debug:
                    debug = self._build_debug(roi_bgr, edges, accepted, rejected, None)
                return self._failure(
                    GeometryFailure.AMBIGUOUS_GEOMETRY,
                    roi=roi,
                    geometry_score=winner.geometry_score,
                    edge_support_score=winner.edge_support_score,
                    structure_score=winner.structure_score,
                    debug=debug,
                )

        corners = winner.corners.copy()
        if self.refine_corners:
            corners = self._subpixel_refine(blurred, corners)
        try:
            corners_tuple = order_corners(corners)
        except ValueError:
            return self._failure(
                GeometryFailure.CORNER_ORDER_FAILED,
                roi=roi,
                geometry_score=winner.geometry_score,
                edge_support_score=winner.edge_support_score,
                structure_score=winner.structure_score,
                debug=debug,
            )

        source_corners = roi.to_source(corners_tuple)
        center = tuple(
            float(value)
            for value in np.asarray(source_corners, dtype=np.float64).mean(axis=0)
        )
        if include_debug:
            debug = self._build_debug(roi_bgr, edges, accepted, rejected, winner)
        return GeometryResult(
            accepted=True,
            corners_px=source_corners,
            center_px=center,
            geometry_score=winner.geometry_score,
            edge_support_score=winner.edge_support_score,
            structure_score=winner.structure_score,
            roi_xyxy_px=roi.xyxy_px,
            failure_reason=None,
            debug=debug,
        )

    def _evaluate_candidate(
        self,
        corners: np.ndarray,
        *,
        gray: np.ndarray,
        primary_edges: np.ndarray,
        model_area: float,
        roi_size: tuple[int, int],
        model_touches_image_boundary: bool,
    ) -> _Candidate | _RejectedCandidate:
        roi_width, roi_height = roi_size
        if model_touches_image_boundary or self._touches_boundary(
            corners, roi_width, roi_height
        ):
            return _RejectedCandidate(corners, GeometryFailure.TRUNCATED_QUADRILATERAL)

        sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        minimum_side = float(np.min(sides))
        minimum_side_score = _clamp01(minimum_side / self.minimum_side_px)
        if minimum_side < self.minimum_side_px:
            return _RejectedCandidate(
                corners,
                GeometryFailure.UNDERSIZED_QUADRILATERAL,
                geometry_score=minimum_side_score,
            )

        candidate_area = abs(float(cv2.contourArea(corners)))
        area_fraction = candidate_area / model_area
        if not self.minimum_area_fraction <= area_fraction <= self.maximum_area_fraction:
            return _RejectedCandidate(
                corners,
                GeometryFailure.NO_VALID_QUADRILATERAL,
                geometry_score=_clamp01(min(area_fraction, 1.0 / max(area_fraction, 1e-6))),
            )
        area_score = _clamp01(min(area_fraction, 1.0 / max(area_fraction, 1e-6)))

        width = 0.5 * float(sides[0] + sides[2])
        height = 0.5 * float(sides[1] + sides[3])
        aspect = min(width, height) / max(width, height)
        relative_aspect_error = abs(aspect - self.expected_aspect_ratio) / max(
            self.expected_aspect_ratio,
            1e-6,
        )
        aspect_score = _clamp01(
            1.0 - relative_aspect_error / max(self.aspect_ratio_tolerance, 1e-6)
        )
        if relative_aspect_error > self.aspect_ratio_tolerance:
            return _RejectedCandidate(
                corners,
                GeometryFailure.INVALID_ASPECT_RATIO,
                geometry_score=aspect_score,
            )

        opposite_score = 0.5 * (
            min(float(sides[0]), float(sides[2]))
            / max(float(sides[0]), float(sides[2]), 1e-6)
            + min(float(sides[1]), float(sides[3]))
            / max(float(sides[1]), float(sides[3]), 1e-6)
        )
        geometry_score = _clamp01(
            (area_score + opposite_score + aspect_score + minimum_side_score) / 4.0
        )
        if geometry_score < self.min_geometry_score:
            return _RejectedCandidate(
                corners,
                GeometryFailure.NO_VALID_QUADRILATERAL,
                geometry_score=geometry_score,
            )

        edge_support = self._edge_support(primary_edges, corners)
        if edge_support < self.min_edge_support:
            return _RejectedCandidate(
                corners,
                GeometryFailure.LOW_EDGE_SUPPORT,
                geometry_score=geometry_score,
                edge_support_score=edge_support,
            )

        structure_score = self._internal_structure(gray, corners)
        if structure_score < self.minimum_structure_score:
            return _RejectedCandidate(
                corners,
                GeometryFailure.LOW_INTERNAL_STRUCTURE,
                geometry_score=geometry_score,
                edge_support_score=edge_support,
                structure_score=structure_score,
            )

        total_score = _clamp01(
            (geometry_score + edge_support + structure_score) / 3.0
        )
        return _Candidate(
            corners=corners,
            geometry_score=geometry_score,
            edge_support_score=edge_support,
            structure_score=structure_score,
            total_score=total_score,
        )

    def _touches_boundary(
        self,
        corners: np.ndarray,
        roi_width: int,
        roi_height: int,
    ) -> bool:
        margin = self.border_margin_px
        return bool(
            np.any(corners[:, 0] <= margin)
            or np.any(corners[:, 1] <= margin)
            or np.any(corners[:, 0] >= (roi_width - 1 - margin))
            or np.any(corners[:, 1] >= (roi_height - 1 - margin))
        )

    @staticmethod
    def _edge_support(edges: np.ndarray, corners: np.ndarray) -> float:
        perimeter = np.zeros_like(edges)
        cv2.polylines(
            perimeter,
            [np.rint(corners).astype(np.int32)],
            True,
            255,
            1,
            cv2.LINE_8,
        )
        nearby_edges = cv2.dilate(
            edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        )
        perimeter_pixels = int(np.count_nonzero(perimeter))
        if perimeter_pixels == 0:
            return 0.0
        supported = np.count_nonzero((perimeter > 0) & (nearby_edges > 0))
        return _clamp01(float(supported) / perimeter_pixels)

    @staticmethod
    def _internal_structure(gray: np.ndarray, corners: np.ndarray) -> float:
        output_width = 240
        output_height = 340
        destination = np.asarray(
            [
                [0.0, 0.0],
                [output_width - 1.0, 0.0],
                [output_width - 1.0, output_height - 1.0],
                [0.0, output_height - 1.0],
            ],
            np.float32,
        )
        transform = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
        warped = cv2.warpPerspective(
            gray,
            transform,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
        )

        yy, xx = np.indices(warped.shape)
        normalized_x = xx / float(output_width - 1)
        normalized_y = yy / float(output_height - 1)
        outer_inset = 0.02
        inner_inset = 0.12
        inside_outer = (
            (normalized_x >= outer_inset)
            & (normalized_x <= 1.0 - outer_inset)
            & (normalized_y >= outer_inset)
            & (normalized_y <= 1.0 - outer_inset)
        )
        inside_inner = (
            (normalized_x >= inner_inset)
            & (normalized_x <= 1.0 - inner_inset)
            & (normalized_y >= inner_inset)
            & (normalized_y <= 1.0 - inner_inset)
        )
        border_band = inside_outer & ~inside_inner
        center_region = (
            (normalized_x >= 0.30)
            & (normalized_x <= 0.70)
            & (normalized_y >= 0.30)
            & (normalized_y <= 0.70)
        )
        border_values = warped[border_band]
        center_values = warped[center_region]
        if border_values.size == 0 or center_values.size == 0:
            return 0.0

        combined = np.concatenate((border_values, center_values)).astype(np.float32)
        low, high = np.percentile(combined, (10.0, 90.0))
        if high - low < 8.0:
            return 0.0
        dark_threshold = 0.5 * float(low + high)
        border_dark = float(np.mean(border_values <= dark_threshold))
        center_dark = float(np.mean(center_values <= dark_threshold))
        return _clamp01((border_dark - center_dark) / 0.60)

    @staticmethod
    def _subpixel_refine(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        height, width = gray.shape[:2]
        safe = corners.copy().astype(np.float32)
        safe[:, 0] = np.clip(safe[:, 0], 5.0, max(5.0, width - 6.0))
        safe[:, 1] = np.clip(safe[:, 1], 5.0, max(5.0, height - 6.0))
        try:
            refined = cv2.cornerSubPix(
                gray,
                safe.reshape(-1, 1, 2),
                (5, 5),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.01,
                ),
            )
        except cv2.error:
            return corners
        if refined is None or not np.isfinite(refined).all():
            return corners
        return refined.reshape(4, 2)

    @staticmethod
    def _is_duplicate(corners: np.ndarray, seen: Sequence[np.ndarray]) -> bool:
        return any(
            float(np.mean(np.linalg.norm(corners - previous, axis=1))) < 3.0
            for previous in seen
        )

    @staticmethod
    def _as_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.uint8, copy=False)
        if image.shape[2] == 1:
            return image[:, :, 0].astype(np.uint8, copy=False)
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        raise ValueError("image must be grayscale, BGR, or BGRA")

    @staticmethod
    def _as_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 1:
            return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 3:
            return image.copy()
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ValueError("image must be grayscale, BGR, or BGRA")

    @staticmethod
    def _best_rejected(
        rejected: Sequence[_RejectedCandidate],
    ) -> _RejectedCandidate | None:
        if not rejected:
            return None
        return max(
            rejected,
            key=lambda item: (
                item.geometry_score,
                item.edge_support_score,
                item.structure_score,
            ),
        )

    @staticmethod
    def _best_failure_reason(
        rejected: Sequence[_RejectedCandidate],
        *,
        saw_contour: bool,
        saw_four_points: bool,
    ) -> GeometryFailure:
        if not rejected:
            return GeometryFailure.NO_VALID_QUADRILATERAL
        # Report the most advanced hard gate reached.  This gives operators a
        # useful cause instead of whichever contour happened to be visited last.
        priority = (
            GeometryFailure.LOW_INTERNAL_STRUCTURE,
            GeometryFailure.LOW_EDGE_SUPPORT,
            GeometryFailure.INVALID_ASPECT_RATIO,
            GeometryFailure.UNDERSIZED_QUADRILATERAL,
            GeometryFailure.TRUNCATED_QUADRILATERAL,
            GeometryFailure.CORNER_ORDER_FAILED,
            GeometryFailure.NO_VALID_QUADRILATERAL,
        )
        reasons = {item.reason for item in rejected}
        for reason in priority:
            if reason in reasons:
                return reason
        if not saw_contour or not saw_four_points:
            return GeometryFailure.NO_VALID_QUADRILATERAL
        return GeometryFailure.NO_VALID_QUADRILATERAL

    @staticmethod
    def _build_debug(
        roi_bgr: np.ndarray,
        edges: np.ndarray,
        accepted: Sequence[_Candidate],
        rejected: Sequence[_RejectedCandidate],
        winner: _Candidate | None,
    ) -> GeometryDebugImages:
        candidates = roi_bgr.copy()
        for candidate in rejected:
            cv2.polylines(
                candidates,
                [np.rint(candidate.corners).astype(np.int32)],
                True,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        for candidate in accepted:
            color = (0, 255, 0) if candidate is winner else (0, 200, 255)
            thickness = 2 if candidate is winner else 1
            cv2.polylines(
                candidates,
                [np.rint(candidate.corners).astype(np.int32)],
                True,
                color,
                thickness,
                cv2.LINE_AA,
            )
        return GeometryDebugImages(
            roi_bgr=roi_bgr.copy(),
            edges=edges.copy(),
            candidates_bgr=candidates,
        )

    @staticmethod
    def _failure(
        reason: GeometryFailure,
        *,
        roi: RoiWindow | None = None,
        geometry_score: float = 0.0,
        edge_support_score: float = 0.0,
        structure_score: float = 0.0,
        debug: GeometryDebugImages | None = None,
    ) -> GeometryResult:
        return GeometryResult(
            accepted=False,
            corners_px=None,
            center_px=None,
            geometry_score=_clamp01(geometry_score),
            edge_support_score=_clamp01(edge_support_score),
            structure_score=_clamp01(structure_score),
            roi_xyxy_px=roi.xyxy_px if roi is not None else (0, 0, 0, 0),
            failure_reason=reason,
            debug=debug,
        )
