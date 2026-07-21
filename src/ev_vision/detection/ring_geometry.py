from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from ev_vision.config import RingGeometryConfig
from ev_vision.detection.failures import DetectionFailure


@dataclass(frozen=True)
class ArcFit:
    center_px: tuple[float, float]
    axes_px: tuple[float, float]
    angle_deg: float
    equivalent_radius_px: float
    coverage: float
    residual_px: float


@dataclass(frozen=True)
class RingGeometryResult:
    valid: bool
    center_px: tuple[float, float] | None
    arcs: tuple[ArcFit, ...]
    visible_arc_count: int
    common_center_score: float
    ratio_score: float
    coverage_score: float
    scale_px_per_mm: float | None
    failure_reason: DetectionFailure | None


def _empty(reason: DetectionFailure) -> RingGeometryResult:
    return RingGeometryResult(False, None, (), 0, 0.0, 0.0, 0.0, None, reason)


def _fit_arc(contour: np.ndarray) -> ArcFit | None:
    if len(contour) < 12:
        return None
    try:
        (cx, cy), (diameter_a, diameter_b), angle = cv2.fitEllipse(contour)
    except cv2.error:
        return None
    semi_a, semi_b = diameter_a * 0.5, diameter_b * 0.5
    if min(semi_a, semi_b) < 6.0 or max(semi_a, semi_b) / min(semi_a, semi_b) > 1.45:
        return None

    points = contour[:, 0, :].astype(np.float64)
    theta = np.deg2rad(angle)
    cosine, sine = np.cos(theta), np.sin(theta)
    delta = points - np.asarray([cx, cy])
    rotated_x = cosine * delta[:, 0] + sine * delta[:, 1]
    rotated_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    normalized_radius = np.sqrt(
        (rotated_x / max(semi_a, 1e-6)) ** 2
        + (rotated_y / max(semi_b, 1e-6)) ** 2
    )
    equivalent_radius = float(np.sqrt(semi_a * semi_b))
    residual = float(np.median(np.abs(normalized_radius - 1.0)) * equivalent_radius)
    if residual > max(3.0, equivalent_radius * 0.10):
        return None

    angular = np.mod(
        np.arctan2(rotated_y / max(semi_b, 1e-6), rotated_x / max(semi_a, 1e-6)),
        2.0 * np.pi,
    )
    occupied = np.unique(np.floor(angular / (2.0 * np.pi) * 72.0).astype(int))
    coverage = min(1.0, len(occupied) / 72.0)
    return ArcFit(
        center_px=(float(cx), float(cy)),
        axes_px=(float(semi_a), float(semi_b)),
        angle_deg=float(angle),
        equivalent_radius_px=equivalent_radius,
        coverage=coverage,
        residual_px=residual,
    )


def _best_center_cluster(
    fits: tuple[ArcFit, ...], config: RingGeometryConfig, expected_center_px: tuple[float, float] | None
) -> tuple[ArcFit, ...]:
    if not fits:
        return ()
    median_radius = float(np.median([fit.equivalent_radius_px for fit in fits]))
    tolerance = max(9.0, config.center_tolerance_fraction * median_radius)
    candidates: list[tuple[ArcFit, ...]] = []
    for seed in fits:
        cluster = tuple(
            fit for fit in fits
            if np.linalg.norm(np.subtract(fit.center_px, seed.center_px)) <= tolerance
        )
        candidates.append(cluster)
    def score(cluster: tuple[ArcFit, ...]) -> tuple[float, float]:
        coverage = sum(item.coverage for item in cluster)
        if expected_center_px is None:
            return (float(len(cluster)), coverage)
        center = np.mean([item.center_px for item in cluster], axis=0)
        distance = float(np.linalg.norm(center - np.asarray(expected_center_px)))
        return (float(len(cluster)) - distance / max(tolerance, 1.0), coverage)
    return max(candidates, key=score)


def _deduplicate_radii(fits: tuple[ArcFit, ...]) -> tuple[ArcFit, ...]:
    ordered = sorted(fits, key=lambda item: item.equivalent_radius_px)
    groups: list[list[ArcFit]] = []
    for fit in ordered:
        if not groups:
            groups.append([fit])
            continue
        reference = float(np.mean([item.equivalent_radius_px for item in groups[-1]]))
        if abs(fit.equivalent_radius_px - reference) <= max(8.0, 0.10 * reference):
            groups[-1].append(fit)
        else:
            groups.append([fit])
    return tuple(
        max(group, key=lambda item: (item.coverage, -item.residual_px))
        for group in groups
    )


def _ratio_match(
    arcs: tuple[ArcFit, ...], config: RingGeometryConfig, expected_scale_px_per_mm: float | None
) -> tuple[float, float | None, tuple[ArcFit, ...]]:
    expected = np.asarray(config.expected_radius_ratios, np.float64)
    count = min(len(arcs), len(expected))
    if count == 0:
        return 0.0, None, ()
    observed_sets = combinations(arcs, count) if len(arcs) > count else (arcs,)
    best: tuple[float, float | None, tuple[ArcFit, ...]] = (0.0, None, ())
    best_rank = -1.0
    for observed_group in observed_sets:
        observed_group = tuple(sorted(observed_group, key=lambda item: item.equivalent_radius_px))
        observed = np.asarray([item.equivalent_radius_px for item in observed_group])
        for expected_indexes in combinations(range(len(expected)), count):
            expected_subset = expected[list(expected_indexes)]
            unit = float(np.dot(observed, expected_subset) / np.dot(expected_subset, expected_subset))
            if unit <= 0.0:
                continue
            relative = np.abs(observed - unit * expected_subset) / np.maximum(unit * expected_subset, 1e-6)
            rms = float(np.sqrt(np.mean(relative ** 2)))
            ratio_score = float(np.clip(1.0 - rms / config.ratio_tolerance, 0.0, 1.0))
            scale = unit / 20.0
            scale_compatibility = 1.0
            if expected_scale_px_per_mm is not None:
                scale_error = abs(scale - expected_scale_px_per_mm) / max(expected_scale_px_per_mm, 1e-6)
                scale_compatibility = float(np.clip(1.0 - scale_error, 0.0, 1.0))
            # History selects the most plausible radius assignment; it must not
            # lower the current frame's independent ring-ratio confidence.
            rank = ratio_score * scale_compatibility
            if rank > best_rank:
                best_rank = rank
                best = (ratio_score, scale, observed_group)
    return best


def detect_concentric_arcs(
    edge_mask: np.ndarray,
    config: RingGeometryConfig,
    *,
    expected_center_px: tuple[float, float] | None = None,
    expected_scale_px_per_mm: float | None = None,
    roi_xyxy: tuple[int, int, int, int] | None = None,
) -> RingGeometryResult:
    if edge_mask.ndim != 2 or edge_mask.size == 0:
        raise ValueError("edge_mask must be a non-empty single-channel image")
    height, width = edge_mask.shape
    x0, y0, x1, y1 = (0, 0, width, height) if roi_xyxy is None else roi_xyxy
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(width, int(x1)), min(height, int(y1))
    if x1 <= x0 or y1 <= y0:
        return _empty(DetectionFailure.RING_RATIO_INVALID)
    working = np.zeros_like(edge_mask)
    working[y0:y1, x0:x1] = edge_mask[y0:y1, x0:x1]
    contours, _ = cv2.findContours(working, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    fits = tuple(
        fit for contour in contours
        if (fit := _fit_arc(contour)) is not None
        and fit.coverage >= config.min_arc_coverage
    )
    clustered = _best_center_cluster(fits, config, expected_center_px)
    arcs = _deduplicate_radii(clustered)
    if not arcs:
        return _empty(DetectionFailure.RING_RATIO_INVALID)

    centers = np.asarray([arc.center_px for arc in arcs], np.float64)
    weights = np.asarray([max(arc.coverage, 0.05) for arc in arcs], np.float64)
    center = np.average(centers, axis=0, weights=weights)
    median_radius = float(np.median([arc.equivalent_radius_px for arc in arcs]))
    spread = float(np.sqrt(np.average(np.sum((centers - center) ** 2, axis=1), weights=weights)))
    center_tolerance = max(9.0, config.center_tolerance_fraction * median_radius)
    common_center_score = float(np.clip(1.0 - spread / (2.0 * center_tolerance), 0.0, 1.0))

    ratio_score, scale, matched = _ratio_match(arcs, config, expected_scale_px_per_mm)
    coverage_score = float(np.mean([arc.coverage for arc in matched or arcs]))
    visible_count = len(matched or arcs)
    valid = bool(
        visible_count >= config.min_multiple_arcs
        and scale is not None
        and np.isfinite(scale)
        and scale > 0.0
        and np.all(np.isfinite(center))
    )
    reason = None if valid else DetectionFailure.RING_RATIO_INVALID
    return RingGeometryResult(
        valid=valid,
        center_px=(float(center[0]), float(center[1])),
        arcs=tuple(matched or arcs),
        visible_arc_count=visible_count,
        common_center_score=common_center_score,
        ratio_score=ratio_score,
        coverage_score=coverage_score,
        scale_px_per_mm=scale,
        failure_reason=reason,
    )

