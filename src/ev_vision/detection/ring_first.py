from __future__ import annotations

from dataclasses import replace
from enum import Enum
import math

from ev_vision.config import RingFirstConfig
from ev_vision.detection.ring_geometry import RingGeometryResult


class RingQuality(str, Enum):
    REJECTED = "REJECTED"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"


def classify_ring(result: RingGeometryResult, config: RingFirstConfig) -> RingQuality:
    if not config.enabled or not result.valid or result.center_px is None:
        return RingQuality.REJECTED
    if not all(math.isfinite(value) for value in result.center_px):
        return RingQuality.REJECTED
    if result.scale_px_per_mm is None or not math.isfinite(result.scale_px_per_mm):
        return RingQuality.REJECTED
    if result.scale_px_per_mm <= 0.0:
        return RingQuality.REJECTED

    if (
        result.visible_arc_count >= config.strong_min_arcs
        and result.common_center_score >= config.strong_common_center_score
        and result.ratio_score >= config.strong_ratio_score
        and result.coverage_score >= config.strong_coverage_score
    ):
        return RingQuality.STRONG
    if (
        config.allow_medium_acquisition
        and result.visible_arc_count >= config.medium_min_arcs
        and result.common_center_score >= config.medium_common_center_score
        and result.ratio_score >= config.medium_ratio_score
        and result.coverage_score >= config.medium_coverage_score
    ):
        return RingQuality.MEDIUM
    return RingQuality.REJECTED


def translate_ring_result(
    result: RingGeometryResult,
    offset_xy: tuple[int, int],
) -> RingGeometryResult:
    offset_x, offset_y = offset_xy
    center = None
    if result.center_px is not None:
        center = (result.center_px[0] + offset_x, result.center_px[1] + offset_y)
    arcs = tuple(
        replace(
            arc,
            center_px=(arc.center_px[0] + offset_x, arc.center_px[1] + offset_y),
        )
        for arc in result.arcs
    )
    return replace(result, center_px=center, arcs=arcs)