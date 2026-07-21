from __future__ import annotations

import pytest

from ev_vision.config import RingFirstConfig
from ev_vision.detection.failures import DetectionFailure
from ev_vision.detection.ring_first import RingQuality, classify_ring, translate_ring_result
from ev_vision.detection.ring_geometry import ArcFit, RingGeometryResult


def ring_result(
    *,
    count: int,
    common_center: float,
    ratio: float,
    coverage: float,
    valid: bool = True,
    center: tuple[float, float] | None = (20.0, 30.0),
    scale: float | None = 1.5,
) -> RingGeometryResult:
    arcs = tuple(
        ArcFit(
            center_px=(20.0 + index, 30.0 - index),
            axes_px=(10.0 + index, 11.0 + index),
            angle_deg=0.0,
            equivalent_radius_px=20.0 * (index + 1),
            coverage=coverage,
            residual_px=0.5,
        )
        for index in range(count)
    )
    return RingGeometryResult(
        valid=valid,
        center_px=center,
        arcs=arcs,
        visible_arc_count=count,
        common_center_score=common_center,
        ratio_score=ratio,
        coverage_score=coverage,
        scale_px_per_mm=scale,
        failure_reason=None if valid else DetectionFailure.RING_RATIO_INVALID,
    )


def test_classifies_strong_medium_and_rejected_rings() -> None:
    config = RingFirstConfig()
    assert classify_ring(
        ring_result(count=3, common_center=0.80, ratio=0.85, coverage=0.25), config
    ) is RingQuality.STRONG
    assert classify_ring(
        ring_result(count=2, common_center=0.62, ratio=0.66, coverage=0.13), config
    ) is RingQuality.MEDIUM
    assert classify_ring(
        ring_result(count=1, common_center=0.95, ratio=0.95, coverage=0.90), config
    ) is RingQuality.REJECTED


@pytest.mark.parametrize(
    "result",
    [
        ring_result(count=3, common_center=0.9, ratio=0.9, coverage=0.4, valid=False),
        ring_result(count=3, common_center=0.9, ratio=0.9, coverage=0.4, center=(float("nan"), 1.0)),
        ring_result(count=3, common_center=0.9, ratio=0.9, coverage=0.4, scale=0.0),
    ],
)
def test_rejects_invalid_center_or_scale(result: RingGeometryResult) -> None:
    assert classify_ring(result, RingFirstConfig()) is RingQuality.REJECTED


def test_translate_ring_result_offsets_center_and_arc_centers() -> None:
    local = ring_result(count=2, common_center=0.75, ratio=0.75, coverage=0.2)
    translated = translate_ring_result(local, (100, 50))
    assert translated.center_px == pytest.approx((120.0, 80.0))
    assert translated.arcs[0].center_px == pytest.approx((120.0, 80.0))
    assert translated.arcs[1].center_px == pytest.approx((121.0, 79.0))
    assert translated.scale_px_per_mm == local.scale_px_per_mm
    assert translated.common_center_score == local.common_center_score
