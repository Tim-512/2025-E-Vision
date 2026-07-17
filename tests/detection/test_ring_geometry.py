from __future__ import annotations

import cv2
import numpy as np
import pytest

from ev_vision.config import ImageNormalizationConfig, RingGeometryConfig
from ev_vision.detection.image_normalization import normalize_frame
from ev_vision.detection.ring_geometry import detect_concentric_arcs
from tests.fixtures.synthetic_board import render_ring_target


def ring_edges(image: np.ndarray) -> np.ndarray:
    return normalize_frame(image, ImageNormalizationConfig()).ring_edge_mask


def test_five_gray_rings_pass_common_center_and_ratio_checks():
    target = render_ring_target(ring_gray=118, perspective=0.08)
    result = detect_concentric_arcs(ring_edges(target.image), RingGeometryConfig())
    assert result.valid is True
    assert result.center_px == pytest.approx(target.center_px, abs=8.0)
    assert len(result.arcs) >= 3
    assert result.ratio_score > 0.75
    assert result.common_center_score > 0.75


def test_non_red_gray_intensities_are_equivalent():
    dark = render_ring_target(ring_gray=70)
    light = render_ring_target(ring_gray=145)
    a = detect_concentric_arcs(ring_edges(dark.image), RingGeometryConfig())
    b = detect_concentric_arcs(ring_edges(light.image), RingGeometryConfig())
    assert a.valid and b.valid
    assert a.center_px == pytest.approx(b.center_px, abs=3.0)


def test_wrong_radius_ratios_are_rejected():
    image = np.zeros((720, 960), np.uint8)
    for radius in (35, 69, 111, 143, 209):
        cv2.circle(image, (480, 360), radius, 255, 2)
    result = detect_concentric_arcs(
        image, RingGeometryConfig(ratio_tolerance=0.10)
    )
    assert result.valid is False
    assert result.ratio_score < 0.7


def test_partial_multiple_arcs_recover_center_near_edge():
    target = render_ring_target(board_center=(5.0, 360.0))
    result = detect_concentric_arcs(
        ring_edges(target.image), RingGeometryConfig(min_arc_coverage=0.12)
    )
    assert result.visible_arc_count >= 2
    assert result.center_px == pytest.approx(target.center_px, abs=15.0)


def test_partial_arcs_keep_ratio_valid_with_small_expected_scale_difference():
    target = render_ring_target(board_center=(0.0, 360.0))
    result = detect_concentric_arcs(
        ring_edges(target.image),
        RingGeometryConfig(min_arc_coverage=0.12),
        expected_scale_px_per_mm=1.5735,
    )

    assert result.valid is True
    assert result.visible_arc_count >= 2
    assert result.center_px == pytest.approx(target.center_px, abs=15.0)
    assert result.scale_px_per_mm == pytest.approx(1.657, abs=0.03)


def test_saturated_spot_does_not_form_multiple_ring_identity():
    image = np.zeros((360, 480), np.uint8)
    cv2.circle(image, (240, 180), 8, 255, -1)
    result = detect_concentric_arcs(image, RingGeometryConfig())
    assert result.valid is False
