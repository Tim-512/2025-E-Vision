from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from ev_vision.config import DetectionConfig, RingGeometryConfig
from ev_vision.detection.classical_board import ClassicalBoardDetector
from ev_vision.detection.contracts import ObservationSource
from tests.fixtures.synthetic_board import render_ring_target


def _confirmed_detector() -> ClassicalBoardDetector:
    defaults = DetectionConfig()
    subject = ClassicalBoardDetector(dataclasses.replace(
        defaults,
        tracking=dataclasses.replace(
            defaults.tracking,
            max_center_jump_px=800.0,
            max_velocity_px_s=100_000.0,
            max_acceleration_px_s2=10_000_000.0,
        ),
    ))
    target = render_ring_target()
    for sequence in range(1, 4):
        result = subject.detect(
            target.image,
            captured_ns=1_000_000_000 + sequence * 10_000_000,
            source_sequence=sequence,
        )
    assert result.target_valid is True
    return subject


def _expired_detector() -> ClassicalBoardDetector:
    subject = _confirmed_detector()
    blank = np.full((720, 960, 3), 35, np.uint8)
    for sequence, timestamp_ns in enumerate(
        (1_050_000_000, 1_100_000_000, 1_151_000_000, 1_210_000_000),
        start=5,
    ):
        result = subject.detect(
            blank,
            captured_ns=timestamp_ns,
            source_sequence=sequence,
        )
    assert result.tracking_state == "LOST"
    return subject


def test_complete_board_confirms_and_uses_a4_center() -> None:
    target = render_ring_target(perspective=0.10, shadow_strength=0.30)
    subject = ClassicalBoardDetector(DetectionConfig())
    subject.detect(target.image, captured_ns=1_000_000_000, source_sequence=1)
    subject.detect(target.image, captured_ns=1_010_000_000, source_sequence=2)
    result = subject.detect(
        target.image,
        captured_ns=1_020_000_000,
        source_sequence=3,
        include_debug=True,
    )

    assert result.target_valid is True
    assert result.observation_source == ObservationSource.FULL_BOARD
    assert result.center_px == pytest.approx(target.center_px, abs=10.0)
    assert result.homography_valid is True
    assert set(result.debug_images) >= {
        "normalized-gray",
        "white-mask",
        "edge-mask",
        "ring-arcs",
        "candidate-scores",
    }


def test_confirmed_detector_tracks_clipped_target() -> None:
    subject = _confirmed_detector()
    target = render_ring_target(board_center=(10.0, 360.0))

    result = subject.detect(
        target.image,
        captured_ns=1_040_000_000,
        source_sequence=4,
    )

    assert result.target_valid is True
    assert result.observation_source in {
        ObservationSource.CONCENTRIC_ARCS,
        ObservationSource.FUSED_PARTIAL,
    }
    assert result.partially_outside is True


def test_lost_detector_rejects_partial_only_reacquisition() -> None:
    subject = _expired_detector()
    target = render_ring_target(board_center=(-30.0, 360.0))

    result = subject.detect(
        target.image,
        captured_ns=2_000_000_000,
        source_sequence=20,
    )

    assert result.target_valid is False
    assert result.tracking_state == "LOST"


def test_config_apply_and_reload_need_no_model() -> None:
    subject = ClassicalBoardDetector(DetectionConfig())
    subject.apply_config(
        dataclasses.replace(
            DetectionConfig(),
            rings=RingGeometryConfig(ratio_tolerance=0.20),
        )
    )
    subject.reload_model()

    assert subject.model_state == "READY"
    assert subject.model_backend == "classical"
    assert subject.model_path is None


def test_preview_does_not_confirm_or_mutate_tracker() -> None:
    subject = ClassicalBoardDetector(DetectionConfig())
    target = render_ring_target()

    preview = subject.detect(
        target.image,
        captured_ns=1_000_000_000,
        source_sequence=1,
        update_tracker=False,
    )
    actual = subject.detect(
        target.image,
        captured_ns=1_010_000_000,
        source_sequence=2,
    )

    assert preview.tracking_state == "CONFIRMING"
    assert actual.tracking_state == "CONFIRMING"
    assert actual.target_valid is False


def test_reset_requires_full_confirmation_again() -> None:
    subject = _confirmed_detector()
    subject.reset()
    target = render_ring_target()

    result = subject.detect(
        target.image,
        captured_ns=2_000_000_000,
        source_sequence=20,
    )

    assert result.tracking_state == "CONFIRMING"
    assert result.target_valid is False


def test_blank_frame_fails_closed_without_debug_crash() -> None:
    subject = ClassicalBoardDetector(DetectionConfig())
    result = subject.detect(
        np.zeros((240, 320, 3), np.uint8),
        captured_ns=1_000_000_000,
        source_sequence=1,
        include_debug=True,
    )

    assert result.target_valid is False
    assert result.center_px is None
    assert set(result.debug_images) >= {
        "normalized-gray",
        "white-mask",
        "edge-mask",
        "ring-arcs",
        "candidate-scores",
    }


def test_detector_source_has_no_red_or_model_dependency() -> None:
    source = inspect.getsource(__import__(
        "ev_vision.detection.classical_board", fromlist=["ClassicalBoardDetector"]
    )).lower()

    assert "ultralytics" not in source
    assert "color_bgr2hsv" not in source
    assert "red_channel" not in source
    assert "red_mask" not in source
