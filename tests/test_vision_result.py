from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from ev_vision.vision_result import VisionTargetResult


CORNERS = (
    (100.0, 80.0),
    (1180.0, 80.0),
    (1180.0, 944.0),
    (100.0, 944.0),
)


def hybrid_result(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "timestamp_ns": 50_000_000,
        "source_sequence": 17,
        "target_valid": True,
        "tracking_state": "TRACKING",
        "combined_score": 0.92,
        "center_px": (700.0, 480.0),
        "corners_px": CORNERS,
        "homography_valid": True,
        "target_x_mm": 0.0,
        "target_y_mm": 0.0,
        "failure_reason": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tracking_result_maps_to_valid_gimbal_semantics() -> None:
    result = VisionTargetResult.from_hybrid(
        hybrid_result(),
        image_size=(1280, 1024),
        now_ns=90_000_000,
    )

    assert result.timestamp_ms == 50
    assert result.frame_sequence == 17
    assert result.target_valid is True
    assert result.tracking_state == "TRACKING"
    assert result.confidence == pytest.approx(0.92)
    assert result.center_x_px == pytest.approx(700.0)
    assert result.center_y_px == pytest.approx(480.0)
    assert result.offset_x_px == pytest.approx(60.0)
    assert result.offset_y_px == pytest.approx(-32.0)
    assert result.target_x_mm == pytest.approx(0.0)
    assert result.target_y_mm == pytest.approx(0.0)
    assert result.corners == CORNERS
    assert result.frame_age_ms == pytest.approx(40.0)
    assert result.laser_permission is False


def test_result_is_immutable() -> None:
    result = VisionTargetResult.from_hybrid(
        hybrid_result(),
        image_size=(1280, 1024),
        now_ns=90_000_000,
    )

    with pytest.raises(FrozenInstanceError):
        result.target_valid = False  # type: ignore[misc]


@pytest.mark.parametrize("state", ["SEARCHING", "CONFIRMING", "PREDICTING", "LOST", "tracking"])
def test_nontracking_result_is_invalid(state: str) -> None:
    result = VisionTargetResult.from_hybrid(
        hybrid_result(tracking_state=state),
        image_size=(1280, 1024),
        now_ns=90_000_000,
    )

    assert result.target_valid is False
    assert result.laser_permission is False


@pytest.mark.parametrize(
    ("overrides", "now_ns"),
    [
        ({"target_valid": False}, 90_000_000),
        ({"center_px": None}, 90_000_000),
        ({"corners_px": None}, 90_000_000),
        ({"corners_px": ()}, 90_000_000),
        ({"corners_px": CORNERS[:3]}, 90_000_000),
        ({"homography_valid": False}, 90_000_000),
        ({"target_x_mm": None}, 90_000_000),
        ({"target_y_mm": None}, 90_000_000),
        ({}, 150_000_001),
    ],
)
def test_missing_or_stale_required_data_is_invalid(
    overrides: dict[str, object], now_ns: int
) -> None:
    result = VisionTargetResult.from_hybrid(
        hybrid_result(**overrides),
        image_size=(1280, 1024),
        now_ns=now_ns,
        max_result_age_ms=100.0,
    )

    assert result.target_valid is False
    assert result.laser_permission is False


def test_result_at_maximum_age_is_still_valid() -> None:
    result = VisionTargetResult.from_hybrid(
        hybrid_result(),
        image_size=(1280, 1024),
        now_ns=150_000_000,
        max_result_age_ms=100.0,
    )

    assert result.frame_age_ms == pytest.approx(100.0)
    assert result.target_valid is True


@pytest.mark.parametrize(
    "failure_reason",
    [
        "STALE_FRAME",
        "AMBIGUOUS_CANDIDATES",
        "MODEL_ERROR",
        "CAMERA_ERROR",
        "EXCESSIVE_POSITION_JUMP",
    ],
)
def test_failure_results_are_invalid_even_if_upstream_marks_target_valid(
    failure_reason: str,
) -> None:
    result = VisionTargetResult.from_hybrid(
        hybrid_result(failure_reason=failure_reason),
        image_size=(1280, 1024),
        now_ns=90_000_000,
    )

    assert result.target_valid is False
    assert result.laser_permission is False


def test_missing_center_produces_transport_neutral_empty_coordinates() -> None:
    result = VisionTargetResult.from_hybrid(
        hybrid_result(
            center_px=None,
            corners_px=None,
            homography_valid=False,
            target_x_mm=None,
            target_y_mm=None,
        ),
        image_size=(1280, 1024),
        now_ns=90_000_000,
    )

    assert result.center_x_px is None
    assert result.center_y_px is None
    assert result.offset_x_px is None
    assert result.offset_y_px is None
    assert result.target_x_mm is None
    assert result.target_y_mm is None
    assert result.corners == ()
    assert result.target_valid is False
    assert result.laser_permission is False


