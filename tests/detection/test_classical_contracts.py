from __future__ import annotations

import dataclasses

from ev_vision.detection.contracts import (
    ClassicalBoardResult,
    ObservationSource,
)
from ev_vision.detection.failures import DetectionFailure


def test_source_priority() -> None:
    order = [
        ObservationSource.FULL_BOARD,
        ObservationSource.CONCENTRIC_ARCS,
        ObservationSource.FUSED_PARTIAL,
        ObservationSource.SINGLE_ARC,
        ObservationSource.WHITE_REGION,
        ObservationSource.PREDICTED,
    ]
    priorities = [item.priority for item in order]

    assert priorities == sorted(priorities, reverse=True)


def test_partial_result_needs_no_four_corners() -> None:
    result = ClassicalBoardResult(
        timestamp_ns=1_000_000,
        source_sequence=7,
        detected=True,
        target_valid=True,
        tracking_state="TRACKING",
        observation_source=ObservationSource.CONCENTRIC_ARCS,
        confidence=0.81,
        center_px=(320.0, 240.0),
        corners_px=(),
    )

    assert result.homography_valid is False


def test_contract_has_no_red_or_laser_fields() -> None:
    names = {item.name for item in dataclasses.fields(ClassicalBoardResult)}

    assert not any(
        name.startswith("red") or "_red" in name or "laser" in name
        for name in names
    )


def test_classical_failure_reasons_are_available() -> None:
    expected = {
        "NO_WHITE_CANDIDATE",
        "WHITE_OCCUPANCY_LOW",
        "A4_GEOMETRY_INVALID",
        "RING_RATIO_INVALID",
        "RING_CENTER_INCONSISTENT",
        "PARTIAL_HISTORY_REQUIRED",
        "SINGLE_ARC_LIMIT",
        "EXCESSIVE_SCALE_JUMP",
        "EXCESSIVE_VELOCITY",
        "PREDICTION_EXPIRED",
    }

    assert expected <= {item.name for item in DetectionFailure}
