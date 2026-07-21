from __future__ import annotations

from dataclasses import replace

import pytest

from ev_vision.detection.contracts import ObservationSource
from ev_vision.gimbal_usb.protocol import GimbalTargetCommand
from ev_vision.gimbal_usb.safety import GateLimits, GimbalControlGate
from ev_vision.tuning.models import DetectionSnapshot


REAL_SOURCES = (
    "FULL_BOARD",
    "CONCENTRIC_ARCS",
    "SINGLE_ARC",
    "WHITE_REGION",
    "FUSED_PARTIAL",
)


class FakeConverter:
    available = True

    def __init__(self, angles: tuple[float, float]) -> None:
        self.angles = angles
        self.calls: list[tuple[tuple[float, float] | None, tuple[tuple[float, float], ...]]] = []

    def convert(
        self,
        center_px: tuple[float, float] | None,
        corners_px: tuple[tuple[float, float], ...] = (),
    ) -> tuple[float, float]:
        self.calls.append((center_px, corners_px))
        return self.angles


class SequenceConverter:
    available = True

    def __init__(self, angles: list[tuple[float, float]]) -> None:
        self._angles = iter(angles)

    def convert(
        self,
        center_px: tuple[float, float] | None,
        corners_px: tuple[tuple[float, float], ...] = (),
    ) -> tuple[float, float]:
        del center_px, corners_px
        return next(self._angles)


class UnavailableConverter:
    available = False
    reason = "calibration unavailable"

    def convert(self, center_px, corners_px=()):
        raise AssertionError("unavailable converter must not be called")


class RaisingConverter:
    available = True

    def convert(self, center_px, corners_px=()):
        del center_px, corners_px
        raise RuntimeError("calibration conversion failed")


def limits() -> GateLimits:
    return GateLimits(
        max_result_age_ms=60.0,
        predicted_control_max_frames=3,
        predicted_control_max_age_us=60_000,
        predicted_max_angle_step_deg=1.5,
        image_size=(1280, 1024),
    )


def snapshot(**changes) -> DetectionSnapshot:
    item = DetectionSnapshot(
        enabled=True,
        detected=True,
        source_sequence=1,
        result_age_ms=20.0,
        error=None,
        target_valid=True,
        tracking_state="TRACKING",
        observation_source="FULL_BOARD",
        predicted_frames=0,
        source_age_us=0,
        failure_reason=None,
        center_px=(640.0, 512.0),
    )
    return replace(item, **changes)


def prediction(**changes) -> DetectionSnapshot:
    item = snapshot(
        tracking_state="PREDICTING",
        observation_source="PREDICTED",
        predicted_frames=1,
        source_age_us=20_000,
    )
    return replace(item, **changes)


@pytest.mark.parametrize("source", REAL_SOURCES)
def test_fresh_real_source_is_immediately_valid(source: str) -> None:
    gate = GimbalControlGate(FakeConverter((1.0, -2.0)), limits())

    decision = gate.evaluate(snapshot(observation_source=source))

    assert decision.command == GimbalTargetCommand(1.0, -2.0, True)
    assert decision.reason == "real_observation"
    assert decision.observation_source == source


def test_only_full_board_passes_corners_to_angle_converter() -> None:
    corners = ((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0))
    converter = FakeConverter((1.0, -2.0))
    gate = GimbalControlGate(converter, limits())

    gate.evaluate(snapshot(observation_source="FULL_BOARD", corners_px=corners))
    gate.evaluate(snapshot(observation_source="CONCENTRIC_ARCS", corners_px=corners))

    assert converter.calls == [
        ((640.0, 512.0), corners),
        ((640.0, 512.0), ()),
    ]


def test_observation_source_enum_is_supported() -> None:
    gate = GimbalControlGate(FakeConverter((1.0, -2.0)), limits())

    decision = gate.evaluate(
        snapshot(observation_source=ObservationSource.CONCENTRIC_ARCS)
    )

    assert decision.command.tracking is True
    assert decision.observation_source == "CONCENTRIC_ARCS"


@pytest.mark.parametrize(
    ("detection", "reason"),
    [
        (None, "detection_missing"),
        (snapshot(enabled=False), "detection_disabled"),
        (snapshot(error="detector stopped"), "detection_error"),
        (snapshot(target_valid=False), "target_invalid"),
        (snapshot(failure_reason="position_jump"), "upstream_failure"),
        (snapshot(result_age_ms=None), "result_stale"),
        (snapshot(result_age_ms=-0.1), "result_stale"),
        (snapshot(result_age_ms=60.1), "result_stale"),
        (snapshot(result_age_ms=float("nan")), "result_stale"),
        (snapshot(result_age_ms=float("inf")), "result_stale"),
        (snapshot(tracking_state="SEARCHING"), "tracking_state"),
        (snapshot(observation_source="NONE"), "observation_source"),
        (snapshot(observation_source="SOMETHING_NEW"), "observation_source"),
        (snapshot(center_px=None), "center_invalid"),
        (snapshot(center_px=(float("nan"), 20.0)), "center_invalid"),
        (snapshot(center_px=(20.0, float("inf"))), "center_invalid"),
        (snapshot(center_px=(-0.1, 20.0)), "center_invalid"),
        (snapshot(center_px=(1280.0, 20.0)), "center_invalid"),
        (snapshot(center_px=(20.0, 1024.0)), "center_invalid"),
    ],
)
def test_invalid_real_observations_are_all_zero(
    detection: DetectionSnapshot | None,
    reason: str,
) -> None:
    gate = GimbalControlGate(FakeConverter((9.0, -7.0)), limits())

    decision = gate.evaluate(detection)

    assert decision.command == GimbalTargetCommand.safe()
    assert decision.reason == reason


def test_exact_real_freshness_and_frame_edges_are_valid() -> None:
    gate = GimbalControlGate(FakeConverter((0.1, -0.2)), limits())

    decision = gate.evaluate(
        snapshot(result_age_ms=60.0, center_px=(1279.999, 1023.999))
    )

    assert decision.command.tracking is True


@pytest.mark.parametrize(
    "angles",
    [
        (float("nan"), 0.0),
        (0.0, float("inf")),
        (float("-inf"), 0.0),
    ],
)
def test_non_finite_converted_angles_fail_closed(
    angles: tuple[float, float],
) -> None:
    decision = GimbalControlGate(FakeConverter(angles), limits()).evaluate(snapshot())

    assert decision.command == GimbalTargetCommand.safe()
    assert decision.reason == "angle_conversion_failed"


@pytest.mark.parametrize("converter", [UnavailableConverter(), RaisingConverter()])
def test_unavailable_or_raising_converter_fails_closed(converter) -> None:
    decision = GimbalControlGate(converter, limits()).evaluate(snapshot())

    assert decision.command == GimbalTargetCommand.safe()
    assert decision.reason == "angle_conversion_failed"


def test_predictions_one_to_three_are_valid_at_inclusive_limits() -> None:
    gate = GimbalControlGate(
        SequenceConverter(
            [
                (1.0, 1.0),
                (2.5, -0.5),
                (4.0, -2.0),
                (5.5, -3.5),
            ]
        ),
        limits(),
    )
    assert gate.evaluate(snapshot()).command.tracking is True

    for frame in range(1, 4):
        decision = gate.evaluate(
            prediction(
                predicted_frames=frame,
                source_age_us=frame * 20_000,
                result_age_ms=60.0,
            )
        )
        assert decision.command.tracking is True
        assert decision.reason == "predicted_observation"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"tracking_state": "TRACKING"}, "tracking_state"),
        ({"predicted_frames": 0}, "prediction_frame_limit"),
        ({"predicted_frames": 4}, "prediction_frame_limit"),
        ({"source_age_us": -1}, "prediction_age_limit"),
        ({"source_age_us": 60_001}, "prediction_age_limit"),
        ({"result_age_ms": 60.1}, "result_stale"),
        ({"target_valid": False}, "target_invalid"),
        ({"error": "detector failed"}, "detection_error"),
        ({"failure_reason": "jump"}, "upstream_failure"),
        ({"center_px": (1280.0, 1.0)}, "center_invalid"),
    ],
)
def test_any_rejected_prediction_latches_until_real(
    changes: dict[str, object],
    reason: str,
) -> None:
    gate = GimbalControlGate(FakeConverter((0.0, 0.0)), limits())
    assert gate.evaluate(snapshot()).command.tracking is True

    rejected = gate.evaluate(prediction(**changes))
    latched = gate.evaluate(prediction(predicted_frames=2, source_age_us=40_000))

    assert rejected.command == GimbalTargetCommand.safe()
    assert rejected.reason == reason
    assert latched.command == GimbalTargetCommand.safe()
    assert latched.reason == "prediction_latched_invalid"


def test_prediction_without_prior_real_latches() -> None:
    gate = GimbalControlGate(FakeConverter((0.0, 0.0)), limits())

    first = gate.evaluate(prediction())
    second = gate.evaluate(prediction(predicted_frames=2, source_age_us=40_000))

    assert first.reason == "prediction_without_real"
    assert second.reason == "prediction_latched_invalid"
    assert first.command == second.command == GimbalTargetCommand.safe()


@pytest.mark.parametrize("angles", [(1.500001, 0.0), (0.0, -1.500001)])
def test_prediction_per_axis_angle_jump_is_rejected_and_latched(
    angles: tuple[float, float],
) -> None:
    gate = GimbalControlGate(SequenceConverter([(0.0, 0.0), angles]), limits())
    assert gate.evaluate(snapshot()).command.tracking is True

    rejected = gate.evaluate(prediction())

    assert rejected.command == GimbalTargetCommand.safe()
    assert rejected.reason == "prediction_angle_step"


def test_rejected_prediction_does_not_retain_previous_nonzero_angles() -> None:
    gate = GimbalControlGate(
        SequenceConverter([(8.0, -4.0), (20.0, -4.0)]),
        limits(),
    )
    assert gate.evaluate(snapshot()).command == GimbalTargetCommand(8.0, -4.0, True)

    rejected = gate.evaluate(prediction())

    assert rejected.command == GimbalTargetCommand.safe()
    assert rejected.command.yaw_deg == 0.0
    assert rejected.command.pitch_deg == 0.0


def test_valid_real_observation_immediately_recovers_without_jump_check() -> None:
    gate = GimbalControlGate(
        SequenceConverter(
            [
                (0.0, 0.0),
                (3.0, 0.0),
                (50.0, -40.0),
                (50.5, -39.0),
            ]
        ),
        limits(),
    )
    assert gate.evaluate(snapshot()).command.tracking is True
    assert gate.evaluate(prediction()).reason == "prediction_angle_step"
    assert gate.evaluate(prediction(predicted_frames=2)).reason == (
        "prediction_latched_invalid"
    )

    recovered = gate.evaluate(
        snapshot(center_px=(700.0, 500.0), observation_source="WHITE_REGION")
    )
    next_prediction = gate.evaluate(
        prediction(predicted_frames=1, source_age_us=20_000)
    )

    assert recovered.command == GimbalTargetCommand(50.0, -40.0, True)
    assert recovered.reason == "real_observation"
    assert next_prediction.command == GimbalTargetCommand(50.5, -39.0, True)


def test_invalid_real_observation_does_not_clear_prediction_latch() -> None:
    gate = GimbalControlGate(
        SequenceConverter([(0.0, 0.0), (3.0, 0.0)]),
        limits(),
    )
    gate.evaluate(snapshot())
    assert gate.evaluate(prediction()).reason == "prediction_angle_step"

    assert gate.evaluate(snapshot(target_valid=False)).reason == "target_invalid"
    assert gate.evaluate(prediction(predicted_frames=2)).reason == (
        "prediction_latched_invalid"
    )


def test_prediction_converter_exception_latches() -> None:
    class RealThenRaise:
        available = True

        def __init__(self) -> None:
            self.calls = 0

        def convert(self, center_px, corners_px=()):
            del center_px, corners_px
            self.calls += 1
            if self.calls == 1:
                return (0.0, 0.0)
            raise RuntimeError("failed")

    gate = GimbalControlGate(RealThenRaise(), limits())
    gate.evaluate(snapshot())

    failed = gate.evaluate(prediction())
    latched = gate.evaluate(prediction(predicted_frames=2))

    assert failed.reason == "angle_conversion_failed"
    assert latched.reason == "prediction_latched_invalid"
