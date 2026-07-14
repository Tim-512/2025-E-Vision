from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from ev_vision.models import BoardObservation
from ev_vision.tuning.diagnostics import compute_diagnostics, render_overlay
from ev_vision.tuning.models import DetectionSnapshot, OverlayOptions


def half_dark_half_bright_checkerboard(height: int = 8, width: int = 12) -> np.ndarray:
    yy, xx = np.indices((height, width))
    checker = (xx + yy) % 2
    gray = np.where(xx < width // 2, checker * 5, 250 + checker * 5).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def detection_snapshot(*, source_sequence: int = 7, error: str | None = None) -> DetectionSnapshot:
    observation = BoardObservation(
        captured_ns=123,
        corners_px=((2.0, 2.0), (17.0, 2.0), (17.0, 17.0), (2.0, 17.0)),
        center_px=(9.5, 9.5),
        confidence=0.9,
        homography_valid=True,
    )
    return DetectionSnapshot(
        enabled=True,
        detected=True,
        source_sequence=source_sequence,
        observation=observation,
        result_age_ms=3.0,
        error=error,
    )


def geometry_only_options() -> OverlayOptions:
    return OverlayOptions(
        show_board_outline=True,
        show_corners=True,
        show_center=True,
        show_crosshair=False,
        show_detection_text=False,
        show_center_roi=False,
    )


def text_only_options() -> OverlayOptions:
    return OverlayOptions(
        show_board_outline=False,
        show_corners=False,
        show_center=False,
        show_crosshair=False,
        show_detection_text=True,
        show_center_roi=False,
    )


def test_compute_diagnostics_reports_histograms_clipping_focus_and_center_roi() -> None:
    image = half_dark_half_bright_checkerboard()

    diagnostics = compute_diagnostics(image, source_sequence=7, computed_ns=456)

    assert diagnostics.source_sequence == 7
    assert diagnostics.computed_ns == 456
    assert diagnostics.roi_px == (3, 2, 6, 4)
    for histogram in (
        diagnostics.gray_histogram,
        diagnostics.blue_histogram,
        diagnostics.green_histogram,
        diagnostics.red_histogram,
    ):
        assert len(histogram) == 256
        assert sum(histogram) == image.shape[0] * image.shape[1]
    assert diagnostics.dark_percent == pytest.approx(50.0)
    assert diagnostics.bright_percent == pytest.approx(50.0)
    assert diagnostics.focus_score > 0.0
    assert np.array_equal(image, half_dark_half_bright_checkerboard())


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 4, 3), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 4), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.float32),
        "not-an-array",
    ],
)
def test_compute_diagnostics_rejects_non_bgr_uint8_images(image: object) -> None:
    with pytest.raises(ValueError, match="non-empty uint8 HxWx3 BGR"):
        compute_diagnostics(image, source_sequence=7, computed_ns=456)


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 4, 3), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 4), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.float32),
        "not-an-array",
    ],
)
def test_render_overlay_rejects_non_bgr_uint8_images(image: object) -> None:
    with pytest.raises(ValueError, match="non-empty uint8 HxWx3 BGR"):
        render_overlay(
            image,
            source_sequence=7,
            detection=DetectionSnapshot(enabled=False, detected=False),
        )


def test_render_overlay_draws_only_geometry_compatible_with_the_frame() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    original = image.copy()
    detection = detection_snapshot(source_sequence=7)

    compatible = render_overlay(
        image,
        source_sequence=7,
        detection=detection,
        options=geometry_only_options(),
    )
    stale = render_overlay(
        image,
        source_sequence=8,
        detection=detection,
        options=geometry_only_options(),
    )

    assert np.any(compatible != original)
    assert np.array_equal(stale, original)
    assert np.array_equal(image, original)
    assert compatible is not image
    assert stale is not image


@pytest.mark.parametrize(
    "observation",
    [
        BoardObservation(
            captured_ns=123,
            corners_px=((2.0, 2.0), (17.0, 2.0), (17.0, 17.0)),
            center_px=(9.5, 9.5),
            confidence=0.9,
            homography_valid=True,
        ),
        BoardObservation(
            captured_ns=123,
            corners_px=((np.nan, 2.0), (17.0, 2.0), (17.0, 17.0), (2.0, 17.0)),
            center_px=(9.5, 9.5),
            confidence=0.9,
            homography_valid=True,
        ),
        BoardObservation(
            captured_ns=123,
            corners_px=((2.0, 2.0), (17.0, 2.0), (17.0, 17.0), (2.0, 17.0)),
            center_px=(np.inf, 9.5),
            confidence=0.9,
            homography_valid=True,
        ),
        BoardObservation(
            captured_ns=123,
            corners_px=((1e100, 2.0), (17.0, 2.0), (17.0, 17.0), (2.0, 17.0)),
            center_px=(9.5, 9.5),
            confidence=0.9,
            homography_valid=True,
        ),
        BoardObservation(
            captured_ns=123,
            corners_px=((2.0, 2.0), (17.0, 2.0), (17.0, 17.0), (2.0, 17.0)),
            center_px=(1e100, 9.5),
            confidence=0.9,
            homography_valid=True,
        ),
    ],
    ids=("wrong-corner-count", "nan-corner", "infinite-center", "huge-corner", "huge-center"),
)
def test_render_overlay_skips_invalid_observation_geometry_and_reports_not_detected(    monkeypatch: pytest.MonkeyPatch,
    observation: BoardObservation,
) -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    drawn_text: list[str] = []

    def record_text(*args, **kwargs):
        drawn_text.append(args[1])
        return args[0]

    monkeypatch.setattr(cv2, "putText", record_text)
    detection = replace(detection_snapshot(), observation=observation)
    options = replace(geometry_only_options(), show_detection_text=True)

    output = render_overlay(
        image,
        source_sequence=7,
        detection=detection,
        options=options,
    )

    assert drawn_text == ["not-detected"]
    assert np.array_equal(output, image)


def test_render_overlay_error_takes_priority_and_suppresses_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    drawn_text: list[str] = []

    def record_text(*args, **kwargs):
        drawn_text.append(args[1])
        return args[0]

    monkeypatch.setattr(cv2, "putText", record_text)
    options = replace(geometry_only_options(), show_detection_text=True)

    output = render_overlay(
        image,
        source_sequence=7,
        detection=detection_snapshot(error="boom"),
        options=options,
    )

    assert drawn_text == ["error"]
    assert np.array_equal(output, image)


@pytest.mark.parametrize(
    ("detection", "expected_text"),
    [
        (detection_snapshot(), "detected"),
        (detection_snapshot(source_sequence=6), "not-detected"),
        (
            replace(detection_snapshot(), detected=False, observation=None),
            "not-detected",
        ),
        (
            replace(detection_snapshot(), detected=False, observation=None, error="boom"),
            "error",
        ),
    ],
)
def test_render_overlay_displays_detection_state(
    monkeypatch: pytest.MonkeyPatch,
    detection: DetectionSnapshot,
    expected_text: str,
) -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    drawn_text: list[str] = []
    real_put_text = cv2.putText

    def record_text(*args, **kwargs):
        drawn_text.append(args[1])
        return real_put_text(*args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_text)

    render_overlay(
        image,
        source_sequence=7,
        detection=detection,
        options=text_only_options(),
    )

    assert drawn_text == [expected_text]
    assert np.count_nonzero(image) == 0


def test_render_overlay_can_draw_crosshair_and_center_roi_without_mutating_input() -> None:
    image = np.zeros((20, 24, 3), dtype=np.uint8)
    original = image.copy()
    options = OverlayOptions(
        show_board_outline=False,
        show_corners=False,
        show_center=False,
        show_crosshair=True,
        show_detection_text=False,
        show_center_roi=True,
    )

    output = render_overlay(
        image,
        source_sequence=7,
        detection=DetectionSnapshot(enabled=False, detected=False),
        options=options,
    )

    assert np.any(output != original)
    assert np.array_equal(image, original)


def test_render_overlay_disabled_returns_independent_copy() -> None:
    image = np.zeros((20, 24, 3), dtype=np.uint8)

    output = render_overlay(
        image,
        source_sequence=7,
        detection=detection_snapshot(),
        options=OverlayOptions(enabled=False),
    )

    assert output is not image
    assert np.array_equal(output, image)
