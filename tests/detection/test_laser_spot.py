from __future__ import annotations

import cv2
import numpy as np
import pytest

from ev_vision.detection.laser_spot import LaserSpotDetector


def background(size=(240, 320)) -> np.ndarray:
    height, width = size
    x = np.linspace(35, 80, width, dtype=np.float32)
    gray = np.repeat(x[None, :], height, axis=0)
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def add_spot(image: np.ndarray, center: tuple[float, float], color=(255, 80, 255), sigma=2.2) -> np.ndarray:
    result = image.astype(np.float32).copy()
    yy, xx = np.indices(image.shape[:2], dtype=np.float32)
    blob = np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2.0 * sigma * sigma))
    result += blob[..., None] * np.asarray(color, np.float32)[None, None, :]
    return np.clip(result, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("color", [(255, 60, 220), (255, 120, 80), (220, 50, 255)])
def test_off_on_difference_detects_405nm_color_variations(color) -> None:
    off = background()
    expected = (151.35, 92.65)
    on = add_spot(off, expected, color=color)
    detector = LaserSpotDetector()
    detector.update_background(off)

    observation = detector.detect(on, captured_ns=42)

    assert observation is not None
    assert observation.captured_ns == 42
    assert np.linalg.norm(np.asarray(observation.position_px) - expected) <= 0.45
    assert observation.confidence >= 0.7


def test_saturated_spot_has_subpixel_centroid() -> None:
    off = background()
    expected = (80.4, 180.7)
    on = add_spot(off, expected, color=(500, 200, 500), sigma=3.0)
    detector = LaserSpotDetector()
    detector.update_background(off)

    observation = detector.detect(on, captured_ns=0)

    assert observation is not None
    assert np.linalg.norm(np.asarray(observation.position_px) - expected) <= 0.6


def test_current_spot_wins_over_lingering_old_mark() -> None:
    off = background()
    old = add_spot(off, (60.0, 70.0), color=(100, 25, 100), sigma=2.5)
    current = add_spot(old, (245.2, 160.8), color=(255, 80, 255), sigma=2.2)
    detector = LaserSpotDetector()
    detector.update_background(off)

    observation = detector.detect(current, captured_ns=0)

    assert observation is not None
    assert np.linalg.norm(np.asarray(observation.position_px) - (245.2, 160.8)) <= 0.7


def test_missing_spot_returns_none() -> None:
    off = background()
    detector = LaserSpotDetector()
    detector.update_background(off)
    assert detector.detect(off.copy(), captured_ns=0) is None


def test_roi_excludes_spot_and_accepts_boundary_spot() -> None:
    off = background()
    detector = LaserSpotDetector()
    detector.update_background(off)
    image = add_spot(off, (199.0, 99.0))

    assert detector.detect(image, captured_ns=0, roi=(20, 20, 150, 80)) is None
    observation = detector.detect(image, captured_ns=0, roi=(150, 70, 201, 101))
    assert observation is not None
    assert np.linalg.norm(np.asarray(observation.position_px) - (199.0, 99.0)) <= 1.0


def test_slow_background_update_does_not_immediately_absorb_live_spot() -> None:
    off = background()
    on = add_spot(off, (110.0, 120.0))
    detector = LaserSpotDetector(background_alpha=0.02)
    detector.update_background(off)
    detector.update_background(on)

    assert detector.detect(on, captured_ns=0) is not None
