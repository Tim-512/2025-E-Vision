from pathlib import Path

import cv2
import numpy as np

from ev_vision.detection.roi_board_geometry import RoiBoardGeometry


FIXTURE = Path(__file__).parents[1] / "fixtures" / "real_board_roi.jpg"


def test_real_capture_geometry_recovers_board_when_given_correct_model_roi() -> None:
    encoded = np.fromfile(FIXTURE, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert image is not None

    # Full-image coordinates minus crop origin (285, 235), with loose model padding.
    expected = np.asarray(
        ((99, 108), (653, 80), (677, 467), (127, 493)),
        np.float32,
    )
    model_box = (75.0, 60.0, 700.0, 515.0)

    result = RoiBoardGeometry().refine(
        image,
        model_box=model_box,
        include_debug=True,
    )

    assert result.accepted is True
    assert result.failure_reason is None
    assert np.allclose(result.corners_px, expected, atol=24.0)
    assert result.debug is not None