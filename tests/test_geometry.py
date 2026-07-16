import math

import numpy as np
import pytest

from ev_vision.geometry import CameraIntrinsics, HomographyError, TargetGeometry, pixel_error_to_angles


def test_standard_target_center_and_circle_radius() -> None:
    geom = TargetGeometry(px_per_cm=40.0)
    assert geom.rectified_size_px == (840, 1188)
    assert geom.center_px == pytest.approx((420.0, 594.0))
    assert geom.cm_to_rectified_px((6.0, 0.0)) == pytest.approx((660.0, 594.0))


def test_target_image_round_trip() -> None:
    corners = ((100.0, 50.0), (500.0, 80.0), (470.0, 700.0), (80.0, 650.0))
    geom = TargetGeometry.from_image_corners(corners, px_per_cm=40.0)
    for point in ((0.0, 0.0), (6.0, 0.0), (-10.5, 14.85)):
        assert geom.image_to_target_cm(geom.target_cm_to_image(point)) == pytest.approx(point, abs=1e-5)


def test_target_image_round_trip_preserves_custom_board_dimensions() -> None:
    corners = ((10.0, 20.0), (410.0, 20.0), (410.0, 220.0), (10.0, 220.0))
    geom = TargetGeometry.from_image_corners(
        corners,
        px_per_cm=25.0,
        width_cm=40.0,
        height_cm=20.0,
    )

    assert geom.width_cm == pytest.approx(40.0)
    assert geom.height_cm == pytest.approx(20.0)
    assert np.asarray(geom.target_corners_cm) == pytest.approx(
        np.asarray(((-20.0, 10.0), (20.0, 10.0), (20.0, -10.0), (-20.0, -10.0)))
    )
    assert geom.image_to_target_cm(corners[0]) == pytest.approx((-20.0, 10.0))
    assert geom.image_to_target_cm(corners[2]) == pytest.approx((20.0, -10.0))

def test_degenerate_quadrilateral_is_rejected() -> None:
    with pytest.raises(HomographyError):
        TargetGeometry.from_image_corners(((0, 0), (1, 0), (2, 0), (3, 0)))


def test_pixel_error_to_angles() -> None:
    intrinsics = CameraIntrinsics(fx=1000.0, fy=800.0, cx=640.0, cy=512.0)
    yaw, pitch = pixel_error_to_angles((740.0, 432.0), intrinsics)
    assert yaw == pytest.approx(math.atan2(100.0, 1000.0))
    assert pitch == pytest.approx(math.atan2(80.0, 800.0))
