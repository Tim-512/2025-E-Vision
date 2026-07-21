from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from ev_vision.calibration import Calibration, CalibrationError, save_calibration
import ev_vision.gimbal_usb.target_angles as module
from ev_vision.gimbal_usb.target_angles import (
    TargetAngleConverter,
    UnavailableTargetAngleConverter,
)


def calibration(
    *,
    image_size: tuple[int, int] = (1280, 1024),
    rms_px: float = 0.31,
    distortion: np.ndarray | None = None,
) -> Calibration:
    return Calibration(
        image_size=image_size,
        camera_matrix=np.array(
            [
                [800.0, 0.0, 640.0],
                [0.0, 800.0, 512.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        distortion=(
            np.zeros(5, dtype=np.float64)
            if distortion is None
            else distortion
        ),
        rms_px=rms_px,
    )


def test_optical_center_converts_to_zero_angles() -> None:
    converter = TargetAngleConverter(calibration(), yaw_sign=1, pitch_sign=1)

    yaw_deg, pitch_deg = converter.convert((640.0, 512.0))

    assert yaw_deg == pytest.approx(0.0, abs=1e-10)
    assert pitch_deg == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(
    ("center_px", "yaw_sign", "pitch_sign"),
    [
        ((720.0, 512.0), 1, 0),
        ((560.0, 512.0), -1, 0),
        ((640.0, 592.0), 0, 1),
        ((640.0, 432.0), 0, -1),
    ],
)
def test_raw_angle_signs_follow_image_axes(
    center_px: tuple[float, float],
    yaw_sign: int,
    pitch_sign: int,
) -> None:
    converter = TargetAngleConverter(calibration(), yaw_sign=1, pitch_sign=1)

    yaw_deg, pitch_deg = converter.convert(center_px)

    assert (yaw_deg > 0) - (yaw_deg < 0) == yaw_sign
    assert (pitch_deg > 0) - (pitch_deg < 0) == pitch_sign


def test_axis_configuration_signs_apply_after_raw_angle(monkeypatch) -> None:
    converter = TargetAngleConverter(calibration(), yaw_sign=-1, pitch_sign=-1)
    monkeypatch.setattr(
        module.cv2,
        "undistortPoints",
        lambda *args: np.array([[[0.1, 0.2]]], dtype=np.float64),
    )

    yaw_deg, pitch_deg = converter.convert((700.0, 600.0))

    assert yaw_deg == pytest.approx(-math.degrees(math.atan(0.1)))
    assert pitch_deg == pytest.approx(-math.degrees(math.atan(0.2)))


def test_fixed_laser_biases_apply_after_axis_signs(monkeypatch) -> None:
    converter = TargetAngleConverter(
        calibration(),
        yaw_sign=-1,
        pitch_sign=1,
        laser_yaw_bias_deg=1.25,
        laser_pitch_bias_deg=-0.75,
    )
    monkeypatch.setattr(
        module.cv2,
        "undistortPoints",
        lambda *args: np.array([[[0.1, 0.2]]], dtype=np.float64),
    )

    yaw_deg, pitch_deg = converter.convert((700.0, 600.0))

    assert yaw_deg == pytest.approx(-math.degrees(math.atan(0.1)) + 1.25)
    assert pitch_deg == pytest.approx(math.degrees(math.atan(0.2)) - 0.75)


def projected_board_corners(
    *,
    center_mm: tuple[float, float, float],
    board_size_mm: tuple[float, float] = (210.0, 297.0),
) -> tuple[tuple[float, float], ...]:
    width, height = board_size_mm
    object_points = np.array(
        [
            [-width / 2.0, -height / 2.0, 0.0],
            [width / 2.0, -height / 2.0, 0.0],
            [width / 2.0, height / 2.0, 0.0],
            [-width / 2.0, height / 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    cal = calibration()
    projected, _ = cv2.projectPoints(
        object_points,
        np.zeros(3, dtype=np.float64),
        np.asarray(center_mm, dtype=np.float64),
        cal.camera_matrix,
        cal.distortion,
    )
    return tuple(
        tuple(float(value) for value in point)
        for point in projected.reshape(4, 2)
    )


def test_xyz_laser_offset_uses_full_board_pose() -> None:
    center_mm = (100.0, -50.0, 2000.0)
    laser_offset_mm = (40.0, -20.0, 10.0)
    cal = calibration()
    center_px, _ = cv2.projectPoints(
        np.zeros((1, 3), dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.asarray(center_mm, dtype=np.float64),
        cal.camera_matrix,
        cal.distortion,
    )
    converter = TargetAngleConverter(
        cal,
        laser_pose_compensation_enabled=True,
        laser_offset_x_mm=laser_offset_mm[0],
        laser_offset_y_mm=laser_offset_mm[1],
        laser_offset_z_mm=laser_offset_mm[2],
        board_size_mm=(210.0, 297.0),
    )

    yaw_deg, pitch_deg = converter.convert(
        tuple(center_px.reshape(2)),
        projected_board_corners(center_mm=center_mm),
    )

    assert yaw_deg == pytest.approx(
        math.degrees(math.atan2(60.0, 1990.0)), abs=0.02
    )
    assert pitch_deg == pytest.approx(
        math.degrees(math.atan2(-30.0, 1990.0)), abs=0.02
    )


def test_invalid_pose_falls_back_to_center_only_angles() -> None:
    converter = TargetAngleConverter(
        calibration(),
        laser_pose_compensation_enabled=True,
        laser_offset_x_mm=100.0,
        board_size_mm=(210.0, 297.0),
    )

    yaw_deg, pitch_deg = converter.convert(
        (720.0, 512.0),
        ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)),
    )

    assert yaw_deg == pytest.approx(math.degrees(math.atan(0.1)))
    assert pitch_deg == pytest.approx(0.0)


def test_disabled_pose_compensation_does_not_call_pnp(monkeypatch) -> None:
    converter = TargetAngleConverter(
        calibration(),
        laser_pose_compensation_enabled=False,
        laser_offset_x_mm=100.0,
    )

    def unexpected_pnp(*args, **kwargs):
        raise AssertionError("solvePnP must not run while pose compensation is disabled")

    monkeypatch.setattr(module.cv2, "solvePnP", unexpected_pnp)

    yaw_deg, pitch_deg = converter.convert(
        (720.0, 512.0),
        projected_board_corners(center_mm=(100.0, 0.0, 2000.0)),
    )

    assert yaw_deg == pytest.approx(math.degrees(math.atan(0.1)))
    assert pitch_deg == pytest.approx(0.0)


@pytest.mark.parametrize("target_z_mm", [99.0, 10001.0])
def test_pose_depth_outside_limits_falls_back_to_center_angles(
    monkeypatch,
    target_z_mm: float,
) -> None:
    converter = TargetAngleConverter(
        calibration(),
        laser_pose_compensation_enabled=True,
        laser_offset_x_mm=100.0,
        pose_min_distance_mm=100.0,
        pose_max_distance_mm=10000.0,
    )
    monkeypatch.setattr(
        module.cv2,
        "solvePnP",
        lambda *args, **kwargs: (
            True,
            np.zeros((3, 1), dtype=np.float64),
            np.array([[0.0], [0.0], [target_z_mm]], dtype=np.float64),
        ),
    )

    yaw_deg, pitch_deg = converter.convert(
        (720.0, 512.0),
        projected_board_corners(center_mm=(0.0, 0.0, 2000.0)),
    )

    assert yaw_deg == pytest.approx(math.degrees(math.atan(0.1)))
    assert pitch_deg == pytest.approx(0.0)


def test_malformed_projected_pose_falls_back_to_center_angles(monkeypatch) -> None:
    converter = TargetAngleConverter(
        calibration(),
        laser_pose_compensation_enabled=True,
        laser_offset_x_mm=100.0,
    )
    monkeypatch.setattr(
        module.cv2,
        "solvePnP",
        lambda *args, **kwargs: (
            True,
            np.zeros((3, 1), dtype=np.float64),
            np.array([[0.0], [0.0], [2000.0]], dtype=np.float64),
        ),
    )
    corners = projected_board_corners(center_mm=(0.0, 0.0, 2000.0))
    monkeypatch.setattr(
        module.cv2,
        "projectPoints",
        lambda *args, **kwargs: (np.zeros((3, 2), dtype=np.float64), None),
    )

    yaw_deg, pitch_deg = converter.convert(
        (720.0, 512.0),
        corners,
    )

    assert yaw_deg == pytest.approx(math.degrees(math.atan(0.1)))
    assert pitch_deg == pytest.approx(0.0)


def test_excessive_pose_reprojection_error_falls_back_to_center_angles(
    monkeypatch,
) -> None:
    converter = TargetAngleConverter(
        calibration(),
        laser_pose_compensation_enabled=True,
        laser_offset_x_mm=100.0,
        pose_max_reprojection_error_px=2.0,
    )
    monkeypatch.setattr(
        module.cv2,
        "solvePnP",
        lambda *args, **kwargs: (
            True,
            np.zeros((3, 1), dtype=np.float64),
            np.array([[0.0], [0.0], [2000.0]], dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        module.cv2,
        "projectPoints",
        lambda *args, **kwargs: (
            np.full((4, 1, 2), 1000.0, dtype=np.float64),
            None,
        ),
    )

    yaw_deg, pitch_deg = converter.convert(
        (720.0, 512.0),
        ((500.0, 400.0), (700.0, 400.0), (700.0, 700.0), (500.0, 700.0)),
    )

    assert yaw_deg == pytest.approx(math.degrees(math.atan(0.1)))
    assert pitch_deg == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"board_size_mm": (210.0,)}, "board_size_mm"),
        ({"board_size_mm": (210.0, 0.0)}, "board height"),
        ({"board_size_mm": (float("nan"), 297.0)}, "board width"),
        ({"laser_offset_x_mm": float("inf")}, "laser_offset_x_mm"),
        ({"laser_yaw_bias_deg": True}, "laser_yaw_bias_deg"),
        ({"pose_min_distance_mm": 0.0}, "pose_min_distance_mm"),
        (
            {
                "pose_min_distance_mm": 1000.0,
                "pose_max_distance_mm": 1000.0,
            },
            "pose_max_distance_mm",
        ),
        ({"pose_max_reprojection_error_px": float("nan")}, "pose_max_reprojection_error_px"),
    ],
)
def test_invalid_pose_compensation_constructor_values_are_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(CalibrationError, match=message):
        TargetAngleConverter(calibration(), **kwargs)


def test_convert_uses_single_point_undistortion(monkeypatch) -> None:
    item = calibration(distortion=np.array([0.1, -0.02, 0.0, 0.0, 0.0]))
    converter = TargetAngleConverter(item)
    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def undistort_points(
        points: np.ndarray,
        matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> np.ndarray:
        calls.append((points.copy(), matrix, distortion))
        return np.array([[[0.25, -0.125]]], dtype=np.float64)

    monkeypatch.setattr(module.cv2, "undistortPoints", undistort_points)

    yaw_deg, pitch_deg = converter.convert((700.0, 400.0))

    assert len(calls) == 1
    points, matrix, distortion = calls[0]
    assert points.shape == (1, 1, 2)
    assert points.dtype == np.float64
    assert np.allclose(points[0, 0], (700.0, 400.0))
    assert matrix is item.camera_matrix
    assert distortion is item.distortion
    assert yaw_deg == pytest.approx(math.degrees(math.atan(0.25)))
    assert pitch_deg == pytest.approx(math.degrees(math.atan(-0.125)))


def test_from_file_loads_valid_calibration(tmp_path: Path) -> None:
    path = tmp_path / "camera.yaml"
    save_calibration(path, calibration())

    converter = TargetAngleConverter.from_file(
        path,
        image_size=(1280, 1024),
        max_rms_px=0.5,
        yaw_sign=-1,
        pitch_sign=1,
    )

    assert converter.calibration.image_size == (1280, 1024)
    assert converter.yaw_sign == -1
    assert converter.pitch_sign == 1


def test_from_file_rejects_runtime_image_size_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "camera.yaml"
    save_calibration(path, calibration())

    with pytest.raises(CalibrationError, match="image size.*runtime"):
        TargetAngleConverter.from_file(
            path,
            image_size=(640, 480),
            max_rms_px=0.5,
            yaw_sign=1,
            pitch_sign=1,
        )


def test_from_file_rejects_excessive_rms(tmp_path: Path) -> None:
    path = tmp_path / "camera.yaml"
    save_calibration(path, calibration(rms_px=0.7))

    with pytest.raises(CalibrationError, match="RMS"):
        TargetAngleConverter.from_file(
            path,
            image_size=(1280, 1024),
            max_rms_px=0.5,
            yaw_sign=1,
            pitch_sign=1,
        )


@pytest.mark.parametrize(
    "yaml_text",
    [
        "not: [valid",
        """schema_version: 1
image_size: [1280, 1024]
camera_matrix: [[800, 0], [0, 800]]
distortion: [0, 0, 0, 0, 0]
rms_px: 0.3
""",
        """schema_version: 1
image_size: [1280, 1024]
camera_matrix: [[800, 0, 640], [0, 800, 512], [0, 0, 1]]
distortion: [0, .nan, 0, 0, 0]
rms_px: 0.3
""",
    ],
)
def test_from_file_rejects_corrupt_matrix_or_distortion(
    tmp_path: Path,
    yaml_text: str,
) -> None:
    path = tmp_path / "camera.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(CalibrationError):
        TargetAngleConverter.from_file(
            path,
            image_size=(1280, 1024),
            max_rms_px=0.5,
            yaw_sign=1,
            pitch_sign=1,
        )


def test_from_file_rejects_missing_calibration(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="cannot load calibration"):
        TargetAngleConverter.from_file(
            tmp_path / "missing.yaml",
            image_size=(1280, 1024),
            max_rms_px=0.5,
            yaw_sign=1,
            pitch_sign=1,
        )


@pytest.mark.parametrize(
    "center_px",
    [
        None,
        (float("nan"), 1.0),
        (1.0, float("inf")),
        (-1.0, 20.0),
        (1280.0, 20.0),
        (20.0, 1024.0),
        (1.0,),
        (1.0, 2.0, 3.0),
    ],
)
def test_invalid_or_out_of_frame_center_is_rejected(center_px) -> None:
    with pytest.raises(CalibrationError, match="target center"):
        TargetAngleConverter(calibration()).convert(center_px)


def test_invalid_intrinsic_focal_length_is_rejected() -> None:
    item = calibration()
    item.camera_matrix[0, 0] = 0.0

    with pytest.raises(CalibrationError, match="focal lengths"):
        TargetAngleConverter(item)


@pytest.mark.parametrize("yaw_sign,pitch_sign", [(0, 1), (1, 2), (True, 1), (1, False)])
def test_invalid_axis_sign_is_rejected(yaw_sign, pitch_sign) -> None:
    with pytest.raises(CalibrationError, match="yaw_sign and pitch_sign"):
        TargetAngleConverter(calibration(), yaw_sign=yaw_sign, pitch_sign=pitch_sign)


def test_opencv_undistortion_failure_is_fail_closed(monkeypatch) -> None:
    converter = TargetAngleConverter(calibration())

    def fail(*args):
        raise module.cv2.error("undistortion failed")

    monkeypatch.setattr(module.cv2, "undistortPoints", fail)

    with pytest.raises(CalibrationError, match="cannot undistort target center"):
        converter.convert((640.0, 512.0))


def test_non_finite_undistorted_point_is_rejected(monkeypatch) -> None:
    converter = TargetAngleConverter(calibration())
    monkeypatch.setattr(
        module.cv2,
        "undistortPoints",
        lambda *args: np.array([[[float("nan"), 0.0]]]),
    )

    with pytest.raises(CalibrationError, match="angle is not finite"):
        converter.convert((640.0, 512.0))


def test_unavailable_converter_is_fail_closed() -> None:
    converter = UnavailableTargetAngleConverter("calibration file is unavailable")

    assert converter.available is False
    assert converter.reason == "calibration file is unavailable"
    with pytest.raises(CalibrationError, match="calibration file is unavailable"):
        converter.convert((640.0, 512.0))
