from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from ev_vision.calibration import Calibration, CalibrationError, load_calibration


class AngleConverter(Protocol):
    @property
    def available(self) -> bool:
        ...

    def convert(
        self,
        center_px: tuple[float, float] | None,
        corners_px: tuple[tuple[float, float], ...] = (),
    ) -> tuple[float, float]:
        ...


@dataclass(frozen=True)
class UnavailableTargetAngleConverter:
    reason: str

    @property
    def available(self) -> bool:
        return False

    def convert(
        self,
        center_px: tuple[float, float] | None,
        corners_px: tuple[tuple[float, float], ...] = (),
    ) -> tuple[float, float]:
        del center_px, corners_px
        raise CalibrationError(self.reason)


@dataclass(frozen=True)
class TargetAngleConverter:
    calibration: Calibration
    yaw_sign: int = 1
    pitch_sign: int = 1
    laser_pose_compensation_enabled: bool = False
    laser_offset_x_mm: float = 0.0
    laser_offset_y_mm: float = 0.0
    laser_offset_z_mm: float = 0.0
    laser_yaw_bias_deg: float = 0.0
    laser_pitch_bias_deg: float = 0.0
    board_size_mm: tuple[float, float] = (210.0, 297.0)
    pose_min_distance_mm: float = 100.0
    pose_max_distance_mm: float = 10000.0
    pose_max_reprojection_error_px: float = 5.0

    def __post_init__(self) -> None:
        self.calibration.validate()
        focal_x = float(self.calibration.camera_matrix[0, 0])
        focal_y = float(self.calibration.camera_matrix[1, 1])
        if focal_x <= 0.0 or focal_y <= 0.0:
            raise CalibrationError("camera_matrix focal lengths must be positive")
        if any(
            isinstance(value, bool) or value not in (-1, 1)
            for value in (self.yaw_sign, self.pitch_sign)
        ):
            raise CalibrationError("yaw_sign and pitch_sign must be -1 or 1")
        if not isinstance(self.laser_pose_compensation_enabled, bool):
            raise CalibrationError("laser_pose_compensation_enabled must be boolean")

        finite_values = {
            "laser_offset_x_mm": self.laser_offset_x_mm,
            "laser_offset_y_mm": self.laser_offset_y_mm,
            "laser_offset_z_mm": self.laser_offset_z_mm,
            "laser_yaw_bias_deg": self.laser_yaw_bias_deg,
            "laser_pitch_bias_deg": self.laser_pitch_bias_deg,
        }
        for name, value in finite_values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise CalibrationError(f"{name} must be finite")

        try:
            board_width_mm, board_height_mm = self.board_size_mm
        except (TypeError, ValueError) as exc:
            raise CalibrationError("board_size_mm must contain width and height") from exc
        positive_values = {
            "board width": board_width_mm,
            "board height": board_height_mm,
            "pose_min_distance_mm": self.pose_min_distance_mm,
            "pose_max_distance_mm": self.pose_max_distance_mm,
            "pose_max_reprojection_error_px": self.pose_max_reprojection_error_px,
        }
        for name, value in positive_values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise CalibrationError(f"{name} must be positive and finite")
        if float(self.pose_max_distance_mm) <= float(self.pose_min_distance_mm):
            raise CalibrationError(
                "pose_max_distance_mm must exceed pose_min_distance_mm"
            )

    @property
    def available(self) -> bool:
        return True

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        image_size: tuple[int, int],
        max_rms_px: float,
        yaw_sign: int,
        pitch_sign: int,
        laser_pose_compensation_enabled: bool = False,
        laser_offset_x_mm: float = 0.0,
        laser_offset_y_mm: float = 0.0,
        laser_offset_z_mm: float = 0.0,
        laser_yaw_bias_deg: float = 0.0,
        laser_pitch_bias_deg: float = 0.0,
        board_size_mm: tuple[float, float] = (210.0, 297.0),
        pose_min_distance_mm: float = 100.0,
        pose_max_distance_mm: float = 10000.0,
        pose_max_reprojection_error_px: float = 5.0,
    ) -> "TargetAngleConverter":
        calibration = load_calibration(path)
        calibration.validate(max_rms_px=max_rms_px)
        runtime_size = tuple(image_size)
        if calibration.image_size != runtime_size:
            raise CalibrationError(
                f"calibration image size {calibration.image_size} "
                f"does not match runtime {runtime_size}"
            )
        return cls(
            calibration,
            yaw_sign=yaw_sign,
            pitch_sign=pitch_sign,
            laser_pose_compensation_enabled=laser_pose_compensation_enabled,
            laser_offset_x_mm=laser_offset_x_mm,
            laser_offset_y_mm=laser_offset_y_mm,
            laser_offset_z_mm=laser_offset_z_mm,
            laser_yaw_bias_deg=laser_yaw_bias_deg,
            laser_pitch_bias_deg=laser_pitch_bias_deg,
            board_size_mm=board_size_mm,
            pose_min_distance_mm=pose_min_distance_mm,
            pose_max_distance_mm=pose_max_distance_mm,
            pose_max_reprojection_error_px=pose_max_reprojection_error_px,
        )

    def _validate_center(
        self, center_px: tuple[float, float] | None
    ) -> tuple[float, float]:
        if center_px is None:
            raise CalibrationError("target center is unavailable")
        try:
            if len(center_px) != 2:
                raise CalibrationError("target center must contain x and y")
            x = float(center_px[0])
            y = float(center_px[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise CalibrationError("target center is invalid") from exc

        width, height = self.calibration.image_size
        if (
            not math.isfinite(x)
            or not math.isfinite(y)
            or not (0.0 <= x < width)
            or not (0.0 <= y < height)
        ):
            raise CalibrationError(
                "target center is invalid or outside calibrated image"
            )
        return x, y

    def _center_angles(self, center_px: tuple[float, float]) -> tuple[float, float]:
        x, y = center_px
        try:
            normalized = cv2.undistortPoints(
                np.array([[[x, y]]], dtype=np.float64),
                self.calibration.camera_matrix,
                self.calibration.distortion,
            )
        except cv2.error as exc:
            raise CalibrationError(f"cannot undistort target center: {exc}") from exc
        try:
            x_normalized, y_normalized = np.asarray(
                normalized, dtype=np.float64
            ).reshape(2)
        except (TypeError, ValueError) as exc:
            raise CalibrationError(
                "calibrated target point has an invalid shape"
            ) from exc
        yaw = math.degrees(math.atan(float(x_normalized)))
        pitch = math.degrees(math.atan(float(y_normalized)))
        if not math.isfinite(yaw) or not math.isfinite(pitch):
            raise CalibrationError("calibrated target angle is not finite")
        return yaw, pitch

    def _board_object_points(self) -> np.ndarray:
        width, height = (float(value) for value in self.board_size_mm)
        return np.array(
            [
                [-width / 2.0, -height / 2.0, 0.0],
                [width / 2.0, -height / 2.0, 0.0],
                [width / 2.0, height / 2.0, 0.0],
                [-width / 2.0, height / 2.0, 0.0],
            ],
            dtype=np.float64,
        )

    def _pose_compensated_angles(
        self,
        corners_px: tuple[tuple[float, float], ...],
    ) -> tuple[float, float] | None:
        if not self.laser_pose_compensation_enabled:
            return None
        try:
            image_points = np.asarray(corners_px, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if image_points.shape != (4, 2) or not np.isfinite(image_points).all():
            return None

        object_points = self._board_object_points()
        try:
            solved, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.calibration.camera_matrix,
                self.calibration.distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None
        if not solved:
            return None
        try:
            rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
            target = np.asarray(tvec, dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(rvec).all() or not np.isfinite(target).all():
            return None

        distance_z = float(target[2])
        if not (
            float(self.pose_min_distance_mm)
            <= distance_z
            <= float(self.pose_max_distance_mm)
        ):
            return None

        try:
            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                target,
                self.calibration.camera_matrix,
                self.calibration.distortion,
            )
        except cv2.error:
            return None
        try:
            projected_points = np.asarray(projected, dtype=np.float64).reshape(4, 2)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(projected_points).all():
            return None
        mean_error = float(
            np.mean(np.linalg.norm(projected_points - image_points, axis=1))
        )
        if (
            not math.isfinite(mean_error)
            or mean_error > float(self.pose_max_reprojection_error_px)
        ):
            return None

        laser_origin = np.array(
            [
                float(self.laser_offset_x_mm),
                float(self.laser_offset_y_mm),
                float(self.laser_offset_z_mm),
            ],
            dtype=np.float64,
        )
        aim = target - laser_origin
        if not np.isfinite(aim).all() or float(aim[2]) <= 0.0:
            return None
        yaw = math.degrees(math.atan2(float(aim[0]), float(aim[2])))
        pitch = math.degrees(math.atan2(float(aim[1]), float(aim[2])))
        if not math.isfinite(yaw) or not math.isfinite(pitch):
            return None
        return yaw, pitch

    def convert(
        self,
        center_px: tuple[float, float] | None,
        corners_px: tuple[tuple[float, float], ...] = (),
    ) -> tuple[float, float]:
        center = self._validate_center(center_px)
        camera_angles = self._pose_compensated_angles(corners_px)
        if camera_angles is None:
            camera_angles = self._center_angles(center)

        yaw_deg = self.yaw_sign * camera_angles[0] + float(self.laser_yaw_bias_deg)
        pitch_deg = (
            self.pitch_sign * camera_angles[1] + float(self.laser_pitch_bias_deg)
        )
        if not math.isfinite(yaw_deg) or not math.isfinite(pitch_deg):
            raise CalibrationError("calibrated target angle is not finite")
        return yaw_deg, pitch_deg
