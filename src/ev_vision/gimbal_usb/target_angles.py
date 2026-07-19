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
    ) -> tuple[float, float]:
        del center_px
        raise CalibrationError(self.reason)


@dataclass(frozen=True)
class TargetAngleConverter:
    calibration: Calibration
    yaw_sign: int = 1
    pitch_sign: int = 1

    def __post_init__(self) -> None:
        self.calibration.validate()
        focal_x = float(self.calibration.camera_matrix[0, 0])
        focal_y = float(self.calibration.camera_matrix[1, 1])
        if focal_x <= 0.0 or focal_y <= 0.0:
            raise CalibrationError(
                "camera_matrix focal lengths must be positive"
            )
        if any(
            isinstance(value, bool) or value not in (-1, 1)
            for value in (self.yaw_sign, self.pitch_sign)
        ):
            raise CalibrationError("yaw_sign and pitch_sign must be -1 or 1")

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
    ) -> "TargetAngleConverter":
        calibration = load_calibration(path)
        calibration.validate(max_rms_px=max_rms_px)
        runtime_size = tuple(image_size)
        if calibration.image_size != runtime_size:
            raise CalibrationError(
                f"calibration image size {calibration.image_size} "
                f"does not match runtime {runtime_size}"
            )
        return cls(calibration, yaw_sign=yaw_sign, pitch_sign=pitch_sign)

    def convert(
        self,
        center_px: tuple[float, float] | None,
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

        try:
            normalized = cv2.undistortPoints(
                np.array([[[x, y]]], dtype=np.float64),
                self.calibration.camera_matrix,
                self.calibration.distortion,
            )
        except cv2.error as exc:
            raise CalibrationError(
                f"cannot undistort target center: {exc}"
            ) from exc
        try:
            x_normalized, y_normalized = np.asarray(
                normalized, dtype=np.float64
            ).reshape(2)
        except (TypeError, ValueError) as exc:
            raise CalibrationError(
                "calibrated target point has an invalid shape"
            ) from exc

        yaw_deg = self.yaw_sign * math.degrees(math.atan(float(x_normalized)))
        pitch_deg = self.pitch_sign * math.degrees(math.atan(float(y_normalized)))
        if not math.isfinite(yaw_deg) or not math.isfinite(pitch_deg):
            raise CalibrationError("calibrated target angle is not finite")
        return yaw_deg, pitch_deg
