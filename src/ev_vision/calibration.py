from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


class CalibrationError(ValueError):
    pass


@dataclass
class Calibration:
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rms_px: float
    _map_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        self.distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        self.validate()

    def validate(self, *, max_rms_px: float | None = None) -> None:
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise CalibrationError("image size must be positive")
        if self.camera_matrix.shape != (3, 3) or not np.isfinite(self.camera_matrix).all():
            raise CalibrationError("camera_matrix must be a finite 3x3 matrix")
        if self.distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(self.distortion).all():
            raise CalibrationError("distortion must contain a supported finite OpenCV coefficient vector")
        if not np.isfinite(self.rms_px) or self.rms_px < 0:
            raise CalibrationError("RMS must be finite and non-negative")
        if max_rms_px is not None and self.rms_px > max_rms_px:
            raise CalibrationError(f"calibration RMS {self.rms_px:.3f}px exceeds {max_rms_px:.3f}px")

    @staticmethod
    def chessboard_object_points(pattern_size: tuple[int, int], square_size_mm: float) -> np.ndarray:
        columns, rows = pattern_size
        if columns <= 1 or rows <= 1 or square_size_mm <= 0:
            raise CalibrationError("invalid chessboard geometry")
        points = np.zeros((columns * rows, 3), np.float32)
        points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * float(square_size_mm)
        return points

    def maps(self, image_size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        size = image_size or self.image_size
        if tuple(size) != tuple(self.image_size):
            raise CalibrationError(f"image size {tuple(size)} does not match calibration {self.image_size}")
        if size not in self._map_cache:
            self._map_cache[size] = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.distortion,
                None,
                self.camera_matrix,
                size,
                cv2.CV_32FC1,
            )
        return self._map_cache[size]

    def undistort(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if (width, height) != self.image_size:
            raise CalibrationError(f"image size {(width, height)} does not match calibration {self.image_size}")
        map_x, map_y = self.maps()
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)


def _mapping(calibration: Calibration) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "image_size": [int(calibration.image_size[0]), int(calibration.image_size[1])],
        "camera_matrix": calibration.camera_matrix.tolist(),
        "distortion": calibration.distortion.tolist(),
        "rms_px": float(calibration.rms_px),
    }


def save_calibration(path: str | Path, calibration: Calibration) -> None:
    calibration.validate()
    Path(path).write_text(yaml.safe_dump(_mapping(calibration), sort_keys=False), encoding="utf-8")


def load_calibration(path: str | Path) -> Calibration:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CalibrationError(f"cannot load calibration: {exc}") from exc
    required = {"schema_version", "image_size", "camera_matrix", "distortion", "rms_px"}
    if not isinstance(raw, dict) or set(raw) != required or raw.get("schema_version") != 1:
        raise CalibrationError("invalid calibration schema")
    try:
        image_size = tuple(int(value) for value in raw["image_size"])
        if len(image_size) != 2:
            raise ValueError
        return Calibration(image_size, raw["camera_matrix"], raw["distortion"], float(raw["rms_px"]))
    except (TypeError, ValueError) as exc:
        raise CalibrationError("invalid calibration values") from exc


@dataclass(frozen=True)
class ChessboardCalibrationResult:
    calibration: Calibration
    input_images: int
    usable_poses: int
    rejected_images: int


def _has_complete_corner_set(corners: np.ndarray | None, expected_count: int) -> bool:
    return corners is not None and np.asarray(corners).shape == (expected_count, 1, 2)


def solve_chessboard_with_report(
    images: list[np.ndarray],
    *,
    pattern_size: tuple[int, int],
    square_size_mm: float,
) -> ChessboardCalibrationResult:
    if not images:
        raise CalibrationError("no calibration images")
    sizes = []
    for image in images:
        if image.ndim not in (2, 3):
            raise CalibrationError("calibration images must be gray or BGR arrays")
        sizes.append((int(image.shape[1]), int(image.shape[0])))
    if len(set(sizes)) != 1:
        raise CalibrationError("all calibration images must have the same size")
    image_size = sizes[0]
    object_template = Calibration.chessboard_object_points(pattern_size, square_size_mm)
    expected_corners = pattern_size[0] * pattern_size[1]
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found or not _has_complete_corner_set(corners, expected_corners):
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if not _has_complete_corner_set(refined, expected_corners):
            continue
        object_points.append(object_template.copy())
        image_points.append(refined)
    usable = len(image_points)
    if usable < 10:
        raise CalibrationError(f"need at least 10 usable chessboard poses, got {usable}")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    value = Calibration(image_size, matrix, distortion, float(rms))
    return ChessboardCalibrationResult(value, len(images), usable, len(images) - usable)


def solve_chessboard(
    images: list[np.ndarray],
    *,
    pattern_size: tuple[int, int],
    square_size_mm: float,
) -> Calibration:
    return solve_chessboard_with_report(
        images,
        pattern_size=pattern_size,
        square_size_mm=square_size_mm,
    ).calibration
