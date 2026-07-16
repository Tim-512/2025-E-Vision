from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


class HomographyError(ValueError):
    pass


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def _polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _solve_homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    matrix = np.asarray(rows, dtype=np.float64)
    _, singular, vh = np.linalg.svd(matrix)
    if singular[-2] <= 1e-10 or singular[0] / singular[-2] > 1e12:
        raise HomographyError("degenerate quadrilateral")
    homography = vh[-1].reshape(3, 3)
    if abs(homography[2, 2]) < 1e-12:
        raise HomographyError("invalid homography scale")
    return homography / homography[2, 2]


def _transform(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    vector = homography @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    if abs(vector[2]) < 1e-12:
        raise HomographyError("point maps to infinity")
    return float(vector[0] / vector[2]), float(vector[1] / vector[2])


@dataclass(frozen=True)
class TargetGeometry:
    px_per_cm: float = 40.0
    width_cm: float = 21.0
    height_cm: float = 29.7
    _target_to_image: np.ndarray | None = None
    _image_to_target: np.ndarray | None = None

    @property
    def rectified_size_px(self) -> tuple[int, int]:
        return round(self.width_cm * self.px_per_cm), round(self.height_cm * self.px_per_cm)

    @property
    def center_px(self) -> tuple[float, float]:
        width, height = self.rectified_size_px
        return width / 2.0, height / 2.0

    @property
    def target_corners_cm(self) -> tuple[tuple[float, float], ...]:
        half_w, half_h = self.width_cm / 2.0, self.height_cm / 2.0
        return ((-half_w, half_h), (half_w, half_h), (half_w, -half_h), (-half_w, -half_h))

    @classmethod
    def from_image_corners(
        cls,
        corners: Iterable[tuple[float, float]],
        px_per_cm: float = 40.0,
        *,
        width_cm: float = 21.0,
        height_cm: float = 29.7,
    ) -> "TargetGeometry":
        image = np.asarray(tuple(corners), dtype=np.float64)
        if image.shape != (4, 2) or _polygon_area(image) < 1.0:
            raise HomographyError("degenerate quadrilateral")
        base = cls(px_per_cm=px_per_cm, width_cm=width_cm, height_cm=height_cm)
        target = np.asarray(base.target_corners_cm, dtype=np.float64)
        target_to_image = _solve_homography(target, image)
        try:
            image_to_target = np.linalg.inv(target_to_image)
        except np.linalg.LinAlgError as exc:
            raise HomographyError("singular homography") from exc
        if np.linalg.cond(target_to_image) > 1e12:
            raise HomographyError("ill-conditioned homography")
        return cls(
            px_per_cm=px_per_cm,
            width_cm=width_cm,
            height_cm=height_cm,
            _target_to_image=target_to_image,
            _image_to_target=image_to_target,
        )

    def cm_to_rectified_px(self, point_cm: tuple[float, float]) -> tuple[float, float]:
        cx, cy = self.center_px
        return cx + point_cm[0] * self.px_per_cm, cy - point_cm[1] * self.px_per_cm

    def rectified_px_to_cm(self, point_px: tuple[float, float]) -> tuple[float, float]:
        cx, cy = self.center_px
        return (point_px[0] - cx) / self.px_per_cm, (cy - point_px[1]) / self.px_per_cm

    def target_cm_to_image(self, point_cm: tuple[float, float]) -> tuple[float, float]:
        if self._target_to_image is None:
            return self.cm_to_rectified_px(point_cm)
        return _transform(point_cm, self._target_to_image)

    def image_to_target_cm(self, point_px: tuple[float, float]) -> tuple[float, float]:
        if self._image_to_target is None:
            return self.rectified_px_to_cm(point_px)
        return _transform(point_px, self._image_to_target)


def pixel_error_to_angles(point_px: tuple[float, float], intrinsics: CameraIntrinsics) -> tuple[float, float]:
    yaw = math.atan2(point_px[0] - intrinsics.cx, intrinsics.fx)
    pitch = math.atan2(intrinsics.cy - point_px[1], intrinsics.fy)
    return yaw, pitch
