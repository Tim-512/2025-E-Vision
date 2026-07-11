from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticBoard:
    image: np.ndarray
    corners: np.ndarray


def render_board(
    corners: np.ndarray | None = None,
    *,
    image_size: tuple[int, int] = (1024, 1280),
    blur_sigma: float = 0.0,
    gradient_strength: float = 0.0,
    distractors: bool = False,
) -> SyntheticBoard:
    height, width = image_size
    if corners is None:
        corners = np.array([[310.0, 170.0], [970.0, 205.0], [930.0, 855.0], [275.0, 810.0]], np.float32)
    corners = np.asarray(corners, dtype=np.float32)

    gray = np.full((height, width), 225, np.uint8)
    cv2.fillConvexPoly(gray, np.rint(corners).astype(np.int32), 12, lineType=cv2.LINE_AA)

    center = corners.mean(axis=0)
    inner = center + 0.94 * (corners - center)
    cv2.fillConvexPoly(gray, np.rint(inner).astype(np.int32), 238, lineType=cv2.LINE_AA)

    if distractors:
        cv2.line(gray, (35, 80), (1180, 115), 20, 20, cv2.LINE_AA)
        cv2.line(gray, (80, 970), (1220, 920), 12, 30, cv2.LINE_AA)
        cv2.rectangle(gray, (45, 270), (205, 470), 25, 10)

    if gradient_strength:
        gradient = np.linspace(1.0 - gradient_strength, 1.0 + gradient_strength, width, dtype=np.float32)
        gray = np.clip(gray.astype(np.float32) * gradient[None, :], 0, 255).astype(np.uint8)

    if blur_sigma:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)

    return SyntheticBoard(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), corners)
