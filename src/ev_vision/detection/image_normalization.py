from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ev_vision.config import ImageNormalizationConfig


@dataclass(frozen=True)
class NormalizedFrame:
    gray: np.ndarray
    denoised_gray: np.ndarray
    normalized_gray: np.ndarray
    white_mask: np.ndarray
    gradient: np.ndarray
    edge_mask: np.ndarray
    saturated_mask: np.ndarray
    ring_edge_mask: np.ndarray


def _disk_kernel(radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return np.ones((1, 1), np.uint8)
    size = radius_px * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def normalize_frame(
    image: np.ndarray,
    config: ImageNormalizationConfig,
    *,
    mask_radius_px: int = 12,
) -> NormalizedFrame:
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("image must be a non-empty gray or BGR array")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("image must be a non-empty gray or BGR array")

    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image.copy()
    )
    denoised = cv2.GaussianBlur(
        gray, (config.gaussian_kernel,) * 2, 0
    )
    local = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(config.clahe_grid_size,) * 2,
    ).apply(denoised)
    background = cv2.GaussianBlur(
        local, (config.illumination_kernel,) * 2, 0
    )
    normalized = cv2.addWeighted(local, 1.0, background, -1.0, 128.0)

    threshold = max(
        0.0,
        float(np.percentile(normalized, config.white_percentile))
        - config.white_local_offset,
    )
    white = cv2.threshold(
        normalized, threshold, 255, cv2.THRESH_BINARY
    )[1]
    white = cv2.morphologyEx(
        white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
    )

    gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    edges = cv2.Canny(normalized, config.canny_low, config.canny_high)
    saturated = cv2.threshold(
        gray, config.saturation_threshold, 255, cv2.THRESH_BINARY
    )[1]
    saturated = cv2.dilate(saturated, _disk_kernel(mask_radius_px))
    ring_edges = cv2.bitwise_and(edges, cv2.bitwise_not(saturated))

    return NormalizedFrame(
        gray=gray,
        denoised_gray=denoised,
        normalized_gray=normalized,
        white_mask=white,
        gradient=gradient,
        edge_mask=edges,
        saturated_mask=saturated,
        ring_edge_mask=ring_edges,
    )
