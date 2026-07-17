from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticBoard:
    image: np.ndarray
    corners: np.ndarray


@dataclass(frozen=True)
class SyntheticRoiBoard:
    image: np.ndarray
    corners: np.ndarray
    model_box: tuple[float, float, float, float]


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


def _roi_background(image_size: tuple[int, int] = (480, 640)) -> np.ndarray:
    height, width = image_size
    gradient = np.linspace(178.0, 204.0, width, dtype=np.float32)
    gray = np.repeat(gradient[None, :], height, axis=0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _draw_roi_frame(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    frame_value: int = 18,
    center_value: int = 235,
    frame_fraction: float = 0.12,
) -> None:
    cv2.fillConvexPoly(
        image,
        np.rint(corners).astype(np.int32),
        (frame_value,) * 3,
        lineType=cv2.LINE_AA,
    )
    center = corners.mean(axis=0)
    inner = center + (1.0 - frame_fraction) * (corners - center)
    cv2.fillConvexPoly(
        image,
        np.rint(inner).astype(np.int32),
        (center_value,) * 3,
        lineType=cv2.LINE_AA,
    )


def synthetic_cluttered_board() -> SyntheticRoiBoard:
    image = _roi_background()
    corners = np.asarray(
        [[220.0, 70.0], [425.0, 92.0], [445.0, 406.0], [198.0, 386.0]],
        np.float32,
    )
    _draw_roi_frame(image, corners)

    # These distractors sit outside the model-guided ROI.  The ROI-only
    # detector must not let a larger unrelated rectangle win.
    cv2.rectangle(image, (18, 25), (130, 180), (10, 10, 10), 8)
    cv2.circle(image, (565, 105), 55, (20, 20, 20), 7)
    cv2.line(image, (20, 445), (610, 455), (35, 35, 35), 5)
    return SyntheticRoiBoard(
        image=image,
        corners=corners,
        model_box=(188.0, 55.0, 455.0, 420.0),
    )


def geometry_failure_fixture(
    fixture_name: str,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    image = _roi_background()

    if fixture_name == "non_convex":
        points = np.asarray(
            [[205, 95], [430, 105], [315, 210], [445, 390], [190, 375]],
            np.int32,
        )
        cv2.polylines(image, [points], True, (12, 12, 12), 12, cv2.LINE_AA)
        return image, (175.0, 65.0, 460.0, 420.0)

    if fixture_name == "truncated":
        corners = np.asarray(
            [[0.0, 55.0], [210.0, 75.0], [220.0, 415.0], [0.0, 395.0]],
            np.float32,
        )
        _draw_roi_frame(image, corners)
        return image, (0.0, 45.0, 225.0, 430.0)

    if fixture_name == "undersized":
        corners = np.asarray(
            [[295.0, 210.0], [320.0, 211.0], [322.0, 247.0], [294.0, 246.0]],
            np.float32,
        )
        _draw_roi_frame(image, corners, frame_fraction=0.25)
        return image, (280.0, 195.0, 338.0, 263.0)

    if fixture_name == "weak_edges":
        corners = np.asarray(
            [[220.0, 70.0], [425.0, 92.0], [445.0, 406.0], [198.0, 386.0]],
            np.float32,
        )
        # Only the lower-threshold Canny pass sees this weak outer contour.
        _draw_roi_frame(
            image,
            corners,
            frame_value=170,
            center_value=220,
            frame_fraction=0.10,
        )
        return image, (188.0, 55.0, 455.0, 420.0)

    if fixture_name == "no_internal_frame":
        corners = np.asarray(
            [[220.0, 70.0], [425.0, 92.0], [445.0, 406.0], [198.0, 386.0]],
            np.float32,
        )
        cv2.fillConvexPoly(
            image,
            np.rint(corners).astype(np.int32),
            (18, 18, 18),
            lineType=cv2.LINE_AA,
        )
        return image, (188.0, 55.0, 455.0, 420.0)

    raise ValueError(f"unknown geometry failure fixture: {fixture_name}")


@dataclass(frozen=True)
class SyntheticRingTarget:
    image: np.ndarray
    center_px: tuple[float, float]
    corners_px: tuple[tuple[float, float], ...]
    radii_px: tuple[float, ...]
    partially_outside: bool


def render_ring_target(
    *,
    image_size: tuple[int, int] = (720, 960),
    board_center: tuple[float, float] | None = None,
    perspective: float = 0.0,
    shadow_strength: float = 0.0,
    blur_sigma: float = 0.0,
    ring_gray: int = 92,
    ring_thickness_px: int = 3,
    gradient_strength: float = 0.0,
    noise_sigma: float = 0.0,
    distractors: bool = False,
    seed: int = 0,
) -> SyntheticRingTarget:
    height, width = image_size
    center = board_center or (width / 2.0, height / 2.0)
    pixels_per_mm = min(width / 420.0, height / 420.0)
    canonical_width = max(2, int(round(210.0 * pixels_per_mm)))
    canonical_height = max(2, int(round(297.0 * pixels_per_mm)))
    canonical = np.full((canonical_height, canonical_width, 3), 242, np.uint8)
    tape_px = max(2, int(round(18.0 * pixels_per_mm)))
    cv2.rectangle(
        canonical,
        (0, 0),
        (canonical_width - 1, canonical_height - 1),
        (18, 18, 18),
        tape_px,
        cv2.LINE_AA,
    )
    canonical_center = (canonical_width // 2, canonical_height // 2)
    radii = tuple(
        float(int(round(radius_mm * pixels_per_mm)))
        for radius_mm in (20.0, 40.0, 60.0, 80.0, 100.0)
    )
    for radius_px in radii:
        cv2.circle(
            canonical,
            canonical_center,
            int(radius_px),
            (ring_gray, ring_gray, ring_gray),
            ring_thickness_px,
            cv2.LINE_AA,
        )

    half_w = canonical_width / 2.0
    half_h = canonical_height / 2.0
    skew_x = perspective * canonical_width
    skew_y = perspective * canonical_height
    destination = np.asarray(
        [
            [center[0] - half_w + skew_x, center[1] - half_h + skew_y],
            [center[0] + half_w - skew_x, center[1] - half_h],
            [center[0] + half_w, center[1] + half_h - skew_y],
            [center[0] - half_w, center[1] + half_h],
        ],
        np.float32,
    )
    source = np.asarray(
        [
            [0.0, 0.0],
            [canonical_width - 1.0, 0.0],
            [canonical_width - 1.0, canonical_height - 1.0],
            [0.0, canonical_height - 1.0],
        ],
        np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    background = np.full((height, width, 3), 35, np.uint8)
    warped = cv2.warpPerspective(
        canonical, transform, (width, height), borderValue=(0, 0, 0)
    )
    alpha = cv2.warpPerspective(
        np.full((canonical_height, canonical_width), 255, np.uint8),
        transform,
        (width, height),
        borderValue=0,
    )
    image = np.where(alpha[..., None] > 0, warped, background)

    if gradient_strength:
        gradient = np.linspace(
            1.0 - gradient_strength, 1.0 + gradient_strength, width, dtype=np.float32
        )
        image = np.clip(image.astype(np.float32) * gradient[None, :, None], 0, 255).astype(np.uint8)
    if shadow_strength:
        shadow = np.ones((height, width), np.float32)
        polygon = np.asarray(
            [[0, 0], [int(width * 0.58), 0], [int(width * 0.42), height], [0, height]],
            np.int32,
        )
        cv2.fillConvexPoly(shadow, polygon, max(0.0, 1.0 - shadow_strength))
        image = np.clip(image.astype(np.float32) * shadow[..., None], 0, 255).astype(np.uint8)
    if distractors:
        cv2.rectangle(image, (25, 25), (135, 125), (215, 215, 215), -1)
        cv2.circle(image, (width - 80, 80), 35, (100, 100, 100), 4, cv2.LINE_AA)
    if noise_sigma:
        noise = np.random.default_rng(seed).normal(0.0, noise_sigma, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if blur_sigma:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)

    projected_center = cv2.perspectiveTransform(
        np.asarray([[canonical_center]], np.float32), transform
    )[0, 0]
    outside = bool(
        np.any(destination[:, 0] < 0)
        or np.any(destination[:, 0] >= width)
        or np.any(destination[:, 1] < 0)
        or np.any(destination[:, 1] >= height)
    )
    return SyntheticRingTarget(
        image=image,
        center_px=(float(projected_center[0]), float(projected_center[1])),
        corners_px=tuple((float(x), float(y)) for x, y in destination),
        radii_px=radii,
        partially_outside=outside,
    )
