from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PoseSignature = tuple[float, float, float, float]


@dataclass(frozen=True)
class ChessboardFrameAnalysis:
    found: bool
    corners: np.ndarray | None
    focus_score: float
    coverage_fraction: float
    edge_margin_ok: bool
    pose_signature: PoseSignature | None
    save_allowed: bool
    reason: str


def _grayscale(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be grayscale or BGR")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("image must be grayscale or BGR")


def analyze_chessboard_frame(
    image: np.ndarray,
    *,
    pattern_size: tuple[int, int],
    min_focus_score: float = 40.0,
    min_coverage_fraction: float = 0.02,
    edge_margin_fraction: float = 0.03,
) -> ChessboardFrameAnalysis:
    gray = _grayscale(image)
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    focus_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    expected_corner_count = pattern_size[0] * pattern_size[1]
    if not found or corners is None or len(corners) != expected_corner_count:
        return ChessboardFrameAnalysis(
            found=False,
            corners=None,
            focus_score=focus_score,
            coverage_fraction=0.0,
            edge_margin_ok=False,
            pose_signature=None,
            save_allowed=False,
            reason="corners_not_found",
        )

    refined = cv2.cornerSubPix(
        gray,
        np.asarray(corners, dtype=np.float32),
        (11, 11),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        ),
    ).reshape(-1, 2)
    x_min, y_min = refined.min(axis=0)
    x_max, y_max = refined.max(axis=0)
    height, width = gray.shape
    coverage_fraction = float(
        max(0.0, float(x_max - x_min))
        * max(0.0, float(y_max - y_min))
        / float(width * height)
    )
    margin_x = width * edge_margin_fraction
    margin_y = height * edge_margin_fraction
    edge_margin_ok = bool(
        x_min >= margin_x
        and x_max < width - margin_x
        and y_min >= margin_y
        and y_max < height - margin_y
    )
    top_left = refined[0]
    top_right = refined[pattern_size[0] - 1]
    top_row_angle = math.degrees(
        math.atan2(
            float(top_right[1] - top_left[1]),
            float(top_right[0] - top_left[0]),
        )
    )
    pose_signature: PoseSignature = (
        float((x_min + x_max) / (2.0 * width)),
        float((y_min + y_max) / (2.0 * height)),
        coverage_fraction,
        top_row_angle,
    )

    if focus_score < min_focus_score:
        reason = "focus_too_low"
    elif coverage_fraction < min_coverage_fraction:
        reason = "coverage_too_small"
    elif not edge_margin_ok:
        reason = "corners_too_close_to_edge"
    else:
        reason = "ready"

    return ChessboardFrameAnalysis(
        found=True,
        corners=refined,
        focus_score=focus_score,
        coverage_fraction=coverage_fraction,
        edge_margin_ok=edge_margin_ok,
        pose_signature=pose_signature,
        save_allowed=reason == "ready",
        reason=reason,
    )


def poses_are_similar(
    first: PoseSignature,
    second: PoseSignature,
    *,
    center_tolerance: float = 0.08,
    area_ratio_tolerance: float = 0.25,
    angle_tolerance_deg: float = 12.0,
) -> bool:
    first_area = max(first[2], 1e-9)
    area_ratio_delta = abs(second[2] / first_area - 1.0)
    angle_delta = abs((second[3] - first[3] + 180.0) % 360.0 - 180.0)
    return bool(
        math.hypot(second[0] - first[0], second[1] - first[1])
        <= center_tolerance
        and area_ratio_delta <= area_ratio_tolerance
        and angle_delta <= angle_tolerance_deg
    )


def calibration_image_path(output: Path, index: int) -> Path:
    return Path(output) / f"calibration-{index:03d}.png"


def save_original_frame(output: Path, image: np.ndarray, *, index: int) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = calibration_image_path(output, index)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write calibration image: {path}")
    return path


def remove_last_saved(session_paths: list[Path]) -> Path | None:
    if not session_paths:
        return None
    path = session_paths.pop()
    path.unlink(missing_ok=False)
    return path


def _infer_pattern_size(corner_count: int) -> tuple[int, int] | None:
    factor_pairs = [
        (columns, corner_count // columns)
        for columns in range(2, int(math.sqrt(corner_count)) + 1)
        if corner_count % columns == 0
    ]
    if not factor_pairs:
        return None
    columns, rows = min(
        factor_pairs,
        key=lambda pair: (abs(pair[0] - pair[1]), -max(pair)),
    )
    return (max(columns, rows), min(columns, rows))


def render_capture_overlay(
    image: np.ndarray,
    analysis: ChessboardFrameAnalysis,
    saved_count: int,
    duplicate_warning: bool,
) -> np.ndarray:
    rendered = image.copy()
    if analysis.corners is not None:
        pattern_size = _infer_pattern_size(len(analysis.corners))
        if pattern_size is not None:
            cv2.drawChessboardCorners(
                rendered,
                pattern_size,
                analysis.corners.reshape(-1, 1, 2),
                analysis.found,
            )

    color = (0, 220, 0) if analysis.save_allowed else (0, 0, 255)
    lines = [
        (
            f"saved={saved_count} focus={analysis.focus_score:.1f} "
            f"coverage={analysis.coverage_fraction:.3f}"
        ),
        f"status={analysis.reason}",
        "SPACE save | R remove last session image | Q/ESC quit",
    ]
    if duplicate_warning:
        lines.append("warning: pose is similar to an image already saved")
    for row, line in enumerate(lines, start=1):
        cv2.putText(
            rendered,
            line,
            (16, 28 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return rendered
