from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from ev_vision.detection.contracts import ClassicalCandidateEvaluation
from ev_vision.detection.image_normalization import NormalizedFrame
from ev_vision.detection.ring_geometry import RingGeometryResult


DEBUG_IMAGE_NAMES = (
    "normalized-gray",
    "white-mask",
    "edge-mask",
    "ring-arcs",
    "candidate-scores",
)


def _bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def render_debug_images(
    normalized: NormalizedFrame,
    *,
    ring_result: RingGeometryResult | None = None,
    candidates: Sequence[ClassicalCandidateEvaluation] = (),
    roi_xyxy: tuple[int, int, int, int] | None = None,
) -> dict[str, np.ndarray]:
    ring_image = _bgr(normalized.normalized_gray)
    if roi_xyxy is not None:
        x0, y0, x1, y1 = roi_xyxy
        cv2.rectangle(ring_image, (x0, y0), (x1, y1), (255, 180, 0), 2)
    if ring_result is not None:
        for arc in ring_result.arcs:
            center = tuple(int(round(value)) for value in arc.center_px)
            axes = tuple(max(1, int(round(value * 0.5))) for value in arc.axes_px)
            cv2.ellipse(ring_image, center, axes, arc.angle_deg, 0, 360, (0, 220, 255), 2)
        if ring_result.center_px is not None:
            center = tuple(int(round(value)) for value in ring_result.center_px)
            cv2.drawMarker(ring_image, center, (0, 255, 0), cv2.MARKER_CROSS, 18, 2)

    score_image = _bgr(normalized.gray)
    for index, item in enumerate(candidates):
        corners = np.asarray(item.corners_px, np.int32).reshape((-1, 1, 2))
        color = (0, 220, 0) if item.accepted else (0, 0, 220)
        cv2.polylines(score_image, [corners], True, color, 2, cv2.LINE_AA)
        anchor = tuple(int(round(value)) for value in item.corners_px[0])
        reason = "OK" if item.accepted else ",".join(item.rejection_reasons) or "REJECT"
        cv2.putText(
            score_image,
            f"{index}:{item.combined_score:.2f} {reason}",
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return {
        "normalized-gray": normalized.normalized_gray.copy(),
        "white-mask": normalized.white_mask.copy(),
        "edge-mask": normalized.edge_mask.copy(),
        "ring-arcs": ring_image,
        "candidate-scores": score_image,
    }
