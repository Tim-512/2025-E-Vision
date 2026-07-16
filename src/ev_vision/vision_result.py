from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ev_vision.config import BoardTrackingConfig


_INVALID_FAILURES = frozenset(
    {
        "STALE_FRAME",
        "AMBIGUOUS_CANDIDATES",
        "MODEL_ERROR",
        "CAMERA_ERROR",
        "EXCESSIVE_POSITION_JUMP",
    }
)


@dataclass(frozen=True)
class VisionTargetResult:
    """Transport-neutral target semantics shared with the gimbal team."""

    timestamp_ms: int
    frame_sequence: int
    target_valid: bool
    tracking_state: str
    confidence: float
    center_x_px: float | None
    center_y_px: float | None
    offset_x_px: float | None
    offset_y_px: float | None
    target_x_mm: float | None
    target_y_mm: float | None
    corners: tuple[tuple[float, float], ...]
    frame_age_ms: float
    laser_permission: bool = False

    @classmethod
    def from_hybrid(
        cls,
        hybrid: Any,
        *,
        image_size: tuple[int, int],
        now_ns: int,
        max_result_age_ms: float = BoardTrackingConfig().max_result_age_ms,
    ) -> "VisionTargetResult":
        """Map a hybrid detector result without defining a wire protocol."""

        timestamp_ns = int(hybrid.timestamp_ns)
        frame_age_ms = max(0.0, (int(now_ns) - timestamp_ns) / 1_000_000.0)

        center = getattr(hybrid, "center_px", None)
        if center is None:
            center_x_px = None
            center_y_px = None
            offset_x_px = None
            offset_y_px = None
        else:
            center_x_px = float(center[0])
            center_y_px = float(center[1])
            image_width, image_height = image_size
            offset_x_px = center_x_px - float(image_width) / 2.0
            offset_y_px = center_y_px - float(image_height) / 2.0

        raw_corners = getattr(hybrid, "corners_px", None)
        corners = (
            tuple((float(x), float(y)) for x, y in raw_corners)
            if raw_corners
            else ()
        )
        target_x_mm = getattr(hybrid, "target_x_mm", None)
        target_y_mm = getattr(hybrid, "target_y_mm", None)
        failure_reason = getattr(hybrid, "failure_reason", None)
        failure_value = getattr(failure_reason, "value", failure_reason)

        target_valid = bool(
            getattr(hybrid, "tracking_state", None) == "TRACKING"
            and getattr(hybrid, "target_valid", False)
            and center is not None
            and len(corners) == 4
            and getattr(hybrid, "homography_valid", False)
            and target_x_mm is not None
            and target_y_mm is not None
            and frame_age_ms <= float(max_result_age_ms)
            and failure_value not in _INVALID_FAILURES
        )

        return cls(
            timestamp_ms=timestamp_ns // 1_000_000,
            frame_sequence=int(hybrid.source_sequence),
            target_valid=target_valid,
            tracking_state=str(hybrid.tracking_state),
            confidence=float(hybrid.combined_score),
            center_x_px=center_x_px,
            center_y_px=center_y_px,
            offset_x_px=offset_x_px,
            offset_y_px=offset_y_px,
            target_x_mm=(float(target_x_mm) if target_x_mm is not None else None),
            target_y_mm=(float(target_y_mm) if target_y_mm is not None else None),
            corners=corners,
            frame_age_ms=frame_age_ms,
            laser_permission=False,
        )


