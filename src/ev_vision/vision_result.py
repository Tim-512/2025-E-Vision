from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from ev_vision.config import BoardTrackingConfig


_INVALID_FAILURES = frozenset(
    {
        "STALE_FRAME",
        "AMBIGUOUS_CANDIDATES",
        "MODEL_ERROR",
        "CAMERA_ERROR",
        "EXCESSIVE_POSITION_JUMP",
        "EXCESSIVE_SCALE_JUMP",
        "EXCESSIVE_VELOCITY",
        "PREDICTION_EXPIRED",
    }
)


def _enum_name(value: object, default: str) -> str:
    raw = getattr(value, "value", value)
    return default if raw is None else str(raw).upper()


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
    observation_source: str = "NONE"
    homography_valid: bool = False
    predicted_frames: int = 0
    near_image_edge: bool = False
    partially_outside: bool = False
    failure_reason: str | None = None
    laser_permission: bool = False  # Deprecated V1-only compatibility.

    @classmethod
    def from_detection(
        cls,
        detection: Any,
        *,
        image_size: tuple[int, int],
        now_ns: int,
        max_result_age_ms: float = BoardTrackingConfig().max_result_age_ms,
        predict_max_frames: int = 3,
        predict_max_ms: float = 150.0,
    ) -> "VisionTargetResult":
        """Map a detector result into protocol-neutral tracking semantics."""

        timestamp_ns = int(detection.timestamp_ns)
        frame_age_ms = max(0.0, (int(now_ns) - timestamp_ns) / 1_000_000.0)
        state = _enum_name(getattr(detection, "tracking_state", None), "SEARCHING")
        source = _enum_name(
            getattr(detection, "observation_source", None), "FULL_BOARD"
        )

        center = getattr(detection, "center_px", None)
        if center is None:
            center_x_px = center_y_px = offset_x_px = offset_y_px = None
        else:
            center_x_px, center_y_px = float(center[0]), float(center[1])
            width, height = image_size
            offset_x_px = center_x_px - float(width) / 2.0
            offset_y_px = center_y_px - float(height) / 2.0

        raw_corners = getattr(detection, "corners_px", None)
        corners = (
            tuple((float(x), float(y)) for x, y in raw_corners)
            if raw_corners
            else ()
        )
        homography_valid = bool(getattr(detection, "homography_valid", False))
        target_x_mm = getattr(detection, "target_x_mm", None)
        target_y_mm = getattr(detection, "target_y_mm", None)
        predicted_frames = int(getattr(detection, "predicted_frames", 0))
        failure = getattr(detection, "failure_reason", None)
        failure_value = _enum_name(failure, "") if failure is not None else None

        fresh = frame_age_ms <= float(max_result_age_ms)
        full_valid = (
            source == "FULL_BOARD"
            and state == "TRACKING"
            and center is not None
            and len(corners) == 4
            and homography_valid
            and target_x_mm is not None
            and target_y_mm is not None
        )
        partial_valid = (
            source
            in {"CONCENTRIC_ARCS", "SINGLE_ARC", "WHITE_REGION", "FUSED_PARTIAL"}
            and state == "TRACKING"
            and center is not None
        )
        predicted_valid = (
            source == "PREDICTED"
            and state == "PREDICTING"
            and center is not None
            and predicted_frames <= int(predict_max_frames)
            and frame_age_ms <= float(predict_max_ms)
        )
        target_valid = bool(
            getattr(detection, "target_valid", False)
            and fresh
            and failure_value not in _INVALID_FAILURES
            and (full_valid or partial_valid or predicted_valid)
        )

        confidence = getattr(detection, "confidence", None)
        if confidence is None:
            confidence = getattr(detection, "combined_score", 0.0)

        return cls(
            timestamp_ms=timestamp_ns // 1_000_000,
            frame_sequence=int(detection.source_sequence),
            target_valid=target_valid,
            tracking_state=state,
            confidence=float(confidence),
            center_x_px=center_x_px,
            center_y_px=center_y_px,
            offset_x_px=offset_x_px,
            offset_y_px=offset_y_px,
            target_x_mm=(float(target_x_mm) if target_x_mm is not None else None),
            target_y_mm=(float(target_y_mm) if target_y_mm is not None else None),
            corners=corners,
            frame_age_ms=frame_age_ms,
            observation_source=source,
            homography_valid=homography_valid,
            predicted_frames=predicted_frames,
            near_image_edge=bool(getattr(detection, "near_image_edge", False)),
            partially_outside=bool(getattr(detection, "partially_outside", False)),
            failure_reason=failure_value,
            laser_permission=False,
        )

    @classmethod
    def from_hybrid(
        cls,
        hybrid: Any,
        *,
        image_size: tuple[int, int],
        now_ns: int,
        max_result_age_ms: float = BoardTrackingConfig().max_result_age_ms,
    ) -> "VisionTargetResult":
        """Compatibility wrapper for the legacy hybrid detector result."""

        values = dict(vars(hybrid))
        values.setdefault("observation_source", "FULL_BOARD")
        # Preserve the legacy contract that only the exact upstream TRACKING state
        # is accepted, while the generic mapper still normalizes enum/string names.
        values["target_valid"] = bool(
            values.get("target_valid", False)
            and values.get("tracking_state") == "TRACKING"
        )
        return cls.from_detection(
            SimpleNamespace(**values),
            image_size=image_size,
            now_ns=now_ns,
            max_result_age_ms=max_result_age_ms,
        )
