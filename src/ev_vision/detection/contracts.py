from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np

from ev_vision.detection.failures import DetectionFailure


class ObservationSource(str, Enum):
    NONE = "NONE"
    FULL_BOARD = "FULL_BOARD"
    CONCENTRIC_ARCS = "CONCENTRIC_ARCS"
    SINGLE_ARC = "SINGLE_ARC"
    WHITE_REGION = "WHITE_REGION"
    FUSED_PARTIAL = "FUSED_PARTIAL"
    PREDICTED = "PREDICTED"

    @property
    def priority(self) -> int:
        return {
            self.NONE: 0,
            self.PREDICTED: 1,
            self.WHITE_REGION: 2,
            self.SINGLE_ARC: 3,
            self.FUSED_PARTIAL: 4,
            self.CONCENTRIC_ARCS: 5,
            self.FULL_BOARD: 6,
        }[self]


@dataclass(frozen=True)
class ClassicalCandidateEvaluation:
    corners_px: tuple[tuple[float, float], ...]
    center_px: tuple[float, float] | None
    white_score: float
    geometry_score: float
    ring_score: float
    border_score: float
    temporal_score: float
    texture_penalty: float
    jump_penalty: float
    combined_score: float
    accepted: bool
    failure_reason: DetectionFailure | None = None
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClassicalBoardResult:
    timestamp_ns: int
    source_sequence: int
    detected: bool
    target_valid: bool
    tracking_state: str
    observation_source: ObservationSource
    confidence: float
    center_px: tuple[float, float] | None
    corners_px: tuple[tuple[float, float], ...] = ()
    scale_px_per_mm: float | None = None
    velocity_px_s: tuple[float, float] | None = None
    predicted_frames: int = 0
    source_age_us: int = 0
    homography_valid: bool = False
    target_x_mm: float | None = None
    target_y_mm: float | None = None
    near_image_edge: bool = False
    partially_outside: bool = False
    failure_reason: DetectionFailure | None = None
    candidates: tuple[ClassicalCandidateEvaluation, ...] = ()
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    debug_images: Mapping[str, np.ndarray] = field(
        default_factory=dict, compare=False
    )
