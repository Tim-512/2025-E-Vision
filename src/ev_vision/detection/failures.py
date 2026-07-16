from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np

from ev_vision.detection.roi_board_geometry import GeometryResult
from ev_vision.detection.yolo_board import BoardSearchResult


class DetectionFailure(str, Enum):
    NO_MODEL_CANDIDATE = "NO_MODEL_CANDIDATE"
    LOW_MODEL_CONFIDENCE = "LOW_MODEL_CONFIDENCE"
    NO_VALID_QUADRILATERAL = "NO_VALID_QUADRILATERAL"
    INVALID_ASPECT_RATIO = "INVALID_ASPECT_RATIO"
    LOW_EDGE_SUPPORT = "LOW_EDGE_SUPPORT"
    LOW_INTERNAL_STRUCTURE = "LOW_INTERNAL_STRUCTURE"
    AMBIGUOUS_CANDIDATES = "AMBIGUOUS_CANDIDATES"
    EXCESSIVE_POSITION_JUMP = "EXCESSIVE_POSITION_JUMP"
    STALE_FRAME = "STALE_FRAME"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_ERROR = "MODEL_ERROR"


@dataclass(frozen=True)
class CandidateEvaluation:
    model: BoardSearchResult
    geometry: GeometryResult
    temporal_score: float
    combined_score: float


@dataclass(frozen=True)
class HybridBoardResult:
    timestamp_ns: int
    source_sequence: int
    detected: bool
    target_valid: bool
    tracking_state: str
    model_state: str
    model_backend: str
    model_path: str | None
    model_confidence: float
    geometry_score: float
    edge_support_score: float
    structure_score: float
    temporal_score: float
    combined_score: float
    candidate_count: int
    corners_px: tuple[tuple[float, float], ...] | None
    center_px: tuple[float, float] | None
    failure_reason: DetectionFailure | None
    inference_ms: float
    geometry_ms: float
    total_ms: float
    candidates: tuple[CandidateEvaluation, ...] = ()
    debug_images: Mapping[str, np.ndarray] = field(default_factory=dict, compare=False)
