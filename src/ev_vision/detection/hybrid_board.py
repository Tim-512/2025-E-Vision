from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import time
from typing import Protocol

import numpy as np

from ev_vision.config import CandidateScoringConfig, DetectionConfig
from ev_vision.detection.failures import (
    CandidateEvaluation,
    DetectionFailure,
    HybridBoardResult,
)
from ev_vision.detection.roi_board_geometry import (
    GeometryFailure,
    GeometryResult,
    RoiBoardGeometry,
)
from ev_vision.detection.yolo_board import BoardSearchResult


class ModelCandidatePort(Protocol):
    def detect_candidates(self, image: np.ndarray) -> Sequence[BoardSearchResult]: ...


class GeometryPort(Protocol):
    def refine(
        self,
        image: np.ndarray,
        model_box: tuple[float, float, float, float],
        *,
        include_debug: bool = False,
    ) -> GeometryResult: ...


class TrackerPort(Protocol):
    def temporal_score(self, center_px: tuple[float, float]) -> float: ...

    def update(self, observation: object) -> object: ...


@dataclass(frozen=True)
class _TrackerObservation:
    timestamp_ns: int
    source_sequence: int
    detected: bool
    center_px: tuple[float, float] | None
    corners_px: tuple[tuple[float, float], ...] | None
    failure_reason: DetectionFailure | None


_GEOMETRY_FAILURE_MAP = {
    GeometryFailure.INVALID_ROI: DetectionFailure.NO_VALID_QUADRILATERAL,
    GeometryFailure.NO_VALID_QUADRILATERAL: DetectionFailure.NO_VALID_QUADRILATERAL,
    GeometryFailure.TRUNCATED_QUADRILATERAL: DetectionFailure.NO_VALID_QUADRILATERAL,
    GeometryFailure.UNDERSIZED_QUADRILATERAL: DetectionFailure.NO_VALID_QUADRILATERAL,
    GeometryFailure.INVALID_ASPECT_RATIO: DetectionFailure.INVALID_ASPECT_RATIO,
    GeometryFailure.LOW_EDGE_SUPPORT: DetectionFailure.LOW_EDGE_SUPPORT,
    GeometryFailure.LOW_INTERNAL_STRUCTURE: DetectionFailure.LOW_INTERNAL_STRUCTURE,
    GeometryFailure.AMBIGUOUS_GEOMETRY: DetectionFailure.AMBIGUOUS_CANDIDATES,
    GeometryFailure.CORNER_ORDER_FAILED: DetectionFailure.NO_VALID_QUADRILATERAL,
}


class HybridBoardDetector:
    def __init__(
        self,
        model: ModelCandidatePort | None,
        geometry: GeometryPort | None = None,
        *,
        config: DetectionConfig | None = None,
        tracker: TrackerPort | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        model_state: str | None = None,
        model_backend: str | None = None,
        model_path: str | None = None,
    ) -> None:
        self.model = model
        self.config = config or DetectionConfig()
        self.geometry = geometry or self._geometry_from_config(self.config)
        self.tracker = tracker
        self.clock_ns = clock_ns
        self._model_state = model_state
        self._model_backend = model_backend
        self._model_path = model_path

    @staticmethod
    def _geometry_from_config(config: DetectionConfig) -> RoiBoardGeometry:
        settings = config.roi_geometry
        return RoiBoardGeometry(
            padding_fraction=settings.padding_fraction,
            canny_low=settings.canny_low,
            canny_high=settings.canny_high,
            min_edge_support=settings.min_edge_support,
            min_geometry_score=settings.min_geometry_score,
            expected_aspect_ratio=settings.expected_aspect_ratio,
            aspect_ratio_tolerance=settings.aspect_ratio_tolerance,
            minimum_side_px=settings.minimum_side_px,
            minimum_area_fraction=settings.minimum_area_fraction,
            maximum_area_fraction=settings.maximum_area_fraction,
        )

    def detect(
        self,
        image: np.ndarray,
        *,
        captured_ns: int,
        source_sequence: int,
        include_debug: bool = False,
        update_tracker: bool = True,
    ) -> HybridBoardResult:
        started_ns = self.clock_ns()
        model_state, model_backend, model_path = self._model_metadata()
        if self._model_unavailable():
            ended_ns = self.clock_ns()
            return self._empty_result(
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                model_state="UNAVAILABLE",
                model_backend=model_backend,
                model_path=model_path,
                failure_reason=DetectionFailure.MODEL_UNAVAILABLE,
                inference_ms=0.0,
                geometry_ms=0.0,
                total_ms=_elapsed_ms(started_ns, ended_ns),
            )

        try:
            candidates = tuple(self.model.detect_candidates(image))  # type: ignore[union-attr]
        except Exception:
            inference_ended_ns = self.clock_ns()
            return self._empty_result(
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                model_state="ERROR",
                model_backend=model_backend,
                model_path=model_path,
                failure_reason=DetectionFailure.MODEL_ERROR,
                inference_ms=_elapsed_ms(started_ns, inference_ended_ns),
                geometry_ms=0.0,
                total_ms=_elapsed_ms(started_ns, inference_ended_ns),
            )
        inference_ended_ns = self.clock_ns()
        inference_ms = _elapsed_ms(started_ns, inference_ended_ns)

        if not candidates:
            result = self._empty_result(
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                model_state=model_state,
                model_backend=model_backend,
                model_path=model_path,
                failure_reason=DetectionFailure.NO_MODEL_CANDIDATE,
                inference_ms=inference_ms,
                geometry_ms=0.0,
                total_ms=_elapsed_ms(started_ns, inference_ended_ns),
            )
            self._update_tracker(result, update_tracker=update_tracker)
            return result

        geometry_started_ns = self.clock_ns()
        evaluations: list[CandidateEvaluation] = []
        debug_images: dict[str, np.ndarray] = {}
        for index, model_candidate in enumerate(candidates):
            geometry_result = self.geometry.refine(
                image,
                model_candidate.xyxy_px,
                include_debug=include_debug,
            )
            temporal_score = self._temporal_score(geometry_result)
            evaluations.append(
                CandidateEvaluation(
                    model=model_candidate,
                    geometry=geometry_result,
                    temporal_score=temporal_score,
                    combined_score=self._combined_score(
                        model_candidate,
                        geometry_result,
                        temporal_score,
                    ),
                )
            )
            if include_debug:
                self._collect_debug(debug_images, index, geometry_result)
        geometry_ended_ns = self.clock_ns()
        geometry_ms = _elapsed_ms(geometry_started_ns, geometry_ended_ns)
        evaluations.sort(key=lambda item: item.combined_score, reverse=True)

        accepted = tuple(item for item in evaluations if item.geometry.accepted)
        best = accepted[0] if accepted else None
        failure_reason: DetectionFailure | None = None
        detected = False
        if best is None:
            failure_reason = _map_geometry_failure(evaluations[0].geometry.failure_reason)
        elif (
            len(accepted) > 1
            and best.combined_score - accepted[1].combined_score
            < self.config.candidate_scoring.ambiguity_margin
        ):
            failure_reason = DetectionFailure.AMBIGUOUS_CANDIDATES
        else:
            detected = True

        result = HybridBoardResult(
            timestamp_ns=captured_ns,
            source_sequence=source_sequence,
            detected=detected,
            target_valid=False,
            tracking_state="SEARCHING",
            model_state=model_state,
            model_backend=model_backend,
            model_path=model_path,
            model_confidence=best.model.confidence if best is not None else 0.0,
            geometry_score=best.geometry.geometry_score if best is not None else 0.0,
            edge_support_score=best.geometry.edge_support_score if best is not None else 0.0,
            structure_score=best.geometry.structure_score if best is not None else 0.0,
            temporal_score=best.temporal_score if best is not None else 0.0,
            combined_score=best.combined_score if best is not None else 0.0,
            candidate_count=len(candidates),
            corners_px=best.geometry.corners_px if detected and best is not None else None,
            center_px=best.geometry.center_px if detected and best is not None else None,
            failure_reason=failure_reason,
            inference_ms=inference_ms,
            geometry_ms=geometry_ms,
            total_ms=_elapsed_ms(started_ns, geometry_ended_ns),
            candidates=tuple(evaluations),
            debug_images=debug_images,
        )
        self._update_tracker(result, update_tracker=update_tracker)
        return result

    def _combined_score(
        self,
        model: BoardSearchResult,
        geometry: GeometryResult,
        temporal_score: float,
    ) -> float:
        if not geometry.accepted:
            return 0.0
        scoring: CandidateScoringConfig = self.config.candidate_scoring
        return (
            scoring.model_weight * _clamp01(model.confidence)
            + scoring.geometry_weight * _clamp01(geometry.geometry_score)
            + scoring.structure_weight * _clamp01(geometry.structure_score)
            + scoring.temporal_weight * temporal_score
        )

    def _temporal_score(self, geometry: GeometryResult) -> float:
        if not geometry.accepted or geometry.center_px is None or self.tracker is None:
            return 0.0
        try:
            return _clamp01(self.tracker.temporal_score(geometry.center_px))
        except Exception:
            return 0.0

    def _model_unavailable(self) -> bool:
        if self.model is None:
            return True
        return hasattr(self.model, "backend") and getattr(self.model, "backend") is None

    def _model_metadata(self) -> tuple[str, str, str | None]:
        backend = getattr(self.model, "backend", None) if self.model is not None else None
        backend_name = self._model_backend or _backend_name(backend)
        path = self._model_path or _backend_path(backend)
        if self._model_state is not None:
            state = self._model_state
        elif self._model_unavailable():
            state = "UNAVAILABLE"
        else:
            state = "READY"
        return state, backend_name, path

    @staticmethod
    def _collect_debug(
        debug_images: dict[str, np.ndarray],
        index: int,
        geometry: GeometryResult,
    ) -> None:
        debug = geometry.debug
        if debug is None:
            return
        debug_images[f"candidate_{index}_roi"] = debug.roi_bgr
        debug_images[f"candidate_{index}_edges"] = debug.edges
        debug_images[f"candidate_{index}_geometry"] = debug.candidates_bgr

    def _update_tracker(
        self,
        result: HybridBoardResult,
        *,
        update_tracker: bool,
    ) -> None:
        if not update_tracker or self.tracker is None:
            return
        self.tracker.update(
            _TrackerObservation(
                timestamp_ns=result.timestamp_ns,
                source_sequence=result.source_sequence,
                detected=result.detected,
                center_px=result.center_px,
                corners_px=result.corners_px,
                failure_reason=result.failure_reason,
            )
        )

    @staticmethod
    def _empty_result(
        *,
        captured_ns: int,
        source_sequence: int,
        model_state: str,
        model_backend: str,
        model_path: str | None,
        failure_reason: DetectionFailure,
        inference_ms: float,
        geometry_ms: float,
        total_ms: float,
    ) -> HybridBoardResult:
        return HybridBoardResult(
            timestamp_ns=captured_ns,
            source_sequence=source_sequence,
            detected=False,
            target_valid=False,
            tracking_state="SEARCHING",
            model_state=model_state,
            model_backend=model_backend,
            model_path=model_path,
            model_confidence=0.0,
            geometry_score=0.0,
            edge_support_score=0.0,
            structure_score=0.0,
            temporal_score=0.0,
            combined_score=0.0,
            candidate_count=0,
            corners_px=None,
            center_px=None,
            failure_reason=failure_reason,
            inference_ms=inference_ms,
            geometry_ms=geometry_ms,
            total_ms=total_ms,
        )


def _map_geometry_failure(reason: GeometryFailure | None) -> DetectionFailure:
    if reason is None:
        return DetectionFailure.NO_VALID_QUADRILATERAL
    return _GEOMETRY_FAILURE_MAP.get(reason, DetectionFailure.NO_VALID_QUADRILATERAL)


def _elapsed_ms(started_ns: int, ended_ns: int) -> float:
    return max(0.0, float(ended_ns - started_ns) / 1_000_000.0)


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def _backend_name(backend: object | None) -> str:
    if backend is None:
        return ""
    artifact_kind = getattr(backend, "artifact_kind", None)
    if isinstance(artifact_kind, str) and artifact_kind:
        return artifact_kind
    return type(backend).__name__


def _backend_path(backend: object | None) -> str | None:
    if backend is None:
        return None
    for attribute in ("artifact_path", "model_path", "path"):
        value = getattr(backend, attribute, None)
        if value is not None:
            return str(value)
    return None
