from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path
import threading
import time
from typing import Protocol

import numpy as np

from ev_vision.config import BoardConfig, CandidateScoringConfig, DetectionConfig
from ev_vision.detection.board_solution import solve_board_plane
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
from ev_vision.detection.yolo_board import BoardSearchResult, InferencePort, YoloBoardDetector
from ev_vision.tracking.board_tracker import (
    TrackObservation,
    TrackedBoardResult,
    TrackingState,
)


class ModelCandidatePort(Protocol):
    """Candidate source; models without backend metadata are considered available.

    Availability is overridden by ``model_state="UNAVAILABLE"``. Backend-aware
    adapters such as ``YoloBoardDetector`` may additionally expose ``backend``;
    an explicitly present ``backend=None`` declares that adapter unavailable.
    """

    def detect_candidates(self, image: np.ndarray) -> Sequence[BoardSearchResult]: ...


class BackendAwareModelCandidatePort(ModelCandidatePort, Protocol):
    backend: object | None


class GeometryPort(Protocol):
    def refine(
        self,
        image: np.ndarray,
        model_box: tuple[float, float, float, float],
        *,
        include_debug: bool = False,
    ) -> GeometryResult: ...


class TrackerPort(Protocol):
    latest: TrackedBoardResult

    def temporal_score(self, center_px: tuple[float, float]) -> float: ...

    def update(
        self,
        observation: TrackObservation,
        *,
        now_ns: int | None = None,
    ) -> TrackedBoardResult: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class DetectionBackendSelection:
    backend: InferencePort | None
    path: Path | None
    artifact_kind: str
    errors: tuple[str, ...]


@dataclass(frozen=True)
class _ModelMetadata:
    state: str
    backend: str
    path: str | None


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
_MISSING_BACKEND = object()


class HybridBoardDetector:
    def __init__(
        self,
        model: ModelCandidatePort | None,
        geometry: GeometryPort | None = None,
        *,
        config: DetectionConfig | None = None,
        tracker: TrackerPort | None = None,
        board: BoardConfig | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        model_state: str | None = None,
        model_backend: str | None = None,
        model_path: str | None = None,
        model_errors: Sequence[str] = (),
        model_loader: Callable[[DetectionConfig], DetectionBackendSelection] | None = None,
    ) -> None:
        self.model = model
        self.config = config or DetectionConfig()
        self.geometry = geometry or self._geometry_from_config(self.config)
        self.tracker = tracker
        self.board = board or BoardConfig()
        self.clock_ns = clock_ns
        self._model_state = model_state
        self._model_backend = model_backend
        self._model_path = model_path
        self._model_errors = tuple(model_errors)
        self._model_loader = model_loader
        self._model_lock = threading.RLock()
        self._reload_call_lock = threading.Lock()
        self._reload_epoch = 0
        self._reload_in_progress = False

    @property
    def model_state(self) -> str:
        return self._model_metadata().state

    @property
    def model_backend(self) -> str:
        return self._model_metadata().backend

    @property
    def model_path(self) -> str | None:
        return self._model_metadata().path

    @property
    def model_errors(self) -> tuple[str, ...]:
        with self._model_lock:
            return self._model_errors

    def reset(self) -> None:
        with self._model_lock:
            self._reset_tracker_locked()

    def apply_config(self, config: DetectionConfig) -> None:
        geometry = self._geometry_from_config(config)
        with self._model_lock:
            self.config = config
            self.geometry = geometry
        self.reset()

    def reload_model(self) -> None:
        with self._reload_call_lock:
            self._reload_model_serialized()

    def _reload_model_serialized(self) -> None:
        with self._model_lock:
            self._reload_epoch += 1
            reload_epoch = self._reload_epoch
            self._reload_in_progress = True
            config = self.config
            self._reset_tracker_locked()

        if self._model_loader is None:
            errors = ("no model loader is configured",)
            self._publish_unavailable(errors, reload_epoch=reload_epoch)
            raise RuntimeError(errors[0])
        try:
            selection = self._model_loader(config)
        except Exception as exc:
            errors = (str(exc),)
            self._publish_unavailable(errors, reload_epoch=reload_epoch)
            raise RuntimeError(errors[0]) from exc
        if selection.backend is None:
            errors = selection.errors or ("no portable model artifact could be loaded",)
            self._publish_unavailable(errors, reload_epoch=reload_epoch)
            raise RuntimeError("; ".join(errors))
        try:
            self._smoke_test_backend(selection.backend)
            model = self._model_from_backend(selection.backend)
        except Exception as exc:
            path = str(selection.path) if selection.path is not None else "model"
            errors = (f"{path}: {exc}",)
            self._publish_unavailable(errors, reload_epoch=reload_epoch)
            raise RuntimeError(errors[0]) from exc
        with self._model_lock:
            if reload_epoch != self._reload_epoch:
                raise RuntimeError("model reload was superseded")
            self.model = model
            self._model_state = "READY"
            self._model_backend = selection.artifact_kind
            self._model_path = str(selection.path) if selection.path is not None else None
            self._model_errors = ()
            self._reload_in_progress = False

    def _publish_unavailable(
        self, errors: Sequence[str], *, reload_epoch: int
    ) -> None:
        with self._model_lock:
            if reload_epoch != self._reload_epoch:
                return
            self.model = None
            self._model_state = "UNAVAILABLE"
            self._model_backend = "classical-diagnostic"
            self._model_path = None
            self._model_errors = tuple(errors)
            self._reload_in_progress = False

    def _reset_tracker_locked(self) -> None:
        if self.tracker is not None:
            try:
                self.tracker.reset()
            except Exception:
                pass

    def _model_from_backend(self, backend: InferencePort) -> YoloBoardDetector:
        settings = self.config.model
        return YoloBoardDetector(
            backend,
            confidence_threshold=settings.confidence_threshold,
            max_candidates=settings.max_candidates,
        )

    @staticmethod
    def _smoke_test_backend(backend: InferencePort) -> None:
        width, height = backend.input_size
        tensor = np.zeros((1, 3, height, width), dtype=np.float32)
        tuple(backend.infer(tensor))

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
        with self._model_lock:
            model = self.model
            metadata = self._model_metadata_locked(model)
            reload_epoch = self._reload_epoch
            model_unavailable = (
                self._reload_in_progress or self._model_unavailable(model)
            )
        if model_unavailable:
            result = self._empty_result(
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                model_state="UNAVAILABLE",
                model_backend=metadata.backend,
                model_path=metadata.path,
                failure_reason=DetectionFailure.MODEL_UNAVAILABLE,
            )
            return replace(result, total_ms=_elapsed_ms(started_ns, self.clock_ns()))

        try:
            candidates = tuple(model.detect_candidates(image))  # type: ignore[union-attr]
        except Exception:
            inference_ended_ns = self.clock_ns()
            result = self._empty_result(
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                model_state="ERROR",
                model_backend=metadata.backend,
                model_path=metadata.path,
                failure_reason=DetectionFailure.MODEL_ERROR,
                inference_ms=_elapsed_ms(started_ns, inference_ended_ns),
            )
            return self._finish(
                result,
                started_ns=started_ns,
                update_tracker=update_tracker,
                reload_epoch=reload_epoch,
            )
        inference_ended_ns = self.clock_ns()
        inference_ms = _elapsed_ms(started_ns, inference_ended_ns)

        if not candidates:
            result = self._empty_result(
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                model_state=metadata.state,
                model_backend=metadata.backend,
                model_path=metadata.path,
                failure_reason=DetectionFailure.NO_MODEL_CANDIDATE,
                inference_ms=inference_ms,
            )
            return self._finish(
                result,
                started_ns=started_ns,
                update_tracker=update_tracker,
                reload_epoch=reload_epoch,
            )

        evaluations: list[CandidateEvaluation] = []
        debug_images: dict[str, np.ndarray] = {}
        geometry_ms = 0.0
        for index, model_candidate in enumerate(candidates):
            refine_started_ns = self.clock_ns()
            try:
                geometry_result = self.geometry.refine(
                    image,
                    model_candidate.xyxy_px,
                    include_debug=include_debug,
                )
            except Exception:
                refine_ended_ns = self.clock_ns()
                geometry_ms += _elapsed_ms(refine_started_ns, refine_ended_ns)
                result = self._empty_result(
                    captured_ns=captured_ns,
                    source_sequence=source_sequence,
                    model_state=metadata.state,
                    model_backend=metadata.backend,
                    model_path=metadata.path,
                    failure_reason=DetectionFailure.MODEL_ERROR,
                    inference_ms=inference_ms,
                    geometry_ms=geometry_ms,
                )
                return self._finish(
                    result,
                    started_ns=started_ns,
                    update_tracker=update_tracker,
                    reload_epoch=reload_epoch,
                )
            refine_ended_ns = self.clock_ns()
            geometry_ms += _elapsed_ms(refine_started_ns, refine_ended_ns)

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
            tracking_state=self._current_tracking_state(),
            model_state=metadata.state,
            model_backend=metadata.backend,
            model_path=metadata.path,
            model_confidence=best.model.confidence if best is not None else 0.0,
            geometry_score=best.geometry.geometry_score if best is not None else 0.0,
            edge_support_score=(
                best.geometry.edge_support_score if best is not None else 0.0
            ),
            structure_score=best.geometry.structure_score if best is not None else 0.0,
            temporal_score=best.temporal_score if best is not None else 0.0,
            combined_score=best.combined_score if best is not None else 0.0,
            candidate_count=len(candidates),
            corners_px=best.geometry.corners_px if detected and best is not None else None,
            center_px=best.geometry.center_px if detected and best is not None else None,
            failure_reason=failure_reason,
            inference_ms=inference_ms,
            geometry_ms=geometry_ms,
            total_ms=0.0,
            candidates=tuple(evaluations),
            debug_images=debug_images,
        )
        return self._finish(
            result,
            started_ns=started_ns,
            update_tracker=update_tracker,
            reload_epoch=reload_epoch,
        )

    def _finish(
        self,
        result: HybridBoardResult,
        *,
        started_ns: int,
        update_tracker: bool,
        reload_epoch: int,
    ) -> HybridBoardResult:
        with self._model_lock:
            if reload_epoch != self._reload_epoch or self._reload_in_progress:
                processed = self._stale_reload_result_locked(result)
            else:
                processed = self._apply_tracker(result, update_tracker=update_tracker)
        ended_ns = self.clock_ns()
        return replace(processed, total_ms=_elapsed_ms(started_ns, ended_ns))

    def _stale_reload_result_locked(
        self, result: HybridBoardResult
    ) -> HybridBoardResult:
        metadata = self._model_metadata_locked(self.model)
        unavailable = self._reload_in_progress or self._model_unavailable(self.model)
        return replace(
            self._empty_result(
                captured_ns=result.timestamp_ns,
                source_sequence=result.source_sequence,
                model_state="UNAVAILABLE" if self._reload_in_progress else metadata.state,
                model_backend=metadata.backend,
                model_path=metadata.path,
                failure_reason=(
                    DetectionFailure.MODEL_UNAVAILABLE
                    if unavailable
                    else DetectionFailure.STALE_FRAME
                ),
                inference_ms=result.inference_ms,
                geometry_ms=result.geometry_ms,
            ),
            debug_images={},
        )

    def _apply_tracker(
        self,
        result: HybridBoardResult,
        *,
        update_tracker: bool,
    ) -> HybridBoardResult:
        if self.tracker is None:
            return replace(result, target_valid=False, tracking_state="SEARCHING")
        if not update_tracker:
            return replace(
                result,
                target_valid=False,
                tracking_state=self._current_tracking_state(),
            )

        observation = TrackObservation(
            timestamp_ns=result.timestamp_ns,
            source_sequence=result.source_sequence,
            detected=result.detected,
            center_px=result.center_px,
            corners_px=result.corners_px,
            failure_reason=result.failure_reason,
        )
        tracker_now_ns = self.clock_ns()
        try:
            tracked = self.tracker.update(observation, now_ns=tracker_now_ns)
        except Exception:
            return self._safe_tracker_failure(result)

        failure_reason = tracked.failure_reason or result.failure_reason
        invalidated = tracked.failure_reason is not None
        board_solution = None
        if (
            not invalidated
            and result.detected
            and result.failure_reason is None
            and result.corners_px is not None
        ):
            board_solution = solve_board_plane(result.corners_px, board=self.board)
        homography_valid = bool(
            board_solution is not None and board_solution.homography_valid
        )
        target_valid = (
            tracked.state is TrackingState.TRACKING
            and tracked.target_valid
            and result.detected
            and result.failure_reason is None
            and not invalidated
            and homography_valid
        )
        target_center_mm = (
            board_solution.target_center_mm if homography_valid else None
        )
        return replace(
            result,
            detected=False if invalidated else result.detected,
            target_valid=target_valid,
            tracking_state=tracked.state.value,
            homography_valid=homography_valid,
            target_x_mm=(target_center_mm[0] if target_center_mm is not None else None),
            target_y_mm=(target_center_mm[1] if target_center_mm is not None else None),
            corners_px=None if invalidated else result.corners_px,
            center_px=(
                board_solution.center_px
                if homography_valid and board_solution is not None
                else (None if invalidated else result.center_px)
            ),
            failure_reason=failure_reason,
        )

    def _safe_tracker_failure(self, result: HybridBoardResult) -> HybridBoardResult:
        if self.tracker is not None:
            try:
                self.tracker.reset()
            except Exception:
                pass
        return replace(
            result,
            detected=False,
            target_valid=False,
            tracking_state=TrackingState.SEARCHING.value,
            homography_valid=False,
            target_x_mm=None,
            target_y_mm=None,
            corners_px=None,
            center_px=None,
            failure_reason=DetectionFailure.MODEL_ERROR,
        )

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

    def _model_unavailable(self, model: ModelCandidatePort | None = None) -> bool:
        candidate = self.model if model is None else model
        if self._model_state is not None and self._model_state.upper() == "UNAVAILABLE":
            return True
        if candidate is None:
            return True
        backend = getattr(candidate, "backend", _MISSING_BACKEND)
        return backend is None

    def _model_metadata(self) -> _ModelMetadata:
        with self._model_lock:
            return self._model_metadata_locked(self.model)

    def _model_metadata_locked(self, model: ModelCandidatePort | None) -> _ModelMetadata:
        backend = getattr(model, "backend", None) if model is not None else None
        backend_name = self._model_backend or _backend_name(backend)
        path = self._model_path or _backend_path(backend)
        if self._model_state is not None:
            state = self._model_state
        elif self._model_unavailable(model):
            state = "UNAVAILABLE"
        else:
            state = "READY"
        return _ModelMetadata(state=state, backend=backend_name, path=path)

    def _current_tracking_state(self) -> str:
        if self.tracker is None:
            return "SEARCHING"
        try:
            return self.tracker.latest.state.value
        except Exception:
            return "SEARCHING"

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

    @staticmethod
    def _empty_result(
        *,
        captured_ns: int,
        source_sequence: int,
        model_state: str,
        model_backend: str,
        model_path: str | None,
        failure_reason: DetectionFailure,
        inference_ms: float = 0.0,
        geometry_ms: float = 0.0,
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
            total_ms=0.0,
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
