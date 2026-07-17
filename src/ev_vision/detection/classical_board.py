from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np

from ev_vision.config import BoardConfig, DetectionConfig
from ev_vision.detection.board_solution import BoardPlaneSolution, solve_board_plane
from ev_vision.detection.candidate_scoring import CandidateEvidence, rank_candidates
from ev_vision.detection.contracts import (
    ClassicalBoardResult,
    ClassicalCandidateEvaluation,
    ObservationSource,
)
from ev_vision.detection.debug_rendering import render_debug_images
from ev_vision.detection.failures import DetectionFailure
from ev_vision.detection.image_normalization import NormalizedFrame, normalize_frame
from ev_vision.detection.partial_board import BoardHistory, fuse_partial_observation, predicted_roi
from ev_vision.detection.ring_geometry import RingGeometryResult, detect_concentric_arcs
from ev_vision.detection.white_board import WhiteBoardCandidate, find_white_board_candidates
from ev_vision.tracking.board_tracker import (
    BoardTracker,
    TrackObservation,
    TrackedBoardResult,
    TrackingState,
)


@dataclass(frozen=True)
class _FullAcquisition:
    observation: TrackObservation
    candidates: tuple[ClassicalCandidateEvaluation, ...]
    solution: BoardPlaneSolution | None
    ring_result: RingGeometryResult | None


class ClassicalBoardDetector:
    model_state = "READY"
    model_backend = "classical"
    model_path = None

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._tracker = BoardTracker(config.tracking)
        self._history: BoardHistory | None = None
        self._board = BoardConfig()

    def detect(
        self,
        image: np.ndarray,
        *,
        captured_ns: int,
        source_sequence: int,
        include_debug: bool = False,
        update_tracker: bool = True,
    ) -> ClassicalBoardResult:
        started = time.perf_counter()
        normalized = normalize_frame(
            image,
            self._config.normalization,
            mask_radius_px=self._config.rings.saturation_mask_radius_px,
        )
        normalized_ms = (time.perf_counter() - started) * 1000.0
        acquisition = self._tracker.latest.state in {
            TrackingState.SEARCHING,
            TrackingState.CONFIRMING,
            TrackingState.LOST,
        }
        roi_xyxy: tuple[int, int, int, int] | None = None
        ring_result: RingGeometryResult | None = None
        solution: BoardPlaneSolution | None = None
        candidates: tuple[ClassicalCandidateEvaluation, ...] = ()
        near_image_edge = False
        partially_outside = False

        detection_started = time.perf_counter()
        if acquisition:
            full = self._acquire_full_board(
                normalized,
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                tracking=False,
            )
            observation = full.observation
            candidates = full.candidates
            solution = full.solution
            ring_result = full.ring_result
        else:
            full = self._acquire_full_board(
                normalized,
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                tracking=True,
                roi_xyxy=self._tracking_roi(captured_ns, normalized.gray.shape),
            )
            candidates = full.candidates
            solution = full.solution
            ring_result = full.ring_result
            if full.observation.detected:
                observation = full.observation
            else:
                observation, ring_result, roi_xyxy, near_image_edge, partially_outside = (
                    self._acquire_partial(
                        normalized,
                        captured_ns=captured_ns,
                        source_sequence=source_sequence,
                    )
                )

        tracked = (
            self._tracker.update(observation, now_ns=captured_ns)
            if update_tracker
            else self._tracker.preview(observation, now_ns=captured_ns)
        )
        if update_tracker and tracked.target_valid and tracked.observation_source is not ObservationSource.PREDICTED:
            self._remember(tracked, solution, captured_ns)
        detection_ms = (time.perf_counter() - detection_started) * 1000.0
        center = tracked.predicted_center_px
        if tracked.observation is not None and tracked.observation.center_px is not None:
            center = tracked.observation.center_px
        corners = ()
        if tracked.observation is not None and tracked.observation.corners_px:
            corners = tracked.observation.corners_px
        elif self._history is not None and tracked.target_valid:
            corners = self._history.corners_px
        homography_valid = bool(
            tracked.observation_source is ObservationSource.FULL_BOARD
            and solution is not None
            and solution.homography_valid
        )
        failure_reason = tracked.failure_reason
        debug_images = (
            render_debug_images(
                normalized,
                ring_result=ring_result,
                candidates=candidates,
                roi_xyxy=roi_xyxy,
            )
            if include_debug
            else {}
        )
        return ClassicalBoardResult(
            timestamp_ns=int(captured_ns),
            source_sequence=int(source_sequence),
            detected=bool(tracked.observation is not None and tracked.observation.detected),
            target_valid=tracked.target_valid,
            tracking_state=tracked.state.value,
            observation_source=tracked.observation_source,
            confidence=(
                tracked.observation.confidence
                if tracked.observation is not None and tracked.observation.detected
                else 0.0
            ),
            center_px=center if tracked.target_valid or tracked.state is TrackingState.CONFIRMING else None,
            corners_px=corners,
            scale_px_per_mm=tracked.scale_px_per_mm,
            velocity_px_s=tracked.velocity_px_s,
            predicted_frames=tracked.predicted_frames,
            source_age_us=tracked.source_age_us,
            homography_valid=homography_valid,
            target_x_mm=0.0 if homography_valid else None,
            target_y_mm=0.0 if homography_valid else None,
            near_image_edge=near_image_edge,
            partially_outside=partially_outside,
            failure_reason=failure_reason,
            candidates=candidates,
            timings_ms={
                "normalization": normalized_ms,
                "detection": detection_ms,
                "total": (time.perf_counter() - started) * 1000.0,
            },
            debug_images=debug_images,
        )

    def reset(self) -> None:
        self._tracker.reset()
        self._history = None

    def apply_config(self, config: DetectionConfig) -> None:
        self._config = config
        self._tracker.apply_config(config.tracking)

    def reload_model(self) -> None:
        return None

    def _tracking_roi(
        self,
        timestamp_ns: int,
        image_shape: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        if self._history is None:
            return None
        height, width = image_shape
        return predicted_roi(
            self._history,
            timestamp_ns=timestamp_ns,
            image_size=(width, height),
        )

    def _acquire_full_board(
        self,
        normalized: NormalizedFrame,
        *,
        captured_ns: int,
        source_sequence: int,
        tracking: bool,
        roi_xyxy: tuple[int, int, int, int] | None = None,
    ) -> _FullAcquisition:
        white_candidates = find_white_board_candidates(
            normalized,
            self._config.white_board,
            roi_xyxy=roi_xyxy,
        )
        if not white_candidates:
            return _FullAcquisition(
                self._miss(captured_ns, source_sequence, DetectionFailure.NO_WHITE_CANDIDATE),
                (),
                None,
                None,
            )

        evidence: list[CandidateEvidence] = []
        ring_results: list[RingGeometryResult] = []
        solutions: list[BoardPlaneSolution] = []
        for candidate in white_candidates:
            ring = detect_concentric_arcs(
                normalized.ring_edge_mask,
                self._config.rings,
                expected_center_px=candidate.center_px,
                roi_xyxy=candidate.bbox_xyxy,
            )
            ring_results.append(ring)
            solution = solve_board_plane(candidate.corners_px, board=self._board)
            solutions.append(solution)
            ring_score = float(np.clip(
                0.40 * ring.common_center_score
                + 0.40 * ring.ratio_score
                + 0.20 * ring.coverage_score,
                0.0,
                1.0,
            )) if ring.valid else 0.0
            temporal_score = (
                self._tracker.temporal_score(candidate.center_px) if tracking else 1.0
            )
            texture_ratio = candidate.texture_std / max(
                self._config.white_board.max_texture_std, 1e-6
            )
            texture_penalty = max(
                0.0,
                min(
                    self._config.classical_scoring.max_texture_penalty,
                    (texture_ratio - 0.75)
                    / 0.25
                    * self._config.classical_scoring.max_texture_penalty,
                ),
            )
            jump_penalty = 0.0 if temporal_score >= 0.5 else (0.55 if tracking else 0.0)
            evidence.append(CandidateEvidence(
                white_score=candidate.white_occupancy,
                geometry_score=candidate.geometry_score,
                ring_score=ring_score,
                border_score=candidate.border_support,
                temporal_score=temporal_score,
                texture_penalty=texture_penalty,
                jump_penalty=jump_penalty,
            ))

        ranked = rank_candidates(evidence, self._config.classical_scoring, tracking=tracking)
        evaluations = tuple(
            self._evaluation(candidate, scored)
            for candidate, scored in zip(white_candidates, ranked.evaluations, strict=True)
        )
        if ranked.best is None:
            return _FullAcquisition(
                self._miss(
                    captured_ns,
                    source_sequence,
                    ranked.failure_reason or DetectionFailure.LOW_INTERNAL_STRUCTURE,
                ),
                evaluations,
                None,
                ring_results[0] if ring_results else None,
            )
        best_index = next(
            index for index, scored in enumerate(ranked.evaluations)
            if scored is ranked.best
        )
        candidate = white_candidates[best_index]
        solution = solutions[best_index]
        ring = ring_results[best_index]
        if not solution.homography_valid or solution.center_px is None:
            return _FullAcquisition(
                self._miss(captured_ns, source_sequence, DetectionFailure.A4_GEOMETRY_INVALID),
                evaluations,
                solution,
                ring,
            )
        scale = self._board_scale(candidate.corners_px)
        observation = TrackObservation(
            timestamp_ns=captured_ns,
            source_sequence=source_sequence,
            detected=True,
            source=ObservationSource.FULL_BOARD,
            center_px=solution.center_px,
            corners_px=candidate.corners_px,
            scale_px_per_mm=scale,
            confidence=ranked.best.combined_score,
        )
        return _FullAcquisition(observation, evaluations, solution, ring)

    def _acquire_partial(
        self,
        normalized: NormalizedFrame,
        *,
        captured_ns: int,
        source_sequence: int,
    ) -> tuple[TrackObservation, RingGeometryResult, tuple[int, int, int, int] | None, bool, bool]:
        if self._history is None:
            empty = detect_concentric_arcs(normalized.ring_edge_mask, self._config.rings)
            return self._miss(captured_ns, source_sequence, DetectionFailure.PARTIAL_HISTORY_REQUIRED), empty, None, False, False
        height, width = normalized.gray.shape
        roi = predicted_roi(
            self._history,
            timestamp_ns=captured_ns,
            image_size=(width, height),
        )
        ring = detect_concentric_arcs(
            normalized.ring_edge_mask,
            self._config.rings,
            expected_center_px=self._history.center_px,
            expected_scale_px_per_mm=self._history.scale_px_per_mm,
            roi_xyxy=roi,
        )
        if not ring.valid:
            fallback = detect_concentric_arcs(
                normalized.ring_edge_mask,
                self._config.rings,
                expected_scale_px_per_mm=self._history.scale_px_per_mm,
            )
            if fallback.valid or fallback.visible_arc_count > ring.visible_arc_count:
                ring = fallback
        white_center, white_confidence = self._local_white_region(normalized, roi)
        arc_confidence = float(np.clip(
            0.45 * ring.common_center_score
            + 0.35 * ring.ratio_score
            + 0.20 * ring.coverage_score,
            0.0,
            1.0,
        ))
        partial = fuse_partial_observation(
            history=self._history,
            timestamp_ns=captured_ns,
            image_size=(width, height),
            arc_center_px=ring.center_px,
            arc_count=ring.visible_arc_count,
            arc_confidence=arc_confidence,
            white_center_px=white_center,
            white_confidence=white_confidence,
            observed_scale_px_per_mm=ring.scale_px_per_mm,
        )
        if not partial.valid or partial.center_px is None:
            if ring.valid and ring.center_px is not None:
                outside = self._translated_history_outside(
                    ring.center_px,
                    image_size=(width, height),
                )
                observation = TrackObservation(
                    timestamp_ns=captured_ns,
                    source_sequence=source_sequence,
                    detected=True,
                    source=ObservationSource.CONCENTRIC_ARCS,
                    center_px=ring.center_px,
                    corners_px=None,
                    scale_px_per_mm=ring.scale_px_per_mm,
                    confidence=arc_confidence,
                )
                return observation, ring, roi, outside, outside
            return self._miss(captured_ns, source_sequence, partial.failure_reason), ring, roi, partial.near_image_edge, partial.partially_outside
        observation = TrackObservation(
            timestamp_ns=captured_ns,
            source_sequence=source_sequence,
            detected=True,
            source=partial.source,
            center_px=partial.center_px,
            corners_px=None,
            scale_px_per_mm=partial.scale_px_per_mm,
            confidence=partial.confidence,
        )
        return observation, ring, roi, partial.near_image_edge, partial.partially_outside

    def _remember(
        self,
        tracked: TrackedBoardResult,
        solution: BoardPlaneSolution | None,
        captured_ns: int,
    ) -> None:
        observation = tracked.observation
        if observation is None or observation.center_px is None:
            return
        if observation.source is ObservationSource.FULL_BOARD:
            corners = observation.corners_px or ()
        elif self._history is not None:
            dx = observation.center_px[0] - self._history.center_px[0]
            dy = observation.center_px[1] - self._history.center_px[1]
            corners = tuple((x + dx, y + dy) for x, y in self._history.corners_px)
        else:
            return
        if len(corners) != 4:
            return
        self._history = BoardHistory(
            center_px=observation.center_px,
            velocity_px_s=tracked.velocity_px_s or (0.0, 0.0),
            scale_px_per_mm=tracked.scale_px_per_mm or self._board_scale(corners),
            corners_px=corners,
            timestamp_ns=captured_ns,
        )

    @staticmethod
    def _local_white_region(
        normalized: NormalizedFrame,
        roi_xyxy: tuple[int, int, int, int],
    ) -> tuple[tuple[float, float] | None, float]:
        x0, y0, x1, y1 = roi_xyxy
        crop = normalized.white_mask[y0:y1, x0:x1]
        gray_crop = normalized.gray[y0:y1, x0:x1]
        if crop.size == 0 or float(np.std(gray_crop)) < 8.0:
            return None, 0.0
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(crop)
        if count <= 1:
            return None, 0.0
        index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = float(stats[index, cv2.CC_STAT_AREA])
        occupancy = area / max(float(crop.size), 1.0)
        if occupancy >= 0.98:
            return None, 0.0
        center = centroids[index]
        return (float(center[0] + x0), float(center[1] + y0)), float(np.clip(occupancy * 3.0, 0.0, 1.0))

    def _translated_history_outside(
        self,
        center_px: tuple[float, float],
        *,
        image_size: tuple[int, int],
    ) -> bool:
        if self._history is None:
            return False
        dx = center_px[0] - self._history.center_px[0]
        dy = center_px[1] - self._history.center_px[1]
        corners = np.asarray(
            [(x + dx, y + dy) for x, y in self._history.corners_px],
            np.float64,
        )
        width, height = image_size
        return bool(
            np.any(corners[:, 0] < 0.0)
            or np.any(corners[:, 0] >= width)
            or np.any(corners[:, 1] < 0.0)
            or np.any(corners[:, 1] >= height)
        )

    @staticmethod
    def _board_scale(corners: tuple[tuple[float, float], ...]) -> float:
        points = np.asarray(corners, np.float64)
        sides = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
        width = 0.5 * (sides[0] + sides[2])
        height = 0.5 * (sides[1] + sides[3])
        return float(0.5 * (width / 210.0 + height / 297.0))

    @staticmethod
    def _evaluation(candidate: WhiteBoardCandidate, scored) -> ClassicalCandidateEvaluation:
        return ClassicalCandidateEvaluation(
            corners_px=candidate.corners_px,
            center_px=candidate.center_px,
            white_score=scored.evidence.white_score,
            geometry_score=scored.evidence.geometry_score,
            ring_score=scored.evidence.ring_score,
            border_score=scored.evidence.border_score,
            temporal_score=scored.evidence.temporal_score,
            texture_penalty=scored.evidence.texture_penalty,
            jump_penalty=scored.evidence.jump_penalty,
            combined_score=scored.combined_score,
            accepted=scored.accepted,
            failure_reason=scored.failure_reason,
            rejection_reasons=scored.rejection_reasons,
        )

    @staticmethod
    def _miss(
        captured_ns: int,
        source_sequence: int,
        reason: DetectionFailure | None,
    ) -> TrackObservation:
        return TrackObservation(
            timestamp_ns=captured_ns,
            source_sequence=source_sequence,
            detected=False,
            source=ObservationSource.NONE,
            center_px=None,
            corners_px=None,
            confidence=0.0,
            failure_reason=reason,
        )
