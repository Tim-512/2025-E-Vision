from __future__ import annotations

from dataclasses import dataclass, replace
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
from ev_vision.detection.ring_first import RingQuality, classify_ring, translate_ring_result
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
    ring_ms: float = 0.0
    white_board_ms: float = 0.0


class ClassicalBoardDetector:
    model_state = "READY"
    model_backend = "classical"
    model_path = None

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._tracker = BoardTracker(config.tracking)
        self._history: BoardHistory | None = None
        self._board = BoardConfig()
        self._tracking_frame_count = 0

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
        acquisition = self._tracker.latest.state in {
            TrackingState.SEARCHING,
            TrackingState.CONFIRMING,
            TrackingState.LOST,
        }

        crop_started = time.perf_counter()
        processing_image = image
        offset_xy = (0, 0)
        roi_xyxy: tuple[int, int, int, int] | None = None
        roi_used = False
        if not acquisition:
            candidate_roi = self._tracking_roi(captured_ns, image.shape[:2])
            if candidate_roi is not None:
                x0, y0, x1, y1 = candidate_roi
                height, width = image.shape[:2]
                if x1 > x0 and y1 > y0 and (x0, y0, x1, y1) != (0, 0, width, height):
                    processing_image = image[y0:y1, x0:x1]
                    offset_xy = (x0, y0)
                    roi_xyxy = candidate_roi
                    roi_used = True
        crop_ms = (time.perf_counter() - crop_started) * 1000.0

        normalization_started = time.perf_counter()
        normalized = normalize_frame(
            processing_image,
            self._config.normalization,
            mask_radius_px=self._config.rings.saturation_mask_radius_px,
        )
        normalization_ms = (time.perf_counter() - normalization_started) * 1000.0

        if acquisition:
            if update_tracker:
                self._tracking_frame_count = 0
            tracking_frame_number = 0
        else:
            tracking_frame_number = self._tracking_frame_count + 1
            if update_tracker:
                self._tracking_frame_count = tracking_frame_number
        interval = max(1, self._config.ring_first.white_board_interval_frames)
        white_board_ran = bool(
            acquisition
            or not self._config.ring_first.enabled
            or tracking_frame_number % interval == 0
        )

        ring_result: RingGeometryResult | None = None
        solution: BoardPlaneSolution | None = None
        candidates: tuple[ClassicalCandidateEvaluation, ...] = ()
        near_image_edge = False
        partially_outside = False
        ring_ms = 0.0
        white_board_ms = 0.0

        detection_started = time.perf_counter()
        if acquisition or white_board_ran:
            full = self._acquire_full_board(
                normalized,
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                tracking=not acquisition,
                offset_xy=offset_xy,
            )
        else:
            full = self._acquire_rings_only(
                normalized,
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                offset_xy=offset_xy,
                failure_reason=DetectionFailure.LOW_INTERNAL_STRUCTURE,
            )
        observation = full.observation
        candidates = full.candidates
        solution = full.solution
        ring_result = full.ring_result
        ring_ms += full.ring_ms
        white_board_ms += full.white_board_ms

        if not acquisition and not observation.detected and roi_used:
            # A target can move outside the predicted crop. Pay the full-frame cost
            # only after the cheap ROI path misses so normal tracking stays fast.
            fallback_normalization_started = time.perf_counter()
            fallback_normalized = normalize_frame(
                image,
                self._config.normalization,
                mask_radius_px=self._config.rings.saturation_mask_radius_px,
            )
            normalization_ms += (
                time.perf_counter() - fallback_normalization_started
            ) * 1000.0
            fallback = self._acquire_rings_only(
                fallback_normalized,
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                failure_reason=observation.failure_reason or DetectionFailure.LOW_INTERNAL_STRUCTURE,
            )
            ring_ms += fallback.ring_ms
            if fallback.observation.detected:
                observation = fallback.observation
                ring_result = fallback.ring_result
                normalized = fallback_normalized
                offset_xy = (0, 0)
                if observation.center_px is not None:
                    outside = self._translated_history_outside(
                        observation.center_px,
                        image_size=(image.shape[1], image.shape[0]),
                    )
                    near_image_edge = outside
                    partially_outside = outside
            else:
                ring_result = fallback.ring_result

        if not acquisition and not observation.detected:
            observation, ring_result, near_image_edge, partially_outside = self._acquire_partial(
                normalized,
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                ring=ring_result,
                offset_xy=offset_xy,
                full_image_shape=image.shape[:2],
                allow_white_region=white_board_ran,
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

        debug_ring = (
            translate_ring_result(ring_result, (-offset_xy[0], -offset_xy[1]))
            if ring_result is not None and roi_used
            else ring_result
        )
        debug_candidates = (
            tuple(self._translate_evaluation(item, (-offset_xy[0], -offset_xy[1])) for item in candidates)
            if roi_used
            else candidates
        )
        debug_images = (
            render_debug_images(
                normalized,
                ring_result=debug_ring,
                candidates=debug_candidates,
                roi_xyxy=None,
            )
            if include_debug
            else {}
        )
        total_ms = (time.perf_counter() - started) * 1000.0
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
                "crop_ms": crop_ms,
                "normalization_ms": normalization_ms,
                "ring_ms": ring_ms,
                "white_board_ms": white_board_ms,
                "detection_ms": detection_ms,
                "total_ms": total_ms,
                "roi_used": 1.0 if roi_used else 0.0,
                "white_board_ran": 1.0 if white_board_ran else 0.0,
                "normalization": normalization_ms,
                "detection": detection_ms,
                "total": total_ms,
            },
            debug_images=debug_images,
        )

    def reset(self) -> None:
        self._tracker.reset()
        self._history = None
        self._tracking_frame_count = 0

    def apply_config(self, config: DetectionConfig) -> None:
        self._config = config
        self._tracker.apply_config(config.tracking)
        self._tracking_frame_count = 0

    def reload_model(self) -> None:
        return None

    def _tracking_roi(
        self,
        timestamp_ns: int,
        image_shape: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        height, width = image_shape
        if self._history is not None:
            return predicted_roi(
                self._history,
                timestamp_ns=timestamp_ns,
                image_size=(width, height),
            )
        center = self._tracker.latest.predicted_center_px
        if center is None and self._tracker.latest.observation is not None:
            center = self._tracker.latest.observation.center_px
        scale = self._tracker.latest.scale_px_per_mm
        if center is None or scale is None or scale <= 0.0:
            return None
        # The outer ring radius is about 100 mm; 15% padding keeps the crop
        # small while retaining the complete ring geometry.
        half_extent = max(64.0, 115.0 * scale)
        x0 = max(0, int(np.floor(center[0] - half_extent)))
        y0 = max(0, int(np.floor(center[1] - half_extent)))
        x1 = min(width, int(np.ceil(center[0] + half_extent)))
        y1 = min(height, int(np.ceil(center[1] + half_extent)))
        return (x0, y0, x1, y1)

    def _acquire_full_board(
        self,
        normalized: NormalizedFrame,
        *,
        captured_ns: int,
        source_sequence: int,
        tracking: bool,
        offset_xy: tuple[int, int] = (0, 0),
    ) -> _FullAcquisition:
        white_started = time.perf_counter()
        local_candidates = find_white_board_candidates(
            normalized,
            self._config.white_board,
            roi_xyxy=None,
        )
        white_board_ms = (time.perf_counter() - white_started) * 1000.0
        if not local_candidates:
            rings_only = self._acquire_rings_only(
                normalized,
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                offset_xy=offset_xy,
                failure_reason=DetectionFailure.NO_WHITE_CANDIDATE,
            )
            return replace(rings_only, white_board_ms=white_board_ms)

        candidates = tuple(
            self._translate_white_candidate(candidate, offset_xy)
            for candidate in local_candidates
        )
        evidence: list[CandidateEvidence] = []
        ring_results: list[RingGeometryResult] = []
        solutions: list[BoardPlaneSolution] = []
        ring_ms = 0.0
        for local_candidate, candidate in zip(local_candidates, candidates, strict=True):
            ring_started = time.perf_counter()
            local_ring = detect_concentric_arcs(
                normalized.ring_edge_mask,
                self._config.rings,
                expected_center_px=local_candidate.center_px,
                roi_xyxy=local_candidate.bbox_xyxy,
            )
            ring_ms += (time.perf_counter() - ring_started) * 1000.0
            ring = translate_ring_result(local_ring, offset_xy)
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
                    (texture_ratio - 0.75) / 0.25
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
            for candidate, scored in zip(candidates, ranked.evaluations, strict=True)
        )
        if ranked.best is None:
            acceptable = [
                ring for ring in ring_results
                if classify_ring(ring, self._config.ring_first) is not RingQuality.REJECTED
            ]
            if acceptable:
                ring = max(
                    acceptable,
                    key=lambda item: (
                        item.visible_arc_count,
                        item.common_center_score + item.ratio_score + item.coverage_score,
                    ),
                )
                observation = self._ring_observation(
                    ring, captured_ns=captured_ns, source_sequence=source_sequence
                )
                return _FullAcquisition(
                    observation, evaluations, None, ring, ring_ms, white_board_ms
                )
            rings_only = self._acquire_rings_only(
                normalized,
                captured_ns=captured_ns,
                source_sequence=source_sequence,
                offset_xy=offset_xy,
                failure_reason=(ranked.failure_reason or DetectionFailure.LOW_INTERNAL_STRUCTURE),
            )
            return _FullAcquisition(
                rings_only.observation, evaluations, rings_only.solution,
                rings_only.ring_result, ring_ms + rings_only.ring_ms, white_board_ms
            )
        best_index = next(
            index for index, scored in enumerate(ranked.evaluations) if scored is ranked.best
        )
        candidate = candidates[best_index]
        solution = solutions[best_index]
        ring = ring_results[best_index]
        if not solution.homography_valid or solution.center_px is None:
            return _FullAcquisition(
                self._miss(captured_ns, source_sequence, DetectionFailure.A4_GEOMETRY_INVALID),
                evaluations, solution, ring, ring_ms, white_board_ms
            )
        scale = self._board_scale(candidate.corners_px)
        observation = TrackObservation(
            timestamp_ns=captured_ns, source_sequence=source_sequence, detected=True,
            source=ObservationSource.FULL_BOARD, center_px=solution.center_px,
            corners_px=candidate.corners_px, scale_px_per_mm=scale,
            confidence=ranked.best.combined_score,
        )
        return _FullAcquisition(
            observation, evaluations, solution, ring, ring_ms, white_board_ms
        )

    def _acquire_rings_only(
        self,
        normalized: NormalizedFrame,
        *,
        captured_ns: int,
        source_sequence: int,
        offset_xy: tuple[int, int] = (0, 0),
        failure_reason: DetectionFailure = DetectionFailure.LOW_INTERNAL_STRUCTURE,
    ) -> _FullAcquisition:
        expected_center = None
        expected_scale = None
        if self._history is not None:
            expected_center = self._history.center_px
            expected_scale = self._history.scale_px_per_mm
        else:
            expected_center = self._tracker.latest.predicted_center_px
            if expected_center is None and self._tracker.latest.observation is not None:
                expected_center = self._tracker.latest.observation.center_px
            expected_scale = self._tracker.latest.scale_px_per_mm
        if expected_center is not None:
            expected_center = (
                expected_center[0] - offset_xy[0], expected_center[1] - offset_xy[1]
            )
        ring_started = time.perf_counter()
        local_ring = detect_concentric_arcs(
            normalized.ring_edge_mask, self._config.rings,
            expected_center_px=expected_center, expected_scale_px_per_mm=expected_scale,
            roi_xyxy=None,
        )
        ring_ms = (time.perf_counter() - ring_started) * 1000.0
        ring = translate_ring_result(local_ring, offset_xy)
        quality = classify_ring(ring, self._config.ring_first)
        if quality is RingQuality.REJECTED or ring.center_px is None:
            return _FullAcquisition(
                self._miss(captured_ns, source_sequence, failure_reason),
                (), None, ring, ring_ms, 0.0
            )
        observation = self._ring_observation(
            ring, captured_ns=captured_ns, source_sequence=source_sequence
        )
        return _FullAcquisition(observation, (), None, ring, ring_ms, 0.0)

    @staticmethod
    def _ring_observation(
        ring: RingGeometryResult,
        *,
        captured_ns: int,
        source_sequence: int,
    ) -> TrackObservation:
        confidence = float(np.clip(
            0.45 * ring.common_center_score
            + 0.35 * ring.ratio_score
            + 0.20 * ring.coverage_score,
            0.0, 1.0,
        ))
        return TrackObservation(
            timestamp_ns=captured_ns, source_sequence=source_sequence, detected=True,
            source=ObservationSource.CONCENTRIC_ARCS, center_px=ring.center_px,
            corners_px=None, scale_px_per_mm=ring.scale_px_per_mm,
            confidence=confidence, acquisition_eligible=True,
        )

    def _acquire_partial(
        self,
        normalized: NormalizedFrame,
        *,
        captured_ns: int,
        source_sequence: int,
        ring: RingGeometryResult | None,
        offset_xy: tuple[int, int],
        full_image_shape: tuple[int, int],
        allow_white_region: bool,
    ) -> tuple[TrackObservation, RingGeometryResult | None, bool, bool]:
        if self._history is None or ring is None:
            return (
                self._miss(captured_ns, source_sequence, DetectionFailure.PARTIAL_HISTORY_REQUIRED),
                ring, False, False,
            )
        height, width = full_image_shape
        if allow_white_region:
            local_roi = (0, 0, normalized.gray.shape[1], normalized.gray.shape[0])
            white_center, white_confidence = self._local_white_region(normalized, local_roi)
            if white_center is not None:
                white_center = (white_center[0] + offset_xy[0], white_center[1] + offset_xy[1])
        else:
            white_center, white_confidence = None, 0.0
        arc_confidence = float(np.clip(
            0.45 * ring.common_center_score + 0.35 * ring.ratio_score
            + 0.20 * ring.coverage_score, 0.0, 1.0,
        ))
        partial = fuse_partial_observation(
            history=self._history, timestamp_ns=captured_ns, image_size=(width, height),
            arc_center_px=ring.center_px, arc_count=ring.visible_arc_count,
            arc_confidence=arc_confidence, white_center_px=white_center,
            white_confidence=white_confidence, observed_scale_px_per_mm=ring.scale_px_per_mm,
        )
        if not partial.valid or partial.center_px is None:
            if ring.valid and ring.center_px is not None:
                outside = self._translated_history_outside(ring.center_px, image_size=(width, height))
                observation = TrackObservation(
                    timestamp_ns=captured_ns, source_sequence=source_sequence, detected=True,
                    source=ObservationSource.CONCENTRIC_ARCS, center_px=ring.center_px,
                    corners_px=None, scale_px_per_mm=ring.scale_px_per_mm,
                    confidence=arc_confidence,
                )
                return observation, ring, outside, outside
            return (self._miss(captured_ns, source_sequence, partial.failure_reason), ring,
                    partial.near_image_edge, partial.partially_outside)
        observation = TrackObservation(
            timestamp_ns=captured_ns, source_sequence=source_sequence, detected=True,
            source=partial.source, center_px=partial.center_px, corners_px=None,
            scale_px_per_mm=partial.scale_px_per_mm, confidence=partial.confidence,
        )
        return observation, ring, partial.near_image_edge, partial.partially_outside

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
    def _translate_white_candidate(
        candidate: WhiteBoardCandidate,
        offset_xy: tuple[int, int],
    ) -> WhiteBoardCandidate:
        offset_x, offset_y = offset_xy
        if offset_x == 0 and offset_y == 0:
            return candidate
        contour = candidate.contour + np.asarray(
            [[[offset_x, offset_y]]], dtype=candidate.contour.dtype
        )
        return replace(
            candidate, contour=contour,
            corners_px=tuple((x + offset_x, y + offset_y) for x, y in candidate.corners_px),
            center_px=(candidate.center_px[0] + offset_x, candidate.center_px[1] + offset_y),
            bbox_xyxy=(candidate.bbox_xyxy[0] + offset_x, candidate.bbox_xyxy[1] + offset_y,
                       candidate.bbox_xyxy[2] + offset_x, candidate.bbox_xyxy[3] + offset_y),
        )

    @staticmethod
    def _translate_evaluation(
        evaluation: ClassicalCandidateEvaluation,
        offset_xy: tuple[int, int],
    ) -> ClassicalCandidateEvaluation:
        offset_x, offset_y = offset_xy
        return replace(
            evaluation,
            corners_px=tuple((x + offset_x, y + offset_y) for x, y in evaluation.corners_px),
            center_px=(evaluation.center_px[0] + offset_x, evaluation.center_px[1] + offset_y),
        )

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
