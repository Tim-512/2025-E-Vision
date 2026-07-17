from __future__ import annotations

from dataclasses import dataclass, replace

from ev_vision.config import ClassicalScoringConfig
from ev_vision.detection.failures import DetectionFailure


@dataclass(frozen=True)
class CandidateEvidence:
    white_score: float
    geometry_score: float
    ring_score: float
    border_score: float
    temporal_score: float
    texture_penalty: float
    jump_penalty: float
    scale_jump_fraction: float = 0.0


@dataclass(frozen=True)
class ScoredCandidate:
    evidence: CandidateEvidence
    combined_score: float
    accepted: bool
    failure_reason: DetectionFailure | None = None
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankedCandidates:
    best: ScoredCandidate | None
    evaluations: tuple[ScoredCandidate, ...]
    failure_reason: DetectionFailure | None = None


def combined_score(item: CandidateEvidence, config: ClassicalScoringConfig) -> float:
    positive = (
        config.white_weight * item.white_score
        + config.geometry_weight * item.geometry_score
        + config.ring_weight * item.ring_score
        + config.border_weight * item.border_score
        + config.temporal_weight * item.temporal_score
    )
    return max(
        0.0,
        min(1.0, positive - item.texture_penalty - item.jump_penalty),
    )


def _evaluate(
    item: CandidateEvidence,
    config: ClassicalScoringConfig,
    *,
    tracking: bool,
) -> ScoredCandidate:
    score = combined_score(item, config)
    reason: DetectionFailure | None = None
    rejection_reasons: list[str] = []

    if tracking and item.jump_penalty >= 0.5:
        reason = DetectionFailure.EXCESSIVE_POSITION_JUMP
        rejection_reasons.append(reason.value)
    elif tracking and item.scale_jump_fraction > 0.30:
        reason = DetectionFailure.EXCESSIVE_SCALE_JUMP
        rejection_reasons.append(reason.value)
    elif not tracking and item.ring_score < 0.50:
        reason = DetectionFailure.LOW_INTERNAL_STRUCTURE
        rejection_reasons.append(reason.value)
    else:
        threshold = (
            config.tracking_threshold if tracking else config.acquisition_threshold
        )
        if score < threshold:
            reason = DetectionFailure.LOW_INTERNAL_STRUCTURE
            rejection_reasons.append(f"SCORE_BELOW_{threshold:.3f}")

    return ScoredCandidate(
        evidence=item,
        combined_score=score,
        accepted=reason is None,
        failure_reason=reason,
        rejection_reasons=tuple(rejection_reasons),
    )


def rank_candidates(
    items: list[CandidateEvidence] | tuple[CandidateEvidence, ...],
    config: ClassicalScoringConfig,
    *,
    tracking: bool,
) -> RankedCandidates:
    evaluations = tuple(
        sorted(
            (_evaluate(item, config, tracking=tracking) for item in items),
            key=lambda evaluation: evaluation.combined_score,
            reverse=True,
        )
    )
    accepted = [evaluation for evaluation in evaluations if evaluation.accepted]
    if not accepted:
        reason = evaluations[0].failure_reason if evaluations else None
        return RankedCandidates(None, evaluations, reason)
    if (
        len(accepted) > 1
        and accepted[0].combined_score - accepted[1].combined_score
        < config.ambiguity_margin
    ):
        ambiguous = tuple(
            replace(
                evaluation,
                accepted=False,
                failure_reason=DetectionFailure.AMBIGUOUS_CANDIDATES,
                rejection_reasons=(DetectionFailure.AMBIGUOUS_CANDIDATES.value,),
            )
            if evaluation in accepted[:2]
            else evaluation
            for evaluation in evaluations
        )
        return RankedCandidates(
            None, ambiguous, DetectionFailure.AMBIGUOUS_CANDIDATES
        )
    return RankedCandidates(accepted[0], evaluations, None)
