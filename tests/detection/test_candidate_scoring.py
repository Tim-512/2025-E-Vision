from __future__ import annotations

import pytest

from ev_vision.config import ClassicalScoringConfig
from ev_vision.detection.candidate_scoring import (
    CandidateEvidence,
    combined_score,
    rank_candidates,
)
from ev_vision.detection.failures import DetectionFailure


def evidence(**changes):
    values = dict(
        white_score=0.90,
        geometry_score=0.85,
        ring_score=0.90,
        border_score=0.40,
        temporal_score=0.0,
        texture_penalty=0.02,
        jump_penalty=0.0,
    )
    values.update(changes)
    return CandidateEvidence(**values)


def test_search_accepts_without_history_but_requires_ring_identity():
    assert rank_candidates(
        [evidence()], ClassicalScoringConfig(), tracking=False
    ).best
    rejected = rank_candidates(
        [evidence(ring_score=0.0)], ClassicalScoringConfig(), tracking=False
    )
    assert rejected.best is None
    assert rejected.evaluations[0].failure_reason == DetectionFailure.LOW_INTERNAL_STRUCTURE


def test_tracking_rejects_large_jump():
    ranked = rank_candidates(
        [evidence(jump_penalty=0.8)], ClassicalScoringConfig(), tracking=True
    )
    assert ranked.best is None
    assert ranked.evaluations[0].failure_reason == DetectionFailure.EXCESSIVE_POSITION_JUMP


def test_tracking_rejects_hard_scale_jump():
    ranked = rank_candidates(
        [evidence(scale_jump_fraction=0.55)],
        ClassicalScoringConfig(),
        tracking=True,
    )
    assert ranked.best is None
    assert ranked.evaluations[0].failure_reason == DetectionFailure.EXCESSIVE_SCALE_JUMP


def test_near_equal_candidates_are_ambiguous():
    ranked = rank_candidates(
        [evidence(ring_score=0.90), evidence(ring_score=0.88)],
        ClassicalScoringConfig(ambiguity_margin=0.08),
        tracking=False,
    )
    assert ranked.best is None
    assert ranked.failure_reason == DetectionFailure.AMBIGUOUS_CANDIDATES


def test_combined_score_is_clamped_after_penalties():
    assert combined_score(evidence(texture_penalty=2.0), ClassicalScoringConfig()) == 0.0
    assert combined_score(evidence(), ClassicalScoringConfig()) == pytest.approx(0.685)

