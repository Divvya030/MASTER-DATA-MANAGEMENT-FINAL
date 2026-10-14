"""
Matching: loads a TRAINED model, runs it over the real (uncorrupted)
STANDARDIZED records for a dataset, using Dedupe's own candidate-generation
(learned blocking predicates) and scoring -- never a custom fuzzy matcher.

Candidate generation and scoring are kept as two explicit, separately
measurable steps (per the brief), even though dedupe.Dedupe.partition()
would do both in one call -- this lets us report real, not fabricated,
candidate-reduction statistics.
"""
from __future__ import annotations
import io
from dataclasses import dataclass


@dataclass
class MatchResult:
    status: str  # COMPLETED | FAILED
    total_records: int = 0
    total_possible_pairs: int = 0
    candidate_pair_count: int = 0
    match_count: int = 0
    non_match_count: int = 0
    uncertain_count: int = 0        # scores in [review_threshold, threshold): sent to steward review, not auto-decided
    pairs: list | None = None       # [{record_id_a, record_id_b, score, decision}, ...]
    raw_scores: object = None       # the raw numpy structured Scores array, for clustering
    error: str | None = None


def run_matching(settings_bytes: bytes, records: dict, threshold: float = 0.5,
                 review_threshold: float | None = None,
                 distinct_threshold: float | None = None) -> MatchResult:
    """
    settings_bytes: a trained model's write_settings() output (from a
                    TrainingRun).
    records: {record_id: {field: value, ...}} -- ONLY the active MATCH
             fields, already coerced to strings, for the REAL (uncorrupted)
             dataset being matched. Caller is responsible for building this
             from the STANDARDIZED layer using the same active ER config
             the model was trained on.
    """
    from app.modules.dedupe_service.dedupe_factory import load_static

    try:
        static_deduper = load_static(settings_bytes)
    except Exception as e:  # noqa: BLE001
        return MatchResult(status="FAILED", error=f"Failed to load trained model: {type(e).__name__}: {e}")

    n = len(records)
    total_possible_pairs = n * (n - 1) // 2

    try:
        pairs = list(static_deduper.pairs(records))       # REAL Dedupe candidate generation (learned blocking)
        candidate_pair_count = len(pairs)
        scores = static_deduper.score(pairs) if pairs else None
    except Exception as e:  # noqa: BLE001
        return MatchResult(status="FAILED", error=f"Candidate generation/scoring failed: {type(e).__name__}: {e}")

    # Decision policy (thresholds come from the ACTIVE model / configuration,
    # never hardcoded):
    #   score >= match_threshold    -> MATCH     (same entity)
    #   score >= distinct_threshold -> UNSURE    (steward decides; never auto-merged)
    #   score <  distinct_threshold -> DISTINCT  (separate entities; record NOT discarded)
    # distinct_threshold defaults to review_threshold so previously validated
    # banding is preserved; "NON_MATCH" was renamed to "DISTINCT".
    if distinct_threshold is None:
        distinct_threshold = review_threshold

    match_pairs = []
    match_count = 0
    non_match_count = 0
    uncertain_count = 0
    if scores is not None:
        for row in scores:
            rid_a, rid_b = int(row["pairs"][0]), int(row["pairs"][1])
            score = float(row["score"])
            if score >= threshold:
                decision = "MATCH"
                match_count += 1
            elif distinct_threshold is not None and score >= distinct_threshold:
                decision = "UNSURE"
                uncertain_count += 1
            else:
                decision = "DISTINCT"
                non_match_count += 1
            match_pairs.append({
                "record_id_a": rid_a, "record_id_b": rid_b,
                "score": round(score, 6), "decision": decision,
            })

    return MatchResult(
        status="COMPLETED",
        total_records=n,
        total_possible_pairs=total_possible_pairs,
        candidate_pair_count=candidate_pair_count,
        match_count=match_count,
        non_match_count=non_match_count,
        uncertain_count=uncertain_count,
        pairs=match_pairs,
        raw_scores=scores,
    )
