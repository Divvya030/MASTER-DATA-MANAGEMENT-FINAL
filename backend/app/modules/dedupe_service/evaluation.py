"""
Blocking / candidate evaluation. Explicit about what each number means:
  * candidate reduction  = a PERFORMANCE/SCALABILITY metric (fraction of the
                           n*(n-1)/2 possible pairs Dedupe's blocking did NOT
                           have to score). It says nothing about accuracy.
  * candidate recall     = whether pairs KNOWN to be duplicates survived
                           blocking. Only computable where ground truth exists
                           (here: steward-confirmed MATCH labels); otherwise
                           reported as null, never guessed.
"""
from __future__ import annotations


def blocking_evaluation(total_possible_pairs: int, candidate_pairs: int, candidate_pair_keys: set,
                        known_match_pairs: list | None) -> dict:
    reduction = round(100 * (1 - candidate_pairs / total_possible_pairs), 2) if total_possible_pairs else None
    out = {
        "total_possible_pairs": total_possible_pairs,
        "candidate_pairs": candidate_pairs,
        "candidate_reduction_pct": reduction,
        "candidate_reduction_meaning": "performance/scalability metric -- NOT accuracy",
        "candidate_recall": None,
    }
    if known_match_pairs:
        found = sum(1 for a, b in known_match_pairs if (min(a, b), max(a, b)) in candidate_pair_keys)
        out["candidate_recall"] = {
            "known_match_pairs": len(known_match_pairs), "found_in_candidates": found,
            "recall": round(found / len(known_match_pairs), 4),
            "meaning": "share of steward-confirmed MATCH pairs that survived Dedupe blocking "
                       "(only over labelled pairs -- not a claim about all true duplicates)",
        }
    else:
        out["candidate_recall_note"] = "No ground truth (steward-confirmed MATCH pairs) for this dataset; candidate recall not computed."
    return out
