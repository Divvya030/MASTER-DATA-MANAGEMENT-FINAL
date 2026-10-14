"""Retraining eligibility + V1-vs-candidate comparison gate. No training happens here.

Constants (application configuration, deliberately not a UI). Only REAL_STEWARD labels count toward
eligibility: AUTO, SYNTHETIC and ASSISTANT_REVIEWED evidence never do.
"""
from __future__ import annotations
from sqlalchemy.orm import Session
from app.db.models import LabeledPair

MIN_NEW_REAL_LABELS = 200
MIN_MATCH = 60
MIN_DISTINCT = 60
MIN_HARD_NEGATIVE_DISTINCT = 30      # DISTINCT labels whose model score was >= the review threshold when queued


def eligibility(db: Session, domain: str, review_threshold: float | None = None) -> dict:
    rows = (db.query(LabeledPair).filter(LabeledPair.domain == domain, LabeledPair.label_source == "REAL_STEWARD",
                                         LabeledPair.label.in_(["MATCH", "DISTINCT"])).all())
    m = sum(1 for r in rows if r.label == "MATCH"); d = sum(1 for r in rows if r.label == "DISTINCT")
    hard = sum(1 for r in rows if r.label == "DISTINCT" and r.model_score is not None
               and (review_threshold is None or r.model_score >= review_threshold))
    checks = {"real_steward_labels": (len(rows), MIN_NEW_REAL_LABELS), "match_labels": (m, MIN_MATCH),
              "distinct_labels": (d, MIN_DISTINCT), "hard_negative_labels": (hard, MIN_HARD_NEGATIVE_DISTINCT)}
    unmet = [k for k, (have, need) in checks.items() if have < need]
    return {"domain": domain, "eligible": not unmet, "unmet": unmet,
            "checks": {k: {"have": h, "need": n} for k, (h, n) in checks.items()},
            "note": "Only REAL_STEWARD (human) labels count. Retraining is never triggered automatically."}


def compare_models(old: dict, new: dict, max_extra_false_merge_rate: float = 0.0) -> dict:
    """Gate: a candidate may replace the ACTIVE model only if it does not raise the false-merge rate
    (precision must not fall by more than `max_extra_false_merge_rate`) and does not lower recall by more
    than 5 points. Inputs: {'precision','recall','false_positives'} from the SAME held-out real labels."""
    reasons = []
    if new["precision"] < old["precision"] - max_extra_false_merge_rate:
        reasons.append("precision (false-merge safety) is lower than the active model")
    if new["false_positives"] > old["false_positives"]:
        reasons.append("more false positives than the active model")
    if new["recall"] < old["recall"] - 0.05:
        reasons.append("recall dropped by more than 5 points")
    return {"approved": not reasons, "reasons": reasons, "old": old, "new": new}
