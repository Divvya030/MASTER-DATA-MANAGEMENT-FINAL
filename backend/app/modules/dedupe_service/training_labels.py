"""Which labelled pairs in the repository will a training run actually use?

One function decides this for BOTH the training endpoint and the "label readiness" preview
shown on the Training screen, so what the UI promises is exactly what training does.

A labelled pair is used for supervised training when
  * its label is MATCH or DISTINCT (UNSURE and not-yet-labelled pairs are never trained on),
  * its label source is REAL_STEWARD (a person decided it), and
  * both stored record snapshots still contain every MATCH field of the model being trained
    (a label made under a different field configuration cannot teach this model anything).
Pairs that fail the last test are reported as excluded -- never silently dropped.
"""
from __future__ import annotations
from sqlalchemy.orm import Session

from app.db.models import LabeledPair
from app.modules.dedupe_service import label_repository as label_repo
from app.modules.dedupe_service.real_labels import split_real_labels


def select_training_labels(db: Session, domain: str, match_fields: list):
    """-> (usable, excluded) where excluded = [{pair_id, missing_fields}] (MATCH/DISTINCT pairs only)."""
    usable, excluded = [], []
    for l in label_repo.usable_real_labels(db, domain):
        va, vb = l.get("values_a") or {}, l.get("values_b") or {}
        missing = [f for f in match_fields if f not in va or f not in vb]
        if missing:
            excluded.append({"pair_id": l["pair_id"], "missing_fields": missing})
        else:
            usable.append(l)
    return usable, excluded


def _class_counts(labels: list) -> dict:
    m = sum(1 for l in labels if l["label"] == "MATCH")
    return {"match": m, "distinct": len(labels) - m}


def label_training_readiness(db: Session, domain: str, match_fields: list, seed: int = 42) -> dict:
    from app.modules.dedupe_service.training_service import (
        MIN_PAIRS_PER_CLASS, MIN_REAL_PER_CLASS, MIN_CALIBRATION_PER_CLASS)

    rows = db.query(LabeledPair).filter(LabeledPair.domain == domain).all()
    labelled = [r for r in rows if r.label is not None]
    usable, excluded = select_training_labels(db, domain, match_fields)
    split = split_real_labels(usable, seed=seed)
    train, val, test = (_class_counts(split[k]) for k in ("train", "validation", "test"))

    origin = {"steward_review": 0, "labeling_screen": 0}
    for l in usable:
        origin["steward_review" if l.get("strategy") == "REVIEW_DECISION" else "labeling_screen"] += 1

    sufficient = train["match"] >= MIN_REAL_PER_CLASS and train["distinct"] >= MIN_REAL_PER_CLASS
    real_only_ok = train["match"] >= MIN_PAIRS_PER_CLASS and train["distinct"] >= MIN_PAIRS_PER_CLASS
    can_calibrate = val["match"] >= MIN_CALIBRATION_PER_CLASS and val["distinct"] >= MIN_CALIBRATION_PER_CLASS

    notes = []
    if not usable:
        notes.append("No usable steward labels yet. Training will rely entirely on synthetic augmentation "
                     "(mode AUTO/ALWAYS) or fail (mode NEVER).")
    elif not real_only_ok:
        notes.append(f"Only {train['match']} MATCH / {train['distinct']} DISTINCT land in the training split; "
                     f"'Never' (real labels only) needs at least {MIN_PAIRS_PER_CLASS} of each.")
    if usable and not sufficient:
        notes.append(f"Fewer than {MIN_REAL_PER_CLASS} real labels per class in the training split, so AUTO mode "
                     "will ADD synthetic pairs next to your real ones.")
    if usable and not can_calibrate:
        notes.append(f"Fewer than {MIN_CALIBRATION_PER_CLASS} MATCH and DISTINCT labels in the validation split: the "
                     "threshold stays PROVISIONAL (0.5) unless you set one.")
    if excluded:
        notes.append(f"{len(excluded)} labelled pair(s) were made without all of this model's MATCH fields and are "
                     "excluded.")
    return {
        "domain": domain, "match_fields": match_fields, "seed": seed,
        "repository": {
            "labelled": len(labelled),
            "match": sum(1 for r in labelled if r.label == "MATCH"),
            "distinct": sum(1 for r in labelled if r.label == "DISTINCT"),
            "unsure_excluded": sum(1 for r in labelled if r.label == "UNSURE"),
            "awaiting_label": len(rows) - len(labelled),
        },
        "usable": {"total": len(usable), **_class_counts(usable), "origin": origin},
        "excluded_missing_match_fields": excluded[:20],
        "excluded_missing_match_fields_count": len(excluded),
        "split_preview": {"train": train, "validation": val, "test": test},
        "thresholds": {"min_per_class_to_skip_synthetic": MIN_REAL_PER_CLASS,
                       "min_per_class_real_only": MIN_PAIRS_PER_CLASS,
                       "min_validation_per_class_to_calibrate": MIN_CALIBRATION_PER_CLASS},
        "verdict": {"real_labels_sufficient": sufficient, "real_only_training_possible": real_only_ok,
                    "threshold_can_be_calibrated": can_calibrate},
        "notes": notes,
    }
