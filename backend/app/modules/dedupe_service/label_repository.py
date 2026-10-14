"""Reusable labelled-pair repository (DB access only; no ER logic)."""
from __future__ import annotations
import datetime as dt
import uuid
from typing import Optional
from sqlalchemy.orm import Session

from app.db.models import LabeledPair

VALID_LABELS = {"MATCH", "DISTINCT", "UNSURE"}


def pair_key(a, b):
    a, b = str(a), str(b)
    return (min(a, b), max(a, b))


def existing_keys(db: Session, domain: str, dataset_id: Optional[str]) -> set:
    q = db.query(LabeledPair).filter(LabeledPair.domain == domain)
    if dataset_id is not None:
        q = q.filter(LabeledPair.dataset_id == dataset_id)
    return {pair_key(p.record_id_a, p.record_id_b) for p in q.all()}


def add_pair(db: Session, *, domain: str, dataset_id: Optional[str], record_id_a, record_id_b, values_a: dict,
             values_b: dict, display_a=None, display_b=None, strategy=None, evidence=None, model_score=None,
             label: Optional[str] = None, label_source: str = "REAL_STEWARD", labeler: Optional[str] = None,
             reason: Optional[str] = None, er_config_version: Optional[str] = None) -> LabeledPair:
    p = LabeledPair(
        pair_id=f"LP-{uuid.uuid4().hex[:12]}", domain=domain, dataset_id=dataset_id,
        record_id_a=str(record_id_a), record_id_b=str(record_id_b), values_a=values_a, values_b=values_b,
        display_a=display_a, display_b=display_b, strategy=strategy, evidence=evidence, model_score=model_score,
        label=label, label_source=label_source, labeler=labeler, reason=reason, er_config_version=er_config_version,
        labeled_at=dt.datetime.utcnow() if label else None,
    )
    db.add(p)
    db.commit()
    return p


def set_label(db: Session, pair: LabeledPair, label: str, labeler: str, reason: Optional[str] = None) -> LabeledPair:
    if label not in VALID_LABELS:
        raise ValueError(f"label must be one of {sorted(VALID_LABELS)}")
    if not labeler or not labeler.strip():
        raise ValueError("labeler (steward identity) is required")
    pair.label, pair.labeler, pair.reason = label, labeler.strip(), reason
    pair.labeled_at = dt.datetime.utcnow()
    db.commit()
    return pair


def pair_to_dict(p: LabeledPair) -> dict:
    return {"pair_id": p.pair_id, "domain": p.domain, "dataset_id": p.dataset_id,
            "record_id_a": p.record_id_a, "record_id_b": p.record_id_b,
            "values_a": p.values_a, "values_b": p.values_b, "display_a": p.display_a, "display_b": p.display_b,
            "strategy": p.strategy, "evidence": p.evidence, "model_score": p.model_score,
            "label": p.label, "label_source": p.label_source, "labeler": p.labeler, "reason": p.reason,
            "er_config_version": p.er_config_version,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "labeled_at": p.labeled_at.isoformat() if p.labeled_at else None}


def usable_real_labels(db: Session, domain: str) -> list:
    """Steward-confirmed MATCH/DISTINCT labels for the domain (UNSURE and
    pending are excluded from supervised training/evaluation)."""
    rows = (db.query(LabeledPair).filter(LabeledPair.domain == domain, LabeledPair.label_source == "REAL_STEWARD",
                                         LabeledPair.label.in_(["MATCH", "DISTINCT"])).all())
    return [pair_to_dict(r) for r in rows]


def label_summary(db: Session, domain: str) -> dict:
    rows = db.query(LabeledPair).filter(LabeledPair.domain == domain).all()
    s = {"domain": domain, "total": len(rows), "pending": 0, "MATCH": 0, "DISTINCT": 0, "UNSURE": 0,
         "by_source": {}}
    for r in rows:
        if r.label is None:
            s["pending"] += 1
        else:
            s[r.label] += 1
            src = s["by_source"].setdefault(r.label_source, {"MATCH": 0, "DISTINCT": 0, "UNSURE": 0})
            src[r.label] += 1
    s["usable_for_training"] = s["MATCH"] + s["DISTINCT"]
    s["note"] = ("UNSURE labels are excluded from supervised training. The number of labels needed is "
                 "dataset-dependent; no fixed count is claimed to give good accuracy.")
    return s
