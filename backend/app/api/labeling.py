"""Human-in-the-loop labelling workflow: candidate generation -> steward
MATCH / DISTINCT / UNSURE -> reusable labelled-pair repository."""
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import Dataset, LabeledPair, ActiveModel, TrainingRun
from app.modules.dedupe_service.config_loading import er_config_version, load_active_er_roles
from app.modules.dedupe_service.record_loading import records_from_standardized, mapping_dict_for_dataset
from app.modules.dedupe_service.labeling_service import generate_label_candidates, identifier_evidence
from app.modules.dedupe_service import label_repository as repo
from app.modules.dedupe_service.field_definitions import coerce_record
from app.modules.dedupe_service.training_labels import label_training_readiness

router = APIRouter(prefix="/api", tags=["labeling"])


def _ds(db, dataset_id):
    ds = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if ds is None:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    return ds


class CandidateRequest(BaseModel):
    n: int = 20
    seed: int = 42
    strategy: str = "AUTO"        # AUTO | BOOTSTRAP | MODEL


@router.post("/datasets/{dataset_id}/er/labeling/candidates")
def create_label_candidates(dataset_id: str, req: CandidateRequest = CandidateRequest(), db: Session = Depends(get_db)):
    ds = _ds(db, dataset_id)
    if not (1 <= req.n <= 200):
        raise HTTPException(422, "n must be between 1 and 200 (a practical labelling batch)")
    roles, rows = load_active_er_roles(db, dataset_id)
    if not rows or not roles.match_fields:
        raise HTTPException(422, "Confirm an ER configuration with at least one MATCH field before labelling.")
    cfg_version = er_config_version(rows)

    model_bytes, threshold, model_used = None, 0.5, None
    if req.strategy in ("AUTO", "MODEL"):
        active = db.query(ActiveModel).filter(ActiveModel.domain == ds.domain).first()
        run = db.query(TrainingRun).filter(TrainingRun.training_run_id == active.training_run_id).first() if active else None
        if run and run.model_path and set(run.match_fields) <= set(roles.match_fields):
            try:
                with open(run.model_path, "rb") as f:
                    model_bytes = f.read()
                threshold, model_used = run.threshold or 0.5, run.training_run_id
            except OSError:
                model_bytes = None
        if req.strategy == "MODEL" and model_bytes is None:
            raise HTTPException(422, "No usable ACTIVE model for this domain/config; use strategy BOOTSTRAP.")

    fields = list(dict.fromkeys(roles.match_fields + roles.identifier_fields))
    records = records_from_standardized(db, dataset_id, fields)
    cands = generate_label_candidates(records, roles, req.n, seed=req.seed,
                                      existing_pair_keys=repo.existing_keys(db, ds.domain, dataset_id),
                                      model_settings_bytes=model_bytes, threshold=threshold)
    all_fields = sorted(mapping_dict_for_dataset(db, dataset_id).keys())
    display = records_from_standardized(db, dataset_id, all_fields) if all_fields else {}
    created = []
    for c in cands:
        a, b = int(c["record_id_a"]), int(c["record_id_b"])
        ev = identifier_evidence(records[a], records[b], roles.identifier_fields)
        p = repo.add_pair(db, domain=ds.domain, dataset_id=dataset_id, record_id_a=a, record_id_b=b,
                          values_a=coerce_record(records[a], fields), values_b=coerce_record(records[b], fields),
                          display_a=display.get(a), display_b=display.get(b), strategy=c["strategy"],
                          evidence={"identifier_agreement": ev, "model_used": model_used},
                          model_score=c["model_score"], er_config_version=cfg_version)
        created.append(repo.pair_to_dict(p))
    return {"dataset_id": dataset_id, "domain": ds.domain, "created": len(created),
            "strategy_used": "MODEL_ASSISTED" if model_bytes else "DEDUPE_ACTIVE_LEARNING_BOOTSTRAP",
            "pairs": created,
            "instruction": "For each pair confirm whether BOTH records represent the same real-world entity: "
                           "MATCH, DISTINCT, or UNSURE."}


@router.get("/datasets/{dataset_id}/er/labeling/pairs")
def list_label_pairs(dataset_id: str, status: str = "PENDING", limit: int = 100, db: Session = Depends(get_db)):
    ds = _ds(db, dataset_id)
    q = db.query(LabeledPair).filter(LabeledPair.domain == ds.domain, LabeledPair.dataset_id == dataset_id)
    if status == "PENDING":
        q = q.filter(LabeledPair.label.is_(None))
    elif status == "LABELED":
        q = q.filter(LabeledPair.label.isnot(None))
    rows = q.order_by(LabeledPair.created_at.asc()).limit(limit).all()
    return {"pairs": [repo.pair_to_dict(r) for r in rows], "count": len(rows)}


class LabelRequest(BaseModel):
    label: str
    labeler: str
    reason: Optional[str] = None


@router.post("/labeling/pairs/{pair_id}/label")
def label_pair(pair_id: str, req: LabelRequest, db: Session = Depends(get_db)):
    p = db.query(LabeledPair).filter(LabeledPair.pair_id == pair_id).first()
    if p is None:
        raise HTTPException(404, f"Labelled pair {pair_id} not found")
    try:
        repo.set_label(db, p, req.label, req.labeler, req.reason)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return repo.pair_to_dict(p)


class ManualPairRequest(BaseModel):
    record_id_a: int
    record_id_b: int
    label: str
    labeler: str
    reason: Optional[str] = None


@router.post("/datasets/{dataset_id}/er/labeling/manual")
def add_manual_label(dataset_id: str, req: ManualPairRequest, db: Session = Depends(get_db)):
    """A steward records a pair they already know (e.g. a known duplicate)."""
    ds = _ds(db, dataset_id)
    if req.record_id_a == req.record_id_b:
        raise HTTPException(422, "A pair needs two different records")
    roles, rows = load_active_er_roles(db, dataset_id)
    if not rows or not roles.match_fields:
        raise HTTPException(422, "Confirm an ER configuration with at least one MATCH field first.")
    fields = list(dict.fromkeys(roles.match_fields + roles.identifier_fields))
    records = records_from_standardized(db, dataset_id, fields)
    if req.record_id_a not in records or req.record_id_b not in records:
        raise HTTPException(422, "Unknown record id for this dataset")
    if repo.pair_key(req.record_id_a, req.record_id_b) in repo.existing_keys(db, ds.domain, dataset_id):
        raise HTTPException(409, "This pair already exists in the labelled-pair repository")
    all_fields = sorted(mapping_dict_for_dataset(db, dataset_id).keys())
    display = records_from_standardized(db, dataset_id, all_fields) if all_fields else {}
    a, b = req.record_id_a, req.record_id_b
    try:
        p = repo.add_pair(db, domain=ds.domain, dataset_id=dataset_id, record_id_a=a, record_id_b=b,
                          values_a=coerce_record(records[a], fields), values_b=coerce_record(records[b], fields),
                          display_a=display.get(a), display_b=display.get(b), strategy="STEWARD_SUPPLIED",
                          evidence={"identifier_agreement": identifier_evidence(records[a], records[b], roles.identifier_fields)},
                          label=None, er_config_version=er_config_version(rows))
        repo.set_label(db, p, req.label, req.labeler, req.reason)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return repo.pair_to_dict(p)


@router.get("/domains/{domain}/labels/summary")
def labels_summary(domain: str, db: Session = Depends(get_db)):
    return repo.label_summary(db, domain)


@router.get("/domains/{domain}/labels")
def list_domain_labels(domain: str, label: Optional[str] = None, limit: int = 500, db: Session = Depends(get_db)):
    q = db.query(LabeledPair).filter(LabeledPair.domain == domain, LabeledPair.label.isnot(None))
    if label:
        q = q.filter(LabeledPair.label == label)
    rows = q.order_by(LabeledPair.labeled_at.desc()).limit(limit).all()
    return {"labels": [repo.pair_to_dict(r) for r in rows], "count": len(rows)}


@router.get("/datasets/{dataset_id}/er/labeling/training-readiness")
def training_label_readiness(dataset_id: str, seed: int = 42, db: Session = Depends(get_db)):
    """Preview of what a training run for this dataset's ACTIVE ER configuration would take from the
    domain's labelled-pair repository (counts, exclusions and the train/validation/test split)."""
    ds = _ds(db, dataset_id)
    roles, rows = load_active_er_roles(db, dataset_id)
    if not rows or not roles.match_fields:
        raise HTTPException(422, "Confirm an ER configuration with at least one MATCH field first.")
    return label_training_readiness(db, ds.domain, list(roles.match_fields), seed=seed)


@router.get("/domains/{domain}/labels/pairs")
def list_labelled_pairs(domain: str, label: Optional[str] = None, origin: Optional[str] = None,
                        limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    """Browse the labelled-pair repository (all datasets of the domain). origin: review | labeling."""
    limit, offset = max(1, min(limit, 200)), max(0, offset)
    q = db.query(LabeledPair).filter(LabeledPair.domain == domain, LabeledPair.label.isnot(None))
    if label:
        q = q.filter(LabeledPair.label == label)
    if origin == "review":
        q = q.filter(LabeledPair.strategy == "REVIEW_DECISION")
    elif origin == "labeling":
        q = q.filter(LabeledPair.strategy != "REVIEW_DECISION")
    total = q.count()
    rows = q.order_by(LabeledPair.labeled_at.desc(), LabeledPair.pair_id.asc()).offset(offset).limit(limit).all()
    return {"pairs": [repo.pair_to_dict(r) for r in rows], "count": len(rows), "total": total,
            "offset": offset, "limit": limit, "has_more": offset + len(rows) < total}
