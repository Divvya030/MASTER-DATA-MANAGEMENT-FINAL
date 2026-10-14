"""
Golden Record (Module 11) + Steward Review (Module 12) + Model Registry/
Activation (Module 7 lifecycle) + Continuous Incoming Batch matching
(Module 13) + Monitoring (Module 10) endpoints.

Deliberately does NOT import from app.main -- app.main imports and mounts
this router, not the other way round. Shared DB-loading logic lives in
app.modules.dedupe_service.record_loading so both files can use it.

Scope note (documented, not hidden): batch/training/matching/clustering
endpoints in this file run synchronously within the HTTP request, same as
the existing train/match/cluster endpoints in app/main.py -- the `Job`
model exists in app/db/models.py but wiring everything to FastAPI
BackgroundTasks end-to-end was not completed in this pass. This matches the
pre-existing, already-documented limitation for the original three
endpoints and is called out again in the implementation report rather than
silently left inconsistent.
"""
from __future__ import annotations
import datetime as dt
import io
import json
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import (
    Dataset, TrainingRun, MatchRun, ClusterRun, ActiveModel,
    Entity, EntityMember, GoldenRecordVersion, StewardReview, PendingReview,
    IncomingBatchRun,
)
from app.modules.dedupe_service.record_loading import records_from_standardized, mapping_dict_for_dataset
from app.modules.dedupe_service.model_store import load_model_metadata
from app.modules.dedupe_service.incremental_matching_impl import run_incremental_matching
from app.modules.golden_record_service import (
    select_golden_values, normalize_config, validate_config, DEFAULT_CONFIG,
)
from app.modules.dedupe_service.model_store import check_model_compatibility
from app.modules.dedupe_service.field_definitions import coerce_record
from app.modules.dedupe_service import label_repository as label_repo
from app.modules.dedupe_service.config_loading import er_config_version, load_active_er_roles
from app.modules.job_service import start_job, finish_job, job_response
from app.domain_schemas import ordered_field_names, order_values, summary_items, display_name
from app.db.models import (SurvivorshipConfig, LabeledPair, TrainingRunLabel, ProcessingJob,
                           IncomingRecordAudit, ERFieldConfig)

router = APIRouter(prefix="/api", tags=["er-workflow"])


# =============================================================================
# Shared helpers
# =============================================================================

def _get_dataset_or_404(db: Session, dataset_id: str) -> Dataset:
    ds = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return ds


def _get_training_run_or_404(db: Session, training_run_id: str) -> TrainingRun:
    run = db.query(TrainingRun).filter(TrainingRun.training_run_id == training_run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Training run {training_run_id} not found")
    return run


def _get_entity_or_404(db: Session, entity_id: str) -> Entity:
    e = db.query(Entity).filter(Entity.entity_id == entity_id).first()
    if e is None:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    return e


def _latest_golden_version(db: Session, entity_id: str) -> Optional[GoldenRecordVersion]:
    return (db.query(GoldenRecordVersion)
            .filter(GoldenRecordVersion.entity_id == entity_id)
            .order_by(GoldenRecordVersion.version_number.desc())
            .first())


def _latest_golden_versions(db: Session, entity_ids) -> dict:
    """{entity_id: latest GoldenRecordVersion} for many entities in ONE query.
    Looping `_latest_golden_version` per entity is an N+1 that made the review
    screens crawl once a domain held thousands of entities."""
    from sqlalchemy import func
    ids = [i for i in set(entity_ids) if i]
    if not ids:
        return {}
    newest = (db.query(GoldenRecordVersion.entity_id.label("eid"),
                       func.max(GoldenRecordVersion.version_number).label("v"))
              .filter(GoldenRecordVersion.entity_id.in_(ids))
              .group_by(GoldenRecordVersion.entity_id).subquery())
    rows = (db.query(GoldenRecordVersion)
            .join(newest, (GoldenRecordVersion.entity_id == newest.c.eid) &
                          (GoldenRecordVersion.version_number == newest.c.v)).all())
    return {r.entity_id: r for r in rows}


def _all_mapped_fields(db: Session, dataset_id: str) -> list:
    return sorted(mapping_dict_for_dataset(db, dataset_id).keys())


class _DatasetRecordCache:
    """Loading a dataset's standardized layer means reading and parsing its
    full parquet file. Golden-record generation touches many individual
    member records, often repeatedly across many entities in one request
    (generate_golden_records over hundreds of clusters; apply_active_model
    over a whole incoming batch) -- without caching, that means re-reading
    the SAME dataset's file from disk once per member touched, which is
    what made golden-record generation on real ~8,000-row data effectively
    hang (confirmed: seconds per lookup x thousands of members). Caching is
    scoped to one request (a fresh instance per endpoint call), keyed by
    dataset_id -- fields are the same every time for a given dataset (all
    its confirmed mapped canonical fields), so dataset_id alone is a safe
    cache key here."""

    def __init__(self, db: Session):
        self.db = db
        self._fields_by_dataset: dict = {}
        self._records_by_dataset: dict = {}

    def values_for(self, dataset_id: str, record_id: str) -> dict:
        if dataset_id not in self._records_by_dataset:
            fields = _all_mapped_fields(self.db, dataset_id)
            self._fields_by_dataset[dataset_id] = fields
            self._records_by_dataset[dataset_id] = (
                records_from_standardized(self.db, dataset_id, fields) if fields else {}
            )
        return self._records_by_dataset[dataset_id].get(int(record_id), {})


def _survivorship_config(db: Session, domain: str) -> dict:
    """Only an ACTIVE (or legacy NULL) configuration generates Golden Records;
    DRAFT / STEWARD_CONFIRMED rows stay out of production until confirmed."""
    row = db.query(SurvivorshipConfig).filter(SurvivorshipConfig.domain == domain).first()
    if row is None:
        return normalize_config(None)
    if row.lifecycle_status in ("DRAFT", "STEWARD_CONFIRMED"):
        return normalize_config(None)
    cfg = normalize_config(row.config)
    cfg["version"] = row.version
    return cfg


def _member_snapshot(cache, m: EntityMember) -> dict:
    """Stored snapshot first (survives dataset deletion); the dataset cache is
    only a fallback for members created before snapshots existed."""
    if m.values:
        return m.values
    return cache.values_for(m.dataset_id, m.record_id)


def _add_member(db: Session, cache, entity_id: str, dataset_id: str, record_id, source: str) -> EntityMember:
    m = EntityMember(entity_id=entity_id, dataset_id=dataset_id, record_id=str(record_id), source=source,
                     values=cache.values_for(dataset_id, record_id) or None)
    db.add(m)
    return m


def _generate_golden_record(db: Session, entity: Entity, reason: str, created_by: str = "system",
                             cache: Optional[_DatasetRecordCache] = None) -> GoldenRecordVersion:
    """Recomputes and appends a new GoldenRecordVersion from the entity's
    CURRENT (non-removed) members using the domain's survivorship config.
    Never mutates a previous version."""
    cache = cache or _DatasetRecordCache(db)
    cfg = _survivorship_config(db, entity.domain)
    active_members = [m for m in entity.members if m.removed_at is None]
    member_records = []
    all_fields = set()
    for m in active_members:
        values = _member_snapshot(cache, m)
        all_fields.update(values.keys())
        member_records.append({"dataset_id": m.dataset_id, "record_id": m.record_id, "values": values})

    needs_review = []
    if not member_records:
        # An entity that lost all its members (e.g. after a SPLIT moved
        # everything out) still gets an explicit empty version rather than
        # silently keeping stale data as "current".
        result_values, result_lineage, result_conflicts, result_conf = {}, {}, {}, {}
    else:
        # source_system / last_updated drive the rules and are recorded in each
        # field's lineage; they are not themselves "values" of the entity.
        all_fields -= {cfg.get("recency_field"), cfg.get("source_system_field")}
        result = select_golden_values(member_records, ordered_field_names(entity.domain, all_fields), config=cfg)
        result_values, result_lineage = result.golden_values, result.lineage
        result_conflicts, result_conf = result.conflicts, result.field_confidence
        needs_review = result.needs_review

    prev = _latest_golden_version(db, entity.entity_id)
    version_number = (prev.version_number + 1) if prev else 1

    gv = GoldenRecordVersion(
        entity_id=entity.entity_id, version_number=version_number,
        golden_values=result_values, lineage=result_lineage,
        conflicts=result_conflicts, field_confidence=result_conf,
        reason=reason, training_run_id=entity.training_run_id,
        er_config_version=entity.er_config_version, created_by=created_by,
        survivorship_rule_version=cfg.get("version", 1),
        steward_status="NEEDS_REVIEW" if needs_review else "PROPOSED",
        needs_review_fields=needs_review, updated_at=dt.datetime.utcnow(),
    )
    db.add(gv)
    entity.updated_at = dt.datetime.utcnow()
    db.commit()
    db.refresh(gv)
    return gv


def _entity_response(db: Session, entity: Entity) -> dict:
    latest = _latest_golden_version(db, entity.entity_id)
    active_members = [m for m in entity.members if m.removed_at is None]
    return {
        "entity_id": entity.entity_id, "domain": entity.domain, "status": entity.status,
        "cluster_run_id": entity.cluster_run_id, "training_run_id": entity.training_run_id,
        "er_config_version": entity.er_config_version,
        "merged_into_entity_id": entity.merged_into_entity_id,
        "member_count": len(active_members),
        "members": [
            {"dataset_id": m.dataset_id, "record_id": m.record_id, "source": m.source,
             "added_at": m.added_at.isoformat() if m.added_at else None}
            for m in active_members
        ],
        "display_name": display_name(entity.domain, latest.golden_values) if latest else None,
        "summary": summary_items(entity.domain, latest.golden_values) if latest else [],
        "golden_record": {
            "version_number": latest.version_number, "values": order_values(entity.domain, latest.golden_values),
            "lineage": latest.lineage, "conflicts": latest.conflicts,
            "field_confidence": latest.field_confidence, "reason": latest.reason,
            "steward_status": latest.steward_status, "needs_review_fields": latest.needs_review_fields or [],
            "survivorship_rule_version": latest.survivorship_rule_version,
            "created_at": latest.created_at.isoformat() if latest.created_at else None,
        } if latest else None,
        "created_at": entity.created_at.isoformat() if entity.created_at else None,
        "updated_at": entity.updated_at.isoformat() if entity.updated_at else None,
    }


# =============================================================================
# Module 7 (cont.): Model Registry / Lifecycle / Activation
# =============================================================================

def _model_response(db: Session, run: TrainingRun) -> dict:
    active = db.query(ActiveModel).filter(ActiveModel.domain == run.domain).first()
    return {
        "training_run_id": run.training_run_id, "domain": run.domain,
        "dataset_id": run.dataset_id, "er_config_version": run.er_config_version,
        "match_fields": run.match_fields, "training_status": run.status,
        "lifecycle_status": run.lifecycle_status,
        "is_active": bool(active and active.training_run_id == run.training_run_id),
        "evaluation_metrics": run.evaluation_metrics,          # SYNTHETIC validation only
        "real_evaluation": run.real_evaluation,                # REAL steward-labelled evaluation
        "label_counts": run.label_counts, "threshold": run.threshold,
        "review_threshold": run.review_threshold, "threshold_calibration": run.threshold_calibration,
        "compatibility_problems": check_model_compatibility(load_model_metadata(run.model_path)) if run.model_path else [],
        "model_metadata": load_model_metadata(run.model_path),
        "activated_at": run.activated_at.isoformat() if run.activated_at else None,
        "retired_at": run.retired_at.isoformat() if run.retired_at else None,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


@router.get("/models")
def list_models(domain: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(TrainingRun).filter(TrainingRun.lifecycle_status.isnot(None))
    if domain:
        q = q.filter(TrainingRun.domain == domain)
    runs = q.order_by(TrainingRun.created_at.desc()).all()
    return {"models": [_model_response(db, r) for r in runs]}


@router.get("/models/{training_run_id}")
def get_model(training_run_id: str, db: Session = Depends(get_db)):
    run = _get_training_run_or_404(db, training_run_id)
    return _model_response(db, run)


def _validate_model(db: Session, training_run_id: str):
    run = _get_training_run_or_404(db, training_run_id)
    if run.status != "TRAINED":
        raise HTTPException(status_code=422, detail=f"Training run is not TRAINED (status={run.status}); nothing to validate.")
    if run.lifecycle_status not in (None, "DRAFT"):
        raise HTTPException(status_code=422, detail=f"Model is already {run.lifecycle_status}; cannot validate again.")
    problems = check_model_compatibility(load_model_metadata(run.model_path))
    if problems:
        raise HTTPException(status_code=422, detail="Model artifact is not compatible with this environment: " + "; ".join(problems))
    syn = run.evaluation_metrics or {}
    real = run.real_evaluation or {}
    has_syn = syn.get("validation_pair_count", 0) > 0
    has_real = (real.get("validation", {}).get("pair_count", 0) + real.get("test", {}).get("pair_count", 0)) > 0
    if not (has_syn or has_real):
        raise HTTPException(status_code=422, detail="No held-out validation evidence exists for this model (neither synthetic "
                                                      "nor real steward-labelled); cannot promote to VALIDATED.")
    run.lifecycle_status = "VALIDATED"
    db.commit()
    return _model_response(db, run)


@router.post("/models/{training_run_id}/validate")
def validate_model(training_run_id: str, db: Session = Depends(get_db)):
    """DRAFT -> VALIDATED: requires held-out evidence and a compatible artifact."""
    return _validate_model(db, training_run_id)


@router.post("/models/{training_run_id}/mark-evaluated")
def mark_model_evaluated(training_run_id: str, db: Session = Depends(get_db)):
    """Backward-compatible alias of /validate (the earlier lifecycle name)."""
    return _validate_model(db, training_run_id)


@router.post("/models/{training_run_id}/activate")
def activate_model(training_run_id: str, db: Session = Depends(get_db)):
    """VALIDATED -> ACTIVE. Refuses a model trained against an ER configuration
    that is no longer the dataset's active one, or whose dependency versions are
    incompatible. Retires the previously ACTIVE model of the domain (RETIRED, not
    deleted). One ACTIVE model per domain is enforced by ActiveModel.domain being
    the primary key."""
    run = _get_training_run_or_404(db, training_run_id)
    if run.lifecycle_status != "VALIDATED":
        raise HTTPException(status_code=422, detail=f"Model must be VALIDATED before activation "
                                                      f"(current lifecycle_status={run.lifecycle_status}). Call /validate first.")
    problems = check_model_compatibility(load_model_metadata(run.model_path))
    if problems:
        raise HTTPException(status_code=422, detail="Incompatible model artifact: " + "; ".join(problems))
    src = db.query(Dataset).filter(Dataset.dataset_id == run.dataset_id).first()
    if src is not None:
        rows = db.query(ERFieldConfig).filter(ERFieldConfig.dataset_id == run.dataset_id).all()
        if rows and er_config_version(rows) != run.er_config_version:
            raise HTTPException(status_code=422, detail="Model was trained against a different ER configuration "
                                                          f"({run.er_config_version}) than the source dataset's current "
                                                          f"active configuration ({er_config_version(rows)}). Retrain.")
    prev = db.query(ActiveModel).filter(ActiveModel.domain == run.domain).first()
    if prev is not None:
        prev_run = _get_training_run_or_404(db, prev.training_run_id)
        prev_run.lifecycle_status = "RETIRED"
        prev_run.retired_at = dt.datetime.utcnow()
        db.delete(prev)
        db.commit()

    run.lifecycle_status = "ACTIVE"
    run.activated_at = dt.datetime.utcnow()
    db.add(ActiveModel(domain=run.domain, training_run_id=run.training_run_id))
    db.commit()
    return _model_response(db, run)


def evaluate_run_on_unseen_real_labels(db: Session, run: TrainingRun) -> dict:
    """REAL steward-labelled evaluation of a persisted model on labels it never
    trained on: labels not in its train split AND sharing no record with any of
    its training labels (so a re-used record cannot leak)."""
    from app.modules.dedupe_service.real_labels import node_key, confusion_metrics, score_labeled_pairs
    import dedupe
    train_ids = {r.pair_id for r in db.query(TrainingRunLabel).filter(
        TrainingRunLabel.training_run_id == run.training_run_id, TrainingRunLabel.split == "train").all()}
    all_labels = label_repo.usable_real_labels(db, run.domain)
    train_nodes = {node_key(l["dataset_id"], l[k]) for l in all_labels if l["pair_id"] in train_ids
                   for k in ("record_id_a", "record_id_b")}
    unseen = [l for l in all_labels if l["pair_id"] not in train_ids and
              node_key(l["dataset_id"], l["record_id_a"]) not in train_nodes and
              node_key(l["dataset_id"], l["record_id_b"]) not in train_nodes and
              all(f in (l["values_a"] or {}) and f in (l["values_b"] or {}) for f in run.match_fields)]
    base = {"kind": "REAL_STEWARD_LABELLED_EVALUATION", "label_source": "REAL_STEWARD",
            "training_run_id": run.training_run_id, "domain": run.domain, "dataset_id": run.dataset_id,
            "threshold": run.threshold if run.threshold is not None else 0.5,
            "threshold_calibration": run.threshold_calibration,
            "excluded_as_training_or_overlapping": len(all_labels) - len(unseen)}
    if not unseen:
        return {**base, "labelled_pair_count": 0, "note": "No real steward labels unseen by this model exist yet; "
                                                            "nothing to evaluate (no accuracy is claimed)."}
    from app.modules.dedupe_service.dedupe_factory import load_static
    with open(run.model_path, "rb") as f:
        static = load_static(f.read())
    scores, ys = score_labeled_pairs(static, [(l["values_a"], l["values_b"], 1 if l["label"] == "MATCH" else 0)
                                              for l in unseen], run.match_fields)
    return {**base, "labelled_pair_count": len(ys), "match_count": sum(ys), "distinct_count": len(ys) - sum(ys),
            **confusion_metrics(scores, ys, base["threshold"]),
            "note": ("Real-world steward-labelled evaluation on labels this model did not train on. "
                     + ("Small label count: indicative only." if len(ys) < 30 else ""))}


@router.post("/models/{training_run_id}/evaluate-real")
def evaluate_model_real(training_run_id: str, db: Session = Depends(get_db)):
    run = _get_training_run_or_404(db, training_run_id)
    if not run.model_path:
        raise HTTPException(status_code=422, detail="This training run has no model artifact.")
    result = evaluate_run_on_unseen_real_labels(db, run)
    real = dict(run.real_evaluation or {})
    real["post_training_unseen"] = result
    run.real_evaluation = real
    db.commit()
    return result


@router.post("/models/{training_run_id}/retire")
def retire_model(training_run_id: str, db: Session = Depends(get_db)):
    run = _get_training_run_or_404(db, training_run_id)
    if run.lifecycle_status != "ACTIVE":
        raise HTTPException(status_code=422, detail=f"Only an ACTIVE model can be retired "
                                                      f"(current lifecycle_status={run.lifecycle_status}).")
    active = db.query(ActiveModel).filter(ActiveModel.domain == run.domain,
                                           ActiveModel.training_run_id == training_run_id).first()
    if active:
        db.delete(active)
    run.lifecycle_status = "RETIRED"
    run.retired_at = dt.datetime.utcnow()
    db.commit()
    return _model_response(db, run)


@router.get("/domains/{domain}/active-model")
def get_active_model(domain: str, db: Session = Depends(get_db)):
    """The core lookup the matching/incoming-batch pipeline depends on:
    domain -> active model, with NO dependency on any particular dataset --
    resolvable even after the training dataset has been deleted."""
    active = db.query(ActiveModel).filter(ActiveModel.domain == domain).first()
    if active is None:
        raise HTTPException(status_code=404, detail=f"No ACTIVE model for domain '{domain}'. "
                                                      f"Train and activate a model first.")
    run = _get_training_run_or_404(db, active.training_run_id)
    return _model_response(db, run)


def _load_active_model_or_404(db: Session, domain: str) -> TrainingRun:
    active = db.query(ActiveModel).filter(ActiveModel.domain == domain).first()
    if active is None:
        raise HTTPException(status_code=422, detail=f"No ACTIVE model for domain '{domain}'. "
                                                      f"Train and activate a model before matching.")
    run = _get_training_run_or_404(db, active.training_run_id)
    if not run.model_path:
        raise HTTPException(status_code=500, detail=f"Active model {run.training_run_id} has no artifact path "
                                                      f"on record -- corrupt registry state.")
    import os
    if not os.path.exists(run.model_path):
        raise HTTPException(status_code=500, detail=f"Active model {run.training_run_id}'s artifact is missing "
                                                      f"on disk at {run.model_path!r} (corrupted/deleted "
                                                      f"artifact). Retrain or restore from backup.")
    problems = check_model_compatibility(load_model_metadata(run.model_path))
    if problems:
        raise HTTPException(status_code=422, detail=f"Active model {run.training_run_id} is incompatible with this "
                                                      f"environment: " + "; ".join(problems))
    return run


# =============================================================================
# Module 11: Golden Record generation from a completed cluster run
# =============================================================================

@router.post("/datasets/{dataset_id}/er/clusters/{cluster_run_id}/golden-records")
def generate_golden_records(dataset_id: str, cluster_run_id: str, db: Session = Depends(get_db)):
    ds = _get_dataset_or_404(db, dataset_id)
    cluster_run = db.query(ClusterRun).filter(ClusterRun.cluster_run_id == cluster_run_id,
                                               ClusterRun.dataset_id == dataset_id).first()
    if cluster_run is None:
        raise HTTPException(status_code=404, detail=f"Cluster run {cluster_run_id} not found")
    if cluster_run.status != "COMPLETED" or not cluster_run.clusters_path:
        raise HTTPException(status_code=422, detail=f"Cluster run {cluster_run_id} is not COMPLETED "
                                                      f"(status={cluster_run.status}); cannot generate "
                                                      f"golden records.")
    training_run = _get_training_run_or_404(db, cluster_run.training_run_id)

    with open(cluster_run.clusters_path) as f:
        clusters = json.load(f)

    created_entities = []
    job = start_job(db, "GOLDEN_RECORDS", domain=ds.domain, dataset_id=dataset_id,
                    model_version=training_run.training_run_id)
    cache = _DatasetRecordCache(db)  # shared across the whole cluster loop -- see _DatasetRecordCache
    for cluster in clusters:
        entity_id = f"ENT-{uuid.uuid4().hex[:12]}"
        entity = Entity(
            entity_id=entity_id, domain=ds.domain, status="ACTIVE",
            cluster_run_id=cluster_run_id, training_run_id=training_run.training_run_id,
            er_config_version=training_run.er_config_version,
        )
        db.add(entity)
        for record_id in cluster["record_ids"]:
            _add_member(db, cache, entity_id, dataset_id, record_id, "BATCH")
        db.commit()
        db.refresh(entity)
        _generate_golden_record(db, entity, reason="INITIAL_GENERATION", cache=cache)
        created_entities.append(entity_id)

    finish_job(db, job, True, artifact_ref=cluster_run_id, result={"entities_created": len(created_entities)})
    return {
        "dataset_id": dataset_id, "cluster_run_id": cluster_run_id, "job_id": job.job_id,
        "training_run_id": training_run.training_run_id,
        "entities_created": len(created_entities),
        "entity_ids": created_entities,
    }


@router.get("/entities")
def list_entities(domain: Optional[str] = None, status: Optional[str] = None,
                  min_member_count: Optional[int] = None, needs_review: bool = False,
                  limit: int = 500, db: Session = Depends(get_db)):
    """min_member_count lets a steward ask for only the entities that actually
    resulted from deduplication (member_count > 1) -- on a real dataset with
    thousands of mostly-singleton entities, an unfiltered listing capped at
    `limit` can otherwise return page after page of singletons and never
    surface the clusters a steward actually needs to look at."""
    from sqlalchemy import func
    q = db.query(Entity)
    if domain:
        q = q.filter(Entity.domain == domain)
    if status:
        q = q.filter(Entity.status == status)
    if min_member_count is not None:
        counts = (db.query(EntityMember.entity_id, func.count(EntityMember.id).label("n"))
                  .filter(EntityMember.removed_at.is_(None)).group_by(EntityMember.entity_id)
                  .having(func.count(EntityMember.id) >= min_member_count).subquery())
        q = q.join(counts, Entity.entity_id == counts.c.entity_id)
    if needs_review:
        from sqlalchemy import func as _func
        latest = (db.query(GoldenRecordVersion.entity_id,
                           _func.max(GoldenRecordVersion.version_number).label("v"))
                  .group_by(GoldenRecordVersion.entity_id).subquery())
        q = (q.join(latest, Entity.entity_id == latest.c.entity_id)
             .join(GoldenRecordVersion, (GoldenRecordVersion.entity_id == latest.c.entity_id) &
                   (GoldenRecordVersion.version_number == latest.c.v))
             .filter(GoldenRecordVersion.steward_status == "NEEDS_REVIEW"))
    entities = q.order_by(Entity.created_at.desc()).limit(min(limit, 2000)).all()
    total = q.order_by(None).count() if hasattr(q, "count") else len(entities)
    return {"entities": [_entity_response(db, e) for e in entities], "count": len(entities), "total_matching": total}


@router.get("/entities/{entity_id}")
def get_entity(entity_id: str, db: Session = Depends(get_db)):
    return _entity_response(db, _get_entity_or_404(db, entity_id))


@router.get("/entities/{entity_id}/golden-record")
def get_golden_record(entity_id: str, db: Session = Depends(get_db)):
    entity = _get_entity_or_404(db, entity_id)
    latest = _latest_golden_version(db, entity_id)
    if latest is None:
        raise HTTPException(status_code=404, detail=f"No golden record generated yet for {entity_id}")
    return {
        "entity_id": entity_id, "version_number": latest.version_number,
        "values": order_values(entity.domain, latest.golden_values), "lineage": latest.lineage,
        "conflicts": latest.conflicts, "field_confidence": latest.field_confidence,
        "steward_status": latest.steward_status, "needs_review_fields": latest.needs_review_fields or [],
        "survivorship_rule_version": latest.survivorship_rule_version,
        "reason": latest.reason, "created_by": latest.created_by,
        "created_at": latest.created_at.isoformat() if latest.created_at else None,
    }


@router.get("/entities/{entity_id}/golden-record/history")
def get_golden_record_history(entity_id: str, db: Session = Depends(get_db)):
    ent = _get_entity_or_404(db, entity_id)
    versions = (db.query(GoldenRecordVersion)
                .filter(GoldenRecordVersion.entity_id == entity_id)
                .order_by(GoldenRecordVersion.version_number.asc()).all())
    return {"entity_id": entity_id, "versions": [
        {"version_number": v.version_number, "values": order_values(ent.domain, v.golden_values), "reason": v.reason,
         "steward_status": v.steward_status,
         "created_by": v.created_by, "created_at": v.created_at.isoformat() if v.created_at else None}
        for v in versions
    ]}


# =============================================================================
# Module 12: Steward Review
# MODEL RESULT (never overwritten) + STEWARD DECISION (StewardReview row) +
# FINAL MASTERED RESULT (entity / golden record versions).
# =============================================================================

class ReviewRequest(BaseModel):
    action: str
    reviewer: str
    reason: Optional[str] = None
    corrected_values: Optional[dict] = None          # CORRECT (multi-field)
    split_members: Optional[list] = None             # SPLIT: [{"dataset_id","record_id"}]
    target_entity_id: Optional[str] = None           # MERGE: absorb this entity INTO target
    pending_review_id: Optional[str] = None          # resolve a queued item alongside the action
    field: Optional[str] = None                      # EDIT_FIELD / RESOLVE_FIELD / REJECT_FIELD_VALUE
    value: Optional[str] = None                      # EDIT_FIELD / RESOLVE_FIELD chosen value
    rejected_value: Optional[str] = None             # REJECT_FIELD_VALUE


VALID_ACTIONS = {"APPROVE", "REJECT", "CORRECT", "SPLIT", "MERGE",
                 "APPROVE_GOLDEN_RECORD", "EDIT_FIELD", "RESOLVE_FIELD", "REJECT_FIELD_VALUE"}


@router.get("/reviews/pending")
def list_pending_reviews(domain: Optional[str] = None, limit: int = 25, offset: int = 0,
                         db: Session = Depends(get_db)):
    """One PAGE of the match-review queue (oldest first). The queue can hold
    thousands of below-threshold pairs, so the page size is bounded and every
    related lookup for the page is batched into a single query."""
    limit, offset = max(1, min(limit, 200)), max(0, offset)
    q = db.query(PendingReview).filter(PendingReview.status == "PENDING")
    if domain:
        q = q.filter(PendingReview.domain == domain)
    total = q.count()
    items = (q.order_by(PendingReview.created_at.asc(), PendingReview.pending_review_id.asc())
              .offset(offset).limit(limit).all())

    golden = _latest_golden_versions(db, [p.entity_id for p in items])

    # record-vs-record items (e.g. UNSURE pairs from a batch match run): fetch
    # every "other" member for the page at once instead of one query per item.
    pair_items = [p for p in items if p.candidate_other_record_id and not p.entity_id]
    members_by_key: dict = {}
    if pair_items:
        rows = (db.query(EntityMember)
                .filter(EntityMember.dataset_id.in_({p.candidate_dataset_id for p in pair_items}),
                        EntityMember.record_id.in_({p.candidate_other_record_id for p in pair_items}),
                        EntityMember.removed_at.is_(None)).all())
        for m in rows:
            members_by_key.setdefault((m.dataset_id, m.record_id), m)

    cache = _DatasetRecordCache(db)   # only touched if a member snapshot is missing
    out = []
    for p in items:
        cand_entity = None
        if p.entity_id:
            latest = golden.get(p.entity_id)
            cand_entity = {"entity_id": p.entity_id,
                           "golden_values": order_values(p.domain, latest.golden_values) if latest else None}
        other_values = None
        if p in pair_items:
            m = members_by_key.get((p.candidate_dataset_id, p.candidate_other_record_id))
            other_values = (m.values if m else None) or cache.values_for(p.candidate_dataset_id, p.candidate_other_record_id)
            other_values = order_values(p.domain, other_values)
        out.append({"pending_review_id": p.pending_review_id, "domain": p.domain, "kind": p.kind,
                    "entity_id": p.entity_id, "candidate_dataset_id": p.candidate_dataset_id,
                    "candidate_record_id": p.candidate_record_id,
                    "candidate_other_record_id": p.candidate_other_record_id,
                    "candidate_values": order_values(p.domain, p.candidate_values), "candidate_entity": cand_entity,
                    "candidate_name": display_name(p.domain, p.candidate_values),
                    "candidate_other_values": other_values,
                    "score": p.score, "match_threshold": p.match_threshold, "review_threshold": p.review_threshold,
                    "model_version": p.training_run_id, "batch_run_id": p.batch_run_id,
                    "created_at": p.created_at.isoformat() if p.created_at else None})
    return {"pending_reviews": out, "count": len(out), "total": total,
            "offset": offset, "limit": limit, "has_more": offset + len(out) < total}


@router.get("/reviews/entities-needing-review")
def entities_needing_review(domain: Optional[str] = None, limit: int = 25, offset: int = 0,
                            db: Session = Depends(get_db)):
    """Golden records whose latest version has fields the survivorship rules
    could not safely resolve. A version is NEEDS_REVIEW exactly when it carries
    unresolved fields, so this is one filtered, paged join -- not a scan of
    every entity with a query per row."""
    from sqlalchemy import func
    limit, offset = max(1, min(limit, 200)), max(0, offset)
    newest = (db.query(GoldenRecordVersion.entity_id.label("eid"),
                       func.max(GoldenRecordVersion.version_number).label("v"))
              .group_by(GoldenRecordVersion.entity_id).subquery())
    q = (db.query(Entity, GoldenRecordVersion)
         .join(newest, Entity.entity_id == newest.c.eid)
         .join(GoldenRecordVersion, (GoldenRecordVersion.entity_id == newest.c.eid) &
                                    (GoldenRecordVersion.version_number == newest.c.v))
         .filter(Entity.status == "ACTIVE", GoldenRecordVersion.steward_status == "NEEDS_REVIEW"))
    if domain:
        q = q.filter(Entity.domain == domain)
    total = q.count()
    rows = q.order_by(Entity.created_at.asc(), Entity.entity_id.asc()).offset(offset).limit(limit).all()
    out = [{"entity_id": e.entity_id, "domain": e.domain, "steward_status": latest.steward_status,
            "needs_review_fields": latest.needs_review_fields,
            "golden_values": order_values(e.domain, latest.golden_values), "version_number": latest.version_number,
            "display_name": display_name(e.domain, latest.golden_values),
            "summary": summary_items(e.domain, latest.golden_values)}
           for e, latest in rows]
    return {"entities": out, "count": len(out), "total": total,
            "offset": offset, "limit": limit, "has_more": offset + len(out) < total}


@router.get("/entities/{entity_id}/review-context")
def get_review_context(entity_id: str, db: Session = Depends(get_db)):
    """Everything a steward needs on one screen: entity, members (with stored
    source snapshots), golden record + conflicts + lineage, pending ambiguous
    items, and the audit history."""
    entity = _get_entity_or_404(db, entity_id)
    entity_resp = _entity_response(db, entity)
    for m_resp, m in zip(entity_resp["members"], [m for m in entity.members if m.removed_at is None]):
        m_resp["values"] = m.values
    pending = db.query(PendingReview).filter(PendingReview.entity_id == entity_id,
                                              PendingReview.status == "PENDING").all()
    history = (db.query(StewardReview).filter(StewardReview.entity_id == entity_id)
               .order_by(StewardReview.created_at.desc()).all())
    entity_resp["pending_review_items"] = [
        {"pending_review_id": p.pending_review_id, "candidate_dataset_id": p.candidate_dataset_id,
         "candidate_record_id": p.candidate_record_id, "candidate_values": p.candidate_values, "score": p.score}
        for p in pending
    ]
    entity_resp["review_history"] = [
        {"review_id": r.review_id, "action": r.action, "reviewer": r.reviewer, "reason": r.reason,
         "model_result": r.model_result, "before_state": r.before_state, "after_state": r.after_state,
         "created_at": r.created_at.isoformat() if r.created_at else None}
        for r in history
    ]
    return entity_resp


def _new_version(db, entity, prev, *, values, lineage, conflicts, confidence, needs_review, reason, reviewer):
    status = "NEEDS_REVIEW" if needs_review else "CORRECTED"
    gv = GoldenRecordVersion(
        entity_id=entity.entity_id, version_number=(prev.version_number + 1) if prev else 1,
        golden_values=values, lineage=lineage, conflicts=conflicts, field_confidence=confidence,
        reason=reason, training_run_id=entity.training_run_id, er_config_version=entity.er_config_version,
        created_by=reviewer, survivorship_rule_version=prev.survivorship_rule_version if prev else None,
        steward_status=status, needs_review_fields=needs_review, updated_at=dt.datetime.utcnow(),
    )
    db.add(gv)
    db.commit()
    return gv


def _field_candidates(prev, field: str) -> dict:
    """{value: {dataset_id, record_id, support}} for the selected value plus all retained conflicts."""
    cands = {}
    sel = prev.golden_values.get(field)
    lin = (prev.lineage or {}).get(field) or {}
    if sel is not None:
        cands[str(sel)] = {"dataset_id": lin.get("dataset_id"), "record_id": lin.get("record_id"), "support": 1}
    for c in (prev.conflicts or {}).get(field, []):
        cands.setdefault(str(c["value"]), {"dataset_id": c.get("dataset_id"), "record_id": c.get("record_id"),
                                           "support": c.get("support", 1)})
    for nr in (prev.needs_review_fields or []):
        if nr["field"] == field:
            for c in nr["candidates"]:
                cands.setdefault(str(c["value"]), {"dataset_id": c.get("dataset_id"), "record_id": c.get("record_id"),
                                                   "support": c.get("support", 1)})
    return cands


def _field_change(db, entity, prev, req: ReviewRequest):
    """Applies EDIT_FIELD / RESOLVE_FIELD / REJECT_FIELD_VALUE / CORRECT as a NEW golden
    record version (history kept). Returns the new version."""
    if prev is None:
        raise HTTPException(status_code=422, detail="This entity has no golden record yet.")
    values = dict(prev.golden_values)
    lineage = {k: (dict(v) if v else v) for k, v in (prev.lineage or {}).items()}
    conflicts = {k: list(v) for k, v in (prev.conflicts or {}).items()}
    conf = dict(prev.field_confidence or {})
    needs = [dict(n) for n in (prev.needs_review_fields or [])]
    now = dt.datetime.utcnow().isoformat()

    def keep_old_as_conflict(f, old):
        if old is not None and str(old) != str(values.get(f)) and \
                str(old) not in {str(c["value"]) for c in conflicts.get(f, [])}:
            l = lineage.get(f) or {}
            conflicts.setdefault(f, []).append({"value": old, "dataset_id": l.get("dataset_id"),
                                                 "record_id": l.get("record_id"), "support": 1})

    if req.action in ("CORRECT", "EDIT_FIELD"):
        changes = req.corrected_values if req.action == "CORRECT" else (
            {req.field: req.value} if req.field is not None and req.value is not None else None)
        if not changes:
            raise HTTPException(status_code=422, detail=f"{req.action} requires " +
                                ("corrected_values" if req.action == "CORRECT" else "field and value"))
        for f, v in changes.items():
            old = values.get(f)
            values[f] = v
            keep_old_as_conflict(f, old)
            lineage[f] = {"value": v, "dataset_id": None, "record_id": None, "rule": "STEWARD_EDIT",
                          "status": "RESOLVED", "confidence": 1.0, "selected_at": now,
                          "reason": f"steward edit by {req.reviewer}"}
            conf[f] = 1.0
            needs = [n for n in needs if n["field"] != f]
    elif req.action == "RESOLVE_FIELD":
        if not req.field or req.value is None:
            raise HTTPException(status_code=422, detail="RESOLVE_FIELD requires field and value")
        cands = _field_candidates(prev, req.field)
        if str(req.value) not in cands:
            raise HTTPException(status_code=422, detail=f"value is not one of the retained candidates for {req.field}: "
                                                          f"{sorted(cands)}")
        old = values.get(req.field)
        values[req.field] = req.value
        keep_old_as_conflict(req.field, old)
        c = cands[str(req.value)]
        lineage[req.field] = {"value": req.value, "dataset_id": c["dataset_id"], "record_id": c["record_id"],
                              "rule": "STEWARD_SELECTION", "status": "RESOLVED", "confidence": 1.0,
                              "selected_at": now, "reason": f"selected among conflicting candidates by {req.reviewer}"}
        conf[req.field] = 1.0
        conflicts[req.field] = [x for x in conflicts.get(req.field, []) if str(x["value"]) != str(req.value)]
        needs = [n for n in needs if n["field"] != req.field]
    elif req.action == "REJECT_FIELD_VALUE":
        if not req.field or req.rejected_value is None:
            raise HTTPException(status_code=422, detail="REJECT_FIELD_VALUE requires field and rejected_value")
        cands = _field_candidates(prev, req.field)
        if str(req.rejected_value) not in cands:
            raise HTTPException(status_code=422, detail="rejected_value is not a known candidate for this field")
        remaining = {v: c for v, c in cands.items() if v != str(req.rejected_value)}
        rejected = list((lineage.get(req.field) or {}).get("rejected_values", [])) + [req.rejected_value]
        conflicts[req.field] = [x for x in conflicts.get(req.field, []) if str(x["value"]) != str(req.rejected_value)]
        if remaining:
            v = sorted(remaining, key=lambda k: (-remaining[k]["support"], k))[0]
            values[req.field] = v
            conflicts[req.field] = [x for x in conflicts.get(req.field, []) if str(x["value"]) != v]
            lineage[req.field] = {"value": v, "dataset_id": remaining[v]["dataset_id"], "record_id": remaining[v]["record_id"],
                                  "rule": "STEWARD_REJECTED_ALTERNATIVE", "status": "RESOLVED", "confidence": 0.8,
                                  "selected_at": now, "rejected_values": rejected,
                                  "reason": f"{req.rejected_value!r} rejected by {req.reviewer}; next best retained candidate used"}
            conf[req.field] = 0.8
        else:
            values[req.field] = None
            lineage[req.field] = {"value": None, "rule": "STEWARD_REJECTED_ALL", "status": "NEEDS_REVIEW",
                                  "rejected_values": rejected, "selected_at": now,
                                  "reason": "all candidate values rejected; field left empty"}
            conf[req.field] = 0.0
        needs = [n for n in needs if n["field"] != req.field]
    return _new_version(db, entity, prev, values=values, lineage=lineage, conflicts=conflicts, confidence=conf,
                        needs_review=needs, reason="STEWARD_CORRECTION", reviewer=req.reviewer)


@router.post("/entities/{entity_id}/review")
def submit_review(entity_id: str, req: ReviewRequest, db: Session = Depends(get_db)):
    if req.action not in VALID_ACTIONS:
        raise HTTPException(status_code=422, detail=f"Unknown action {req.action!r}; must be one of {sorted(VALID_ACTIONS)}")
    if not req.reviewer or not req.reviewer.strip():
        raise HTTPException(status_code=422, detail="reviewer (steward identity) is required")
    entity = _get_entity_or_404(db, entity_id)
    before_latest = _latest_golden_version(db, entity_id)
    before_state = {"status": entity.status, "golden_values": before_latest.golden_values if before_latest else None,
                    "golden_version": before_latest.version_number if before_latest else None,
                    "steward_status": before_latest.steward_status if before_latest else None}
    # what the MODEL / SYSTEM had produced -- preserved verbatim next to the human decision
    model_result = {"cluster_run_id": entity.cluster_run_id, "training_run_id": entity.training_run_id,
                    "er_config_version": entity.er_config_version,
                    "survivorship_rule_version": before_latest.survivorship_rule_version if before_latest else None,
                    "proposed_lineage": before_latest.lineage if before_latest else None,
                    "needs_review_fields": before_latest.needs_review_fields if before_latest else None}
    after_state = None

    if req.action == "APPROVE":
        after_state = before_state          # cluster/membership confirmed; the log entry IS the approval

    elif req.action == "APPROVE_GOLDEN_RECORD":
        if before_latest is None:
            raise HTTPException(status_code=422, detail="No golden record to approve.")
        if before_latest.needs_review_fields:
            fields = [n["field"] for n in before_latest.needs_review_fields]
            raise HTTPException(status_code=422, detail=f"Cannot approve: unresolved conflicting fields require review first: {fields}")
        before_latest.steward_status = "APPROVED"
        before_latest.updated_at = dt.datetime.utcnow()
        db.commit()
        after_state = {"golden_version": before_latest.version_number, "steward_status": "APPROVED"}

    elif req.action == "REJECT":
        entity.status = "REJECTED"
        db.commit()
        after_state = {"status": entity.status}

    elif req.action in ("CORRECT", "EDIT_FIELD", "RESOLVE_FIELD", "REJECT_FIELD_VALUE"):
        gv = _field_change(db, entity, before_latest, req)
        after_state = {"golden_values": gv.golden_values, "golden_version": gv.version_number,
                       "steward_status": gv.steward_status, "needs_review_fields": gv.needs_review_fields}

    elif req.action == "SPLIT":
        if not req.split_members:
            raise HTTPException(status_code=422, detail="SPLIT requires split_members")
        target_keys = {(m["dataset_id"], str(m["record_id"])) for m in req.split_members}
        movable = [m for m in entity.members if m.removed_at is None and (m.dataset_id, m.record_id) in target_keys]
        if not movable:
            raise HTTPException(status_code=422, detail="None of the given split_members are currently attached members of this entity")
        if len(movable) == len([m for m in entity.members if m.removed_at is None]):
            raise HTTPException(status_code=422, detail="Cannot split ALL members out of an entity -- leave at least one member behind")
        new_entity_id = f"ENT-{uuid.uuid4().hex[:12]}"
        new_entity = Entity(entity_id=new_entity_id, domain=entity.domain, status="ACTIVE",
                             training_run_id=entity.training_run_id, er_config_version=entity.er_config_version)
        db.add(new_entity)
        db.commit()
        for m in movable:
            m.removed_at = dt.datetime.utcnow()
            db.add(EntityMember(entity_id=new_entity_id, dataset_id=m.dataset_id, record_id=m.record_id,
                                 source="STEWARD_MANUAL", values=m.values))
        db.commit()
        db.refresh(entity)
        db.refresh(new_entity)
        _generate_golden_record(db, entity, reason="STEWARD_CORRECTION", created_by=req.reviewer)
        _generate_golden_record(db, new_entity, reason="STEWARD_CORRECTION", created_by=req.reviewer)
        after_state = {"original_entity_id": entity_id, "new_entity_id": new_entity_id, "members_moved": len(movable)}

    elif req.action == "MERGE":
        if not req.target_entity_id:
            raise HTTPException(status_code=422, detail="MERGE requires target_entity_id")
        target = _get_entity_or_404(db, req.target_entity_id)
        if target.entity_id == entity.entity_id:
            raise HTTPException(status_code=422, detail="Cannot merge an entity into itself")
        for m in entity.members:
            if m.removed_at is None:
                m.removed_at = dt.datetime.utcnow()
                db.add(EntityMember(entity_id=target.entity_id, dataset_id=m.dataset_id, record_id=m.record_id,
                                     source="STEWARD_MANUAL", values=m.values))
        entity.status = "MERGED"
        entity.merged_into_entity_id = target.entity_id
        db.commit()
        db.refresh(target)
        _generate_golden_record(db, target, reason="STEWARD_CORRECTION", created_by=req.reviewer)
        after_state = {"merged_into_entity_id": target.entity_id}

    review = StewardReview(
        review_id=f"REV-{uuid.uuid4().hex[:12]}", entity_id=entity_id, action=req.action,
        before_state=before_state, after_state=after_state, model_result=model_result, reviewer=req.reviewer.strip(),
        reason=req.reason, training_run_id=entity.training_run_id, er_config_version=entity.er_config_version,
    )
    db.add(review)
    if req.pending_review_id:
        pr = db.query(PendingReview).filter(PendingReview.pending_review_id == req.pending_review_id).first()
        if pr:
            pr.status = "RESOLVED"
            pr.resolved_review_id = review.review_id
    db.commit()

    return {"review_id": review.review_id, "entity_id": entity_id, "action": req.action,
            "before_state": before_state, "after_state": after_state, "model_result": model_result,
            "reviewer": req.reviewer, "created_at": review.created_at.isoformat()}


def _will_be_final(db: Session, entity_id: Optional[str]) -> Optional[bool]:
    """Would this entity be listed in Final Master Data right now? (same rule as master_data())."""
    if not entity_id:
        return None
    e = db.query(Entity).filter(Entity.entity_id == entity_id).first()
    latest = _latest_golden_version(db, entity_id)
    if e is None or latest is None or e.status != "ACTIVE":
        return False
    n = len([m for m in e.members if m.removed_at is None])
    return latest.steward_status == "APPROVED" or (n == 1 and latest.steward_status in ("PROPOSED", "CORRECTED"))


class ResolvePendingRequest(BaseModel):
    decision: str            # CONFIRM_MATCH | REJECT_MATCH (mark distinct) | UNSURE
    reviewer: str
    reason: Optional[str] = None


@router.post("/reviews/pending/{pending_review_id}/resolve")
def resolve_pending_review(pending_review_id: str, req: ResolvePendingRequest, db: Session = Depends(get_db)):
    """Resolves an ambiguous incoming-record match. The steward's answer is stored as
    an audit event AND as a REAL_STEWARD labelled pair for future training/evaluation
    (no retraining is triggered)."""
    if req.decision not in ("CONFIRM_MATCH", "REJECT_MATCH", "UNSURE"):
        raise HTTPException(status_code=422, detail="decision must be CONFIRM_MATCH, REJECT_MATCH or UNSURE")
    if not req.reviewer.strip():
        raise HTTPException(status_code=422, detail="reviewer (steward identity) is required")
    p = db.query(PendingReview).filter(PendingReview.pending_review_id == pending_review_id).first()
    if p is None:
        raise HTTPException(status_code=404, detail=f"Pending review {pending_review_id} not found")
    if p.status != "PENDING":
        raise HTTPException(status_code=409, detail="This item was already resolved")
    run = db.query(TrainingRun).filter(TrainingRun.training_run_id == p.training_run_id).first() if p.training_run_id else None
    cache = _DatasetRecordCache(db)
    cand_values = p.candidate_values or cache.values_for(p.candidate_dataset_id, p.candidate_record_id)
    model_result = {"decision": "AMBIGUOUS", "score": p.score, "match_threshold": p.match_threshold,
                    "review_threshold": p.review_threshold, "model_version": p.training_run_id,
                    "candidate_entity_id": p.entity_id, "candidate_other_record_id": p.candidate_other_record_id,
                    "batch_run_id": p.batch_run_id}

    target_entity: Optional[Entity] = _get_entity_or_404(db, p.entity_id) if p.entity_id else None
    other_values, other_id = None, None
    if target_entity is None and p.candidate_other_record_id:
        m = (db.query(EntityMember).filter(EntityMember.dataset_id == p.candidate_dataset_id,
                                           EntityMember.record_id == p.candidate_other_record_id,
                                           EntityMember.removed_at.is_(None)).first())
        if m is not None:
            target_entity = db.query(Entity).filter(Entity.entity_id == m.entity_id).first()
        other_values = (m.values if m else None) or cache.values_for(p.candidate_dataset_id, p.candidate_other_record_id)
        other_id = p.candidate_other_record_id

    latest = _latest_golden_version(db, target_entity.entity_id) if target_entity else None
    counterpart_values = (latest.golden_values if latest else None) or other_values or {}
    counterpart_id = f"ENTITY::{target_entity.entity_id}" if target_entity else (other_id or "-")
    er_cfg = run.er_config_version if run else None
    label_map = {"CONFIRM_MATCH": "MATCH", "REJECT_MATCH": "DISTINCT", "UNSURE": "UNSURE"}
    label_repo.add_pair(db, domain=p.domain, dataset_id=p.candidate_dataset_id, record_id_a=p.candidate_record_id,
                        record_id_b=counterpart_id, values_a=cand_values or {}, values_b=counterpart_values,
                        strategy="REVIEW_DECISION", model_score=p.score, label=label_map[req.decision],
                        labeler=req.reviewer, reason=req.reason, er_config_version=er_cfg,
                        evidence={"pending_review_id": pending_review_id, "model_version": p.training_run_id})
    if req.decision == "UNSURE":
        return {"pending_review_id": pending_review_id, "decision": "UNSURE", "status": "PENDING",
                "note": "Recorded as UNSURE (excluded from supervised training); the item stays in the queue."}

    def _entity_for(record_values):
        eid = f"ENT-{uuid.uuid4().hex[:12]}"
        e = Entity(entity_id=eid, domain=p.domain, status="ACTIVE", training_run_id=p.training_run_id, er_config_version=er_cfg)
        db.add(e)
        db.add(EntityMember(entity_id=eid, dataset_id=p.candidate_dataset_id, record_id=p.candidate_record_id,
                             source="INCOMING", values=record_values or None))
        db.commit()
        db.refresh(e)
        return e

    before = {"golden_values": latest.golden_values if latest else None,
              "golden_version": latest.version_number if latest else None}
    other_entity_id = None        # entity that ends up holding the OTHER record of a record-vs-record pair
    if req.decision == "CONFIRM_MATCH":
        already = target_entity is not None and any(
            m.removed_at is None and m.dataset_id == p.candidate_dataset_id and str(m.record_id) == str(p.candidate_record_id)
            for m in target_entity.members)
        if target_entity is not None and already:
            gv = _latest_golden_version(db, target_entity.entity_id) or _generate_golden_record(
                db, target_entity, reason="STEWARD_CONFIRMED_MATCH", created_by=req.reviewer)
            result_entity = target_entity
            other_entity_id = target_entity.entity_id
        elif target_entity is not None:
            db.add(EntityMember(entity_id=target_entity.entity_id, dataset_id=p.candidate_dataset_id,
                                 record_id=p.candidate_record_id, source="INCOMING", values=cand_values or None))
            db.commit()
            db.refresh(target_entity)
            gv = _generate_golden_record(db, target_entity, reason="STEWARD_CONFIRMED_MATCH", created_by=req.reviewer)
            result_entity = target_entity
            other_entity_id = target_entity.entity_id
        else:
            result_entity = _entity_for(cand_values)
            if other_id is not None:
                db.add(EntityMember(entity_id=result_entity.entity_id, dataset_id=p.candidate_dataset_id,
                                     record_id=other_id, source="INCOMING", values=other_values or None))
                db.commit()
                db.refresh(result_entity)
            gv = _generate_golden_record(db, result_entity, reason="STEWARD_CONFIRMED_MATCH", created_by=req.reviewer)
            other_entity_id = result_entity.entity_id
    else:
        # REJECT_MATCH -- DISTINCT: the two records must NOT share an entity.
        # Each side keeps (or gets) its own singleton entity and its own
        # Golden Record: a DISTINCT record is never discarded from master data.
        result_entity = _entity_for(cand_values)
        gv = _generate_golden_record(db, result_entity, reason="INITIAL_GENERATION", created_by=req.reviewer)
        if target_entity is None and other_id is not None and p.candidate_other_record_id:
            other_entity_id = f"ENT-{uuid.uuid4().hex[:12]}"
            other_entity = Entity(entity_id=other_entity_id, domain=p.domain, status="ACTIVE",
                                  training_run_id=p.training_run_id, er_config_version=er_cfg)
            db.add(other_entity)
            db.add(EntityMember(entity_id=other_entity_id, dataset_id=p.candidate_dataset_id,
                                 record_id=p.candidate_other_record_id, source="INCOMING",
                                 values=other_values or None))
            db.commit()
            db.refresh(other_entity)
            _generate_golden_record(db, other_entity, reason="INITIAL_GENERATION", created_by=req.reviewer)
            other_entity_id = other_entity.entity_id
        elif target_entity is not None:
            other_entity_id = target_entity.entity_id     # the other side already has its own entity

    review = StewardReview(
        review_id=f"REV-{uuid.uuid4().hex[:12]}", entity_id=result_entity.entity_id, action=req.decision,
        before_state=before, after_state={"entity_id": result_entity.entity_id, "golden_version": gv.version_number,
                                          "golden_values": gv.golden_values},
        model_result=model_result, reviewer=req.reviewer.strip(), reason=req.reason,
        training_run_id=p.training_run_id, er_config_version=er_cfg)
    db.add(review)
    p.status, p.resolved_review_id = "RESOLVED", review.review_id
    audit = db.query(IncomingRecordAudit).filter(IncomingRecordAudit.pending_review_id == pending_review_id).first()
    if audit:
        audit.steward_decision, audit.steward_decided_at = req.decision, dt.datetime.utcnow()
        audit.resulting_entity_id, audit.golden_record_version = result_entity.entity_id, gv.version_number

    # MIRROR items. For a record-vs-record ambiguity inside one batch the matcher emits one queue item per
    # direction (A vs B *and* B vs A). Answering one answers the other: left open, the mirror let a steward
    # create a second entity for the same record (DISTINCT) or attach it twice (MATCH), and could even record
    # a contradictory label for the same pair.
    mirrors_resolved = 0
    if p.candidate_other_record_id and not p.entity_id:
        mirrors = (db.query(PendingReview)
                   .filter(PendingReview.status == "PENDING", PendingReview.domain == p.domain,
                           PendingReview.candidate_dataset_id == p.candidate_dataset_id,
                           PendingReview.candidate_record_id == p.candidate_other_record_id,
                           PendingReview.candidate_other_record_id == p.candidate_record_id,
                           PendingReview.pending_review_id != p.pending_review_id).all())
        for mr in mirrors:
            mr.status, mr.resolved_review_id = "RESOLVED", review.review_id
            m_audit = db.query(IncomingRecordAudit).filter(IncomingRecordAudit.pending_review_id == mr.pending_review_id).first()
            if m_audit:
                m_audit.steward_decision, m_audit.steward_decided_at = req.decision, dt.datetime.utcnow()
                m_audit.resulting_entity_id = other_entity_id or result_entity.entity_id
            mirrors_resolved += 1
    db.commit()
    return {"pending_review_id": pending_review_id, "decision": req.decision, "review_id": review.review_id,
            "resulting_entity_id": result_entity.entity_id, "golden_record_version": gv.version_number,
            "model_result": model_result, "label_recorded": label_map[req.decision],
            "mirror_items_resolved": mirrors_resolved,
            "other_entity_id": other_entity_id,
            "will_appear_in_final_master_data": _will_be_final(db, result_entity.entity_id),
            "counterpart_will_appear_in_final_master_data": _will_be_final(db, other_entity_id) if other_entity_id else None}


# =============================================================================
# Module 13: Continuous incoming data. The SAME service handles a single record
# or a batch. Records arrive through the existing ingestion/mapping/
# standardization modules (never a second standardization); matching uses the
# domain's ACTIVE model against mastered ENTITY golden records; nothing here
# ever trains or retrains a model.
# =============================================================================

class ApplyActiveModelRequest(BaseModel):
    match_threshold: Optional[float] = None       # None -> the active model version's stored threshold
    ambiguous_threshold: Optional[float] = None   # None -> the model's stored review (uncertain-band) threshold
    force: bool = False                           # re-process a dataset that was already processed


def apply_active_model_impl(db: Session, dataset_id: str, match_threshold: Optional[float] = None,
                            review_threshold: Optional[float] = None, force: bool = False) -> dict:
    from app.modules.ingestion import load_layer, IngestionError
    import time
    ds = _get_dataset_or_404(db, dataset_id)
    active_run = _load_active_model_or_404(db, ds.domain)

    match_threshold = match_threshold if match_threshold is not None else (
        active_run.threshold if active_run.threshold is not None else 0.5)
    review_threshold = review_threshold if review_threshold is not None else (
        active_run.review_threshold if active_run.review_threshold is not None else max(0.0, match_threshold - 0.15))
    if review_threshold > match_threshold:
        raise HTTPException(status_code=422, detail="ambiguous_threshold must be <= match_threshold")
    try:
        load_layer(dataset_id, "standardized")
    except IngestionError:
        raise HTTPException(status_code=422, detail="This dataset has not been standardized. Run the existing standardization "
                                                      "step first -- incoming data is never matched on RAW values.")
    mapped = set(mapping_dict_for_dataset(db, dataset_id).keys())
    missing = [f for f in active_run.match_fields if f not in mapped]
    if missing:
        raise HTTPException(status_code=422, detail=f"Dataset has no confirmed mapping for the active model's MATCH fields: {missing}")
    if not force and db.query(IncomingBatchRun).filter(IncomingBatchRun.dataset_id == dataset_id,
                                                        IncomingBatchRun.training_run_id == active_run.training_run_id,
                                                        IncomingBatchRun.status == "COMPLETED").first():
        raise HTTPException(status_code=409, detail="This dataset was already processed with the active model. "
                                                      "Pass force=true to process it again.")

    batch_run_id = f"BR-{uuid.uuid4().hex[:12]}"
    batch = IncomingBatchRun(batch_run_id=batch_run_id, dataset_id=dataset_id, domain=ds.domain,
                              training_run_id=active_run.training_run_id, threshold=match_threshold,
                              review_threshold=review_threshold, status="CREATED")
    db.add(batch)
    db.commit()
    job = start_job(db, "INCOMING_BATCH", domain=ds.domain, dataset_id=dataset_id, model_version=active_run.training_run_id)
    t0 = time.time()

    try:
        batch.status = "MATCHING"
        db.commit()
        match_fields = active_run.match_fields
        new_records = {rid: coerce_record(r, match_fields)
                       for rid, r in records_from_standardized(db, dataset_id, match_fields).items()}
        if not new_records:
            raise ValueError("Incoming batch is empty (no records).")
        cache = _DatasetRecordCache(db)
        n = len(new_records)
        batch.field_missingness = {f: round(sum(1 for r in new_records.values() if r.get(f) is None) / n, 4) for f in match_fields}

        existing_entities = db.query(Entity).filter(Entity.domain == ds.domain, Entity.status == "ACTIVE").all()
        representatives = {}
        for e in existing_entities:
            latest = _latest_golden_version(db, e.entity_id)
            if latest and latest.golden_values:
                representatives[e.entity_id] = coerce_record(latest.golden_values, match_fields)

        with open(active_run.model_path, "rb") as f:
            settings_bytes = f.read()
        result = run_incremental_matching(settings_bytes, representatives, new_records,
                                          match_threshold=match_threshold, ambiguous_threshold=review_threshold)
        if result.status != "COMPLETED":
            raise RuntimeError(result.error or "incremental matching failed")

        # union-find so a chain of NEW records that match each other lands in ONE new entity
        parent = {rid: rid for rid in new_records.keys()}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for rid, d in result.decisions.items():
            if d["decision"] == "MATCHES_ANOTHER_NEW_RECORD" and int(d["other_record_id"]) in parent:
                ra, rb = find(int(rid)), find(int(d["other_record_id"]))
                if ra != rb:
                    parent[rb] = ra

        attached = created = ambiguous = 0
        group_to_entity: dict = {}
        audits = []

        def _audit(rid, d, decision, entity_id, gv, pending_id=None):
            a = IncomingRecordAudit(
                audit_id=f"AUD-{uuid.uuid4().hex[:12]}", batch_run_id=batch_run_id, domain=ds.domain,
                source_dataset_id=dataset_id, source_record_id=str(rid),
                standardized_values=cache.values_for(dataset_id, rid), training_run_id=active_run.training_run_id,
                match_threshold=match_threshold, review_threshold=review_threshold,
                candidate_count=d.get("candidate_count", 0), selected_entity_id=d.get("entity_id"),
                match_score=d.get("score"), decision=decision, resulting_entity_id=entity_id,
                golden_record_version=gv.version_number if gv else None, pending_review_id=pending_id)
            db.add(a)
            audits.append(a)

        for rid_str, d in result.decisions.items():
            rid = int(rid_str)
            decision = d["decision"]
            if decision == "MATCH":
                entity = _get_entity_or_404(db, d["entity_id"])
                _add_member(db, cache, entity.entity_id, dataset_id, rid, "INCOMING")
                db.commit()
                db.refresh(entity)
                gv = _generate_golden_record(db, entity, reason="INCOMING_BATCH_UPDATE", cache=cache)
                _audit(rid, d, "MATCH", entity.entity_id, gv)
                attached += 1
            elif decision == "AMBIGUOUS":
                pid = f"PR-{uuid.uuid4().hex[:12]}"
                db.add(PendingReview(
                    pending_review_id=pid, domain=ds.domain, kind="AMBIGUOUS_MATCH", entity_id=d.get("entity_id"),
                    candidate_dataset_id=dataset_id, candidate_record_id=str(rid),
                    candidate_other_record_id=str(d["other_record_id"]) if d.get("other_record_id") is not None else None,
                    candidate_values=cache.values_for(dataset_id, rid), score=d.get("score"),
                    training_run_id=active_run.training_run_id, batch_run_id=batch_run_id,
                    match_threshold=match_threshold, review_threshold=review_threshold))
                db.commit()
                _audit(rid, d, "AMBIGUOUS", None, None, pending_id=pid)
                ambiguous += 1
            elif decision in ("NEW_ENTITY", "MATCHES_ANOTHER_NEW_RECORD"):
                # NOTE on counting: `created` counts RECORDS assigned to a new
                # (batch-originated) entity, not the number of new entities --
                # several new records that match each other (chained
                # MATCHES_ANOTHER_NEW_RECORD) land in ONE new entity but each
                # record still counts here, so
                # attached_to_existing_count + new_entity_count + ambiguous_count
                # always equals record_count.
                group = find(rid)
                if group in group_to_entity:
                    entity = _get_entity_or_404(db, group_to_entity[group])
                    _add_member(db, cache, entity.entity_id, dataset_id, rid, "INCOMING")
                    db.commit()
                    db.refresh(entity)
                    gv = _generate_golden_record(db, entity, reason="INCOMING_BATCH_UPDATE", cache=cache)
                else:
                    new_entity_id = f"ENT-{uuid.uuid4().hex[:12]}"
                    entity = Entity(entity_id=new_entity_id, domain=ds.domain, status="ACTIVE",
                                     training_run_id=active_run.training_run_id,
                                     er_config_version=active_run.er_config_version)
                    db.add(entity)
                    _add_member(db, cache, new_entity_id, dataset_id, rid, "INCOMING")
                    db.commit()
                    db.refresh(entity)
                    gv = _generate_golden_record(db, entity, reason="INITIAL_GENERATION", cache=cache)
                    group_to_entity[group] = new_entity_id
                created += 1
                _audit(rid, d, "NEW_ENTITY", entity.entity_id, gv)

        batch.status = "COMPLETED"
        batch.record_count = n
        batch.attached_to_existing_count = attached
        batch.new_entity_count = created
        batch.ambiguous_count = ambiguous
        batch.candidate_pair_count = result.candidate_pair_count
        batch.processing_seconds = round(time.time() - t0, 3)
        batch.completed_at = dt.datetime.utcnow()
        db.commit()
        finish_job(db, job, True, artifact_ref=batch_run_id,
                   result={"records": n, "attached": attached, "new_entities": created, "ambiguous": ambiguous})
    except Exception as e:  # noqa: BLE001
        db.rollback()
        batch = db.query(IncomingBatchRun).filter(IncomingBatchRun.batch_run_id == batch_run_id).first()
        batch.status = "FAILED"
        batch.error = f"{type(e).__name__}: {e}"
        batch.processing_seconds = round(time.time() - t0, 3)
        batch.completed_at = dt.datetime.utcnow()
        db.commit()
        finish_job(db, job, False, error=batch.error, artifact_ref=batch_run_id)

    return _batch_run_response(batch)


@router.post("/datasets/{dataset_id}/er/apply-active-model")
def apply_active_model(dataset_id: str, req: ApplyActiveModelRequest = ApplyActiveModelRequest(),
                       db: Session = Depends(get_db)):
    """Match an already ingested + standardized dataset (a batch, or a single
    record) against the domain's ACTIVE model and the mastered entities. NEVER trains."""
    return apply_active_model_impl(db, dataset_id, req.match_threshold, req.ambiguous_threshold, req.force)


def _batch_run_response(batch: IncomingBatchRun) -> dict:
    return {
        "batch_run_id": batch.batch_run_id, "dataset_id": batch.dataset_id, "domain": batch.domain,
        "training_run_id": batch.training_run_id, "model_version": batch.training_run_id,
        "threshold": batch.threshold, "review_threshold": batch.review_threshold, "status": batch.status,
        "record_count": batch.record_count, "attached_to_existing_count": batch.attached_to_existing_count,
        "new_entity_count": batch.new_entity_count, "ambiguous_count": batch.ambiguous_count,
        "candidate_pair_count": batch.candidate_pair_count, "processing_seconds": batch.processing_seconds,
        "field_missingness": batch.field_missingness, "error": batch.error,
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
    }


@router.get("/datasets/{dataset_id}/er/incoming-batches/{batch_run_id}")
def get_incoming_batch(dataset_id: str, batch_run_id: str, db: Session = Depends(get_db)):
    batch = db.query(IncomingBatchRun).filter(IncomingBatchRun.batch_run_id == batch_run_id,
                                               IncomingBatchRun.dataset_id == dataset_id).first()
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Incoming batch run {batch_run_id} not found")
    return _batch_run_response(batch)


def _audit_response(a: IncomingRecordAudit) -> dict:
    return {"audit_id": a.audit_id, "batch_run_id": a.batch_run_id, "source_dataset_id": a.source_dataset_id,
            "source_record_id": a.source_record_id, "ingested_at": a.ingested_at.isoformat() if a.ingested_at else None,
            "standardized_values": a.standardized_values, "model_version": a.training_run_id,
            "match_threshold": a.match_threshold, "review_threshold": a.review_threshold,
            "candidate_count": a.candidate_count, "selected_entity_id": a.selected_entity_id,
            "match_score": a.match_score, "decision": a.decision, "resulting_entity_id": a.resulting_entity_id,
            "golden_record_version": a.golden_record_version, "pending_review_id": a.pending_review_id,
            "steward_decision": a.steward_decision,
            "steward_decided_at": a.steward_decided_at.isoformat() if a.steward_decided_at else None}


@router.get("/datasets/{dataset_id}/er/incoming-batches/{batch_run_id}/results")
def get_incoming_batch_results(dataset_id: str, batch_run_id: str, limit: int = 500, db: Session = Depends(get_db)):
    batch = db.query(IncomingBatchRun).filter(IncomingBatchRun.batch_run_id == batch_run_id,
                                               IncomingBatchRun.dataset_id == dataset_id).first()
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Incoming batch run {batch_run_id} not found")
    rows = db.query(IncomingRecordAudit).filter(IncomingRecordAudit.batch_run_id == batch_run_id).limit(limit).all()
    return {"batch": _batch_run_response(batch), "records": [_audit_response(a) for a in rows], "count": len(rows)}


@router.get("/domains/{domain}/incoming-batches")
def list_incoming_batches(domain: str, db: Session = Depends(get_db)):
    rows = db.query(IncomingBatchRun).filter(IncomingBatchRun.domain == domain).order_by(
        IncomingBatchRun.created_at.desc()).limit(100).all()
    return {"batches": [_batch_run_response(b) for b in rows]}


@router.get("/domains/{domain}/incoming-audit")
def incoming_audit(domain: str, decision: Optional[str] = None, limit: int = 200, db: Session = Depends(get_db)):
    q = db.query(IncomingRecordAudit).filter(IncomingRecordAudit.domain == domain)
    if decision:
        q = q.filter(IncomingRecordAudit.decision == decision)
    rows = q.order_by(IncomingRecordAudit.ingested_at.desc()).limit(limit).all()
    return {"records": [_audit_response(a) for a in rows], "count": len(rows)}


# =============================================================================
# Module 10: Monitoring -- three clearly separate categories:
#   DATA QUALITY  (are the inputs drifting?)
#   MODEL QUALITY (is the model right? real steward-labelled evidence only)
#   OPERATIONAL PERFORMANCE (is the pipeline healthy/fast?)
# The earlier keys (training_validation / operational_matching /
# steward_feedback) are kept for backward compatibility.
# =============================================================================

@router.get("/monitoring/summary")
def monitoring_summary(domain: str, db: Session = Depends(get_db)):
    # Queried by each run's OWN stored domain column, never via a join to Dataset
    # (a join would silently lose stats once the source dataset is deleted).
    training_runs = db.query(TrainingRun).filter(TrainingRun.domain == domain).all()
    match_runs = db.query(MatchRun).filter(MatchRun.domain == domain).all()
    cluster_runs = db.query(ClusterRun).filter(ClusterRun.domain == domain).all()
    batch_runs = db.query(IncomingBatchRun).filter(IncomingBatchRun.domain == domain).order_by(IncomingBatchRun.created_at.asc()).all()
    entities = db.query(Entity).filter(Entity.domain == domain).all()
    reviews = db.query(StewardReview).filter(StewardReview.entity_id.in_([e.entity_id for e in entities])).all() if entities else []
    pending = db.query(PendingReview).filter(PendingReview.domain == domain, PendingReview.status == "PENDING").all()
    active = db.query(ActiveModel).filter(ActiveModel.domain == domain).first()
    jobs = db.query(ProcessingJob).filter(ProcessingJob.domain == domain).all()

    review_action_counts: dict = {}
    for r in reviews:
        review_action_counts[r.action] = review_action_counts.get(r.action, 0) + 1

    labels = label_repo.label_summary(db, domain)
    review_labels = db.query(LabeledPair).filter(LabeledPair.domain == domain, LabeledPair.strategy == "REVIEW_DECISION").all()

    # ---- MODEL QUALITY: real steward-labelled evaluation of the ACTIVE model
    real_eval = None
    if active is not None:
        run = db.query(TrainingRun).filter(TrainingRun.training_run_id == active.training_run_id).first()
        if run is not None and run.model_path:
            try:
                real_eval = evaluate_run_on_unseen_real_labels(db, run)
            except Exception as e:  # noqa: BLE001
                real_eval = {"error": f"{type(e).__name__}: {e}"}

    # ---- DATA QUALITY: missingness drift, training baseline vs each incoming batch
    baseline = {}
    if active is not None:
        md = load_model_metadata(db.query(TrainingRun).filter(TrainingRun.training_run_id == active.training_run_id).first().model_path)
        baseline = (md or {}).get("training_field_stats") or {}
    drift = []
    for b in batch_runs:
        if b.status != "COMPLETED" or not b.field_missingness:
            continue
        rec = b.record_count or 0
        drift.append({
            "batch_run_id": b.batch_run_id, "records": rec,
            "field_missingness": b.field_missingness,
            "missingness_change_vs_training": {f: round(v - baseline.get(f, v), 4) for f, v in b.field_missingness.items()},
            "match_rate": round((b.attached_to_existing_count or 0) / rec, 4) if rec else None,
            "candidate_rate": round((b.candidate_pair_count or 0) / rec, 4) if rec else None,
        })

    done = [b for b in batch_runs if b.status == "COMPLETED"]
    failed_jobs = [j for j in jobs if j.status == "FAILED"]
    proc_times = [b.processing_seconds for b in done if b.processing_seconds is not None]

    return {
        "domain": domain,
        "active_model": {"training_run_id": active.training_run_id, "activated_at": active.activated_at.isoformat()} if active else None,
        # ------------------------------------------------------------------ new, categorised
        "data_quality": {
            "note": "Input drift only -- says nothing about correctness. Compares match-field missingness of each "
                    "incoming batch to the training data, plus per-batch match and candidate rates.",
            "training_baseline_missingness": baseline, "batches": drift,
        },
        "model_quality": {
            "note": "Only REAL steward-labelled evidence counts here. Synthetic validation lives under training_validation.",
            "real_labelled_evaluation": real_eval,
            "labels": labels,
            "steward_confirmed_matches": sum(1 for l in review_labels if l.label == "MATCH"),
            "steward_rejected_matches": sum(1 for l in review_labels if l.label == "DISTINCT"),
            "false_positive_rate": (real_eval or {}).get("false_positive_rate"),
            "false_negative_rate": (real_eval or {}).get("false_negative_rate"),
        },
        "operational_performance": {
            "note": "Volumes, timing and failures -- not correctness.",
            "incoming_batches": len(batch_runs), "records_processed": sum(b.record_count or 0 for b in done),
            "records_matched": sum(b.attached_to_existing_count or 0 for b in done),
            "new_entities": sum(b.new_entity_count or 0 for b in done),
            "ambiguous_records": sum(b.ambiguous_count or 0 for b in done),
            "candidate_pairs": sum(b.candidate_pair_count or 0 for b in done),
            "avg_processing_seconds": round(sum(proc_times) / len(proc_times), 3) if proc_times else None,
            "jobs_total": len(jobs), "jobs_failed": len(failed_jobs),
            "recent_failures": [{"job_id": j.job_id, "job_type": j.job_type, "error": j.error} for j in failed_jobs[-5:]],
        },
        # ------------------------------------------------------------------ kept for compatibility
        "training_validation": {
            "note": "SYNTHETIC validation (controlled corruptions of real records) and REAL held-out evaluation, "
                    "reported separately per run. Neither is a production-accuracy claim.",
            "runs": [{"training_run_id": r.training_run_id, "status": r.status, "lifecycle_status": r.lifecycle_status,
                      "evaluation_metrics": r.evaluation_metrics, "real_evaluation": r.real_evaluation,
                      "label_counts": r.label_counts, "threshold": r.threshold,
                      "threshold_calibration": r.threshold_calibration} for r in training_runs],
        },
        "operational_matching": {
            "note": "Real statistics from actual runs over real (uncorrupted) data -- volumes only, not a correctness claim.",
            "match_runs": len(match_runs),
            "total_candidate_pairs": sum(m.candidate_pair_count or 0 for m in match_runs),
            "total_match_count": sum(m.match_count or 0 for m in match_runs),
            "total_non_match_count": sum(m.non_match_count or 0 for m in match_runs),
            "total_uncertain_count": sum(m.uncertain_count or 0 for m in match_runs),
            "cluster_runs": len(cluster_runs),
            "total_clusters": sum(c.cluster_count or 0 for c in cluster_runs),
            "total_singletons": sum(c.singleton_count or 0 for c in cluster_runs),
            "incoming_batches": len(batch_runs),
            "incoming_records_attached_to_existing": sum(b.attached_to_existing_count or 0 for b in batch_runs),
            "incoming_new_entities_created": sum(b.new_entity_count or 0 for b in batch_runs),
            "incoming_ambiguous_flagged": sum(b.ambiguous_count or 0 for b in batch_runs),
        },
        "steward_feedback": {
            "note": "Human-confirmed evidence, only for what has actually been reviewed so far.",
            "total_reviews": len(reviews), "by_action": review_action_counts, "pending_review_count": len(pending),
        },
        "entities": {"total": len(entities), "active": sum(1 for e in entities if e.status == "ACTIVE"),
                     "merged": sum(1 for e in entities if e.status == "MERGED"),
                     "rejected": sum(1 for e in entities if e.status == "REJECTED")},
    }


# =============================================================================
# Survivorship configuration (per domain, versioned) and jobs
# =============================================================================

class SurvivorshipRequest(BaseModel):
    rule_order: Optional[list] = None
    field_rules: Optional[dict] = None
    source_priority: Optional[dict] = None
    recency_field: Optional[str] = None
    timestamp_dayfirst: Optional[bool] = None
    no_completeness_substrings: Optional[list] = None
    updated_by: str = "steward"


@router.get("/domains/{domain}/survivorship")
def get_survivorship(domain: str, db: Session = Depends(get_db)):
    row = db.query(SurvivorshipConfig).filter(SurvivorshipConfig.domain == domain).first()
    lifecycle = (row.lifecycle_status or "ACTIVE") if row else None
    if row is None:
        cfg = normalize_config(None)
    else:
        cfg = normalize_config(row.config)
        cfg["version"] = row.version
    return {"domain": domain, "config": cfg, "is_default": row is None,
            "lifecycle_status": lifecycle,
            # only an ACTIVE (or legacy) configuration drives production Golden Records
            "generates_golden_records": lifecycle in ("ACTIVE", None),
            "updated_by": row.updated_by if row else None,
            "updated_at": row.updated_at.isoformat() if row and row.updated_at else None}


@router.get("/domains/{domain}/survivorship/detected")
def detect_survivorship_inputs(domain: str, db: Session = Depends(get_db)):
    """What the data actually offers survivorship: the distinct source systems found in
    every dataset of this domain (via the confirmed `source_system` mapping) and whether
    a last-updated column is mapped and parseable. Lets the steward set priorities from
    real values instead of typing names from memory."""
    from collections import Counter
    from app.modules.ingestion import load_layer
    from app.modules.golden_record_service import _parse_ts
    cfg = _survivorship_config(db, domain)
    sys_field, rec_field = cfg.get("source_system_field"), cfg.get("recency_field")
    systems: Counter = Counter()
    datasets = []
    for ds in db.query(Dataset).filter(Dataset.domain == domain).all():
        mapping = mapping_dict_for_dataset(db, ds.dataset_id)
        info = {"dataset_id": ds.dataset_id, "rows": ds.row_count,
                "source_system_mapped": sys_field in mapping, "recency_mapped": bool(rec_field) and rec_field in mapping,
                "timestamps_parsed_pct": None}
        try:
            df = load_layer(ds.dataset_id, "standardized")
        except Exception:  # noqa: BLE001
            try:
                df = load_layer(ds.dataset_id, "raw")
            except Exception:  # noqa: BLE001
                df = None
        if df is not None:
            col = mapping.get(sys_field)
            if col in df.columns:
                for v, n in df[col].dropna().astype(str).str.strip().value_counts().items():
                    if v:
                        systems[v] += int(n)
            col = mapping.get(rec_field) if rec_field else None
            if col in df.columns:
                sample = df[col].dropna().head(500)
                ok = sum(1 for v in sample if _parse_ts(v, cfg.get("timestamp_dayfirst", True)) is not None)
                info["timestamps_parsed_pct"] = round(100.0 * ok / len(sample), 1) if len(sample) else None
        datasets.append(info)
    return {"domain": domain, "source_system_field": sys_field, "recency_field": rec_field,
            "source_systems": [{"name": k, "records": v} for k, v in systems.most_common()],
            "datasets": datasets}


@router.put("/domains/{domain}/survivorship")
def put_survivorship(domain: str, req: SurvivorshipRequest, db: Session = Depends(get_db)):
    """Changing the rules bumps the config version; existing golden record
    versions keep the version they were produced under and are never rewritten."""
    current = _survivorship_config(db, domain)
    proposed = dict(current)
    for k in ("rule_order", "field_rules", "source_priority", "recency_field", "timestamp_dayfirst",
              "no_completeness_substrings"):
        v = getattr(req, k)
        if v is not None:
            proposed[k] = v
    errs = validate_config(proposed)
    from app.domain_schemas import get_schema
    try:
        known = set(get_schema(domain).field_names())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if proposed.get("recency_field") and proposed["recency_field"] not in known:
        errs.append(f"recency_field {proposed['recency_field']!r} is not a canonical {domain} field; "
                    f"allowed: {sorted(known)}")
    if errs:
        raise HTTPException(status_code=422, detail="; ".join(errs))
    row = db.query(SurvivorshipConfig).filter(SurvivorshipConfig.domain == domain).first()
    stored = {k: proposed[k] for k in DEFAULT_CONFIG if k != "version"}
    if row is None:
        row = SurvivorshipConfig(domain=domain, config=stored, version=2,
                                 lifecycle_status="DRAFT", updated_by=req.updated_by)
        db.add(row)
    else:
        row.config, row.version, row.updated_by = stored, row.version + 1, req.updated_by
        row.lifecycle_status = "DRAFT"   # every edit re-enters the lifecycle as a draft
        row.updated_at = dt.datetime.utcnow()
    db.commit()
    return get_survivorship(domain, db)


@router.post("/domains/{domain}/survivorship/submit")
def submit_survivorship(domain: str, updated_by: str = "steward", db: Session = Depends(get_db)):
    """DRAFT -> STEWARD_CONFIRMED (a steward has reviewed the draft rules)."""
    row = db.query(SurvivorshipConfig).filter(SurvivorshipConfig.domain == domain).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No survivorship configuration saved for domain '{domain}'.")
    if row.lifecycle_status not in ("DRAFT", None):
        raise HTTPException(status_code=422, detail=f"Config is {row.lifecycle_status}; only a DRAFT can be submitted for confirmation.")
    row.lifecycle_status = "STEWARD_CONFIRMED"
    row.updated_by = updated_by
    row.updated_at = dt.datetime.utcnow()
    db.commit()
    return get_survivorship(domain, db)


@router.post("/domains/{domain}/survivorship/confirm")
def confirm_survivorship(domain: str, updated_by: str = "steward", db: Session = Depends(get_db)):
    """STEWARD_CONFIRMED -> ACTIVE. Only an ACTIVE configuration generates
    production Golden Records."""
    row = db.query(SurvivorshipConfig).filter(SurvivorshipConfig.domain == domain).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No survivorship configuration saved for domain '{domain}'.")
    if row.lifecycle_status != "STEWARD_CONFIRMED":
        raise HTTPException(status_code=422, detail=f"Config is {row.lifecycle_status}; submit it for steward confirmation first.")
    row.lifecycle_status = "ACTIVE"
    row.updated_by = updated_by
    row.updated_at = dt.datetime.utcnow()
    db.commit()
    return get_survivorship(domain, db)


@router.get("/domains/{domain}/master-data")
def master_data(domain: str, limit: int = 100, offset: int = 0, scope: str = "final", kind: str = "all",
                q: Optional[str] = None, db: Session = Depends(get_db)):
    """FINAL MASTER DATA: one current Golden Record per ACTIVE entity.

    A record is final when
      * a steward APPROVED its latest Golden Record, or
      * it is a SINGLETON entity (one source record -- an unmatched or
        steward-confirmed DISTINCT record) whose Golden Record has no
        unresolved conflicts. There is nothing to merge or reconcile for a
        single record, so it must not sit outside master data waiting for a
        manual approval click.
    Merged entities (2+ records) still need explicit approval, and any entity
    with unresolved conflicting fields (NEEDS_REVIEW) is never final.

    scope   final (default) | awaiting -- the ACTIVE entities that are NOT final yet, each with
            the reason (NEEDS_REVIEW / PENDING_APPROVAL / RE_APPROVAL). Without this list a record
            that is waiting for approval (e.g. the other side of a steward DISTINCT decision that
            is part of an unapproved merged cluster) is simply invisible.
    kind    all | merged | distinct  (member_count > 1 / == 1)
    q       case-insensitive text search over entity id and Golden Record values.

    Newest-changed first (an approval counts as a change), so a record a steward has just approved
    is on the first page instead of being buried under older/newer singletons.
    Pagination is done in SQL; no per-row queries.
    """
    from sqlalchemy import func, and_, or_, not_, String, cast
    limit, offset = max(1, min(limit, 1000)), max(0, offset)
    if scope not in ("final", "awaiting"):
        raise HTTPException(status_code=422, detail="scope must be 'final' or 'awaiting'")
    if kind not in ("all", "merged", "distinct"):
        raise HTTPException(status_code=422, detail="kind must be 'all', 'merged' or 'distinct'")

    newest = (db.query(GoldenRecordVersion.entity_id.label("eid"),
                       func.max(GoldenRecordVersion.version_number).label("v"))
              .group_by(GoldenRecordVersion.entity_id).subquery())
    members = (db.query(EntityMember.entity_id.label("eid"), func.count(EntityMember.id).label("n"))
               .filter(EntityMember.removed_at.is_(None))
               .group_by(EntityMember.entity_id).subquery())
    member_n = func.coalesce(members.c.n, 0)

    base = (db.query(Entity, GoldenRecordVersion, member_n.label("n"))
            .join(newest, Entity.entity_id == newest.c.eid)
            .join(GoldenRecordVersion, (GoldenRecordVersion.entity_id == newest.c.eid) &
                                       (GoldenRecordVersion.version_number == newest.c.v))
            .outerjoin(members, members.c.eid == Entity.entity_id)
            .filter(Entity.domain == domain, Entity.status == "ACTIVE"))

    is_final = or_(GoldenRecordVersion.steward_status == "APPROVED",
                   and_(member_n == 1, GoldenRecordVersion.steward_status.in_(["PROPOSED", "CORRECTED"])))
    final_q = base.filter(is_final)
    awaiting_q = base.filter(not_(is_final), member_n >= 1)
    total_final = final_q.count()
    singleton_final = final_q.filter(member_n == 1).count()
    total_awaiting = awaiting_q.count()
    total_entities = db.query(Entity).filter(Entity.domain == domain, Entity.status == "ACTIVE").count()

    qry = final_q if scope == "final" else awaiting_q
    if kind == "merged":
        qry = qry.filter(member_n > 1)
    elif kind == "distinct":
        qry = qry.filter(member_n == 1)
    if q and q.strip():
        like = f"%{q.strip()}%"
        qry = qry.filter(or_(Entity.entity_id.ilike(like), cast(GoldenRecordVersion.golden_values, String).ilike(like)))
    matching = qry.count()

    changed_at = func.coalesce(GoldenRecordVersion.updated_at, GoldenRecordVersion.created_at)
    rows = (qry.order_by(changed_at.desc(), Entity.entity_id.asc()).offset(offset).limit(limit).all())

    ever_approved = set()
    if scope == "awaiting" and rows:
        ids = [e.entity_id for e, _, _ in rows]
        ever_approved = {r[0] for r in db.query(GoldenRecordVersion.entity_id)
                         .filter(GoldenRecordVersion.entity_id.in_(ids),
                                 GoldenRecordVersion.steward_status == "APPROVED").distinct().all()}

    out = []
    for e, latest, n in rows:
        approved = latest.steward_status == "APPROVED"
        rec = {
            "golden_record_id": f"GR-{e.entity_id}", "entity_id": e.entity_id,
            "domain": e.domain, "version": latest.version_number, "status": latest.steward_status,
            "approval_basis": "STEWARD_APPROVED" if approved else "SINGLE_RECORD_NO_CONFLICT",
            "values": order_values(e.domain, latest.golden_values),
            "display_name": display_name(e.domain, latest.golden_values),
            "summary": summary_items(e.domain, latest.golden_values),
            "lineage": latest.lineage, "field_confidence": latest.field_confidence,
            "member_count": int(n),
            "approved_by": latest.created_by if approved else None,
            "created_at": latest.created_at.isoformat() if latest.created_at else None,
            "updated_at": latest.updated_at.isoformat() if latest.updated_at else None,
        }
        if scope == "awaiting":
            if latest.steward_status == "NEEDS_REVIEW" or latest.needs_review_fields:
                rec["awaiting_reason"] = "NEEDS_REVIEW"
                rec["awaiting_detail"] = "Conflicting field values must be resolved before this record can be approved."
            elif e.entity_id in ever_approved:
                rec["awaiting_reason"] = "RE_APPROVAL"
                rec["awaiting_detail"] = "This record was approved before, but its members or values changed - approve the new version."
            else:
                rec["awaiting_reason"] = "PENDING_APPROVAL"
                rec["awaiting_detail"] = f"Merged from {int(n)} source records - a steward must approve it."
            rec["approval_basis"] = None
        out.append(rec)
    return {"domain": domain, "scope": scope, "kind": kind, "q": q, "records": out, "count": total_final,
            "matching": matching, "returned": len(out),
            "offset": offset, "limit": limit, "has_more": offset + len(out) < matching,
            "total_entities": total_entities,
            "distinct_record_count": singleton_final,
            "merged_record_count": total_final - singleton_final,
            "awaiting_count": total_awaiting,
            "awaiting_review_count": total_entities - total_final}


@router.get("/jobs")
def list_jobs(domain: Optional[str] = None, job_type: Optional[str] = None, status: Optional[str] = None,
              limit: int = 100, db: Session = Depends(get_db)):
    q = db.query(ProcessingJob)
    if domain:
        q = q.filter(ProcessingJob.domain == domain)
    if job_type:
        q = q.filter(ProcessingJob.job_type == job_type)
    if status:
        q = q.filter(ProcessingJob.status == status)
    rows = q.order_by(ProcessingJob.created_at.desc()).limit(limit).all()
    return {"jobs": [job_response(j) for j in rows], "count": len(rows)}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    j = db.query(ProcessingJob).filter(ProcessingJob.job_id == job_id).first()
    if j is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job_response(j)
