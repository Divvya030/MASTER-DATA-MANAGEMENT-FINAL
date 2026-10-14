from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import threading
import uuid
from typing import Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db, get_session_factory
from app.db.models import Dataset, FieldMapping, Profile, ERFieldConfig, Job, TrainingRun, MatchRun, ClusterRun
from app.domain_schemas import get_schema, SCHEMAS
from app.modules.ingestion import (ingest_file, IngestionError, load_raw, load_layer, save_layer,
                                    compute_file_sha256, delete_dataset_files, DATA_DIR as DATA_DIR_FOR_DEDUPE)
from app.modules.mapping import (suggest_mappings, ColumnEvidence, validate_and_persist_mapping,
                                  MappingValidationError)
from app.modules.profiling.profiling_service import profile_dataset
from app.modules.standardization import standardize_dataset
from app.modules.reprofiling import compare_profiles, build_signal_audit_report
from app.modules.er_config import (suggest_roles, validate_role_configuration, build_final_configuration,
                                    FieldRoleAssignment, RoleValidationError)
from app.modules.readiness import assess_readiness, build_dedupe_ready_configuration
from app.modules.nullsafe import series_to_values
from app.modules.dedupe_service.field_definitions import ERFieldRoles, InsufficientERConfigError
from app.modules.dedupe_service.training_service import run_training
from app.modules.dedupe_service.matching_service import run_matching
from app.modules.dedupe_service.clustering_service import run_clustering
from app.modules.dedupe_service.model_store import (
    build_model_metadata, save_model_artifact, load_model_metadata,
)
from app.modules.dedupe_service.record_loading import records_from_standardized, mapping_dict_for_dataset
from app.api.er_workflow import router as er_workflow_router
from app.api.labeling import router as labeling_router
from app.modules.dedupe_service import label_repository as label_repo
from app.modules.dedupe_service.training_labels import select_training_labels
from app.modules.dedupe_service.config_loading import er_config_version as _shared_cfg_version
from app.modules.dedupe_service.model_store import check_model_compatibility
from app.modules.dedupe_service.field_definitions import coerce_record
from app.modules.dedupe_service.evaluation import blocking_evaluation
from app.modules.job_service import start_job, finish_job, job_response
from app.db.models import ActiveModel, TrainingRunLabel, ProcessingJob, LabeledPair, PendingReview

app = FastAPI(title="MDM / Entity Resolution Platform", version="0.1.0")

# Coarse, single-process race-condition guard for the upload duplicate-check.
# Serializes the "check sha256 exists -> decide -> insert" critical section so
# two simultaneous uploads of the same file bytes cannot both pass the
# not-yet-committed duplicate check and create two dataset rows. This protects
# a single-process deployment only -- it does NOT protect a multi-worker /
# multi-instance deployment, which would need a DB-level constraint or
# distributed lock instead. Documented as a known limitation in the report.
_upload_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Golden Record / Model Registry / Steward Review / Continuous Incoming Batch /
# Monitoring endpoints live in their own router (this file was already large
# before those modules existed). See app/api/er_workflow.py.
app.include_router(er_workflow_router)
app.include_router(labeling_router)

from app.api.find_duplicates import router as find_duplicates_router  # noqa: E402
from app.modules.dedupe_service.bundled_models import seed_bundled_models  # noqa: E402
app.include_router(find_duplicates_router)


@app.on_event("startup")
def _seed_bundled_v1_models():
    """Register + activate the shipped V1 models (idempotent; never overrides a later choice)."""
    db = get_session_factory()()
    try:
        seed_bundled_models(db)
    finally:
        db.close()


# ---------- helpers ----------

def _get_dataset_or_404(db: Session, dataset_id: str) -> Dataset:
    ds = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return ds


def _column_evidence(df, columns) -> dict[str, ColumnEvidence]:
    ev = {}
    for c in columns:
        values = series_to_values(df[c])
        non_null = [v for v in values if v is not None and str(v).strip() != ""]
        distinct = len(set(non_null)) if non_null else 0
        ev[c] = ColumnEvidence(
            source_column=c,
            non_null_count=len(non_null),
            total_rows=int(len(df)),
            distinct_count=distinct,
            sample_values=non_null[:3],
        )
    return ev


def _mapping_dict(db: Session, dataset_id: str) -> dict[str, str]:
    rows = db.query(FieldMapping).filter(FieldMapping.dataset_id == dataset_id).all()
    return {r.canonical_field: r.source_column for r in rows}


def _identifier_hints(domain: str) -> dict[str, bool]:
    schema = get_schema(domain)
    return {f.name: f.identifier_hint for f in schema.fields}


def _metadata_hints(domain: str) -> dict[str, bool]:
    schema = get_schema(domain)
    return {f.name: f.metadata_hint for f in schema.fields}


# ---------- Module 1: Ingestion ----------

@app.get("/api/domains")
def list_domains():
    return {"domains": list(SCHEMAS.keys())}


@app.get("/api/domains/{domain}/schema")
def get_domain_schema(domain: str):
    """The canonical fields configured for a domain, in schema order, with
    display labels -- so data-entry forms show every field the domain defines
    instead of a hard-coded handful."""
    from app.domain_schemas import field_label, input_type_for
    if domain not in SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Unknown domain '{domain}'.")
    schema = get_schema(domain)
    return {"domain": domain, "fields": [
        {"name": f.name, "label": field_label(f.name), "input_type": input_type_for(f.name),
         "identifier": f.identifier_hint, "metadata": f.metadata_hint, "description": f.description,
         "aliases": list(f.aliases)}
        for f in schema.fields]}


@app.post("/api/datasets/upload")
async def upload_dataset(
    domain: str = Form(...),
    file: UploadFile = File(...),
    on_duplicate: str = Form("ask"),  # ask | use_existing | allow_duplicate | replace
    db: Session = Depends(get_db),
):
    if domain not in SCHEMAS:
        raise HTTPException(status_code=400, detail=f"Unknown domain '{domain}'. Known: {list(SCHEMAS)}")
    if on_duplicate not in ("ask", "use_existing", "allow_duplicate", "replace"):
        raise HTTPException(status_code=400, detail=f"Invalid on_duplicate value: {on_duplicate!r}")

    raw_bytes = await file.read()
    sha256 = compute_file_sha256(raw_bytes)  # over the ORIGINAL bytes, before any parsing

    # Everything from the duplicate check through to persisting a new row is
    # serialized, so two simultaneous uploads of identical bytes can't both
    # observe "no existing match" and each create their own dataset. See the
    # _upload_lock docstring above for the scope/limits of this guard.
    with _upload_lock:
        existing = db.query(Dataset).filter(Dataset.sha256 == sha256).all()

        if existing and on_duplicate == "ask":
            raise HTTPException(status_code=409, detail={
                "error": "duplicate_file",
                "message": "A dataset with identical file content already exists.",
                "sha256": sha256,
                "existing_datasets": [_dataset_response(d) for d in existing],
            })

        if existing and on_duplicate == "use_existing":
            return _dataset_response(existing[0])

        if existing and on_duplicate == "replace":
            # No dataset-delete mechanism exists anywhere in this codebase today.
            # Building one would be a new capability, not a hardening fix, so this
            # is deliberately reported as unsupported rather than silently faked.
            raise HTTPException(status_code=501, detail={
                "error": "replace_not_supported",
                "message": "Delete-and-replace is not implemented -- no dataset deletion "
                            "mechanism currently exists. Use 'use_existing' or 'allow_duplicate' instead.",
            })

        # No match, or on_duplicate == "allow_duplicate": ingest as a new dataset.
        try:
            ingested, df = ingest_file(file.filename, raw_bytes, domain)
        except IngestionError as e:
            raise HTTPException(status_code=422, detail=str(e))

        ds = Dataset(
            dataset_id=ingested.dataset_id,
            original_filename=ingested.original_filename,
            domain=ingested.domain,
            row_count=ingested.row_count,
            column_count=ingested.column_count,
            columns=ingested.columns,
            status=ingested.status,
            raw_path=ingested.raw_path,
            header_warnings=ingested.header_warnings,
            sha256=ingested.sha256,
        )
        db.add(ds)
        db.commit()
        return _dataset_response(ds)


def _dataset_response(ds: Dataset) -> dict:
    return {
        "dataset_id": ds.dataset_id,
        "original_filename": ds.original_filename,
        "domain": ds.domain,
        "row_count": ds.row_count,
        "column_count": ds.column_count,
        "columns": ds.columns,
        "status": ds.status,
        "header_warnings": ds.header_warnings or [],
        "sha256": ds.sha256,
    }


@app.get("/api/datasets")
def list_datasets(db: Session = Depends(get_db)):
    rows = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    return {"datasets": [_dataset_response(r) for r in rows]}


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    ds = _get_dataset_or_404(db, dataset_id)
    return _dataset_response(ds)


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """
    Deletes a dataset by its internal dataset_id -- never by filename, never
    by matching other datasets with the same content. Order matters for
    safety: the DB row (and its cascade-related FieldMapping/Profile/
    ERFieldConfig/Job rows, all already configured with cascade="all,
    delete-orphan" in db/models.py) is deleted and committed FIRST. Only then
    is the on-disk directory removed. This ordering means a filesystem
    failure after a successful DB delete leaves, at worst, an orphaned
    directory with no DB reference (a harmless storage leak, recoverable by
    re-running cleanup) -- never the more dangerous inverse (a DB row that
    still claims to exist but whose backing files are gone, which would
    crash any later attempt to load_raw/load_layer for it).

    Idempotent in the sense that the end state is always "this dataset_id has
    no data anywhere" -- but a second call returns 404 rather than a fake 200,
    since the resource genuinely no longer exists at that point; this is
    intentional REST semantics, not a partial-cleanup bug.
    """
    ds = _get_dataset_or_404(db, dataset_id)

    db.delete(ds)
    db.commit()

    try:
        files_removed = delete_dataset_files(dataset_id)
        filesystem_status = "removed" if files_removed else "already_absent"
    except OSError as e:
        # DB state is already correctly clean at this point -- report the
        # filesystem issue rather than raising, since the dataset is already
        # gone from the application's perspective and retrying the DB delete
        # would just 404.
        filesystem_status = f"failed: {e}"

    return {
        "dataset_id": dataset_id,
        "database_status": "deleted",
        "filesystem_status": filesystem_status,
    }


# ---------- Module 2: Mapping ----------

@app.get("/api/datasets/{dataset_id}/mapping/suggestions")
def get_mapping_suggestions(dataset_id: str, db: Session = Depends(get_db)):
    ds = _get_dataset_or_404(db, dataset_id)
    df = load_raw(dataset_id)
    schema = get_schema(ds.domain)
    evidence = _column_evidence(df, ds.columns)

    empty_columns = [c for c, e in evidence.items() if e.is_empty]
    suggestions = suggest_mappings(schema, ds.columns, evidence)

    out = {}
    for canonical_field, candidates in suggestions.items():
        out[canonical_field] = [
            {
                "source_column": c.source_column,
                "confidence": c.confidence,
                "match_type": c.match_type,
                "reason": c.reason,
                "completeness": evidence[c.source_column].completeness,
                "distinct_count": evidence[c.source_column].distinct_count,
                "sample_values": evidence[c.source_column].sample_values,
                "is_empty": evidence[c.source_column].is_empty,
            }
            for c in candidates
        ]

    return {
        "dataset_id": dataset_id,
        "canonical_fields": schema.field_names(),
        "suggestions": out,
        "empty_ignored_columns": empty_columns,
    }


class MappingItem(BaseModel):
    canonical_field: str
    source_column: str


class MappingRequest(BaseModel):
    mappings: list[MappingItem]


@app.get("/api/datasets/{dataset_id}/mapping")
def get_confirmed_mapping(dataset_id: str, db: Session = Depends(get_db)):
    _get_dataset_or_404(db, dataset_id)
    return {"dataset_id": dataset_id, "mapping": _mapping_dict(db, dataset_id)}


@app.post("/api/datasets/{dataset_id}/mapping")
def confirm_mapping(dataset_id: str, req: MappingRequest, db: Session = Depends(get_db)):
    ds = _get_dataset_or_404(db, dataset_id)
    schema = get_schema(ds.domain)
    try:
        confirmed = validate_and_persist_mapping(
            schema, ds.columns, [m.dict() for m in req.mappings]
        )
    except MappingValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    db.query(FieldMapping).filter(FieldMapping.dataset_id == dataset_id).delete()
    for m in confirmed:
        db.add(FieldMapping(dataset_id=dataset_id, canonical_field=m.canonical_field,
                             source_column=m.source_column))
    ds.status = "MAPPED"
    db.commit()
    return {"dataset_id": dataset_id, "mapping": {m.canonical_field: m.source_column for m in confirmed}}


# ---------- Module 3: Profiling ----------

def _run_profiling(db: Session, ds: Dataset, stage: str) -> dict:
    mapping = _mapping_dict(db, ds.dataset_id)
    if not mapping:
        raise HTTPException(status_code=422, detail="No confirmed mapping exists for this dataset yet.")
    df = load_layer(ds.dataset_id, "raw" if stage == "RAW" else "standardized")
    hints = _identifier_hints(ds.domain)
    profiles = profile_dataset(df, mapping, hints, stage=stage, metadata_hints=_metadata_hints(ds.domain))

    db.query(Profile).filter(Profile.dataset_id == ds.dataset_id, Profile.stage == stage).delete()
    for canonical_field, p in profiles.items():
        db.add(Profile(
            dataset_id=ds.dataset_id, canonical_field=canonical_field, stage=stage,
            data_quality=p.data_quality, cleansing_signals=p.cleansing_signals,
            er_readiness=p.er_readiness, er_recommendation=p.er_recommendation,
            blocking_assessment="",  # legacy column -- application-level blocking removed
            steward_explanation=p.steward_explanation, reasons=p.reasons,
        ))
    if stage == "RAW":
        ds.status = "PROFILED"
    db.commit()
    return {cf: p.to_dict() for cf, p in profiles.items()}


@app.post("/api/datasets/{dataset_id}/profile")
def run_profile(dataset_id: str, db: Session = Depends(get_db)):
    ds = _get_dataset_or_404(db, dataset_id)
    result = _run_profiling(db, ds, "RAW")
    return {"dataset_id": dataset_id, "stage": "RAW", "profiles": result}


@app.get("/api/datasets/{dataset_id}/profile")
def get_profile(dataset_id: str, stage: str = "RAW", db: Session = Depends(get_db)):
    _get_dataset_or_404(db, dataset_id)
    rows = db.query(Profile).filter(Profile.dataset_id == dataset_id, Profile.stage == stage).all()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No {stage} profile found. Run profiling first.")
    return {
        "dataset_id": dataset_id,
        "stage": stage,
        "profiles": {
            r.canonical_field: {
                "canonical_field": r.canonical_field, "source_column": None, "stage": r.stage,
                "data_quality": r.data_quality, "cleansing_signals": r.cleansing_signals,
                "er_readiness": r.er_readiness, "er_recommendation": r.er_recommendation,
                "steward_explanation": r.steward_explanation, "reasons": r.reasons,
            } for r in rows
        },
    }


# ---------- Module 4: Standardization ----------

@app.post("/api/datasets/{dataset_id}/standardize")
def run_standardize(dataset_id: str, db: Session = Depends(get_db)):
    ds = _get_dataset_or_404(db, dataset_id)
    mapping = _mapping_dict(db, dataset_id)
    raw_profiles = db.query(Profile).filter(Profile.dataset_id == dataset_id, Profile.stage == "RAW").all()
    if not raw_profiles:
        raise HTTPException(status_code=422, detail="Run RAW profiling before standardization.")
    signals_by_field = {r.canonical_field: r.cleansing_signals for r in raw_profiles}

    raw_df = load_raw(dataset_id)
    result = standardize_dataset(raw_df, mapping, signals_by_field)
    save_layer(dataset_id, "standardized", result.df)

    ds.status = "STANDARDIZED"
    db.commit()
    return {"dataset_id": dataset_id, "transformation_summary": result.summary(),
            "transformation_count": len(result.log)}


# ---------- Module 5: Re-profiling ----------

@app.post("/api/datasets/{dataset_id}/reprofile")
def run_reprofile(dataset_id: str, db: Session = Depends(get_db)):
    ds = _get_dataset_or_404(db, dataset_id)
    result = _run_profiling(db, ds, "STANDARDIZED")
    ds.status = "VALIDATED"
    db.commit()
    return {"dataset_id": dataset_id, "stage": "STANDARDIZED", "profiles": result}


@app.get("/api/datasets/{dataset_id}/quality-comparison")
def quality_comparison(dataset_id: str, db: Session = Depends(get_db)):
    _get_dataset_or_404(db, dataset_id)
    raw_rows = db.query(Profile).filter(Profile.dataset_id == dataset_id, Profile.stage == "RAW").all()
    std_rows = db.query(Profile).filter(Profile.dataset_id == dataset_id, Profile.stage == "STANDARDIZED").all()
    if not raw_rows or not std_rows:
        raise HTTPException(status_code=404, detail="Both RAW and STANDARDIZED profiles are required.")

    from app.modules.profiling.profiling_service import FieldProfile as FP

    def to_fp(r):
        return FP(r.canonical_field, "", r.stage, r.data_quality, r.cleansing_signals,
                   r.er_readiness, r.er_recommendation, r.reasons, r.steward_explanation or "")

    raw_profiles = {r.canonical_field: to_fp(r) for r in raw_rows}
    std_profiles = {r.canonical_field: to_fp(r) for r in std_rows}
    comparisons = compare_profiles(raw_profiles, std_profiles)
    audit_report = build_signal_audit_report(raw_profiles, std_profiles)
    return {
        "dataset_id": dataset_id,
        "comparisons": {
            cf: {"canonical_field": c.canonical_field, "before": c.before, "after": c.after,
                 "improved": c.improved, "notes": c.notes}
            for cf, c in comparisons.items()
        },
        "signal_audit_report": [e.to_dict() for e in audit_report],
    }


# ---------- Module 6: ER field role configuration ----------

@app.get("/api/datasets/{dataset_id}/er-config/suggestions")
def er_config_suggestions(dataset_id: str, db: Session = Depends(get_db)):
    _get_dataset_or_404(db, dataset_id)
    rows = db.query(Profile).filter(Profile.dataset_id == dataset_id, Profile.stage == "STANDARDIZED").all()
    if not rows:
        rows = db.query(Profile).filter(Profile.dataset_id == dataset_id, Profile.stage == "RAW").all()
    if not rows:
        raise HTTPException(status_code=422, detail="Run profiling before requesting ER role suggestions.")

    out = {}
    for r in rows:
        suggested = suggest_roles(r.er_recommendation)
        out[r.canonical_field] = {
            "canonical_field": r.canonical_field,
            "er_recommendation": r.er_recommendation,
            "er_readiness": r.er_readiness,
            "suggested_roles": suggested,
            "steward_explanation": r.steward_explanation,
            "completeness_pct": round(100 - r.data_quality["completeness"]["null_percentage"], 2),
            "reasons": r.reasons,
        }
    return {"dataset_id": dataset_id, "fields": out}


class RoleAssignmentItem(BaseModel):
    canonical_field: str
    role: str


class RoleConfigRequest(BaseModel):
    assignments: list[RoleAssignmentItem]


@app.post("/api/datasets/{dataset_id}/er-config")
def submit_er_config(dataset_id: str, req: RoleConfigRequest, db: Session = Depends(get_db)):
    ds = _get_dataset_or_404(db, dataset_id)
    rows = db.query(Profile).filter(Profile.dataset_id == dataset_id, Profile.stage == "STANDARDIZED").all()
    if not rows:
        rows = db.query(Profile).filter(Profile.dataset_id == dataset_id, Profile.stage == "RAW").all()
    evidence = {
        r.canonical_field: {
            "er_recommendation": r.er_recommendation,
            "completeness_pct": round(100 - r.data_quality["completeness"]["null_percentage"], 2),
        } for r in rows
    }

    assignments = [FieldRoleAssignment(a.canonical_field, a.role, "USER_OVERRIDE") for a in req.assignments]
    try:
        warnings = validate_role_configuration(assignments, evidence)
    except RoleValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    db.query(ERFieldConfig).filter(ERFieldConfig.dataset_id == dataset_id).delete()
    for a in assignments:
        db.add(ERFieldConfig(dataset_id=dataset_id, canonical_field=a.canonical_field,
                              role=a.role, source=a.source))
    ds.status = "ER_CONFIGURED"
    db.commit()

    final_config = build_final_configuration(assignments)
    return {
        "dataset_id": dataset_id,
        "configuration": final_config,
        "warnings": [w.__dict__ for w in warnings],
    }


# ---------- Module 7: Final readiness ----------

@app.get("/api/datasets/{dataset_id}/readiness")
def get_readiness(dataset_id: str, db: Session = Depends(get_db)):
    ds = _get_dataset_or_404(db, dataset_id)
    mapping = _mapping_dict(db, dataset_id)
    role_rows = db.query(ERFieldConfig).filter(ERFieldConfig.dataset_id == dataset_id).all()
    er_config = {"identifier_fields": [], "match_fields": [], "blocking_fields": [], "ignored_fields": []}
    key_by_role = {"IDENTIFIER": "identifier_fields", "MATCH": "match_fields",
                   "BLOCKING": "blocking_fields", "IGNORE": "ignored_fields"}
    for r in role_rows:
        er_config[key_by_role[r.role]].append(r.canonical_field)

    std_profiles = db.query(Profile).filter(Profile.dataset_id == dataset_id, Profile.stage == "STANDARDIZED").all()
    profile_source = std_profiles or db.query(Profile).filter(
        Profile.dataset_id == dataset_id, Profile.stage == "RAW").all()

    field_warnings = []
    date_ambiguity_unresolved = False
    for p in profile_source:
        completeness_pct = round(100 - p.data_quality["completeness"]["null_percentage"], 2)
        if completeness_pct < 50 and p.canonical_field in (
                er_config["identifier_fields"] + er_config["match_fields"]):
            field_warnings.append(f"{p.canonical_field} has low completeness ({completeness_pct}%).")
        da = p.data_quality.get("date_ambiguity")
        if da and da.get("is_ambiguous"):
            date_ambiguity_unresolved = True

    evidence = {}
    df = None
    try:
        df = load_raw(dataset_id)
    except Exception:
        pass
    empty_field_count = 0
    if df is not None:
        ev = _column_evidence(df, ds.columns)
        empty_field_count = sum(1 for e in ev.values() if e.is_empty)

    stage_flags = {
        "mapped": bool(mapping),
        "profiled": len(db.query(Profile).filter(Profile.dataset_id == dataset_id, Profile.stage == "RAW").all()) > 0,
        "standardized": ds.status in ("STANDARDIZED", "VALIDATED", "ER_CONFIGURED", "READY_FOR_DEDUPE"),
        "revalidated": len(std_profiles) > 0,
    }

    report = assess_readiness(
        dataset_id=dataset_id, domain=ds.domain, row_count=ds.row_count,
        total_source_columns=ds.column_count, mapped_field_count=len(mapping),
        empty_field_count=empty_field_count, stage_flags=stage_flags,
        er_config=er_config, field_warnings=field_warnings,
        date_ambiguity_unresolved=date_ambiguity_unresolved,
        unmapped_canonical_fields=[f for f in get_schema(ds.domain).field_names() if f not in mapping],
    )

    if report.is_ready:
        ds.status = "READY_FOR_DEDUPE"
        db.commit()

    final_config = build_dedupe_ready_configuration(
        dataset_id, ds.domain, er_config,
        standardization_stage="STANDARDIZED" if stage_flags["standardized"] else "RAW",
    )

    return {"readiness": report.to_dict(), "dedupe_handoff_configuration": final_config}


# ============================================================================
# ENTITY RESOLUTION: Dedupe Training -> Matching -> Clustering
#
# Consumes ONLY the ACTIVE, steward-confirmed ERFieldConfig rows for a
# dataset -- never raw profiling recommendations. IGNORE-role fields are
# never passed to Dedupe. Uses the real dedupe==3.0.3 API throughout (see
# app/modules/dedupe_service/) -- no custom fuzzy matching anywhere.
# ============================================================================

DEDUPE_DIR_NAME = "dedupe"


def _er_config_version(role_rows) -> str:
    """Stable content hash of the active ER configuration, so every training/
    match/cluster run can be traced back to exactly which field roles
    produced it."""
    return _shared_cfg_version(role_rows)


def _load_active_er_roles(db: Session, dataset_id: str) -> ERFieldRoles:
    rows = db.query(ERFieldConfig).filter(ERFieldConfig.dataset_id == dataset_id).all()
    key_to_list = {"IDENTIFIER": [], "MATCH": [], "BLOCKING": [], "IGNORE": []}
    for r in rows:
        if r.role in key_to_list:
            key_to_list[r.role].append(r.canonical_field)
    return ERFieldRoles(key_to_list["IDENTIFIER"], key_to_list["MATCH"],
                         key_to_list["BLOCKING"], key_to_list["IGNORE"]), rows


def _records_from_standardized(dataset_id: str, fields: list[str]) -> dict:
    """Thin wrapper kept for the existing call sites in this file (train/
    match endpoints below) -- the real implementation now lives in
    record_loading.py so app/api/er_workflow.py can share it without a
    circular import."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        return records_from_standardized(db, dataset_id, fields)
    finally:
        db.close()


def _mapping_dict_for_dataset(dataset_id: str) -> dict:
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        return mapping_dict_for_dataset(db, dataset_id)
    finally:
        db.close()


def _dedupe_dir(dataset_id: str) -> str:
    d = os.path.join(DATA_DIR_FOR_DEDUPE, dataset_id, DEDUPE_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


class TrainingRequest(BaseModel):
    seed: int = 42
    clones_per_entity: int = 2
    max_entities: int = 400
    # AUTO: use REAL steward labels from the repository and add controlled
    # SYNTHETIC augmentation only when real train labels are too few;
    # ALWAYS: always augment; NEVER: real labels only.
    synthetic_mode: str = "AUTO"
    use_real_labels: bool = True
    threshold: float | None = None      # None -> calibrate on real validation labels if enough, else provisional 0.5
    review_margin: float = 0.15


@app.post("/api/datasets/{dataset_id}/er/training")
def create_training_run(dataset_id: str, req: TrainingRequest = TrainingRequest(), db: Session = Depends(get_db)):
    ds = _get_dataset_or_404(db, dataset_id)
    er_roles, role_rows = _load_active_er_roles(db, dataset_id)
    if not role_rows:
        raise HTTPException(status_code=422, detail="No active ER configuration exists for this dataset yet. "
                                                      "Confirm ER field roles (Module 6) before training.")
    if not er_roles.match_fields:
        raise HTTPException(status_code=422, detail="The active ER configuration has no MATCH fields. "
                                                      "Dedupe requires at least one MATCH field to train.")

    training_run_id = f"TR-{uuid.uuid4().hex[:12]}"
    run = TrainingRun(
        training_run_id=training_run_id, dataset_id=dataset_id, domain=ds.domain,
        er_config_version=_er_config_version(role_rows), match_fields=er_roles.match_fields,
        status="CREATED",
        training_options={"seed": req.seed, "synthetic_mode": req.synthetic_mode,
                          "use_real_labels": req.use_real_labels, "threshold": req.threshold,
                          "review_margin": req.review_margin},
    )
    db.add(run)
    db.commit()
    job = start_job(db, "TRAINING", domain=ds.domain, dataset_id=dataset_id, model_version=training_run_id)

    try:
        base_records = _records_from_standardized(dataset_id, er_roles.match_fields)
        run.status = "PAIRS_GENERATED"
        db.commit()

        # REAL steward labels are the preferred evidence. Only labels whose
        # record values still cover this model's MATCH fields are usable.
        # The same selection the Training screen previews (training_labels.py): MATCH/DISTINCT steward labels
        # whose snapshots carry every MATCH field; UNSURE / unlabelled / other-config pairs are excluded.
        real_labels, excluded_labels = [], []
        if req.use_real_labels:
            real_labels, excluded_labels = select_training_labels(db, ds.domain, list(er_roles.match_fields))

        result = run_training(base_records, er_roles, seed=req.seed,
                               clones_per_entity=req.clones_per_entity, max_entities=req.max_entities,
                               real_labels=real_labels, synthetic_mode=req.synthetic_mode,
                               threshold=req.threshold, review_margin=req.review_margin,
                               base_dataset_id=dataset_id)

        run.split_config = result.split_config
        run.pair_counts = result.pair_counts
        run.evaluation_metrics = result.evaluation_metrics
        run.label_counts = result.label_counts
        if run.label_counts is not None and req.use_real_labels:
            run.label_counts = {**run.label_counts, "excluded_missing_match_fields": len(excluded_labels),
                                "repository_labels_used": len(real_labels)}
        run.real_evaluation = result.real_evaluation
        run.threshold = result.threshold
        run.review_threshold = result.review_threshold
        run.threshold_calibration = result.threshold_calibration
        run.status = result.status
        run.error = result.error

        if result.status == "TRAINED":
            # Model artifacts are persisted OUTSIDE the dataset's own storage
            # directory, keyed by (domain, training_run_id): deleting the source
            # dataset never destroys the model, and the model is addressable by
            # domain so it can be applied to NEW datasets of the same entity type.
            n = max(len(base_records), 1)
            training_field_stats = {f: round(sum(1 for r in base_records.values() if r.get(f) in (None, "")) / n, 4)
                                    for f in er_roles.match_fields}
            metadata = build_model_metadata(
                training_run_id=training_run_id, domain=ds.domain, source_dataset_id=dataset_id,
                er_config_version=run.er_config_version, match_fields=run.match_fields,
                split_config=result.split_config, pair_counts=result.pair_counts,
                evaluation_metrics=result.evaluation_metrics,
                extra={"label_counts": result.label_counts, "threshold": result.threshold,
                       "review_threshold": result.review_threshold,
                       "threshold_calibration": result.threshold_calibration,
                       "real_evaluation": result.real_evaluation,
                       "identifier_fields": er_roles.identifier_fields, "blocking_fields": er_roles.blocking_fields,
                       # the mapping used at training time is a reusable template for incoming
                       # datasets from the same source (canonical field -> source column name)
                       "source_mapping": _mapping_dict_for_dataset(dataset_id),
                       "training_field_stats": training_field_stats,
                       "standardization": "existing STANDARDIZED layer (not re-standardized for training)"},
            )
            paths = save_model_artifact(
                domain=ds.domain, training_run_id=training_run_id,
                settings_bytes=result.settings_bytes, training_bytes=result.training_bytes,
                metadata=metadata,
            )
            run.model_path = paths["model_path"]
            # DRAFT: trained but not promoted. VALIDATED/ACTIVE are explicit,
            # separate calls (app/api/er_workflow.py) -- training never activates.
            run.lifecycle_status = "DRAFT"
            for split_name, pair_ids in (result.label_splits or {}).items():
                for pid in pair_ids:
                    db.add(TrainingRunLabel(training_run_id=training_run_id, pair_id=pid, split=split_name))

        run.completed_at = dt.datetime.utcnow()
        db.commit()
        finish_job(db, job, run.status == "TRAINED", artifact_ref=training_run_id, error=run.error,
                   result={"status": run.status, "label_counts": run.label_counts})
    except Exception as e:  # noqa: BLE001 -- never leave a run silently stuck, always FAILED with a real reason
        run.status = "FAILED"
        run.error = f"{type(e).__name__}: {e}"
        run.completed_at = dt.datetime.utcnow()
        db.commit()
        finish_job(db, job, False, error=run.error, artifact_ref=training_run_id)

    return _training_run_response(run)


def _training_run_response(run: TrainingRun) -> dict:
    return _training_run_response_impl(run)


def _training_run_response_impl(run: TrainingRun) -> dict:
    # model_metadata is read back from the persisted metadata.json (see
    # model_store.py), not from the DB -- it's the independent identity
    # record for the model artifact itself: domain, training_run_id,
    # er_config_version, creation timestamp, and the exact dependency
    # versions (dedupe/BTrees/numpy/scikit-learn) it was trained under.
    return {
        "training_run_id": run.training_run_id, "dataset_id": run.dataset_id, "domain": run.domain,
        "er_config_version": run.er_config_version, "match_fields": run.match_fields,
        "status": run.status, "split_config": run.split_config, "pair_counts": run.pair_counts,
        "evaluation_metrics": run.evaluation_metrics,        # SYNTHETIC validation only
        "real_evaluation": run.real_evaluation,              # REAL steward-labelled held-out evaluation
        "label_counts": run.label_counts, "threshold": run.threshold,
        "review_threshold": run.review_threshold, "threshold_calibration": run.threshold_calibration,
        "lifecycle_status": run.lifecycle_status, "error": run.error,
        "model_available": run.model_path is not None,
        "model_metadata": load_model_metadata(run.model_path),
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


@app.get("/api/datasets/{dataset_id}/er/training/{training_run_id}")
def get_training_run(dataset_id: str, training_run_id: str, db: Session = Depends(get_db)):
    run = db.query(TrainingRun).filter(TrainingRun.training_run_id == training_run_id,
                                        TrainingRun.dataset_id == dataset_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Training run {training_run_id} not found")
    return _training_run_response(run)


@app.get("/api/datasets/{dataset_id}/er/models")
def list_training_runs(dataset_id: str, db: Session = Depends(get_db)):
    _get_dataset_or_404(db, dataset_id)
    runs = db.query(TrainingRun).filter(TrainingRun.dataset_id == dataset_id).order_by(
        TrainingRun.created_at.desc()).all()
    return {"training_runs": [_training_run_response(r) for r in runs]}


class MatchRequest(BaseModel):
    # None -> use the domain's ACTIVE model (works for datasets the model was
    # never trained on). A specific run may still be named explicitly.
    training_run_id: str | None = None
    threshold: float | None = None          # None -> the model version's stored threshold
    review_threshold: float | None = None   # None -> the model's stored uncertain-band edge
    distinct_threshold: float | None = None  # None -> review_threshold (pairs below it are DISTINCT, not discarded)


@app.post("/api/datasets/{dataset_id}/er/match")
def create_match_run(dataset_id: str, req: MatchRequest, db: Session = Depends(get_db)):
    ds = _get_dataset_or_404(db, dataset_id)
    if req.training_run_id is None:
        active = db.query(ActiveModel).filter(ActiveModel.domain == ds.domain).first()
        if active is None:
            raise HTTPException(status_code=422, detail=f"No ACTIVE model for domain '{ds.domain}'. "
                                                          f"Train, validate and activate a model, or name a training_run_id.")
        run_id = active.training_run_id
    else:
        run_id = req.training_run_id
    training_run = db.query(TrainingRun).filter(TrainingRun.training_run_id == run_id).first()
    if training_run is None:
        raise HTTPException(status_code=404, detail=f"Training run {run_id} not found")
    is_active = db.query(ActiveModel).filter(ActiveModel.domain == training_run.domain,
                                              ActiveModel.training_run_id == run_id).first() is not None
    if training_run.dataset_id != dataset_id and not (is_active and training_run.domain == ds.domain):
        raise HTTPException(status_code=422, detail="This model was trained on a different dataset and is not the "
                                                      "ACTIVE model for this dataset's domain.")
    if training_run.status != "TRAINED" or not training_run.model_path:
        raise HTTPException(status_code=422, detail=f"Training run {run_id} is not in a "
                                                      f"TRAINED state (status={training_run.status}); "
                                                      f"cannot match with an untrained/failed model.")
    problems = check_model_compatibility(load_model_metadata(training_run.model_path))
    if problems:
        raise HTTPException(status_code=422, detail="Model is not compatible with this environment: " + "; ".join(problems))
    mapped = set(_mapping_dict_for_dataset(dataset_id).keys())
    missing = [f for f in training_run.match_fields if f not in mapped]
    name_fields = (load_model_metadata(training_run.model_path) or {}).get("name_fields") or training_run.match_fields[:1]
    # A model field the file simply does not have is scored as MISSING (Dedupe has_missing=True),
    # never as a mismatch. Only a file with none of the model's name fields cannot be matched.
    if missing and not any(f in mapped for f in name_fields):
        raise HTTPException(status_code=422, detail=f"Dataset has no confirmed mapping for any of the model's name fields {name_fields}; cannot match.")

    threshold = req.threshold if req.threshold is not None else (training_run.threshold if training_run.threshold is not None else 0.5)
    review_threshold = req.review_threshold if req.review_threshold is not None else training_run.review_threshold
    distinct_threshold = req.distinct_threshold if req.distinct_threshold is not None else review_threshold

    match_run_id = f"MR-{uuid.uuid4().hex[:12]}"
    run = MatchRun(match_run_id=match_run_id, training_run_id=run_id, dataset_id=dataset_id,
                   domain=training_run.domain, status="CREATED", threshold=threshold,
                   review_threshold=review_threshold)
    db.add(run)
    db.commit()
    job = start_job(db, "MATCHING", domain=ds.domain, dataset_id=dataset_id, model_version=run_id)

    try:
        run.status = "MATCHING"
        db.commit()
        with open(training_run.model_path, "rb") as f:
            settings_bytes = f.read()
        records = _records_from_standardized(dataset_id, training_run.match_fields)
        coerced = {rid: coerce_record(rec, training_run.match_fields) for rid, rec in records.items()}

        result = run_matching(settings_bytes, coerced, threshold=threshold, review_threshold=review_threshold,
                              distinct_threshold=distinct_threshold)
        run.status = result.status
        run.error = result.error
        run.total_records = result.total_records
        run.total_possible_pairs = result.total_possible_pairs
        run.candidate_pair_count = result.candidate_pair_count
        run.match_count = result.match_count
        run.non_match_count = result.non_match_count
        run.uncertain_count = result.uncertain_count

        if result.status == "COMPLETED":
            run_dir = os.path.join(_dedupe_dir(dataset_id), "matching", match_run_id)
            os.makedirs(run_dir, exist_ok=True)
            results_path = os.path.join(run_dir, "results.json")
            with open(results_path, "w") as f:
                json.dump(result.pairs, f)
            run.results_path = results_path
            if result.raw_scores is not None:
                scores_path = os.path.join(run_dir, "scores.npy")
                np.save(scores_path, result.raw_scores)
                run.scores_path = scores_path
            known = [(int(l["record_id_a"]), int(l["record_id_b"])) for l in label_repo.usable_real_labels(db, ds.domain)
                     if l["label"] == "MATCH" and l["dataset_id"] == dataset_id
                     and str(l["record_id_a"]).isdigit() and str(l["record_id_b"]).isdigit()]
            keys = {(min(p["record_id_a"], p["record_id_b"]), max(p["record_id_a"], p["record_id_b"])) for p in (result.pairs or [])}
            run.blocking_evaluation = blocking_evaluation(result.total_possible_pairs, result.candidate_pair_count, keys, known)

            # UNSURE pairs are never auto-decided: they are queued for Match
            # Review (record A vs record B) where a steward decides MATCH,
            # DISTINCT or UNSURE. DISTINCT records are NOT discarded -- each
            # becomes its own singleton entity with its own Golden Record.
            queued: set = set()
            for p_ in (result.pairs or []):
                if p_["decision"] != "UNSURE":
                    continue
                key = (min(p_["record_id_a"], p_["record_id_b"]), max(p_["record_id_a"], p_["record_id_b"]))
                if key in queued:
                    continue
                queued.add(key)
                db.add(PendingReview(
                    pending_review_id=f"PR-{uuid.uuid4().hex[:12]}", domain=ds.domain,
                    kind="AMBIGUOUS_MATCH", entity_id=None, candidate_dataset_id=dataset_id,
                    candidate_record_id=str(p_["record_id_a"]),
                    candidate_other_record_id=str(p_["record_id_b"]),
                    candidate_values=records.get(p_["record_id_a"]),
                    score=p_["score"], training_run_id=run_id,
                    match_threshold=threshold, review_threshold=review_threshold))
            if queued:
                db.commit()

        run.completed_at = dt.datetime.utcnow()
        db.commit()
        finish_job(db, job, run.status == "COMPLETED", artifact_ref=match_run_id, error=run.error,
                   result={"candidate_pairs": run.candidate_pair_count, "matches": run.match_count,
                           "uncertain": run.uncertain_count})
    except Exception as e:  # noqa: BLE001
        run.status = "FAILED"
        run.error = f"{type(e).__name__}: {e}"
        run.completed_at = dt.datetime.utcnow()
        db.commit()
        finish_job(db, job, False, error=run.error, artifact_ref=match_run_id)

    return _match_run_response(run)


def _match_run_response(run: MatchRun) -> dict:
    return {
        "match_run_id": run.match_run_id, "training_run_id": run.training_run_id,
        "dataset_id": run.dataset_id, "status": run.status, "threshold": run.threshold,
        "total_records": run.total_records, "total_possible_pairs": run.total_possible_pairs,
        "candidate_pair_count": run.candidate_pair_count,
        "candidate_reduction_pct": (
            round(100 * (1 - run.candidate_pair_count / run.total_possible_pairs), 2)
            if run.total_possible_pairs else None
        ),
        "match_count": run.match_count, "non_match_count": run.non_match_count,
        "uncertain_count": run.uncertain_count, "review_threshold": run.review_threshold,
        "blocking_evaluation": run.blocking_evaluation, "domain": run.domain,
        "error": run.error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


@app.get("/api/datasets/{dataset_id}/er/matches/{match_run_id}")
def get_match_run(dataset_id: str, match_run_id: str, limit: int = 100, db: Session = Depends(get_db)):
    run = db.query(MatchRun).filter(MatchRun.match_run_id == match_run_id,
                                     MatchRun.dataset_id == dataset_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Match run {match_run_id} not found")
    resp = _match_run_response(run)
    if run.results_path and os.path.exists(run.results_path):
        with open(run.results_path) as f:
            all_pairs = json.load(f)
        resp["pairs"] = [p for p in all_pairs if p["decision"] in ("MATCH", "UNSURE")][:limit]
    return resp


class ClusterRequest(BaseModel):
    match_run_id: str
    threshold: float | None = None      # None -> the match run's own decision threshold


@app.post("/api/datasets/{dataset_id}/er/cluster")
def create_cluster_run(dataset_id: str, req: ClusterRequest, db: Session = Depends(get_db)):
    _get_dataset_or_404(db, dataset_id)
    match_run = db.query(MatchRun).filter(MatchRun.match_run_id == req.match_run_id,
                                           MatchRun.dataset_id == dataset_id).first()
    if match_run is None:
        raise HTTPException(status_code=404, detail=f"Match run {req.match_run_id} not found")
    if match_run.status != "COMPLETED" or not match_run.scores_path:
        raise HTTPException(status_code=422, detail=f"Match run {req.match_run_id} is not COMPLETED "
                                                      f"(status={match_run.status}); cannot cluster.")

    if req.threshold is None:
        req.threshold = match_run.threshold
    cluster_run_id = f"CR-{uuid.uuid4().hex[:12]}"
    run = ClusterRun(cluster_run_id=cluster_run_id, match_run_id=req.match_run_id,
                      training_run_id=match_run.training_run_id, dataset_id=dataset_id,
                      domain=match_run.domain, status="CREATED", threshold=req.threshold)
    db.add(run)
    db.commit()
    job = start_job(db, "CLUSTERING", domain=match_run.domain, dataset_id=dataset_id, model_version=match_run.training_run_id)

    try:
        run.status = "CLUSTERING"
        db.commit()
        raw_scores = np.load(match_run.scores_path)
        # Every record that was part of the matched dataset, whether or not
        # it ended up in any candidate pair, so unmatched records still get
        # a traceable singleton cluster (see _add_singletons).
        all_record_ids = list(range(match_run.total_records)) if match_run.total_records else []

        result = run_clustering(raw_scores, all_record_ids, threshold=req.threshold)
        run.status = result.status
        run.error = result.error
        run.cluster_count = result.cluster_count
        run.singleton_count = result.singleton_count
        run.largest_cluster_size = result.largest_cluster_size

        if result.status == "COMPLETED":
            run_dir = os.path.join(_dedupe_dir(dataset_id), "clustering", cluster_run_id)
            os.makedirs(run_dir, exist_ok=True)
            clusters_path = os.path.join(run_dir, "clusters.json")
            with open(clusters_path, "w") as f:
                json.dump(result.clusters, f)
            run.clusters_path = clusters_path

        run.completed_at = dt.datetime.utcnow()
        db.commit()
        finish_job(db, job, run.status == "COMPLETED", artifact_ref=cluster_run_id, error=run.error,
                   result={"clusters": run.cluster_count, "singletons": run.singleton_count})
    except Exception as e:  # noqa: BLE001
        run.status = "FAILED"
        run.error = f"{type(e).__name__}: {e}"
        run.completed_at = dt.datetime.utcnow()
        db.commit()
        finish_job(db, job, False, error=run.error, artifact_ref=cluster_run_id)

    return _cluster_run_response(run)


def _cluster_run_response(run: ClusterRun) -> dict:
    return {
        "cluster_run_id": run.cluster_run_id, "match_run_id": run.match_run_id,
        "training_run_id": run.training_run_id, "dataset_id": run.dataset_id,
        "status": run.status, "threshold": run.threshold,
        "cluster_count": run.cluster_count, "singleton_count": run.singleton_count,
        "largest_cluster_size": run.largest_cluster_size, "error": run.error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


@app.get("/api/datasets/{dataset_id}/er/clusters/{cluster_run_id}")
def get_cluster_run(dataset_id: str, cluster_run_id: str, limit: int = 100, db: Session = Depends(get_db)):
    run = db.query(ClusterRun).filter(ClusterRun.cluster_run_id == cluster_run_id,
                                       ClusterRun.dataset_id == dataset_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Cluster run {cluster_run_id} not found")
    resp = _cluster_run_response(run)
    if run.clusters_path and os.path.exists(run.clusters_path):
        with open(run.clusters_path) as f:
            all_clusters = json.load(f)
        non_singletons = [c for c in all_clusters if c["cluster_size"] > 1]
        resp["clusters"] = non_singletons[:limit]
        resp["clusters_returned"] = len(non_singletons[:limit])
        resp["non_singleton_cluster_count"] = len(non_singletons)
    return resp


# ---------------------------------------------------------------------------
# Continuous incoming data through the EXISTING ingestion -> mapping ->
# profiling -> standardization modules (no second pipeline), then the shared
# matching service (er_workflow.apply_active_model_impl). Never retrains.
# ---------------------------------------------------------------------------
from app.api.er_workflow import apply_active_model_impl  # noqa: E402


class IncomingRecordsRequest(BaseModel):
    records: list[dict]                      # each dict: {source column name: value}
    source_name: str = "incoming_api"
    match_threshold: float | None = None
    ambiguous_threshold: float | None = None


async def _ingest_incoming_bytes(db: Session, domain: str, filename: str, content: bytes,
                                 match_threshold, ambiguous_threshold) -> dict:
    import io
    from fastapi import UploadFile
    active = db.query(ActiveModel).filter(ActiveModel.domain == domain).first()
    if active is None:
        raise HTTPException(status_code=422, detail=f"No ACTIVE model for domain '{domain}'. Train, validate and activate one first.")
    run = db.query(TrainingRun).filter(TrainingRun.training_run_id == active.training_run_id).first()
    template = (load_model_metadata(run.model_path) or {}).get("source_mapping") or {}

    ds_resp = await upload_dataset(domain=domain, file=UploadFile(file=io.BytesIO(content), filename=filename),
                                   on_duplicate="allow_duplicate", db=db)
    dataset_id = ds_resp["dataset_id"]
    cols = set(ds_resp["columns"])
    # the mapping used when the model was trained is the template for this source
    mappings = [{"canonical_field": cf, "source_column": sc} for cf, sc in template.items() if sc in cols]
    if not all(f in {m["canonical_field"] for m in mappings} for f in run.match_fields):
        # fall back to the ordinary mapping suggestions (top candidate) for what the template lacks
        sugg = get_mapping_suggestions(dataset_id, db)["suggestions"]
        have = {m["canonical_field"] for m in mappings}
        used = {m["source_column"] for m in mappings}
        for cf, cands in sugg.items():
            if cf not in have and cands and cands[0]["source_column"] not in used:
                mappings.append({"canonical_field": cf, "source_column": cands[0]["source_column"]})
                used.add(cands[0]["source_column"])
    confirm_mapping(dataset_id, MappingRequest(mappings=mappings), db)
    run_profile(dataset_id, db)
    run_standardize(dataset_id, db)          # the ONE existing standardization step
    result = apply_active_model_impl(db, dataset_id, match_threshold, ambiguous_threshold)
    return {"dataset_id": dataset_id, "batch": result}


@app.post("/api/domains/{domain}/incoming")
async def submit_incoming_records(domain: str, req: IncomingRecordsRequest, db: Session = Depends(get_db)):
    """Single record or a small batch as JSON."""
    import csv, io
    if domain not in SCHEMAS:
        raise HTTPException(status_code=400, detail=f"Unknown domain '{domain}'.")
    if not req.records:
        raise HTTPException(status_code=422, detail="Incoming batch is empty.")
    cols = list(dict.fromkeys(k for r in req.records for k in r.keys()))
    if not cols:
        raise HTTPException(status_code=422, detail="Incoming records contain no fields.")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in req.records:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return await _ingest_incoming_bytes(db, domain, f"{req.source_name}.csv", buf.getvalue().encode("utf-8"),
                                        req.match_threshold, req.ambiguous_threshold)


@app.post("/api/domains/{domain}/incoming/file")
async def submit_incoming_file(domain: str, file: UploadFile = File(...), match_threshold: float | None = Form(None),
                               ambiguous_threshold: float | None = Form(None), db: Session = Depends(get_db)):
    """A CSV/XLSX batch of incoming records."""
    if domain not in SCHEMAS:
        raise HTTPException(status_code=400, detail=f"Unknown domain '{domain}'.")
    content = await file.read()
    return await _ingest_incoming_bytes(db, domain, file.filename, content, match_threshold, ambiguous_threshold)
