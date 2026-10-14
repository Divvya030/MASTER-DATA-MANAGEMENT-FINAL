"""One-step 'upload a dataset -> find duplicates' workflow, driven by the domain's ACTIVE
model (the bundled V1 unless a v2 has been activated). It composes the EXISTING modules
(ingestion -> mapping -> profiling -> the single standardization step -> active-model
matching -> Dedupe clustering). It NEVER trains, labels or retrains anything."""
from __future__ import annotations
import csv
import io
import json
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import ActiveModel, ClusterRun, MatchRun, TrainingRun, FieldMapping
from app.modules.dedupe_service.model_store import load_model_metadata
from app.modules.ingestion import load_raw
from app.modules.dedupe_service.record_loading import records_from_standardized

router = APIRouter(prefix="/api", tags=["find-duplicates"])


def _active_run(db: Session, domain: str) -> TrainingRun:
    active = db.query(ActiveModel).filter(ActiveModel.domain == domain).first()
    if active is None:
        raise HTTPException(status_code=422, detail=f"No ACTIVE model for domain '{domain}'.")
    return db.query(TrainingRun).filter(TrainingRun.training_run_id == active.training_run_id).first()


@router.post("/domains/{domain}/find-duplicates")
async def find_duplicates(domain: str, file: UploadFile = File(...), threshold: float | None = Form(None),
                          db: Session = Depends(get_db)):
    from app import main as m
    run = _active_run(db, domain)
    md = load_model_metadata(run.model_path) or {}
    ds = await m.upload_dataset(domain=domain, file=file, on_duplicate="allow_duplicate", db=db)
    dataset_id, cols = ds["dataset_id"], set(ds["columns"])

    template = md.get("source_mapping") or {}
    mappings = [{"canonical_field": cf, "source_column": sc} for cf, sc in template.items()
                if sc in cols and cf in run.match_fields]
    have, used = {x["canonical_field"] for x in mappings}, {x["source_column"] for x in mappings}
    sugg = m.get_mapping_suggestions(dataset_id, db)["suggestions"]
    for cf in run.match_fields:
        if cf in have:
            continue
        for cand in sugg.get(cf) or []:
            if cand["source_column"] not in used:
                mappings.append({"canonical_field": cf, "source_column": cand["source_column"]})
                used.add(cand["source_column"]); have.add(cf)
                break
    names = md.get("name_fields") or run.match_fields[:1]
    if not any(f in have for f in names):
        raise HTTPException(status_code=422, detail={
            "error": "no_name_column",
            "message": f"Could not find a {' / '.join(names)} column in this file, which the model needs. "
                       f"Rename a column or use the Advanced pipeline to map it manually.",
            "columns": sorted(cols)})
    m.confirm_mapping(dataset_id, m.MappingRequest(mappings=mappings), db)
    m.run_profile(dataset_id, db)
    m.run_standardize(dataset_id, db)

    match = m.create_match_run(dataset_id, m.MatchRequest(threshold=threshold), db)
    if match["status"] != "COMPLETED":
        raise HTTPException(status_code=500, detail=f"Matching failed: {match.get('error')}")
    mr_row = db.query(MatchRun).filter(MatchRun.match_run_id == match["match_run_id"]).first()
    if not mr_row.scores_path:
        # blocking proposed no candidate pairs at all (e.g. a file with a name column but no
        # address/contact evidence): nothing to cluster, every record is unique
        return {
            "dataset_id": dataset_id, "filename": ds["original_filename"], "row_count": ds["row_count"],
            "model": {"training_run_id": run.training_run_id, "version": md.get("model_version"),
                      "threshold": match["threshold"], "review_threshold": match.get("review_threshold")},
            "mapping_used": {x["canonical_field"]: x["source_column"] for x in mappings},
            "model_fields_not_in_file": [f for f in run.match_fields if f not in have],
            "match_run": match, "cluster_run": None,
            "summary": {"records": ds["row_count"], "duplicate_groups": 0, "records_in_duplicate_groups": 0,
                        "unique_entities": ds["row_count"], "uncertain_pairs": 0, "largest_group": 1,
                        "note": "The model proposed no candidate duplicate pairs for this file."},
        }
    clus = m.create_cluster_run(dataset_id, m.ClusterRequest(match_run_id=match["match_run_id"]), db)
    if clus["status"] != "COMPLETED":
        raise HTTPException(status_code=500, detail=f"Clustering failed: {clus.get('error')}")
    groups = _load_groups(clus["cluster_run_id"], db)
    dup_records = sum(g["cluster_size"] for g in groups)
    return {
        "dataset_id": dataset_id, "filename": ds["original_filename"], "row_count": ds["row_count"],
        "model": {"training_run_id": run.training_run_id, "version": md.get("model_version"),
                  "threshold": match["threshold"], "review_threshold": match.get("review_threshold")},
        "mapping_used": {x["canonical_field"]: x["source_column"] for x in mappings},
        "model_fields_not_in_file": [f for f in run.match_fields if f not in have],
        "match_run": match, "cluster_run": clus,
        "summary": {"records": ds["row_count"], "duplicate_groups": len(groups), "records_in_duplicate_groups": dup_records,
                    "unique_entities": clus["cluster_count"], "uncertain_pairs": match.get("uncertain_count"),
                    "largest_group": clus["largest_cluster_size"]},
    }


def _load_groups(cluster_run_id: str, db: Session) -> list[dict]:
    cr = db.query(ClusterRun).filter(ClusterRun.cluster_run_id == cluster_run_id).first()
    if cr is None or not cr.clusters_path or not os.path.exists(cr.clusters_path):
        raise HTTPException(status_code=404, detail=f"Cluster run {cluster_run_id} not found")
    with open(cr.clusters_path) as f:
        cl = json.load(f)
    return sorted((c for c in cl if c["cluster_size"] > 1), key=lambda c: (-c["cluster_size"], min(c["scores"])))


def _mapped_columns(db: Session, dataset_id: str) -> dict:
    return {r.canonical_field: r.source_column for r in db.query(FieldMapping).filter(FieldMapping.dataset_id == dataset_id).all()}


def _cell(v):
    return None if v is None or (isinstance(v, float) and v != v) else str(v)


@router.get("/find-duplicates/{dataset_id}/{cluster_run_id}/groups")
def duplicate_groups(dataset_id: str, cluster_run_id: str, offset: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    groups = _load_groups(cluster_run_id, db)
    raw = load_raw(dataset_id)
    mapping = _mapped_columns(db, dataset_id)
    cols = list(dict.fromkeys(mapping.values()))
    fields = list(mapping.keys())
    std = records_from_standardized(db, dataset_id, fields)     # the values the model actually compared
    out = []
    for i, g in enumerate(groups[offset: offset + limit], start=offset):
        recs = [{"row_number": rid + 1, **{c: _cell(raw[c].iloc[rid]) for c in cols if c in raw.columns},
                 "_std": {f: _cell(std[rid].get(f)) for f in fields}}
                for rid in sorted(g["record_ids"])]
        score = round(min(g["scores"]), 4)
        out.append({"group_id": f"G{i + 1:05d}", "size": g["cluster_size"],
                    "confidence": score, "match_score": score, "records": recs})
    # Which canonical fields describe the RECORD (source system, last-updated) rather than the
    # entity -- the UI shows these in each record's header instead of as a comparison row.
    from app.db.models import Dataset, SurvivorshipConfig
    from app.domain_schemas import metadata_field_names
    from app.modules.golden_record_service import normalize_config
    ds = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    try:
        meta_fields = sorted(metadata_field_names(ds.domain)) if ds else []
    except ValueError:
        meta_fields = []
    row = db.query(SurvivorshipConfig).filter(SurvivorshipConfig.domain == ds.domain).first() if ds else None
    scfg = normalize_config(row.config if row else None)
    return {"total_groups": len(groups), "offset": offset, "limit": limit, "columns": cols,
            "standardized_columns": fields, "mapping": mapping, "metadata_fields": meta_fields,
            "source_system_field": scfg.get("source_system_field"), "recency_field": scfg.get("recency_field"),
            "groups": out}


@router.get("/find-duplicates/{dataset_id}/{cluster_run_id}/export.csv")
def export_csv(dataset_id: str, cluster_run_id: str, db: Session = Depends(get_db)):
    groups = _load_groups(cluster_run_id, db)
    cr = db.query(ClusterRun).filter(ClusterRun.cluster_run_id == cluster_run_id).first()
    mr = db.query(MatchRun).filter(MatchRun.match_run_id == cr.match_run_id).first()
    review = set()
    if mr and mr.results_path and os.path.exists(mr.results_path):
        with open(mr.results_path) as f:
            for p in json.load(f):
                if p["decision"] == "UNCERTAIN":
                    review.update((p["record_id_a"], p["record_id_b"]))
    info = {}
    for i, g in enumerate(groups):
        for rid, sc in zip(g["record_ids"], g["scores"]):
            info[rid] = (f"G{i + 1:05d}", g["cluster_size"], round(float(sc), 4))
    raw = load_raw(dataset_id)

    def gen():
        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow(["row_number", "duplicate_group_id", "group_size", "member_confidence", "is_duplicate", "needs_review"] + list(raw.columns))
        yield buf.getvalue(); buf.seek(0); buf.truncate()
        for idx in range(len(raw)):
            gi = info.get(idx)
            w.writerow([idx + 1, gi[0] if gi else "", gi[1] if gi else 1, gi[2] if gi else "", bool(gi), idx in review]
                       + ["" if _cell(v) is None else _cell(v) for v in raw.iloc[idx].tolist()])
            if idx % 500 == 499:
                yield buf.getvalue(); buf.seek(0); buf.truncate()
        yield buf.getvalue()
    return StreamingResponse(gen(), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="duplicates_{dataset_id}.csv"'})


@router.get("/domains/{domain}/retraining/eligibility")
def retraining_eligibility(domain: str, db: Session = Depends(get_db)):
    """Is there enough REAL_STEWARD evidence to justify training a V2? Never triggers training."""
    from app.modules.dedupe_service.retraining import eligibility
    run = _active_run(db, domain)
    return {**eligibility(db, domain, run.review_threshold), "active_model": run.training_run_id}
