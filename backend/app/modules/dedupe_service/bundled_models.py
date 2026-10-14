"""Bundled V1 models: seeded into the model registry at application start-up.

The V1 Customer and Supplier models are built OFFLINE (scripts/build_v1_models.py) and ship
inside `models_bundled/<domain>/v1/`. On start-up this module (idempotently) copies each
artifact into the normal model store, registers a TrainingRun row for it and, ONLY if the
domain has no ACTIVE model yet, marks it ACTIVE. So a fresh install can find duplicates
immediately, without any training.

Never overwrites a user's later choice: if a v2 has been activated for a domain, or the
user retired V1, nothing here changes that. The app never trains V1 itself.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import TrainingRun, ActiveModel
from app.modules.dedupe_service.model_store import model_dir

BUNDLED_ROOT = Path(__file__).resolve().parents[3] / "models_bundled"
BUNDLED_DATASET_ID = "BUNDLED-V1"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def seed_bundled_models(db: Session) -> list[dict]:
    report = []
    if not BUNDLED_ROOT.exists():
        return report
    for domain_dir in sorted(p for p in BUNDLED_ROOT.iterdir() if p.is_dir()):
        v1 = domain_dir / "v1"
        meta_p = v1 / "metadata.json"
        if not meta_p.exists() or not (v1 / "model.settings").exists():
            continue
        md = json.loads(meta_p.read_text())
        run_id, domain = md["training_run_id"], md["domain"]
        dest = Path(model_dir(domain, run_id))
        for name in ("model.settings", "training.json", "metadata.json"):
            src, dst = v1 / name, dest / name
            if src.exists() and (not dst.exists() or _sha(src) != _sha(dst)):
                shutil.copyfile(src, dst)
        run = db.query(TrainingRun).filter(TrainingRun.training_run_id == run_id).first()
        action = "present"
        if run is None:
            summ = md.get("training_summary") or {}
            run = TrainingRun(
                training_run_id=run_id, dataset_id=BUNDLED_DATASET_ID, domain=domain,
                er_config_version=md.get("er_config_version", "bundled-v1"), match_fields=md["match_fields"],
                status="TRAINED", split_config={"origin": "OFFLINE_BUNDLED_V1", "model_version": md.get("model_version")},
                pair_counts=summ.get("training_pairs"), model_path=str(dest / "model.settings"),
                evaluation_metrics={"kind": "BUNDLED_V1_OFFLINE_EVALUATION", "headline": md.get("evaluation_summary"),
                                    "note": "See evaluation.json / V1_EVALUATION_REPORT.md for the full validation."},
                label_counts={"bundled_v1": summ.get("training_pairs"), "weak_labels": summ.get("weak_labels")},
                threshold=md["threshold"], review_threshold=md["review_threshold"],
                threshold_calibration=md.get("threshold_calibration"), lifecycle_status="DRAFT",
                training_options={"bundled": True, "model_version": md.get("model_version")},
                created_at=dt.datetime.utcnow(), completed_at=dt.datetime.utcnow(),
            )
            db.add(run)
            db.commit()
            action = "registered"
        elif run.dataset_id == BUNDLED_DATASET_ID and (run.threshold != md["threshold"] or run.review_threshold != md["review_threshold"]):
            # a newer build of the shipped model re-calibrated its thresholds: refresh the registry row
            summ = md.get("training_summary") or {}
            run.threshold, run.review_threshold = md["threshold"], md["review_threshold"]
            run.threshold_calibration = md.get("threshold_calibration")
            run.evaluation_metrics = {"kind": "BUNDLED_V1_OFFLINE_EVALUATION", "headline": md.get("evaluation_summary"),
                                      "note": "See evaluation.json / V1_EVALUATION_REPORT.md for the full validation."}
            db.commit()
            action = "thresholds-refreshed"
        active = db.query(ActiveModel).filter(ActiveModel.domain == domain).first()
        if active is None and run.lifecycle_status in ("DRAFT", "VALIDATED", None):
            run.lifecycle_status = "ACTIVE"
            run.activated_at = dt.datetime.utcnow()
            db.add(ActiveModel(domain=domain, training_run_id=run_id))
            db.commit()
            action += "+activated"
        report.append({"domain": domain, "training_run_id": run_id, "action": action})
    return report
