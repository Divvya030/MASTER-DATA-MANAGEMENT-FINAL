"""
Job abstraction (job_id / status / progress / result / error).

The MVP executes work synchronously in-process, but every long-running
operation is wrapped in a ProcessingJob row, and callers only ever use
start_job()/update_job()/finish_job(). Replacing local execution with a real
worker queue later means changing where the work runs -- not the table, the
API (/api/jobs) or the endpoints' contracts.
"""
from __future__ import annotations
import datetime as dt
import uuid
from typing import Optional
from sqlalchemy.orm import Session

from app.db.models import ProcessingJob


def start_job(db: Session, job_type: str, *, domain: Optional[str] = None, dataset_id: Optional[str] = None,
              model_version: Optional[str] = None) -> ProcessingJob:
    job = ProcessingJob(job_id=f"JOB-{uuid.uuid4().hex[:12]}", job_type=job_type, status="RUNNING", progress=0.0,
                        domain=domain, dataset_id=dataset_id, model_version=model_version,
                        started_at=dt.datetime.utcnow())
    db.add(job)
    db.commit()
    return job


def update_job(db: Session, job: ProcessingJob, progress: float, **fields) -> None:
    job.progress = max(0.0, min(1.0, progress))
    for k, v in fields.items():
        setattr(job, k, v)
    db.commit()


def finish_job(db: Session, job: ProcessingJob, ok: bool, *, result: Optional[dict] = None,
               error: Optional[str] = None, artifact_ref: Optional[str] = None,
               model_version: Optional[str] = None) -> ProcessingJob:
    job.status = "DONE" if ok else "FAILED"
    job.progress = 1.0 if ok else job.progress
    job.result = result
    job.error = error
    job.artifact_ref = artifact_ref
    if model_version:
        job.model_version = model_version
    job.completed_at = dt.datetime.utcnow()
    db.commit()
    return job


def job_response(j: ProcessingJob) -> dict:
    dur = (j.completed_at - j.started_at).total_seconds() if j.completed_at and j.started_at else None
    return {"job_id": j.job_id, "job_type": j.job_type, "status": j.status, "progress": j.progress,
            "domain": j.domain, "dataset_id": j.dataset_id, "model_version": j.model_version,
            "artifact_ref": j.artifact_ref, "result": j.result, "error": j.error,
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            "duration_seconds": dur}
