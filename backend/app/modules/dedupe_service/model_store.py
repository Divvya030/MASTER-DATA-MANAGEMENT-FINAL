"""
Model persistence for the Entity Resolution pipeline.

Deliberately kept SEPARATE from per-dataset storage (app/modules/ingestion.py's
DATA_DIR/{dataset_id}/...): a dataset can be deleted (delete_dataset_files)
without destroying a model trained from it, and -- looking ahead to the
"reuse a trained Customer model against a NEW Customer dataset" requirement --
a model must be addressable by DOMAIN/entity type, not physically nested
inside the one dataset directory it happened to be trained from.

Layout on disk:

    {MODELS_DIR}/{domain}/{training_run_id}/
        model.settings    -- deduper.write_settings() bytes (the actual
                              trained Dedupe model: learned blocking
                              predicates + classifier weights)
        training.json     -- deduper.write_training() bytes (labeled pairs
                              used; lets a future run resume/inspect training)
        metadata.json     -- identity + provenance record (see
                              build_model_metadata below) -- this is what
                              answers "which model is this, and can I trust
                              it", independent of the dataset row in the DB.

Nothing here talks to the DB or FastAPI; app/main.py wires this to
TrainingRun/MatchRun persistence, same pattern as training_service.py etc.
"""
from __future__ import annotations
import datetime as dt
import json
import os
import sys
from importlib.metadata import version as _pkg_version, PackageNotFoundError
from typing import Optional

# Bump this if the on-disk layout or metadata.json schema changes shape in a
# way old artifacts wouldn't have -- lets a loader refuse/upgrade old models
# instead of silently misreading them.
MODEL_FORMAT_VERSION = 1

# The exact set of packages whose versions actually affect whether a
# persisted `model.settings` blob can be correctly reloaded and scored
# (Dedupe's settings format is a pickle of internals from these libraries).
_TRACKED_PACKAGES = ["dedupe", "BTrees", "numpy", "scikit-learn", "scipy", "persistent"]


def models_root() -> str:
    """Root directory for ALL trained-model artifacts, independent of any
    dataset's own storage directory. Overridable via MDM_MODELS_DIR for
    deployments that want it on separate storage/volume from raw data."""
    default = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
    d = os.environ.get("MDM_MODELS_DIR", default)
    os.makedirs(d, exist_ok=True)
    return d


def model_dir(domain: str, training_run_id: str) -> str:
    """{models_root}/{domain}/{training_run_id}/ -- created on demand.
    Keyed by DOMAIN (entity type, e.g. 'customer', 'supplier'), not by
    dataset_id, so multiple training runs for the same domain -- whether
    from the same dataset retrained, or eventually a different dataset of
    the same entity type -- live side by side under one discoverable path."""
    d = os.path.join(models_root(), domain, training_run_id)
    os.makedirs(d, exist_ok=True)
    return d


def dependency_versions() -> dict:
    """The ACTUAL installed versions at training time, read from the live
    environment (never hardcoded) -- this is what lets a future reload
    verify/flag environment drift instead of guessing."""
    versions = {"python": sys.version.split()[0]}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def build_model_metadata(
    *,
    training_run_id: str,
    domain: str,
    source_dataset_id: str,
    er_config_version: str,
    match_fields: list,
    split_config: dict,
    pair_counts: dict,
    evaluation_metrics: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """The full identity record for a trained model, per the traceability
    requirement: domain/entity type, training run, ER config version,
    creation timestamp, and the dependency versions it was trained under --
    kept separate from (though it references) the dataset it happened to be
    trained from."""
    return {
        "model_format_version": MODEL_FORMAT_VERSION,
        "training_run_id": training_run_id,
        "domain": domain,
        # Provenance only -- NOT the storage key. A model is looked up by
        # (domain, training_run_id), never by dataset_id, so it can outlive
        # the dataset it was trained from and (in a later phase) be applied
        # to a different dataset of the same domain.
        "source_dataset_id": source_dataset_id,
        "er_config_version": er_config_version,
        "match_fields": match_fields,
        "split_config": split_config,
        "pair_counts": pair_counts,
        "evaluation_metrics": evaluation_metrics,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dependency_versions": dependency_versions(),
        **(extra or {}),
    }


def save_model_artifact(
    *,
    domain: str,
    training_run_id: str,
    settings_bytes: bytes,
    training_bytes: bytes,
    metadata: dict,
) -> dict:
    """Persists the three files for a TRAINED run and returns their paths.
    Called only when training actually succeeded (status == TRAINED)."""
    run_dir = model_dir(domain, training_run_id)

    model_path = os.path.join(run_dir, "model.settings")
    with open(model_path, "wb") as f:
        f.write(settings_bytes)

    training_data_path = os.path.join(run_dir, "training.json")
    with open(training_data_path, "wb") as f:
        f.write(training_bytes)

    metadata_path = os.path.join(run_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {
        "model_path": model_path,
        "training_data_path": training_data_path,
        "metadata_path": metadata_path,
    }


def load_model_metadata(model_path: Optional[str]) -> Optional[dict]:
    """Given a stored model_path (…/model.settings), loads the sibling
    metadata.json. Returns None if there is no model, or no metadata file
    (e.g. an artifact persisted before this module existed) -- callers must
    treat that as "identity unknown", not an error."""
    if not model_path:
        return None
    metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path) as f:
        return json.load(f)


def check_model_compatibility(metadata: Optional[dict]) -> list:
    """Compares the dependency versions a model was TRAINED under with the
    live environment. Dedupe settings are a pickle of library internals, so a
    mismatch can mean a silently wrong (or unloadable) model. Returns a list
    of human-readable problems (empty list = compatible)."""
    if not metadata:
        return ["model metadata.json is missing -- cannot verify dependency compatibility"]
    trained = metadata.get("dependency_versions") or {}
    now = dependency_versions()
    problems = []

    def major(v):
        return (v or "").split(".")[0]

    def major_minor(v):
        return ".".join((v or "").split(".")[:2])

    if trained.get("dedupe") != now.get("dedupe"):
        problems.append(f"dedupe version differs (trained {trained.get('dedupe')}, installed {now.get('dedupe')})")
    if major(trained.get("BTrees")) != major(now.get("BTrees")):
        problems.append(f"BTrees major version differs (trained {trained.get('BTrees')}, installed {now.get('BTrees')})")
    if major(trained.get("numpy")) != major(now.get("numpy")):
        problems.append(f"numpy major version differs (trained {trained.get('numpy')}, installed {now.get('numpy')})")
    if major_minor(trained.get("scikit-learn")) != major_minor(now.get("scikit-learn")):
        problems.append(f"scikit-learn version differs (trained {trained.get('scikit-learn')}, installed {now.get('scikit-learn')})")
    return problems
