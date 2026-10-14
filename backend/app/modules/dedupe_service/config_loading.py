"""Shared ER-configuration helpers (used by main.py and the newer routers so
the config-version hash has exactly one implementation)."""
from __future__ import annotations
import hashlib
import json
from sqlalchemy.orm import Session

from app.db.models import ERFieldConfig
from app.modules.dedupe_service.field_definitions import ERFieldRoles


def er_config_version(role_rows) -> str:
    pairs = sorted((r.canonical_field, r.role) for r in role_rows)
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()[:16]


def load_active_er_roles(db: Session, dataset_id: str):
    rows = db.query(ERFieldConfig).filter(ERFieldConfig.dataset_id == dataset_id).all()
    buckets = {"IDENTIFIER": [], "MATCH": [], "BLOCKING": [], "IGNORE": []}
    for r in rows:
        if r.role in buckets:
            buckets[r.role].append(r.canonical_field)
    return ERFieldRoles(buckets["IDENTIFIER"], buckets["MATCH"], buckets["BLOCKING"], buckets["IGNORE"]), rows
