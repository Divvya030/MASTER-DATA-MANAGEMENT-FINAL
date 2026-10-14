"""
Shared "load records for ER" helper. Extracted from app/main.py so it can be
used by both the original train/match/cluster endpoints and the newer
golden-record/incremental-matching endpoints without a circular import.
Pure I/O glue -- no ER logic lives here.
"""
from __future__ import annotations
from sqlalchemy.orm import Session

from app.db.models import FieldMapping
from app.modules.ingestion import load_layer, load_raw, IngestionError


def mapping_dict_for_dataset(db: Session, dataset_id: str) -> dict:
    rows = db.query(FieldMapping).filter(FieldMapping.dataset_id == dataset_id).all()
    return {r.canonical_field: r.source_column for r in rows}


def records_from_standardized(db: Session, dataset_id: str, fields: list[str]) -> dict:
    """{row_index: {field: value}} restricted to the given canonical fields,
    using confirmed mappings to resolve canonical field -> source column.
    Falls back to RAW if STANDARDIZED doesn't exist yet."""
    mapping = mapping_dict_for_dataset(db, dataset_id)
    try:
        df = load_layer(dataset_id, "standardized")
    except IngestionError:
        df = load_raw(dataset_id)
    out = {}
    for idx in range(len(df)):
        row = {}
        for f in fields:
            source_col = mapping.get(f)
            if source_col and source_col in df.columns:
                v = df[source_col].iloc[idx]
                row[f] = None if (v is None or (isinstance(v, float) and v != v)) else v
            else:
                row[f] = None
        out[idx] = row
    return out
