"""
MODULE 1 -- Data Ingestion

Robust CSV / Excel ingestion. Produces a Dataset record + an immutable RAW layer
(stored as a pickled/parquet snapshot on disk, never mutated afterward) plus
metadata used by later modules.
"""
from __future__ import annotations
import csv
import hashlib
import io
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

DATA_DIR = os.environ.get("MDM_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)

_dataset_counter_state = {"year": None, "seq": 0}


class IngestionError(ValueError):
    pass


def generate_dataset_id(year: Optional[int] = None) -> str:
    year = year or datetime.utcnow().year
    if _dataset_counter_state["year"] != year:
        _dataset_counter_state["year"] = year
        _dataset_counter_state["seq"] = 0
    _dataset_counter_state["seq"] += 1
    return f"DS-{year}-{_dataset_counter_state['seq']:04d}"


def _detect_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except csv.Error:
        return ","


def _strip_bom(raw_bytes: bytes) -> bytes:
    for bom in (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff"):
        if raw_bytes.startswith(bom):
            return raw_bytes[len(bom):]
    return raw_bytes


def _check_raw_header_duplicates(header_line: str, delim: str) -> None:
    raw_headers = [h.strip() for h in header_line.split(delim)]
    if len(raw_headers) != len(set(raw_headers)):
        raise IngestionError("Duplicate column headers detected in source file.")


def _read_csv_robust(raw_bytes: bytes) -> pd.DataFrame:
    raw_bytes = _strip_bom(raw_bytes)
    # Order matters: utf-8 variants first (most common, and they correctly reject
    # invalid byte sequences rather than silently mis-decoding). cp1252 comes
    # before latin-1 deliberately -- cp1252 is latin-1's superset for printable
    # characters in the 0x80-0x9F range (curly quotes, em-dashes, etc, common in
    # Windows-exported CSVs) but REJECTS a handful of genuinely undefined byte
    # values there, so it fails loudly on non-cp1252 data. latin-1 never raises
    # on ANY byte sequence, so it must be the last, most-permissive fallback --
    # placing it earlier would silently swallow cp1252 files as mojibake instead
    # of ever giving cp1252 a chance to decode them correctly.
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            text = raw_bytes.decode(enc)
            sample = text[:8192]
            delim = _detect_delimiter(sample)
            first_line = text.split("\n", 1)[0].rstrip("\r")
            _check_raw_header_duplicates(first_line, delim)
            df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str,
                              keep_default_na=False, na_values=[""],
                              engine="python", on_bad_lines="warn")
            return df
        except IngestionError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise IngestionError(f"Unable to decode/parse CSV file: {last_err}")


def _read_excel_robust(raw_bytes: bytes) -> pd.DataFrame:
    try:
        df = pd.read_excel(io.BytesIO(raw_bytes), dtype=str, keep_default_na=False, na_values=[""])
        return df
    except Exception as e:  # noqa: BLE001
        raise IngestionError(f"Unable to parse Excel file: {e}")


def _validate_headers(df: pd.DataFrame) -> None:
    if df.shape[1] == 0:
        raise IngestionError("File has no columns / could not be parsed.")
    cols = list(df.columns)
    if any(str(c).strip() == "" or str(c).startswith("Unnamed:") for c in cols):
        # Not fatal -- but flag via a warning list the caller can surface.
        pass
    if len(cols) != len(set(cols)):
        raise IngestionError("Duplicate column headers detected in source file.")


@dataclass
class IngestedDataset:
    dataset_id: str
    original_filename: str
    domain: str
    row_count: int
    column_count: int
    columns: list[str]
    status: str
    raw_path: str
    header_warnings: list[str]
    sha256: str


def compute_file_sha256(raw_bytes: bytes) -> str:
    """SHA-256 over the ORIGINAL uploaded bytes, before any decoding/parsing/
    transformation. File-level exact-duplicate detection only -- never used
    for record/entity-level Dedupe matching."""
    return hashlib.sha256(raw_bytes).hexdigest()


def ingest_file(filename: str, raw_bytes: bytes, domain: str) -> tuple[IngestedDataset, pd.DataFrame]:
    # Computed FIRST, over the literal bytes as received -- before BOM stripping,
    # encoding detection, or any parsing touches them.
    sha256 = compute_file_sha256(raw_bytes)

    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "csv":
        df = _read_csv_robust(raw_bytes)
    elif ext in ("xlsx", "xls"):
        df = _read_excel_robust(raw_bytes)
    else:
        raise IngestionError(f"Unsupported file type: .{ext}. Supported: .csv, .xlsx")

    warnings = []
    for c in df.columns:
        if str(c).strip() == "" or str(c).startswith("Unnamed:"):
            warnings.append(f"Blank/unnamed header detected: {c!r}")

    _validate_headers(df)

    dataset_id = generate_dataset_id()
    raw_dir = os.path.join(DATA_DIR, dataset_id)
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, "raw.parquet")
    df.to_parquet(raw_path, index=False)  # immutable snapshot

    ingested = IngestedDataset(
        dataset_id=dataset_id,
        original_filename=filename,
        domain=domain,
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        status="UPLOADED",
        raw_path=raw_path,
        header_warnings=warnings,
        sha256=sha256,
    )
    return ingested, df


def load_raw(dataset_id: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, dataset_id, "raw.parquet")
    if not os.path.exists(path):
        raise IngestionError(f"No raw data found for dataset {dataset_id}")
    return pd.read_parquet(path)


def save_layer(dataset_id: str, layer: str, df: pd.DataFrame) -> str:
    """layer in {'standardized', 'cleansed'}. RAW is never written here -- see ingest_file."""
    if layer == "raw":
        raise IngestionError("Refusing to overwrite the RAW layer -- it is immutable.")
    d = os.path.join(DATA_DIR, dataset_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{layer}.parquet")
    df.to_parquet(path, index=False)
    return path


def load_layer(dataset_id: str, layer: str) -> pd.DataFrame:
    if layer == "raw":
        return load_raw(dataset_id)
    path = os.path.join(DATA_DIR, dataset_id, f"{layer}.parquet")
    if not os.path.exists(path):
        raise IngestionError(f"No {layer} data found for dataset {dataset_id}")
    return pd.read_parquet(path)


def delete_dataset_files(dataset_id: str) -> bool:
    """Remove the entire on-disk directory for a dataset (raw.parquet,
    standardized.parquet, and any future per-dataset artifact -- everything
    for a dataset already lives under one directory by construction, see
    ingest_file()/save_layer() above, so recursive removal is complete by
    design rather than needing to enumerate individual file types).

    Idempotent: returns True if the directory existed and was removed, False
    if it was already absent (e.g. a repeated delete call) -- either way the
    end state (no files for this dataset_id) is the same, so this never
    raises for a simple "already gone" case. Genuine OS-level failures
    (permissions, etc.) still propagate so the caller can report them rather
    than silently pretending cleanup succeeded."""
    d = os.path.join(DATA_DIR, dataset_id)
    if not os.path.isdir(d):
        return False
    shutil.rmtree(d)
    return True
