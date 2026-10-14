"""
MODULE 3 -- Type inference. Infers types from VALUES, never from field names.
Fully generic -- no domain or dataset conditionals.
"""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^\+?[\d\s().\-]{6,20}$")
NUMERIC_ID_RE = re.compile(r"^\d+$")
ALPHANUMERIC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-_/]*$")
DATE_LIKE_RE = re.compile(
    r"^\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?$"
    r"|^\d{1,2}/\d{1,2}/\d{2,4}$"
    r"|^\d{1,2}-\d{1,2}-\d{2,4}$"
)
SCI_NOTATION_RE = re.compile(r"^\d(\.\d+)?[eE][+\-]?\d+$")

TYPES = ["EMAIL", "DATE_LIKE", "NUMERIC_ID", "PHONE_LIKE", "ALPHANUMERIC_ID", "FREE_TEXT"]


def _classify_value(v: str) -> str:
    v = v.strip()
    if not v:
        return "FREE_TEXT"
    if EMAIL_RE.match(v):
        return "EMAIL"
    if DATE_LIKE_RE.match(v):
        return "DATE_LIKE"
    if NUMERIC_ID_RE.match(v):
        return "NUMERIC_ID"
    if SCI_NOTATION_RE.match(v):
        return "NUMERIC_ID"  # symptom of a corrupted numeric identifier
    if PHONE_RE.match(v) and any(ch.isdigit() for ch in v) and len(re.sub(r"\D", "", v)) >= 7:
        return "PHONE_LIKE"
    # Require an actual digit for ALPHANUMERIC_ID: pure-alphabetic strings
    # ("Chennai", "Active", "Supplier") are structurally indistinguishable
    # from ordinary words at the single-value level and must not be
    # classified as identifier-like on structure alone. Genuine mixed
    # alphanumeric identifiers ("SUP-10293", "ABC12345", "IN_98231",
    # "A12B34") all contain at least one digit, so this rule change does not
    # weaken detection of them. A pure-alphabetic identifier column (e.g. all
    # values from a small fixed alpha code set) is not completely prohibited
    # by the wider system -- see identifier_hint and format_consistency
    # evidence in recommend_er_role(), which can still support treating such
    # a field as identifier-like from column-level evidence even though
    # type_inference alone will label individual values FREE_TEXT.
    if ALPHANUMERIC_ID_RE.match(v) and len(v) <= 40 and " " not in v and any(c.isdigit() for c in v):
        return "ALPHANUMERIC_ID"
    return "FREE_TEXT"


@dataclass
class TypeInferenceResult:
    inferred_type: str
    type_confidence: float  # % of non-null values supporting dominant type
    type_distribution: dict[str, int]


def infer_type(values: list[str]) -> TypeInferenceResult:
    non_null = [v for v in values if v is not None and str(v).strip() != ""]
    if not non_null:
        return TypeInferenceResult("FREE_TEXT", 0.0, {})

    counts = Counter(_classify_value(str(v)) for v in non_null)
    dominant_type, dominant_count = counts.most_common(1)[0]
    confidence = round(100.0 * dominant_count / len(non_null), 2)
    return TypeInferenceResult(dominant_type, confidence, dict(counts))
