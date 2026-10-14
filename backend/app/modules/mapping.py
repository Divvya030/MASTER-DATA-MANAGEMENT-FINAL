"""
MODULE 2 -- Schema / Field Mapping

Generic, deterministic mapping of arbitrary source columns to a canonical domain
schema. Contains NO dataset-specific or domain-name conditionals. Everything that
varies by domain lives in app.domain_schemas.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

from app.domain_schemas import CanonicalSchema, CanonicalField

# Guarded substrings: canonical field -> tokens that must appear as whole tokens,
# not raw substrings, to prevent false positives like "state" matching
# "statement_cycle_name". This table is generic vocabulary, not dataset-specific.
_GUARDED_SUBSTRING_MIN_TOKEN_LEN = 3


_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def normalize_name(raw: str) -> str:
    """Lowercase, split camelCase boundaries, strip non-alphanumerics, collapse
    separators -> single normalized token string."""
    if raw is None:
        return ""
    s = raw.strip()
    s = _CAMEL_BOUNDARY_RE.sub(" ", s)  # firstName -> first Name
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(raw: str) -> list[str]:
    return [t for t in normalize_name(raw).split(" ") if t]


_TRAILING_DIGIT_RE = re.compile(r"^([a-z]+)\d+$")


def _stem_token(tok: str) -> str:
    """Strip a trailing digit suffix used as a repeat-column index, e.g.
    'address1' -> 'address', 'phone2' -> 'phone'. Used only for fuzzy
    (token-subset / guarded-substring) matching, never for exact/alias matching,
    so it cannot cause false EXACT/ALIAS matches."""
    m = _TRAILING_DIGIT_RE.match(tok)
    return m.group(1) if m else tok


def tokenize_stemmed(raw: str) -> list[str]:
    return [_stem_token(t) for t in tokenize(raw)]


@dataclass
class MappingCandidate:
    source_column: str
    canonical_field: str
    confidence: float
    match_type: str  # EXACT, ALIAS, TOKEN_SUBSET, GUARDED_SUBSTRING
    reason: str


@dataclass
class ColumnEvidence:
    """Evidence about a source column, used to enrich mapping candidates for the UI."""
    source_column: str
    non_null_count: int
    total_rows: int
    distinct_count: int
    sample_values: list
    is_empty: bool = field(init=False)

    def __post_init__(self):
        self.is_empty = self.non_null_count == 0

    @property
    def completeness(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return round(100.0 * self.non_null_count / self.total_rows, 2)


CONFIDENCE_EXACT = 1.0
CONFIDENCE_ALIAS = 0.85
CONFIDENCE_TOKEN_SUBSET = 0.65
CONFIDENCE_GUARDED_SUBSTRING = 0.45


def _exact_match(col_norm: str, canon_norm: str) -> bool:
    return col_norm == canon_norm


def _alias_match(col_norm: str, field_def: CanonicalField) -> bool:
    return col_norm in {normalize_name(a) for a in field_def.aliases}


def _token_subset_match(col_tokens: set[str], canon_tokens: set[str]) -> bool:
    if not canon_tokens:
        return False
    # canonical tokens are a subset of column tokens (or vice versa) and overlap is
    # substantial -- avoids matching on a single generic shared token like "id".
    if canon_tokens.issubset(col_tokens) or col_tokens.issubset(canon_tokens):
        overlap = canon_tokens & col_tokens
        return len(overlap) >= max(1, min(len(canon_tokens), len(col_tokens)))
    return False


def _guarded_substring_match(col_norm: str, canon_norm: str) -> bool:
    """
    Only allow substring matches when the canonical name appears as a WHOLE TOKEN
    inside the column's token list (never a raw substring of a longer word).
    This is what prevents 'state' from matching 'statement_cycle_name'.
    """
    if len(canon_norm) < _GUARDED_SUBSTRING_MIN_TOKEN_LEN:
        return False
    col_tokens = col_norm.split(" ")
    canon_tokens = canon_norm.split(" ")
    if len(canon_tokens) == 1:
        return canon_tokens[0] in col_tokens
    # multi-token canonical name: must appear as contiguous subsequence of tokens
    n, m = len(col_tokens), len(canon_tokens)
    for i in range(n - m + 1):
        if col_tokens[i:i + m] == canon_tokens:
            return True
    return False


def suggest_mappings(
    schema: CanonicalSchema,
    source_columns: list[str],
    evidence_by_column: Optional[dict[str, ColumnEvidence]] = None,
) -> dict[str, list[MappingCandidate]]:
    """
    Returns canonical_field_name -> list of MappingCandidate, sorted by confidence desc.
    Columns with is_empty=True evidence are excluded from candidates (EMPTY_COLUMN rule)
    but the caller is responsible for still surfacing them separately in the UI.
    """
    evidence_by_column = evidence_by_column or {}
    results: dict[str, list[MappingCandidate]] = {f.name: [] for f in schema.fields}

    for col in source_columns:
        ev = evidence_by_column.get(col)
        if ev is not None and ev.is_empty:
            continue  # EMPTY_COLUMN rule: never a mapping candidate

        col_norm = normalize_name(col)
        col_tokens = set(tokenize(col))
        col_tokens_stemmed = set(tokenize_stemmed(col))

        for cfield in schema.fields:
            canon_norm = normalize_name(cfield.name)
            canon_tokens = set(tokenize(cfield.name))

            if _exact_match(col_norm, canon_norm):
                results[cfield.name].append(MappingCandidate(
                    col, cfield.name, CONFIDENCE_EXACT, "EXACT",
                    "Column name normalizes to exactly the canonical field name."))
                continue

            if _alias_match(col_norm, cfield):
                results[cfield.name].append(MappingCandidate(
                    col, cfield.name, CONFIDENCE_ALIAS, "ALIAS",
                    "Column name matches a known alias of this canonical field."))
                continue

            if _token_subset_match(col_tokens, canon_tokens) or _token_subset_match(col_tokens_stemmed, canon_tokens):
                results[cfield.name].append(MappingCandidate(
                    col, cfield.name, CONFIDENCE_TOKEN_SUBSET, "TOKEN_SUBSET",
                    "Column tokens substantially overlap with the canonical field tokens."))
                continue

            if _guarded_substring_match(col_norm, canon_norm):
                results[cfield.name].append(MappingCandidate(
                    col, cfield.name, CONFIDENCE_GUARDED_SUBSTRING, "GUARDED_SUBSTRING",
                    "Canonical field name appears as a whole token/phrase within the column name."))

    for cfield_name in results:
        results[cfield_name].sort(key=lambda c: c.confidence, reverse=True)

    return results


@dataclass
class ConfirmedMapping:
    canonical_field: str
    source_column: str


class MappingValidationError(ValueError):
    pass


def validate_and_persist_mapping(
    schema: CanonicalSchema,
    source_columns: list[str],
    proposed: list[dict],
) -> list[ConfirmedMapping]:
    """
    proposed: list of {"canonical_field": ..., "source_column": ...}
    Rejects: unknown source columns, unknown canonical fields, duplicate canonical
    assignments, invalid structures.
    """
    known_columns = set(source_columns)
    known_fields = set(schema.field_names())
    seen_canonical: set[str] = set()
    confirmed: list[ConfirmedMapping] = []

    for i, item in enumerate(proposed):
        if not isinstance(item, dict) or "canonical_field" not in item or "source_column" not in item:
            raise MappingValidationError(f"Invalid mapping structure at index {i}: {item!r}")
        cfield = item["canonical_field"]
        scol = item["source_column"]
        if cfield not in known_fields:
            raise MappingValidationError(f"Unknown canonical field: {cfield!r}")
        if scol not in known_columns:
            raise MappingValidationError(f"Unknown source column: {scol!r}")
        if cfield in seen_canonical:
            raise MappingValidationError(f"Duplicate canonical field assignment: {cfield!r}")
        seen_canonical.add(cfield)
        confirmed.append(ConfirmedMapping(canonical_field=cfield, source_column=scol))

    return confirmed
