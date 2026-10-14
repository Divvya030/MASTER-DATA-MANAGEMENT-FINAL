"""
MODULE 3 -- profiling_service. Orchestrates type inference + metrics +
recommendation into one per-field profile. Framework-independent (plain
dataclasses/dicts, no FastAPI/SQLAlchemy imports here).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

from . import type_inference as ti
from . import metrics as m
from . import recommendation as rec
from app.modules.nullsafe import series_to_values


def build_steward_explanation(canonical_field: str, er_recommendation: str, data_quality: dict) -> str:
    """One short, human-readable sentence a steward can act on, derived from the
    SAME underlying statistics as the recommendation (completeness, uniqueness,
    type, missing tokens) -- the raw numbers stay available in data_quality and
    the reasons list; this is only the plain-language rendering of them."""
    dq = data_quality or {}
    comp = dq.get("completeness") or {}
    card = dq.get("cardinality") or {}
    ti = dq.get("type_inference") or {}
    fill = round(100.0 - (comp.get("null_percentage") or 0.0), 1)
    uniq = card.get("uniqueness_ratio")
    ftype = (ti.get("inferred_type") or "").replace("_", " ").lower()
    missing_note = ""
    mtb = comp.get("missing_token_breakdown") or {}
    missing_variants = sorted({k for k in mtb if k and k.strip()})
    if missing_variants:
        missing_note = f" Some rows use placeholder values such as {', '.join(repr(m) for m in missing_variants[:3])} instead of leaving the field empty."

    if er_recommendation == "INSUFFICIENT_DATA":
        return (f"Only {fill}% of rows have a value here, which is too sparse to rely on for matching."
                + missing_note)
    if er_recommendation == "STRONG_IDENTIFIER_CANDIDATE":
        return (f"Nearly every row has a different, consistently formatted {ftype or 'value'} "
                f"({fill}% filled), so this field looks like an identifier. Confirm it is one before treating it as such."
                + missing_note)
    if er_recommendation == "LOW_VALUE_FOR_ER":
        return (f"Almost all rows say the same thing, so this field cannot tell records apart."
                + missing_note)
    if er_recommendation == "GOOD_MATCH_FIELD":
        return (f"Well filled ({fill}% of rows) with mostly distinct {ftype or 'values'}, so it is a strong "
                f"signal that two records are the same entity." + missing_note)
    return (f"Filled for {fill}% of rows and moderately varied -- useful for matching when combined "
            f"with other fields." + missing_note)


@dataclass
class FieldProfile:
    canonical_field: str
    source_column: str
    stage: str  # RAW | STANDARDIZED
    data_quality: dict
    cleansing_signals: list[str]
    er_readiness: str  # HIGH | MEDIUM | LOW | INSUFFICIENT
    er_recommendation: str
    reasons: list[str]
    steward_explanation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _readiness_label(er_recommendation: str, completeness_pct: float) -> str:
    if er_recommendation == "INSUFFICIENT_DATA":
        return "INSUFFICIENT"
    if er_recommendation in ("GOOD_MATCH_FIELD", "STRONG_IDENTIFIER_CANDIDATE") and completeness_pct >= 90:
        return "HIGH"
    if er_recommendation == "LOW_VALUE_FOR_ER":
        return "LOW"
    return "MEDIUM"


def profile_field(
    canonical_field: str,
    source_column: str,
    values: list,
    stage: str = "RAW",
    identifier_hint: bool = False,
    thresholds: rec.Thresholds = rec.DEFAULT_THRESHOLDS,
    metadata_hint: bool = False,
) -> FieldProfile:
    completeness = m.compute_completeness(values)
    cardinality = m.compute_cardinality(values)
    type_result = ti.infer_type(values)
    string_quality = m.compute_string_quality(values)
    format_patterns = m.compute_format_patterns(values)

    is_identifier_like_type = type_result.inferred_type in ("NUMERIC_ID", "ALPHANUMERIC_ID")
    id_quality = None
    if is_identifier_like_type or identifier_hint:
        id_quality = m.compute_identifier_quality(values)

    date_ambiguity = None
    if type_result.inferred_type == "DATE_LIKE":
        date_ambiguity = m.compute_date_ambiguity(values)

    completeness_pct = round(100.0 - completeness.null_percentage, 2)

    # Share of the SINGLE most common value among non-null values -- distinct
    # from cardinality.top_value_share, which sums the top-5 combined (a
    # different, coarser signal). This is the correct measure of "does one
    # value dominate almost everything" used by the LOW_VALUE_FOR_ER gate
    # below; derived from data already computed, no new pass over the values.
    dominant_value_share = (
        cardinality.top_values[0][1] / completeness.non_null_count
        if cardinality.top_values and completeness.non_null_count else 0.0
    )

    evidence = rec.FieldProfileEvidence(
        identifier_hint=identifier_hint,
        inferred_type=type_result.inferred_type,
        type_confidence=type_result.type_confidence,
        completeness_pct=completeness_pct,
        uniqueness_ratio=cardinality.uniqueness_ratio,
        is_constant=cardinality.is_constant,
        format_consistency=(id_quality.format_consistency if id_quality else None),
        is_ambiguous_date=(date_ambiguity.is_ambiguous if date_ambiguity else False),
        dominant_value_share=dominant_value_share,
    )

    er_recommendation, er_reasons = rec.recommend_er_role(evidence, thresholds)
    if metadata_hint:
        # Record metadata (source system, last-updated timestamp) describes the
        # ROW, not the entity. A near-unique timestamp would otherwise look like
        # a perfect match field. It is consumed by survivorship, never by matching.
        er_recommendation = "LOW_VALUE_FOR_ER"
        er_reasons = ["Record metadata field (source / timestamp): used by survivorship rules "
                      "(source priority, recency) and lineage, not for matching records to entities."]

    # MIXED_FORMAT_PATTERNS is only a meaningful cleansing signal for
    # structured/identifier-like fields (IDs, phone numbers, dates) where a
    # single consistent format is actually expected. For FREE_TEXT fields
    # (names, descriptions) differing structural patterns are normal -- e.g. a
    # two-word company name and a three-word one simply produce different
    # character skeletons -- so raising this as a cleansing signal there is a
    # false positive with no safe fix. The underlying computed evidence
    # (format_patterns.is_mixed_format) is still exposed in data_quality for
    # transparency; only the derived SIGNAL is suppressed for FREE_TEXT.
    mixed_format_signal = format_patterns.is_mixed_format and type_result.inferred_type != "FREE_TEXT"

    cleansing_signals = rec.derive_cleansing_signals(
        missing_token_breakdown=completeness.missing_token_breakdown,
        case_variation=string_quality.case_variation_same_value,
        leading_trailing_ws_count=string_quality.leading_trailing_whitespace_count,
        repeated_internal_ws_count=string_quality.repeated_internal_whitespace_count,
        is_mixed_format=mixed_format_signal,
        scientific_notation_count=(id_quality.scientific_notation_count if id_quality else 0),
        is_ambiguous_date=(date_ambiguity.is_ambiguous if date_ambiguity else False),
        length_outlier_count=string_quality.length_outlier_count,
    )

    data_quality = {
        "completeness": asdict(completeness),
        "cardinality": {
            **asdict(cardinality),
            "top_values": [{"value": v, "count": c} for v, c in cardinality.top_values],
        },
        "type_inference": asdict(type_result),
        "string_quality": asdict(string_quality),
        "format_patterns": {
            "format_patterns": [{"pattern": p, "count": c} for p, c in format_patterns.format_patterns],
            "is_mixed_format": format_patterns.is_mixed_format,
        },
        "identifier_quality": asdict(id_quality) if id_quality else None,
        "date_ambiguity": asdict(date_ambiguity) if date_ambiguity else None,
    }

    readiness = _readiness_label(er_recommendation, completeness_pct)

    return FieldProfile(
        canonical_field=canonical_field,
        source_column=source_column,
        stage=stage,
        data_quality=data_quality,
        cleansing_signals=cleansing_signals,
        er_readiness=readiness,
        er_recommendation=er_recommendation,
        reasons=er_reasons,
        steward_explanation=(
            "Record metadata -- used by survivorship (source priority / recency), not for entity matching."
            if metadata_hint else
            build_steward_explanation(canonical_field, er_recommendation, data_quality)),
    )


def profile_dataset(
    df: pd.DataFrame,
    mapping: dict[str, str],  # canonical_field -> source_column
    identifier_hints: dict[str, bool],
    stage: str = "RAW",
    metadata_hints: dict[str, bool] | None = None,
) -> dict[str, FieldProfile]:
    profiles = {}
    for canonical_field, source_column in mapping.items():
        if source_column not in df.columns:
            continue
        values = series_to_values(df[source_column])
        profiles[canonical_field] = profile_field(
            canonical_field, source_column, values, stage=stage,
            identifier_hint=identifier_hints.get(canonical_field, False),
            metadata_hint=(metadata_hints or {}).get(canonical_field, False),
        )
    return profiles
