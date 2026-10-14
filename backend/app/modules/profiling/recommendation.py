"""
MODULE 3 -- ER recommendation + blocking assessment + cleansing signal derivation.

Decisions here branch ONLY on generic evidence (identifier_hint, inferred_type,
computed metrics) -- never on domain name or specific field/dataset identity.
All thresholds are centralized in `Thresholds` (configurable, not magic numbers
scattered through the logic).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    min_completeness_for_er: float = 60.0          # % non-null needed to be usable at all
    high_completeness: float = 90.0
    high_uniqueness: float = 0.85
    identifier_min_uniqueness: float = 0.90
    identifier_min_format_consistency: float = 0.85
    # NOTE: low_value_max_uniqueness is kept ONLY for backward-compat reference
    # in older callers/tests; it is NOT used to gate MATCH-usability anymore --
    # see low_value_max_top_share below and the Item-3 correction notes on
    # recommend_er_role(). Raw uniqueness_ratio (distinct/total) is a poor
    # measure of "is this field informative": a categorical field like city or
    # country will ALWAYS have a tiny uniqueness_ratio regardless of how useful
    # it is, since it conflates "few real-world categories" with "no
    # information". A single-value CONCENTRATION measure is the correct one.
    low_value_max_uniqueness: float = 0.05
    # A field is genuinely near-constant (little/no ER value) only when its
    # single most common value accounts for this share or more of all values --
    # e.g. a status flag that's 99% "Active". A low-cardinality-but-distributed
    # categorical field (city: 28 distinct values, no single value over ~10%)
    # must NOT be caught by this and excluded from MATCH consideration.
    low_value_max_top_share: float = 0.98


DEFAULT_THRESHOLDS = Thresholds()

ER_RECOMMENDATIONS = [
    "GOOD_MATCH_FIELD",
    "SUPPORTING_MATCH_FIELD",
    "LOW_VALUE_FOR_ER",
    "STRONG_IDENTIFIER_CANDIDATE",  # statistical identifier-LIKENESS -- see recommend_er_role()
    "INSUFFICIENT_DATA",
]


@dataclass
class FieldProfileEvidence:
    """Minimal generic evidence bundle recommendation logic needs."""
    identifier_hint: bool
    inferred_type: str
    type_confidence: float
    completeness_pct: float  # 100 - null_percentage
    uniqueness_ratio: float
    is_constant: bool
    format_consistency: float | None  # only meaningful for identifier-like fields
    is_ambiguous_date: bool = False
    dominant_value_share: float = 0.0  # share of non-null values equal to the single most common value


def recommend_er_role(ev: FieldProfileEvidence, t: Thresholds = DEFAULT_THRESHOLDS) -> tuple[str, list[str]]:
    reasons: list[str] = []

    # 1. Insufficient evidence
    if ev.completeness_pct < t.min_completeness_for_er:
        reasons.append(f"Completeness {ev.completeness_pct}% is below the "
                        f"minimum usable threshold ({t.min_completeness_for_er}%).")
        return "INSUFFICIENT_DATA", reasons

    # 2. Strong identifier candidate (evidence-driven; identifier_hint is only a HINT).
    # NOTE ON SEMANTICS: STRONG_IDENTIFIER_CANDIDATE means "statistically
    # identifier-like based on observed data characteristics" (high uniqueness +
    # consistent format) -- it is a CANDIDATE label, not proof. A field being
    # 95% unique does not make it a confirmed business identifier (e.g. a highly
    # varied customer_name column can be statistically unique without being a
    # real identifier). Confirming a field as a true business identifier is a
    # human/steward decision made in Module 6, never an automatic promotion here.
    if (ev.uniqueness_ratio >= t.identifier_min_uniqueness
            and ev.format_consistency is not None
            and ev.format_consistency >= t.identifier_min_format_consistency):
        reasons.append(f"Uniqueness ratio {ev.uniqueness_ratio} and format consistency "
                        f"{ev.format_consistency} both meet identifier-quality thresholds -- "
                        f"this field is statistically identifier-like based on observed data, "
                        f"not confirmed as a true business identifier.")
        if ev.identifier_hint:
            reasons.append("Canonical field is also identifier-hinted, reinforcing this -- "
                            "but the hint alone never triggers this recommendation on its own.")
        return "STRONG_IDENTIFIER_CANDIDATE", reasons

    # 3. Poor-value / repetitive field.
    #
    # ITEM-3 CORRECTION (see recommendation history): this used to gate on raw
    # uniqueness_ratio <= 0.05, which incorrectly excluded genuinely useful
    # low-cardinality categorical fields (city, state, country, postal_code)
    # from MATCH entirely -- distinct/total is always tiny for such fields
    # regardless of how informative they are. LOW UNIQUENESS != NO ER VALUE.
    # The correct test for "is this field basically constant/uninformative" is
    # single-value CONCENTRATION (top_value_share), not category count. A
    # field is only excluded here if it is truly constant or one value
    # dominates almost everything -- not merely because it has few categories.
    if ev.is_constant or ev.dominant_value_share >= t.low_value_max_top_share:
        reasons.append(f"A single value accounts for {ev.dominant_value_share:.1%} of all values "
                        f"(or the field is fully constant), providing little discriminative "
                        f"value for matching regardless of how many rows share it.")
        return "LOW_VALUE_FOR_ER", reasons

    # 4. Strong match field
    if ev.completeness_pct >= t.high_completeness and ev.uniqueness_ratio >= t.high_uniqueness:
        reasons.append(f"High completeness ({ev.completeness_pct}%) and uniqueness "
                        f"({ev.uniqueness_ratio}) make this a strong standalone match signal.")
        return "GOOD_MATCH_FIELD", reasons

    # 5. Supporting match field (default productive case)
    reasons.append(f"Completeness ({ev.completeness_pct}%) and uniqueness ({ev.uniqueness_ratio}) "
                    f"are usable but not strong enough alone -- best used alongside other fields.")
    return "SUPPORTING_MATCH_FIELD", reasons


def derive_cleansing_signals(
    missing_token_breakdown: dict,
    case_variation: bool,
    leading_trailing_ws_count: int,
    repeated_internal_ws_count: int,
    is_mixed_format: bool,
    scientific_notation_count: int,
    is_ambiguous_date: bool,
    length_outlier_count: int,
) -> list[str]:
    """
    NOTE on signals Module 4 never "resolves":
    - LENGTH_OUTLIERS is intentionally diagnostic-only. An unusually long or
      short value can be a perfectly valid real value (e.g. a long legal
      entity name) or a genuine data error -- there is no safe, generic
      transformation that can tell those apart, so this signal is meant to
      flag rows for human review and will legitimately still be present after
      standardization succeeds on everything else. This is expected behavior,
      not a bug.
    - MIXED_FORMAT_PATTERNS is suppressed upstream (see profiling_service.py)
      for FREE_TEXT fields before it ever reaches this function, since
      differing structural patterns are normal for free text and not a
      quality problem there.
    """
    signals = []
    if any(k for k in missing_token_breakdown if k != "" ) or missing_token_breakdown.get("", 0) > 0:
        if len(missing_token_breakdown) > 0:
            signals.append("MISSING_TOKEN_VARIANTS")
    if case_variation:
        signals.append("CASE_VARIATION")
    if leading_trailing_ws_count > 0:
        signals.append("LEADING_TRAILING_WHITESPACE")
    if repeated_internal_ws_count > 0:
        signals.append("REPEATED_INTERNAL_WHITESPACE")
    if is_mixed_format:
        signals.append("MIXED_FORMAT_PATTERNS")
    if scientific_notation_count > 0:
        signals.append("SCIENTIFIC_NOTATION_DETECTED")
    if is_ambiguous_date:
        signals.append("DATE_FORMAT_AMBIGUOUS")
    if length_outlier_count > 0:
        signals.append("LENGTH_OUTLIERS")
    return signals
