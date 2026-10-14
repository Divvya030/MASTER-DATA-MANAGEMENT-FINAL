"""
MODULE 5 -- Re-Profiling / Quality Validation

Runs the SAME profiling engine (Module 3) again on the STANDARDIZED layer and
produces a before/after comparison. RAW profiling results are never overwritten;
both are kept side by side.
"""
from __future__ import annotations
from dataclasses import dataclass

from app.modules.profiling.profiling_service import FieldProfile


@dataclass
class FieldComparison:
    canonical_field: str
    before: dict
    after: dict
    improved: bool
    notes: list[str]


def _extract_headline_metrics(profile: FieldProfile) -> dict:
    dq = profile.data_quality
    return {
        "completeness_pct": round(100 - dq["completeness"]["null_percentage"], 2),
        "whitespace_issues": (
            dq["string_quality"]["leading_trailing_whitespace_count"]
            + dq["string_quality"]["repeated_internal_whitespace_count"]
        ),
        "is_mixed_format": dq["format_patterns"]["is_mixed_format"],
        "scientific_notation_count": (
            dq["identifier_quality"]["scientific_notation_count"] if dq["identifier_quality"] else 0
        ),
        "cleansing_signal_count": len(profile.cleansing_signals),
        "er_recommendation": profile.er_recommendation,
    }


def compare_profiles(
    raw_profiles: dict[str, FieldProfile],
    standardized_profiles: dict[str, FieldProfile],
) -> dict[str, FieldComparison]:
    comparisons = {}
    for canonical_field, raw_profile in raw_profiles.items():
        std_profile = standardized_profiles.get(canonical_field)
        if std_profile is None:
            continue
        before = _extract_headline_metrics(raw_profile)
        after = _extract_headline_metrics(std_profile)

        notes = []
        improved = False
        if after["whitespace_issues"] < before["whitespace_issues"]:
            notes.append(f"Whitespace issues reduced from {before['whitespace_issues']} to {after['whitespace_issues']}.")
            improved = True
        if before["is_mixed_format"] and not after["is_mixed_format"]:
            notes.append("Mixed format patterns resolved into a single consistent format.")
            improved = True
        if after["scientific_notation_count"] < before["scientific_notation_count"]:
            notes.append(
                f"Scientific-notation corruption reduced from {before['scientific_notation_count']} "
                f"to {after['scientific_notation_count']} values."
            )
            improved = True
        resolved_signals = set(raw_profile.cleansing_signals) - set(std_profile.cleansing_signals)
        if resolved_signals:
            notes.append(f"Resolved cleansing signal(s): {', '.join(sorted(resolved_signals))}.")
            improved = True
        if not notes:
            notes.append("No measurable quality change detected for this field.")

        comparisons[canonical_field] = FieldComparison(
            canonical_field=canonical_field, before=before, after=after,
            improved=improved, notes=notes,
        )
    return comparisons


# ---------------------------------------------------------------------------
# Per-signal standardization audit report.
#
# Builds the {field, signal, detected_before, changed, fixed_count,
# remaining_count, status, explanation} record set, reusing ONLY the RAW and
# STANDARDIZED FieldProfile objects that Modules 3/5 already persist -- no new
# storage, no dependency on an ephemeral in-request transformation log.
# fixed_count is derived as (how many instances of this signal existed in
# RAW) - (how many remain in STANDARDIZED); this is more robust than counting
# individual transformation-log rows, since it reflects the NET reduction
# regardless of which specific transformation produced it.
# ---------------------------------------------------------------------------

_EXPLANATIONS = {
    ("MISSING_TOKEN_VARIANTS", "FIXED"):
        "Recognized missing-value tokens (e.g. 'N/A', '-', blank) were converted to true nulls.",
    ("MISSING_TOKEN_VARIANTS", "PARTIALLY_FIXED"):
        "Recognized missing-value tokens were converted to true nulls; some remaining values were "
        "not recognized as a known missing-value token.",
    ("LEADING_TRAILING_WHITESPACE", "FIXED"):
        "Leading and trailing whitespace was trimmed.",
    ("REPEATED_INTERNAL_WHITESPACE", "FIXED"):
        "Repeated internal whitespace was collapsed to a single space.",
    ("CASE_VARIATION", "FIXED"):
        "Case variation was normalized using conservative, deterministic case normalization "
        "(only unambiguous ALL-CAPS/all-lowercase values were re-cased; already mixed-case "
        "values were left untouched).",
    ("CASE_VARIATION", "PARTIALLY_FIXED"):
        "Case variation was normalized where safe to do so; some values were already in a "
        "mixed-case form that could not be safely rewritten without risking real data.",
    ("SCIENTIFIC_NOTATION_DETECTED", "FIXED"):
        "Scientific-notation corruption was safely repaired back to the original integer "
        "(only where the fractional digits round-tripped exactly with no precision loss).",
    ("SCIENTIFIC_NOTATION_DETECTED", "PRESERVED"):
        "Scientific-notation values were preserved because the original precision could not be "
        "safely recovered -- guessing the lost digits would risk corrupting the identifier.",
    ("LENGTH_OUTLIERS", "PRESERVED"):
        "Length outliers were preserved. There is no safe deterministic correction -- truncating "
        "could destroy genuinely valid data (e.g. a long but real legal entity name).",
    ("DATE_FORMAT_AMBIGUOUS", "PRESERVED"):
        "Dates were preserved because their day/month ordering cannot be determined without a "
        "locale decision for this column -- that decision belongs to steward configuration, not "
        "automatic standardization. NOTE: individual values elsewhere in this field that were "
        "unambiguous (a named month, or a value >12 that can only be a day) may still have been "
        "safely converted to YYYY-MM-DD even though this column-level signal remains -- see the "
        "transformation log for row-level detail.",
    ("MIXED_FORMAT_PATTERNS", "PRESERVED"):
        "Mixed formats were preserved because no safe universal normalization rule exists for "
        "this field (e.g. postal codes and phone numbers legitimately vary by country/system).",
}

_DEFAULT_EXPLANATIONS = {
    "FIXED": "This issue was automatically and safely resolved.",
    "PARTIALLY_FIXED": "This issue was partially resolved; the remainder could not be safely fixed.",
    "PRESERVED": "No safe deterministic transformation exists for this issue, so the original value was preserved.",
    "NOT_APPLICABLE": "No actual row-level instances of this issue were found; the signal reflects a "
                       "broader dataset-level pattern rather than a fixable per-value condition.",
}


def _signal_row_count(signal: str, profile: FieldProfile) -> int:
    """Best-effort extraction of a row-level 'how many values exhibit this
    signal' count from an already-computed FieldProfile."""
    dq = profile.data_quality
    if signal == "MISSING_TOKEN_VARIANTS":
        breakdown = dq["completeness"]["missing_token_breakdown"]
        return sum(c for k, c in breakdown.items() if k != "")
    if signal == "LEADING_TRAILING_WHITESPACE":
        return dq["string_quality"]["leading_trailing_whitespace_count"]
    if signal == "REPEATED_INTERNAL_WHITESPACE":
        return dq["string_quality"]["repeated_internal_whitespace_count"]
    if signal == "CASE_VARIATION":
        return dq["string_quality"].get("case_variation_affected_row_count", 0)
    if signal == "SCIENTIFIC_NOTATION_DETECTED":
        return dq["identifier_quality"]["scientific_notation_count"] if dq["identifier_quality"] else 0
    if signal == "LENGTH_OUTLIERS":
        return dq["string_quality"]["length_outlier_count"]
    if signal == "DATE_FORMAT_AMBIGUOUS":
        return dq["date_ambiguity"]["ambiguous_count"] if dq["date_ambiguity"] else 0
    if signal == "MIXED_FORMAT_PATTERNS":
        # Approximation only: top_values stores the top-N patterns, not the
        # full distribution, so this is a lower bound ("non-null rows minus
        # the single most common pattern's count"), documented as such.
        top = dq["format_patterns"]["format_patterns"]
        non_null = dq["completeness"]["non_null_count"]
        if not top:
            return 0
        return max(non_null - top[0]["count"], 0)
    return 0


@dataclass
class SignalAuditEntry:
    field: str
    signal: str
    detected_before: bool
    changed: bool
    fixed_count: int
    remaining_count: int
    status: str  # FIXED | PARTIALLY_FIXED | PRESERVED | NOT_APPLICABLE
    explanation: str
    # True when fixed_count/remaining_count are derived/estimated rather than
    # an exact row-level count -- currently only MIXED_FORMAT_PATTERNS, whose
    # remaining count is a lower-bound approximation (top-N patterns only,
    # not the full distribution). Surfaced explicitly so the UI/API never
    # implies false precision.
    is_estimated: bool = False

    def to_dict(self) -> dict:
        return {
            "field": self.field, "signal": self.signal,
            "detected_before": self.detected_before, "changed": self.changed,
            "fixed_count": self.fixed_count, "remaining_count": self.remaining_count,
            "status": self.status, "explanation": self.explanation,
            "is_estimated": self.is_estimated,
        }


def build_signal_audit_report(
    raw_profiles: dict[str, FieldProfile],
    standardized_profiles: dict[str, FieldProfile],
) -> list[SignalAuditEntry]:
    """One entry per (field, signal) that was actually DETECTED in RAW
    profiling -- signals never present in the first place are not reported,
    since there is nothing to fix or preserve. Callable any time both RAW and
    STANDARDIZED profiles exist; no dependency on request-scoped state."""
    report: list[SignalAuditEntry] = []
    for canonical_field, raw_profile in raw_profiles.items():
        std_profile = standardized_profiles.get(canonical_field)
        if std_profile is None:
            continue
        for signal in raw_profile.cleansing_signals:
            before_count = _signal_row_count(signal, raw_profile)
            after_count = _signal_row_count(signal, std_profile)
            fixed_count = max(before_count - after_count, 0)
            remaining_count = after_count
            changed = fixed_count > 0
            if changed and remaining_count == 0:
                status = "FIXED"
            elif changed and remaining_count > 0:
                status = "PARTIALLY_FIXED"
            elif not changed and remaining_count == 0 and before_count == 0:
                # The signal was technically "detected" (e.g. MISSING_TOKEN_VARIANTS
                # firing on ordinary blank cells with zero actual token variants),
                # but there was nothing to fix at the row level in the first place --
                # distinct from a genuine preserved-because-unsafe case below.
                status = "NOT_APPLICABLE"
            else:
                status = "PRESERVED"
            explanation = _EXPLANATIONS.get((signal, status), _DEFAULT_EXPLANATIONS[status])
            report.append(SignalAuditEntry(
                field=canonical_field, signal=signal, detected_before=True,
                changed=changed, fixed_count=fixed_count, remaining_count=remaining_count,
                status=status, explanation=explanation,
                is_estimated=(signal == "MIXED_FORMAT_PATTERNS"),
            ))
    return report
