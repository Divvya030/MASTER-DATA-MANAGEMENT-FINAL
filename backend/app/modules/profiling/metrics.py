"""
MODULE 3 -- profiling metrics: completeness, cardinality, string quality,
identifier quality, format patterns, date ambiguity. All generic / value-driven.
"""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field
import pandas as pd

MISSING_TOKENS = {"n/a", "na", "-", "--", "null", "none", "unknown", "n.a.", "#n/a", "nil", ""}
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SCI_NOTATION_RE = re.compile(r"^\d(\.\d+)?[eE][+\-]?\d+$")


def _is_missing_token(v) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in MISSING_TOKENS


def _is_na_defensive(v) -> bool:
    """Defense-in-depth null check: correctly treats None, float NaN, and
    pandas NA/NaT as missing even if this function is ever called directly
    on values that bypassed the app-wide series_to_values() sanitization
    boundary (app/modules/nullsafe.py). Must not raise on non-scalar/odd
    inputs -- pd.isna() on a plain string or number is always a safe,
    well-defined False."""
    try:
        result = pd.isna(v)
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


@dataclass
class CompletenessResult:
    total_rows: int
    non_null_count: int
    null_count: int
    null_percentage: float
    missing_token_breakdown: dict[str, int]


def compute_completeness(values: list) -> CompletenessResult:
    total = len(values)
    breakdown: Counter = Counter()
    non_null = 0
    for v in values:
        if v is None or _is_na_defensive(v) or str(v).strip() == "":
            breakdown[""] += 1
            continue
        s = str(v).strip()
        if s.lower() in MISSING_TOKENS and s.lower() != "":
            breakdown[s] += 1
            continue
        non_null += 1
    null_count = total - non_null
    pct = round(100.0 * null_count / total, 2) if total else 0.0
    return CompletenessResult(total, non_null, null_count,
                               pct, dict(breakdown))


@dataclass
class CardinalityResult:
    distinct_count: int
    uniqueness_ratio: float
    # RENAMED from the old ambiguous "duplicate_value_count": this is the
    # count of extra ROW occurrences beyond each value's first appearance
    # (e.g. a value seen 3 times contributes 2 here). Kept under its old name
    # too, as an explicit alias, for any existing caller/test still using it.
    duplicate_row_count: int
    # NEW, correctly-named per the audit: the number of DISTINCT VALUES that
    # occur more than once -- a fundamentally different question from
    # duplicate_row_count above (e.g. one value appearing 100 times gives
    # duplicate_row_count=99 but duplicate_value_count=1).
    duplicate_value_count: int
    is_constant: bool
    top_values: list[tuple[str, int]]
    top_value_share: float
    # Block-distribution evidence: if this field were used as a blocking key,
    # each distinct value forms one "block" (the set of rows sharing it). This
    # is descriptive evidence for the steward to weigh -- see assess_blocking()
    # for why it's a heuristic, not a guarantee of blocking effectiveness.
    singleton_value_count: int
    largest_block_size: int
    avg_block_size: float


def compute_cardinality(values: list, top_n: int = 5) -> CardinalityResult:
    clean = [str(v).strip() for v in values if v is not None and str(v).strip() != ""
             and str(v).strip().lower() not in MISSING_TOKENS]
    n = len(clean)
    counts = Counter(clean)
    distinct = len(counts)
    uniqueness = round(distinct / n, 4) if n else 0.0
    duplicate_row_count = n - distinct
    duplicate_value_count = sum(1 for c in counts.values() if c > 1)
    is_constant = distinct == 1 and n > 0
    top = counts.most_common(top_n)
    top_share = round(sum(c for _, c in top) / n, 4) if n else 0.0

    singleton_value_count = sum(1 for c in counts.values() if c == 1)
    largest_block_size = max(counts.values()) if counts else 0
    avg_block_size = round(n / distinct, 2) if distinct else 0.0

    return CardinalityResult(distinct, uniqueness, duplicate_row_count, duplicate_value_count,
                              is_constant, top, top_share,
                              singleton_value_count, largest_block_size, avg_block_size)


@dataclass
class StringQualityResult:
    leading_trailing_whitespace_count: int
    case_variation_detected: bool
    repeated_internal_whitespace_count: int
    unusual_control_char_count: int
    length_min: int
    length_max: int
    length_mean: float
    length_outlier_count: int
    # Split out from case_variation_detected: these are two genuinely different
    # kinds of evidence that were previously conflated into one boolean.
    case_variation_same_value: bool
    case_variation_convention_mixed: bool
    # Row-level count (not just boolean) of values that belong to a
    # same-value-different-casing group -- used for the standardization audit
    # report's fixed/remaining counts.
    case_variation_affected_row_count: int


def compute_string_quality(values: list) -> StringQualityResult:
    clean = [str(v) for v in values if v is not None and str(v).strip() != ""]
    lt_ws = sum(1 for v in clean if v != v.strip())
    rep_ws = sum(1 for v in clean if re.search(r"\s{2,}", v))
    control = sum(1 for v in clean if CONTROL_CHAR_RE.search(v))

    lengths = [len(v.strip()) for v in clean]
    if lengths:
        mean_len = sum(lengths) / len(lengths)
        std = (sum((l - mean_len) ** 2 for l in lengths) / len(lengths)) ** 0.5
        outliers = sum(1 for l in lengths if abs(l - mean_len) > 3 * std) if std > 0 else 0
        lmin, lmax = min(lengths), max(lengths)
    else:
        mean_len, outliers, lmin, lmax = 0.0, 0, 0, 0

    # (1) SAME normalized-lower value appears with >1 distinct casing style --
    # this is the genuinely fixable, meaningful problem: it means the exact
    # same underlying value is represented inconsistently across rows.
    casing_by_lower: dict[str, set[str]] = {}
    for v in clean:
        stripped = v.strip()
        if not stripped.isalpha() and not any(c.isalpha() for c in stripped):
            continue
        casing_by_lower.setdefault(stripped.lower(), set()).add(stripped)
    case_variation_same_value = any(len(variants) > 1 for variants in casing_by_lower.values())
    affected_lowers = {k for k, variants in casing_by_lower.items() if len(variants) > 1}
    case_variation_affected_row_count = sum(
        1 for v in clean if v.strip().lower() in affected_lowers
    ) if affected_lowers else 0

    # (2) DIFFERENT values happen to use different casing conventions across the
    # column (e.g. one company name legitimately contains an uppercase
    # initialism like "LLP", another doesn't). This is normal diversity for
    # free-text fields, not a data-quality problem -- kept as separate
    # evidence, not blended into the actionable signal below.
    conventions = set()
    for v in clean:
        s = v.strip()
        if not s or not any(c.isalpha() for c in s):
            continue
        if s.isupper():
            conventions.add("UPPER")
        elif s.islower():
            conventions.add("LOWER")
        elif s.istitle():
            conventions.add("TITLE")
        else:
            conventions.add("MIXED")
    case_variation_convention_mixed = len(conventions) > 1

    # Combined flag kept for anyone reading raw evidence who wants "any case
    # inconsistency at all" -- but see profiling_service.py: only
    # case_variation_same_value drives the CASE_VARIATION cleansing signal.
    case_variation_detected = case_variation_same_value or case_variation_convention_mixed

    return StringQualityResult(lt_ws, case_variation_detected, rep_ws, control,
                                lmin, lmax, round(mean_len, 2), outliers,
                                case_variation_same_value, case_variation_convention_mixed,
                                case_variation_affected_row_count)


def _structural_pattern(v: str) -> str:
    """Maps each character to a structural class: letters -> 'A', digits ->
    '9', everything else kept literal (so punctuation/separators remain
    meaningful in the pattern, e.g. 'TX-00231' -> 'AA-99999')."""
    out = []
    for ch in v:
        if ch.isalpha():
            out.append("A")
        elif ch.isdigit():
            out.append("9")
        else:
            out.append(ch)
    return "".join(out)


@dataclass
class FormatPatternResult:
    format_patterns: list[tuple[str, int]]
    is_mixed_format: bool


def compute_format_patterns(values: list, top_n: int = 5) -> FormatPatternResult:
    clean = [str(v).strip() for v in values if v is not None and str(v).strip() != ""]
    patterns = Counter(_structural_pattern(v) for v in clean)
    top = patterns.most_common(top_n)
    is_mixed = len(patterns) > 1
    return FormatPatternResult(top, is_mixed)


@dataclass
class IdentifierQualityResult:
    format_consistency: float  # share of values matching the dominant format pattern
    # RENAMED from "invalid_count": this only ever counted control characters,
    # not any broader notion of semantic identifier validity (no such generic
    # validation mechanism exists), so the old name overclaimed what it measures.
    control_character_count: int
    scientific_notation_count: int
    length_outlier_count: int


def compute_identifier_quality(values: list) -> IdentifierQualityResult:
    clean = [str(v).strip() for v in values if v is not None and str(v).strip() != ""]
    if not clean:
        return IdentifierQualityResult(0.0, 0, 0, 0)

    sci_count = sum(1 for v in clean if SCI_NOTATION_RE.match(v))
    patterns = Counter(_structural_pattern(v) for v in clean)
    dominant_count = patterns.most_common(1)[0][1]
    consistency = round(dominant_count / len(clean), 4)

    control_character_count = sum(1 for v in clean if CONTROL_CHAR_RE.search(v))

    lengths = [len(v) for v in clean]
    mean_len = sum(lengths) / len(lengths)
    std = (sum((l - mean_len) ** 2 for l in lengths) / len(lengths)) ** 0.5
    length_outliers = sum(1 for l in lengths if abs(l - mean_len) > 3 * std) if std > 0 else 0

    return IdentifierQualityResult(consistency, control_character_count, sci_count, length_outliers)


@dataclass
class DateAmbiguityResult:
    parseable_rate: float
    dd_mm_evidence_count: int
    mm_dd_evidence_count: int
    ambiguous_count: int
    is_ambiguous: bool
    mixed_format_suspected: bool
    detected_format_candidates: list[str]
    # NEW: date-LIKE structure is not proof of calendar validity (e.g.
    # "2025-99-99" or "31/02/2025" are shaped like dates but describe no real
    # calendar day). Tracked separately from ambiguous_count -- "ambiguous"
    # means multiple valid readings exist; "invalid" means none do.
    invalid_date_count: int = 0


_SLASH_DATE_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$")
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})")


def _expand_year(year_str: str) -> int:
    if len(year_str) == 4:
        return int(year_str)
    yy = int(year_str)
    return 2000 + yy if yy < 70 else 1900 + yy


def _is_valid_calendar_date(year: int, month: int, day: int) -> bool:
    import datetime
    try:
        datetime.date(year, month, day)
        return True
    except (ValueError, OverflowError):
        return False


def compute_date_ambiguity(values: list) -> DateAmbiguityResult:
    clean = [str(v).strip() for v in values if v is not None and str(v).strip() != ""]
    total = len(clean)
    if total == 0:
        return DateAmbiguityResult(0.0, 0, 0, 0, False, False, [], 0)

    parseable = 0
    dd_mm_evidence = 0
    mm_dd_evidence = 0
    ambiguous = 0
    invalid = 0
    formats_seen = set()

    for v in clean:
        m = _SLASH_DATE_RE.match(v)
        if m:
            a, b, year_str = int(m.group(1)), int(m.group(2)), m.group(3)
            year = _expand_year(year_str)
            formats_seen.add("D/M/Y or M/D/Y (slash-separated)")
            # Ground truth via actual calendar validity, not just numeric
            # range checks -- this correctly subsumes the old ">12" forcing
            # rule as the special case where one reading is calendar-invalid
            # because the "month" would exceed 12, while also catching
            # invalid days-in-month (e.g. Feb 31) that range checks alone miss.
            dd_mm_valid = _is_valid_calendar_date(year, b, a)   # day=a, month=b
            mm_dd_valid = _is_valid_calendar_date(year, a, b)   # month=a, day=b
            if dd_mm_valid and mm_dd_valid:
                parseable += 1
                ambiguous += 1
            elif dd_mm_valid:
                parseable += 1
                dd_mm_evidence += 1
            elif mm_dd_valid:
                parseable += 1
                mm_dd_evidence += 1
            else:
                invalid += 1  # neither reading is a real calendar date
        elif _ISO_DATE_RE.match(v):
            m2 = _ISO_DATE_RE.match(v)
            year, month, day = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            if _is_valid_calendar_date(year, month, day):
                parseable += 1
                formats_seen.add("ISO Y-M-D")
            else:
                invalid += 1

    parseable_rate = round(100.0 * parseable / total, 2)
    is_ambiguous = ambiguous > 0 and dd_mm_evidence == 0 and mm_dd_evidence == 0
    mixed_format_suspected = len(formats_seen) > 1

    return DateAmbiguityResult(
        parseable_rate, dd_mm_evidence, mm_dd_evidence, ambiguous,
        is_ambiguous, mixed_format_suspected, sorted(formats_seen), invalid,
    )
