"""
MODULE 4 -- Standardization / Cleansing

Transforms the mapped, canonical-field columns of a RAW dataframe into a
STANDARDIZED dataframe, guided by the cleansing_signals produced by profiling.
Never touches the RAW layer. Every transformation is logged so the "before ->
after" comparison in Module 5 / the UI is explainable.

Identifiers are always treated as strings; leading zeros are preserved; no
value is invented; scientific-notation corruption is repaired only when the
original integer value is unambiguously recoverable (i.e. it round-trips
exactly back to a whole number with no fractional part lost).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Optional, Callable

import pandas as pd

from app.modules.nullsafe import series_to_values

# No real-world identifier plausibly needs more digits than this. Bounding
# the exponent before Decimal ever parses it prevents a corrupted/malicious
# value with an absurd exponent from making to_integral_value() try to
# materialize a multi-million-digit integer (confirmed to hang).
_MAX_SAFE_SCI_NOTATION_EXPONENT = 50

MISSING_TOKENS = {"n/a", "na", "-", "--", "null", "none", "unknown", "n.a.", "#n/a", "nil",
                  # Form/system placeholders observed in the real customer extract (V1 build): left as
                  # values they make unrelated records look like they agree on city/address/postcode.
                  "please select", "please update", "billing street missing", "tbc", "tbd", "?",
                  "unspecified", "not specified", "not available", "111111", "123 main st."}
SCI_NOTATION_RE = re.compile(r"^(\d)(\.(\d+))?[eE]\+?(\d+)$")
_WS_RE = re.compile(r"\s+")


@dataclass
class TransformationLogEntry:
    canonical_field: str
    row_index: int
    transformation: str
    before: Optional[str]
    after: Optional[str]


@dataclass
class StandardizationResult:
    df: pd.DataFrame
    log: list[TransformationLogEntry] = field(default_factory=list)

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for entry in self.log:
            counts[entry.transformation] = counts.get(entry.transformation, 0) + 1
        return counts


def _normalize_missing_token(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in MISSING_TOKENS:
        return None
    return v


def _trim_and_collapse_ws(v):
    if v is None:
        return v
    s = str(v)
    trimmed = s.strip()
    collapsed = _WS_RE.sub(" ", trimmed)
    return collapsed


def _try_repair_scientific_notation(v):
    """Repair values like '1.00E+14' back to the integer they represent, but
    ONLY when the mantissa's fractional digits are all zero (i.e. no
    precision was actually lost in the original string) AND exact integer
    recovery can be proven with Decimal's exact arithmetic -- never with
    float(), which silently diverges from the true integer for sufficiently
    large exponents (confirmed: float-based recovery of '1.00E+23' yields
    99999999999999991611392, not the true 100000000000000000000000 -- a
    24-digit identifier would be silently corrupted). Decimal has no such
    binary-representation error for any realistic exponent.

    The exponent is bounded BEFORE Decimal ever sees it: Decimal is
    arbitrary-precision, so an absurd exponent (corrupted data, or a
    malicious value) would otherwise make to_integral_value() try to
    materialize a multi-million-digit integer -- confirmed to hang for
    tens of seconds+ on a single value with a ~7-digit exponent. No real
    identifier needs more than a few dozen digits, so this bound costs
    nothing legitimate while closing a real resource-exhaustion path.
    """
    if v is None:
        return v, False
    s = str(v).strip()
    m = SCI_NOTATION_RE.match(s)
    if not m:
        return v, False
    mantissa_int, _, mantissa_frac, exponent_str = m.groups()
    frac = mantissa_frac or ""
    if frac.strip("0") != "":
        # fractional precision was lost -- NOT safely recoverable
        return v, False
    try:
        exponent = int(exponent_str)
    except ValueError:
        return v, False
    if exponent > _MAX_SAFE_SCI_NOTATION_EXPONENT:
        return v, False
    try:
        exact = Decimal(s)
        integral = exact.to_integral_value()
        if exact != integral:
            # Defensive guard only -- should not occur given the all-zero
            # fractional-digit check above, but never guess if it somehow did.
            return v, False
        return str(int(integral)), True
    except (InvalidOperation, ValueError, OverflowError):
        return v, False


def _normalize_email(v):
    if v is None:
        return v
    return str(v).strip().lower()


_PHONE_EXTENSION_RE = re.compile(r"\b(ext\.?|extension|x)\s*\d+\s*$", re.IGNORECASE)


def _normalize_phone(v):
    """Deterministic, safe formatting only: strip whitespace and common
    harmless separators (spaces, dashes, dots, parens) to produce a clean
    digit string, preserving a leading '+' if present. Never invents a
    country code, never adds/removes digits -- purely separator cleanup.

    Extension-like suffixes ("ext 123", "x123") are explicitly detected and
    left UNTOUCHED rather than normalized: naively stripping separators would
    otherwise concatenate the extension's digits onto the main number,
    silently corrupting it into a longer, wrong phone number. Safely
    separating a main number from its extension would require a real phone
    parser, which is deliberately out of scope here."""
    if v is None:
        return v
    s = str(v).strip()
    if _PHONE_EXTENSION_RE.search(s):
        return s
    has_plus = s.startswith("+")
    digits = re.sub(r"[^\d]", "", s)
    if not digits:
        return s  # nothing recognizable as digits -- leave untouched
    return ("+" if has_plus else "") + digits


def _normalize_website(v):
    """Trim always. Lowercase only the clearly case-insensitive scheme+domain
    portion (per URL spec, hostnames are case-insensitive); a URL's
    path/query/fragment can be genuinely case-sensitive (a token, a
    case-sensitive route) and must be left exactly as given. A bare domain
    with no scheme (e.g. 'EXAMPLE.COM', or an email-like value) has no path
    component at all, so it is safe to lowercase in full."""
    if v is None:
        return v
    s = str(v).strip()
    if not s:
        return s
    if "://" not in s:
        return s.lower()
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(s)
    lowered = parts._replace(scheme=parts.scheme.lower(), netloc=parts.netloc.lower())
    return urlunsplit(lowered)


import calendar

_MONTH_LOOKUP = {}
for _i, _name in enumerate(calendar.month_name):
    if _name:
        _MONTH_LOOKUP[_name.lower()] = _i
for _i, _abbr in enumerate(calendar.month_abbr):
    if _abbr:
        _MONTH_LOOKUP[_abbr.lower()] = _i

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YMD_NUMERIC_RE = re.compile(r"^(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})$")
_DMY_NUMERIC_RE = re.compile(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})$")
_DAY_MONTHNAME_YEAR_RE = re.compile(r"^(\d{1,2})[\s\-]([A-Za-z]{3,9})[\s\-](\d{2,4})$")
_MONTHNAME_DAY_YEAR_RE = re.compile(r"^([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{2,4})$")


def _expand_year(year_str: str) -> Optional[int]:
    if len(year_str) == 4:
        return int(year_str)
    if len(year_str) == 2:
        yy = int(year_str)
        return 2000 + yy if yy < 70 else 1900 + yy
    return None


def _valid_ymd(year: int, month: int, day: int) -> bool:
    import datetime as _dt
    try:
        _dt.date(year, month, day)
        return True
    except ValueError:
        return False


def _normalize_date_if_safe(v):
    """Convert a date value to canonical YYYY-MM-DD ONLY when no locale
    guessing is required -- i.e. day/month ordering is unambiguous from the
    value itself (a named month, or a numeric position >12 that can only be
    a day). Genuinely ambiguous values (e.g. '03/04/2024', where both
    readings are valid dates) are left completely untouched -- DATE_AMBIGUOUS
    (see Module 3's date-ambiguity evidence) is the authoritative signal that
    a column-wide locale decision would be required, and this function never
    makes that decision. It only acts where no decision is needed at all."""
    if v is None:
        return v
    s = str(v).strip()
    if not s:
        return v

    if _ISO_DATE_RE.match(s):
        return s  # already canonical

    m = _DAY_MONTHNAME_YEAR_RE.match(s)
    if m:
        day_s, month_name, year_s = m.groups()
        month = _MONTH_LOOKUP.get(month_name.lower())
        year = _expand_year(year_s)
        if month and year is not None:
            try:
                day = int(day_s)
                if _valid_ymd(year, month, day):
                    return f"{year:04d}-{month:02d}-{day:02d}"
            except ValueError:
                pass
        return v

    m = _MONTHNAME_DAY_YEAR_RE.match(s)
    if m:
        month_name, day_s, year_s = m.groups()
        month = _MONTH_LOOKUP.get(month_name.lower())
        year = _expand_year(year_s)
        if month and year is not None:
            try:
                day = int(day_s)
                if _valid_ymd(year, month, day):
                    return f"{year:04d}-{month:02d}-{day:02d}"
            except ValueError:
                pass
        return v

    m = _YMD_NUMERIC_RE.match(s)
    if m:
        year_s, a_s, b_s = m.groups()
        year, a, b = int(year_s), int(a_s), int(b_s)
        if a > 12 and b <= 12 and _valid_ymd(year, b, a):
            return f"{year:04d}-{b:02d}-{a:02d}"      # a=day, b=month (forced)
        if b > 12 and a <= 12 and _valid_ymd(year, a, b):
            return f"{year:04d}-{a:02d}-{b:02d}"      # a=month, b=day (forced)
        return v  # both <=12 -- genuinely ambiguous even with year leading, do not guess

    m = _DMY_NUMERIC_RE.match(s)
    if m:
        a_s, b_s, year_s = m.groups()
        year = _expand_year(year_s)
        a, b = int(a_s), int(b_s)
        if year is None:
            return v
        if a > 12 and b <= 12 and _valid_ymd(year, b, a):
            return f"{year:04d}-{b:02d}-{a:02d}"      # a=day, b=month (forced)
        if b > 12 and a <= 12 and _valid_ymd(year, a, b):
            return f"{year:04d}-{a:02d}-{b:02d}"      # a=month, b=day (forced)
        return v  # both <=12 -- genuinely ambiguous, preserve

    return v  # unrecognized format -- preserve


def _normalize_identifier(v):
    """Identifiers stay strings; preserve leading zeros; just trim whitespace."""
    if v is None:
        return v
    return str(v).strip()


# Genuine letter-by-letter initialisms (pronounced as individual letters, e.g.
# "L-L-C") that are conventionally kept fully uppercase in business writing.
# Deliberately narrow: common abbreviations like Inc/Ltd/Pvt/Corp are
# conventionally Title-Case in normal prose, not all-caps acronyms -- verified
# against real data (see standardization fix notes), where already-correctly-
# formatted values consistently write "Inc"/"Pvt"/"Ltd" in Title Case but
# "LLP" fully uppercase, 100% of the time either way. Getting this list wrong
# in the other direction (too broad) actively creates NEW case inconsistency
# instead of resolving it, so it is kept deliberately conservative/narrow.
_LEGAL_SUFFIX_TOKENS = {"llc", "llp", "plc", "lp", "pllc"}


def _normalize_name_case(v):
    """Conservative case normalization for free-text person/company name
    fields. Only re-cases values that are UNAMBIGUOUSLY all-uppercase or
    all-lowercase -- i.e. there is no existing intentional mixed-case
    formatting to risk destroying. A value that already mixes case (e.g.
    'Sterling Automotive Inc') is left completely untouched, since telling
    intentional formatting apart from an actual error requires business
    judgment this platform does not have -- see the "no unsafe semantic
    transformations" principle. Known legal-entity suffix tokens are kept
    fully uppercase rather than being flattened by naive title-casing."""
    if v is None:
        return v
    s = str(v)
    if not (s.isupper() or s.islower()):
        return s
    words = s.split(" ")
    out = []
    for w in words:
        if not w:
            out.append(w)
            continue
        core = w.strip(".,")
        if core.lower() in _LEGAL_SUFFIX_TOKENS:
            out.append(w.upper())
        else:
            out.append(w.capitalize())
    return " ".join(out)


IDENTIFIER_FIELDS = {"customer_id", "supplier_id", "duns", "tax_identifier", "registration_number"}
EMAIL_FIELDS = {"email"}
PHONE_FIELDS = {"phone"}
WEBSITE_FIELDS = {"website"}
# Free-text person/company name fields shared across canonical schemas (see
# domain_schemas.py) -- not specific to any one dataset or domain.
NAME_FIELDS = {"customer_name", "legal_name", "first_name", "last_name", "supplier_name", "trade_name"}
# Free-text location fields that also benefit from the same conservative case
# normalization (e.g. "CHENNAI" -> "Chennai", "123 MAIN STREET" -> "123 Main
# Street"). Kept separate from NAME_FIELDS since they never need legal-suffix
# preservation, but reuse the identical, equally conservative
# all-upper/all-lower-only rule.
GEO_CASE_FIELDS = {"city", "state", "country", "address"}
# Canonical date-typed fields (only "dob" exists in either schema today; add
# more here if a canonical date field is ever added to domain_schemas.py --
# consistent with how IDENTIFIER_FIELDS/EMAIL_FIELDS/etc. are declared).
DATE_FIELDS = {"dob"}


def standardize_column(
    canonical_field: str,
    values: list,
    cleansing_signals: list[str],
) -> tuple[list, list[TransformationLogEntry]]:
    out = []
    log: list[TransformationLogEntry] = []
    is_identifier = canonical_field in IDENTIFIER_FIELDS

    for idx, raw_v in enumerate(values):
        v = raw_v
        original = v

        if "MISSING_TOKEN_VARIANTS" in cleansing_signals:
            new_v = _normalize_missing_token(v)
            if new_v != v:
                log.append(TransformationLogEntry(canonical_field, idx, "NORMALIZE_MISSING_TOKEN",
                                                    str(v) if v is not None else None, None))
                v = new_v

        if v is None:
            out.append(None)
            continue

        if is_identifier and "SCIENTIFIC_NOTATION_DETECTED" in cleansing_signals:
            repaired, changed = _try_repair_scientific_notation(v)
            if changed:
                log.append(TransformationLogEntry(canonical_field, idx, "REPAIR_SCIENTIFIC_NOTATION",
                                                    str(v), str(repaired)))
                v = repaired

        if "LEADING_TRAILING_WHITESPACE" in cleansing_signals or "REPEATED_INTERNAL_WHITESPACE" in cleansing_signals:
            new_v = _trim_and_collapse_ws(v)
            if new_v != v:
                log.append(TransformationLogEntry(canonical_field, idx, "TRIM_COLLAPSE_WHITESPACE",
                                                    str(v), str(new_v)))
                v = new_v

        if canonical_field in EMAIL_FIELDS:
            new_v = _normalize_email(v)
            if new_v != v:
                log.append(TransformationLogEntry(canonical_field, idx, "NORMALIZE_EMAIL_CASE",
                                                    str(v), str(new_v)))
                v = new_v
        elif canonical_field in WEBSITE_FIELDS:
            new_v = _normalize_website(v)
            if new_v != v:
                log.append(TransformationLogEntry(canonical_field, idx, "NORMALIZE_WEBSITE",
                                                    str(v), str(new_v)))
                v = new_v
        elif canonical_field in PHONE_FIELDS:
            new_v = _normalize_phone(v)
            if new_v != v:
                log.append(TransformationLogEntry(canonical_field, idx, "NORMALIZE_PHONE_SEPARATORS",
                                                    str(v), str(new_v)))
                v = new_v
        elif is_identifier:
            new_v = _normalize_identifier(v)
            if new_v != v:
                log.append(TransformationLogEntry(canonical_field, idx, "NORMALIZE_IDENTIFIER_STRING",
                                                    str(v), str(new_v)))
                v = new_v
        elif canonical_field in (NAME_FIELDS | GEO_CASE_FIELDS):
            # Unconditional, like email/phone/website: the transformation is
            # already safe-by-construction (only re-cases unambiguous
            # ALL-CAPS/all-lowercase values; no-op on anything else), so it
            # doesn't need to be gated behind the CASE_VARIATION signal --
            # that signal is deliberately narrower (only genuine same-value
            # duplicate mismatches, to avoid false-positiving on legitimately
            # diverse different names) and is used for PROFILING/reporting,
            # not as a gate on this always-safe transformation.
            new_v = _normalize_name_case(v)
            if new_v != v:
                log.append(TransformationLogEntry(canonical_field, idx, "NORMALIZE_NAME_CASE",
                                                    str(v), str(new_v)))
                v = new_v
        elif canonical_field in DATE_FIELDS:
            new_v = _normalize_date_if_safe(v)
            if new_v != v:
                log.append(TransformationLogEntry(canonical_field, idx, "NORMALIZE_DATE_ISO",
                                                    str(v), str(new_v)))
                v = new_v

        out.append(v)

    return out, log


def standardize_dataset(
    raw_df: pd.DataFrame,
    mapping: dict[str, str],  # canonical_field -> source_column
    cleansing_signals_by_field: dict[str, list[str]],
) -> StandardizationResult:
    """Returns a NEW dataframe (RAW is never mutated). Only mapped canonical
    columns are standardized; all other raw columns pass through unchanged."""
    df = raw_df.copy(deep=True)
    all_log: list[TransformationLogEntry] = []

    for canonical_field, source_column in mapping.items():
        if source_column not in df.columns:
            continue
        signals = cleansing_signals_by_field.get(canonical_field, [])
        values = series_to_values(df[source_column])
        new_values, log = standardize_column(canonical_field, values, signals)
        df[source_column] = new_values
        all_log.extend(log)

    return StandardizationResult(df=df, log=all_log)
