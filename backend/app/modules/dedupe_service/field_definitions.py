"""
Builds dedupe.variables field definitions from the ACTIVE, steward-confirmed
ER configuration -- never from profiling recommendations directly, and never
hardcoded to a specific domain's field names.

Only MATCH-role fields become Dedupe comparison variables: Dedupe's learned
comparison model is what "MATCH" means in this pipeline (see Module 6). A
field with BOTH IDENTIFIER and MATCH roles still gets exactly ONE Dedupe
variable definition -- roles are metadata about how the field is USED in the
wider ER configuration, not a request for multiple/duplicate Dedupe
variables for the same field. BLOCKING-only and IGNORE fields are excluded
entirely from the fields passed to Dedupe.
"""
from __future__ import annotations
from dataclasses import dataclass


class InsufficientERConfigError(ValueError):
    pass


@dataclass
class ERFieldRoles:
    identifier_fields: list[str]
    match_fields: list[str]
    blocking_fields: list[str]
    ignored_fields: list[str]


def build_dedupe_variables(er_roles: ERFieldRoles):
    """Returns a list of dedupe.variables.String(...) instances, one per
    unique MATCH-role field. Import of dedupe is deferred to call time so
    that modules which only need ERFieldRoles (e.g. tests of pure config
    logic) never require the dedupe package to be installed."""
    from dedupe.variables import String

    if not er_roles.match_fields:
        raise InsufficientERConfigError(
            "No MATCH fields are configured in the active ER configuration. "
            "Dedupe requires at least one MATCH field to build a comparison "
            "model -- IDENTIFIER-only or BLOCKING-only configurations are "
            "not sufficient on their own."
        )
    # De-duplicate while preserving order, in case a field somehow appears
    # twice in match_fields (defensive -- Module 6 already prevents this).
    seen = set()
    unique_match_fields = []
    for f in er_roles.match_fields:
        if f not in seen:
            seen.add(f)
            unique_match_fields.append(f)

    # has_missing=True is required: real datasets have genuinely missing
    # field values, and Dedupe's own string-distance function divides by the
    # combined length of both compared strings -- passing "" for a missing
    # value instead of using Dedupe's own missing-value handling causes a
    # ZeroDivisionError the moment two records are BOTH missing the same
    # field (confirmed against real customer.csv data, not hypothetical).
    return [String(field=f, has_missing=True) for f in unique_match_fields]


def load_active_er_roles(role_rows) -> ERFieldRoles:
    """role_rows: iterable of objects with .canonical_field and .role
    attributes (ERFieldConfig rows). Pure function, no DB/session coupling,
    so it is trivially testable without a database."""
    roles = ERFieldRoles([], [], [], [])
    key_to_list = {
        "IDENTIFIER": roles.identifier_fields,
        "MATCH": roles.match_fields,
        "BLOCKING": roles.blocking_fields,
        "IGNORE": roles.ignored_fields,
    }
    for row in role_rows:
        target = key_to_list.get(row.role)
        if target is not None:
            target.append(row.canonical_field)
    return roles


def coerce_for_dedupe(value):
    """The ONE place a STANDARDIZED value is turned into what Dedupe's String
    variable accepts. Only genuinely missing values (None / NaN / pandas NA /
    empty or whitespace-only) become None, so Dedupe's has_missing handling
    applies; they are NEVER turned into the literal strings "nan"/"None"/"".
    No content is altered (no re-casing, no trimming of real text) -- that is
    the STANDARDIZATION layer's job and is deliberately not repeated here."""
    if value is None:
        return None
    try:
        import pandas as pd
        if value is pd.NA or (not isinstance(value, str) and pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value)
    return None if s.strip() == "" else s


def coerce_record(record: dict, fields: list) -> dict:
    return {f: coerce_for_dedupe(record.get(f)) for f in fields}
