"""
MODULE 6 -- ER Field / Match Configuration

Profiling RECOMMENDS a role; the user/configuration CONFIRMS it. This module
never auto-activates a role purely from the profiler's recommendation.

Role semantics:
  IDENTIFIER  -- independent ER role
  MATCH       -- independent ER role
  IGNORE      -- mutually exclusive with all active roles

NOTE: the former BLOCKING role was removed from the application-level ER
configuration. Candidate generation during matching uses the trained model's
own learned blocking predicates (see app/modules/dedupe_service), so a
user-picked blocking field never influenced matching anyway. Existing
database rows with role="BLOCKING" are tolerated (never crash) but are no
longer produced by the UI or emitted into any configuration object.
"""
from __future__ import annotations
from dataclasses import dataclass

VALID_ROLES = {"IDENTIFIER", "MATCH", "IGNORE"}
# Legacy roles that may still exist in persisted rows; accepted on read but
# never suggested, validated as active roles, or emitted into configuration.
LEGACY_ROLES = {"BLOCKING"}


@dataclass
class FieldRoleAssignment:
    canonical_field: str
    role: str
    source: str  # "SUGGESTED" or "USER_OVERRIDE"


class RoleValidationError(ValueError):
    pass


def suggest_roles(er_recommendation: str) -> list[str]:
    """Per-field role suggestions -- NOT auto-applied. IDENTIFIER and MATCH are
    mutually exclusive at the suggestion level because they derive from
    Module 3's single-valued `er_recommendation` category (a field is profiled
    as EITHER identifier-like OR match-worthy, never both, by that
    recommendation tree)."""
    if er_recommendation == "STRONG_IDENTIFIER_CANDIDATE":
        return ["IDENTIFIER"]
    if er_recommendation in ("GOOD_MATCH_FIELD", "SUPPORTING_MATCH_FIELD"):
        return ["MATCH"]
    return ["IGNORE"]


@dataclass
class RoleValidationWarning:
    canonical_field: str
    message: str


def validate_role_configuration(
    assignments: list[FieldRoleAssignment],
    field_evidence: dict[str, dict],  # canonical_field -> {"er_recommendation":..., "completeness_pct":...}
) -> list[RoleValidationWarning]:
    """
    Validates structural rules and produces WARNINGS (not hard failures) for
    questionable configurations, e.g. assigning IDENTIFIER to a field with no
    identifier evidence. The final decision always remains human-configurable --
    these are warnings, never automatic overrides.

    A field may carry more than one role. The one structural rule: IGNORE is
    mutually exclusive with every other role for that field.
    """
    warnings: list[RoleValidationWarning] = []
    seen_pairs: set[tuple[str, str]] = set()
    roles_by_field: dict[str, set[str]] = {}

    for a in assignments:
        if a.role in LEGACY_ROLES:
            raise RoleValidationError(
                f"Role {a.role!r} is no longer part of the ER configuration "
                f"(candidate generation uses the model's learned blocking predicates). "
                f"Valid roles: {sorted(VALID_ROLES)}")
        if a.role not in VALID_ROLES:
            raise RoleValidationError(f"Invalid role {a.role!r} for field {a.canonical_field!r}. "
                                       f"Valid roles: {sorted(VALID_ROLES)}")
        pair = (a.canonical_field, a.role)
        if pair in seen_pairs:
            raise RoleValidationError(
                f"Duplicate role assignment: {a.canonical_field!r} -> {a.role!r} was submitted more than once.")
        seen_pairs.add(pair)
        roles_by_field.setdefault(a.canonical_field, set()).add(a.role)

        evidence = field_evidence.get(a.canonical_field)
        if evidence is None:
            continue

        if a.role == "IDENTIFIER" and evidence.get("er_recommendation") != "STRONG_IDENTIFIER_CANDIDATE":
            warnings.append(RoleValidationWarning(
                a.canonical_field,
                f"Field assigned IDENTIFIER role but profiling evidence suggests "
                f"'{evidence.get('er_recommendation')}', not STRONG_IDENTIFIER_CANDIDATE. "
                f"Note that STRONG_IDENTIFIER_CANDIDATE itself means statistically "
                f"identifier-like based on observed data, not a confirmed business "
                f"identifier -- confirm this assignment reflects real business knowledge."))

        if a.role in ("MATCH", "IDENTIFIER") and evidence.get("completeness_pct", 100) < 50:
            warnings.append(RoleValidationWarning(
                a.canonical_field,
                f"Field has low completeness ({evidence.get('completeness_pct')}%) for an active ER role."))

    # Structural rule: IGNORE is mutually exclusive with every other role for the
    # same field -- a field cannot be simultaneously ignored and actively used.
    for field_name, roles in roles_by_field.items():
        if "IGNORE" in roles and len(roles) > 1:
            other = sorted(roles - {"IGNORE"})
            raise RoleValidationError(
                f"Field {field_name!r} cannot be assigned IGNORE together with other roles {other}. "
                f"IGNORE must be the field's only role.")

    return warnings


def build_final_configuration(assignments: list[FieldRoleAssignment]) -> dict[str, list[str]]:
    """A field with multiple role assignments naturally appears in multiple
    buckets below -- one list membership per assignment row, never collapsed
    to a single role. blocking_fields is kept as an (always-empty) key so
    existing consumers of the hand-off configuration keep working."""
    result = {"identifier_fields": [], "match_fields": [], "blocking_fields": [], "ignored_fields": []}
    role_to_key = {
        "IDENTIFIER": "identifier_fields",
        "MATCH": "match_fields",
        "IGNORE": "ignored_fields",
    }
    for a in assignments:
        key = role_to_key.get(a.role)
        if key:
            result[key].append(a.canonical_field)
    return result
