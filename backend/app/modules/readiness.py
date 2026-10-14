"""
MODULE 7 -- Final Pre-Dedupe Review / ER Configuration

Produces an explainable (never fake-ML) readiness assessment and the clean,
deterministic configuration object the future Dedupe module will consume.
Stops firmly BEFORE any actual matching/clustering logic.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ReadinessCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class ReadinessReport:
    dataset_id: str
    domain: str
    row_count: int
    field_count: int
    mapped_field_count: int
    unmapped_field_count: int
    empty_field_count: int
    checks: list[ReadinessCheck]
    warnings: list[str]
    is_ready: bool
    # Canonical schema fields (e.g. 'email', 'phone') for which NO source column
    # was mapped at all -- distinct from unmapped_field_count above, which counts
    # unmapped SOURCE columns (legitimate business attributes outside the schema).
    unmapped_canonical_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "domain": self.domain,
            "row_count": self.row_count,
            "field_count": self.field_count,
            "mapped_field_count": self.mapped_field_count,
            "unmapped_field_count": self.unmapped_field_count,
            "empty_field_count": self.empty_field_count,
            "checks": [c.__dict__ for c in self.checks],
            "warnings": self.warnings,
            "is_ready": self.is_ready,
            "unmapped_canonical_fields": self.unmapped_canonical_fields,
        }


def assess_readiness(
    dataset_id: str,
    domain: str,
    row_count: int,
    total_source_columns: int,
    mapped_field_count: int,
    empty_field_count: int,
    stage_flags: dict[str, bool],  # e.g. {"mapped":True,"profiled":True,"standardized":True,"revalidated":True}
    er_config: dict[str, list[str]],  # identifier_fields/match_fields/blocking_fields/ignored_fields
    field_warnings: list[str],  # e.g. "DUNS completeness is low" collected from Module 6 validation / profiling
    date_ambiguity_unresolved: bool = False,
    unmapped_canonical_fields: list[str] | None = None,
) -> ReadinessReport:
    checks: list[ReadinessCheck] = []
    warnings: list[str] = list(field_warnings)

    checks.append(ReadinessCheck("Dataset ingested", row_count > 0,
                                  f"{row_count} rows ingested." if row_count > 0 else "No rows found."))
    checks.append(ReadinessCheck("Mapping confirmed", stage_flags.get("mapped", False),
                                  f"{mapped_field_count} canonical fields mapped."))
    checks.append(ReadinessCheck("Profiling complete", stage_flags.get("profiled", False), ""))
    checks.append(ReadinessCheck("Standardization complete", stage_flags.get("standardized", False), ""))
    checks.append(ReadinessCheck("Re-profile complete", stage_flags.get("revalidated", False), ""))

    has_match = len(er_config.get("match_fields", [])) > 0
    checks.append(ReadinessCheck("Match fields configured", has_match,
                                  f"{len(er_config.get('match_fields', []))} match field(s)."))

    if not er_config.get("identifier_fields"):
        warnings.append("No strong identifier configured -- matching will rely solely on MATCH fields.")
    if date_ambiguity_unresolved:
        warnings.append("Date interpretation is unresolved for one or more DATE_LIKE fields.")

    hard_fail = any(not c.passed for c in checks if c.name in
                     ("Dataset ingested", "Mapping confirmed", "Profiling complete",
                      "Standardization complete", "Re-profile complete"))

    # A dataset with zero MATCH fields is NOT a meaningful matching
    # configuration and must not reach READY_FOR_DEDUPE, even though every
    # pipeline stage completed successfully.
    is_ready = (not hard_fail) and has_match
    if not has_match:
        warnings.append(
            "No MATCH field configured -- at least one MATCH field is required for Dedupe readiness.")

    return ReadinessReport(
        dataset_id=dataset_id,
        domain=domain,
        row_count=row_count,
        field_count=total_source_columns,
        mapped_field_count=mapped_field_count,
        unmapped_field_count=max(total_source_columns - mapped_field_count, 0),
        empty_field_count=empty_field_count,
        checks=checks,
        warnings=warnings,
        is_ready=is_ready,
        unmapped_canonical_fields=unmapped_canonical_fields or [],
    )


def build_dedupe_ready_configuration(
    dataset_id: str,
    domain: str,
    er_config: dict[str, list[str]],
    standardization_stage: str = "STANDARDIZED",
) -> dict:
    """The clean, deterministic hand-off object for the future Dedupe module.
    NOTE: this function performs NO matching, training, or clustering -- it only
    packages configuration that a later, separate Dedupe module will consume."""
    return {
        "dataset_id": dataset_id,
        "domain": domain,
        "identifier_fields": list(er_config.get("identifier_fields", [])),
        "match_fields": list(er_config.get("match_fields", [])),
        "ignored_fields": list(er_config.get("ignored_fields", [])),
        "standardization_stage": standardization_stage,
    }
