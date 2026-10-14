"""
Incremental matching (Module 8 continued / continuous incoming data).

Matches a batch of NEW records against the population of EXISTING entities,
using the domain's currently ACTIVE trained model -- never against every
historical raw record, and never by retraining.

Approach: build one "representative" pseudo-record per existing entity from
its current golden record (this is the whole point of having a golden
record -- it IS the canonical comparison point for an entity, not any one
of its raw members). Then run this through the SAME real Dedupe candidate
generation (learned blocking predicates) and scoring used by
matching_service.py, over the representatives + new records combined.
Dedupe's blocking is what keeps this sub-quadratic -- we are not doing a
manual nested loop over history.

Decision bands (configurable):
  score >= match_threshold             -> MATCH (attach to existing entity,
                                            or merge two new records into one
                                            new entity)
  ambiguous_threshold <= score < match_threshold -> AMBIGUOUS (do not decide;
                                            surface to Steward Review)
  score < ambiguous_threshold           -> DISTINCT (no relationship)
A brand-new record with no candidate pairs at or above ambiguous_threshold
against anything becomes a new entity outright.

NOTE: this module is physically named `incremental_matching_impl.py` (not
`_service.py`) because of a filesystem anomaly in the build environment that
made the originally-named file impossible to read/write via standard OS file
APIs. Content is otherwise identical; `incremental_matching_service.py` is
kept as a deprecated re-export shim for anything importing the old name.
"""
from __future__ import annotations
import io
from dataclasses import dataclass, field


ENTITY_PREFIX = "ENTITY::"   # marks a representative-record key as an existing entity, not a raw new record


@dataclass
class IncrementalMatchResult:
    status: str  # COMPLETED | FAILED
    # {new_record_id: {"decision": "MATCH", "entity_id": ..., "score": ...}}
    # {new_record_id: {"decision": "AMBIGUOUS", "entity_id": ..., "score": ...}}
    # {new_record_id: {"decision": "NEW_ENTITY"}}
    # {new_record_id: {"decision": "MATCHES_ANOTHER_NEW_RECORD", "other_record_id": ..., "score": ...}}
    decisions: dict = field(default_factory=dict)
    candidate_pair_count: int = 0
    error: str | None = None


def run_incremental_matching(
    settings_bytes: bytes,
    existing_entity_representatives: dict,   # {entity_id: {field: value}}
    new_records: dict,                        # {record_id: {field: value}}
    match_threshold: float = 0.5,
    ambiguous_threshold: float = 0.35,
) -> IncrementalMatchResult:
    from app.modules.dedupe_service.dedupe_factory import load_static

    try:
        static_deduper = load_static(settings_bytes)
    except Exception as e:  # noqa: BLE001
        return IncrementalMatchResult(status="FAILED", error=f"Failed to load active model: {type(e).__name__}: {e}")

    combined = {}
    for entity_id, values in existing_entity_representatives.items():
        combined[f"{ENTITY_PREFIX}{entity_id}"] = values
    for record_id, values in new_records.items():
        combined[str(record_id)] = values

    try:
        pairs = list(static_deduper.pairs(combined)) if len(combined) > 1 else []
        scores = static_deduper.score(pairs) if pairs else None
    except Exception as e:  # noqa: BLE001
        return IncrementalMatchResult(status="FAILED", error=f"Candidate generation/scoring failed: {type(e).__name__}: {e}")

    new_record_ids = set(str(r) for r in new_records.keys())
    # best candidate seen so far for each new record: (score, kind, other_id)
    best: dict = {rid: None for rid in new_record_ids}
    cand_counts: dict = {rid: 0 for rid in new_record_ids}   # candidate pairs Dedupe generated per new record

    if scores is not None:
        for row in scores:
            a, b = str(row["pairs"][0]), str(row["pairs"][1])
            score = float(row["score"])
            a_is_entity, b_is_entity = a.startswith(ENTITY_PREFIX), b.startswith(ENTITY_PREFIX)
            for x, is_ent in ((a, a_is_entity), (b, b_is_entity)):
                if not is_ent and x in cand_counts:
                    cand_counts[x] += 1

            if a_is_entity == b_is_entity:
                # entity<->entity pairs are irrelevant here; new<->new pairs
                # are handled below (candidate for merging into one new entity).
                if not a_is_entity and not b_is_entity:
                    for me, other in ((a, b), (b, a)):
                        cur = best.get(me)
                        if cur is None or score > cur[0]:
                            best[me] = (score, "NEW", other)
                continue

            new_id, entity_key = (a, b) if not a_is_entity else (b, a)
            entity_id = entity_key[len(ENTITY_PREFIX):]
            cur = best.get(new_id)
            # Existing-entity matches always take priority over new<->new
            # candidates at the same score band, since attaching to an
            # established entity is preferable to spawning a fresh one.
            if cur is None or score > cur[0] or (cur[1] == "NEW" and score >= cur[0]):
                best[new_id] = (score, "ENTITY", entity_id)

    decisions = {}
    for rid in new_record_ids:
        b = best.get(rid)
        _cc = cand_counts.get(rid, 0)
        if b is None:
            decisions[rid] = {"decision": "NEW_ENTITY"}
            continue
        score, kind, other = b
        if score >= match_threshold:
            if kind == "ENTITY":
                decisions[rid] = {"decision": "MATCH", "entity_id": other, "score": round(score, 6)}
            else:
                decisions[rid] = {"decision": "MATCHES_ANOTHER_NEW_RECORD", "other_record_id": other,
                                   "score": round(score, 6)}
        elif score >= ambiguous_threshold:
            decisions[rid] = {
                "decision": "AMBIGUOUS",
                "entity_id": other if kind == "ENTITY" else None,
                "other_record_id": other if kind == "NEW" else None,
                "score": round(score, 6),
            }
        else:
            decisions[rid] = {"decision": "NEW_ENTITY"}

    for rid, d in decisions.items():
        d["candidate_count"] = cand_counts.get(rid, 0)
    return IncrementalMatchResult(
        status="COMPLETED", decisions=decisions, candidate_pair_count=len(pairs),
    )
