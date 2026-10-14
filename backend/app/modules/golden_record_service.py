"""
Golden Record survivorship (Module 11).

Matching answers "which records are the same entity"; survivorship answers
"which VALUE represents that entity". Deterministic, configurable and
explainable -- no ML/LLM, no random choice.

For every field, candidate values (from the cluster's member records) pass
through an ORDERED chain of rules. After each rule, if exactly one candidate
remains the field is RESOLVED by that rule. If several candidates survive
every rule they are genuinely conflicting: the field is flagged
NEEDS_REVIEW (a provisional value is still proposed so the record is usable,
but it is never presented as settled). Every non-selected value is kept under
`conflicts` -- nothing is silently discarded or overwritten.

Rules:
  source_priority  keep values from the highest-priority source dataset(s)
                   (config: source_priority {dataset_id: int})
  recency          keep values from the most recent record, only where a valid
                   source timestamp exists (config: recency_field)
  validity         keep values that pass a structural check for the field
                   (email / phone / website) -- i.e. prefer the standardized,
                   well-formed representation
  frequency        keep the value most member records agree on (consensus),
                   only if strictly more common than the others
  completeness     keep the longest / most complete value, only if strictly
                   longer than the others. NOT applied by default to
                   identifier/contact-like fields, where "longer" does not
                   mean "more correct" (two different valid phones must be
                   reviewed, not silently resolved).
Rule order and per-field overrides are configuration (SurvivorshipConfig).
"""
from __future__ import annotations
import copy
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

MISSING_TOKENS = {None, "", "nan", "none", "null", "n/a", "na"}

RULES = ("source_priority", "recency", "validity", "frequency", "completeness")

DEFAULT_CONFIG = {
    "version": 1,
    "rule_order": ["source_priority", "recency", "validity", "frequency", "completeness"],
    # fields where "longer" is not "more correct": completeness is skipped
    "no_completeness_substrings": ["phone", "email", "website", "id", "identifier", "number",
                                   "duns", "tax", "registration", "gstin", "vat", "code"],
    "field_rules": {},            # {field: [rule, ...]} overrides the default chain for that field
    # SOURCE-SYSTEM PRIORITY (configuration, never a hardcoded assumption):
    # {source_system: int} -- higher wins. Any source system not listed gets
    # priority 0, so uploading customer.csv with SAP/SALESFORCE/ORACLE_FUSION/CRM
    # tomorrow keeps working without code changes. Legacy dataset_id keys are
    # still honored as a fallback for records whose source system is unknown.
    "source_priority": {},        # {"ORACLE": 1, "SAP": 2, ...}  higher wins
    "source_system_field": "source_system",  # canonical field carrying a record's source system
    # canonical field holding the source record's last-changed timestamp. Defaulted
    # to the canonical `last_updated` field: when it is mapped the recency rule is
    # live; when a dataset has no such column the rule simply has no effect.
    "recency_field": "last_updated",
    # How to read ambiguous numeric dates such as 05/01/2024: True = day first
    # (5 January), False = month first (1 May). ISO dates are never ambiguous.
    "timestamp_dayfirst": True,
}


def normalize_config(cfg: Optional[dict]) -> dict:
    out = copy.deepcopy(DEFAULT_CONFIG)
    if cfg:
        # A stored None/"" must not wipe the defaults for these two fields
        # (older saved configs have recency_field=None).
        out.update({k: v for k, v in cfg.items()
                    if k in DEFAULT_CONFIG
                    and not (k in ("recency_field", "source_system_field") and not v)})
    return out


def validate_config(cfg: dict) -> list:
    errs = []
    for r in cfg.get("rule_order", []):
        if r not in RULES:
            errs.append(f"unknown rule {r!r} in rule_order; allowed: {list(RULES)}")
    for f, rules in (cfg.get("field_rules") or {}).items():
        for r in rules:
            if r not in RULES:
                errs.append(f"unknown rule {r!r} for field {f!r}; allowed: {list(RULES)}")
    sp = cfg.get("source_priority") or {}
    if not all(isinstance(v, (int, float)) for v in sp.values()):
        errs.append("source_priority values must be numbers")
    if not isinstance(cfg.get("timestamp_dayfirst", True), bool):
        errs.append("timestamp_dayfirst must be true or false")
    return errs


def _is_present(value) -> bool:
    if value is None:
        return False
    try:
        if value != value:      # NaN
            return False
    except Exception:  # noqa: BLE001
        pass
    return str(value).strip().lower() not in MISSING_TOKENS


def _valid(field_name: str, v: str) -> bool:
    fn = field_name.lower()
    v = v.strip()
    if "email" in fn:
        return "@" in v and "." in v.split("@")[-1] and " " not in v
    if "phone" in fn:
        return sum(c.isdigit() for c in v) >= 7
    if "website" in fn:
        return "." in v and " " not in v
    return True


_TS_FORMATS_DAY_FIRST = (
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%Y",
    "%d.%m.%Y %H:%M:%S", "%d.%m.%Y",
)
_TS_FORMATS_MONTH_FIRST = (
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%m-%d-%Y %H:%M:%S", "%m-%d-%Y %H:%M", "%m-%d-%Y",
)
_TS_FORMATS_COMMON = (
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d", "%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%d %b %Y %H:%M:%S", "%d %b %Y",
    "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%d-%b-%y",
)


def _parse_ts(v, dayfirst: bool = True) -> Optional[dt.datetime]:
    """Parse a source timestamp. ISO-8601 first (``2026-06-02 11:03:18``,
    ``2026-06-02T11:03:18Z``), then common regional formats. Unparseable values
    return None (the record is then ignored by the recency rule, never guessed)."""
    if not _is_present(v):
        return None
    text = str(v).strip()
    try:
        t = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        return t.replace(tzinfo=None) if t.tzinfo else t
    except ValueError:
        pass
    order = (_TS_FORMATS_DAY_FIRST + _TS_FORMATS_MONTH_FIRST) if dayfirst else (_TS_FORMATS_MONTH_FIRST + _TS_FORMATS_DAY_FIRST)
    for fmt in order + _TS_FORMATS_COMMON:
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


@dataclass
class GoldenRecordResult:
    golden_values: dict
    lineage: dict
    conflicts: dict
    field_confidence: dict
    needs_review: list = field(default_factory=list)   # [{field, message, candidates}]
    rule_version: int = 1


def _first(occurrences):
    """The newest occurrence of a value (ties broken deterministically)."""
    return sorted(occurrences, key=lambda o: (o[3] is None, -(o[3].timestamp() if o[3] else 0), str(o[1]), str(o[2])))[0]


def _chain_for(field_name: str, cfg: dict) -> list:
    if field_name in (cfg.get("field_rules") or {}):
        return list(cfg["field_rules"][field_name])
    chain = list(cfg["rule_order"])
    if any(s in field_name.lower() for s in cfg.get("no_completeness_substrings", [])):
        chain = [r for r in chain if r != "completeness"]
    return chain


def select_golden_values(member_records: list, fields: list, source_priority: Optional[dict] = None,
                         config: Optional[dict] = None) -> GoldenRecordResult:
    """member_records: [{dataset_id, record_id, values: {field: value}}]"""
    cfg = normalize_config(config)
    if source_priority:
        cfg["source_priority"] = {**cfg["source_priority"], **source_priority}
    prio = cfg["source_priority"]
    recency_field = cfg.get("recency_field")
    dayfirst = bool(cfg.get("timestamp_dayfirst", True))
    now = dt.datetime.utcnow().isoformat()
    n_members = max(len(member_records), 1)

    golden, lineage, conflicts, confidence, needs_review = {}, {}, {}, {}, []

    for f in fields:
        # value -> {"occ": [(source_system, dataset_id, record_id, ts)], ...}
        cands: dict = {}
        for rec in member_records:
            raw = rec["values"].get(f)
            if not _is_present(raw):
                continue
            v = str(raw).strip()
            ts = _parse_ts(rec["values"].get(recency_field), dayfirst) if recency_field else None
            src_sys = (rec.get("source_system")
                       or rec["values"].get(cfg.get("source_system_field") or "source_system")
                       or None)
            cands.setdefault(v, []).append((src_sys, rec["dataset_id"], rec["record_id"], ts))

        if not cands:
            golden[f], lineage[f], conflicts[f], confidence[f] = None, None, [], 0.0
            continue

        remaining = dict(cands)
        decided_by = None
        trace = []
        for rule in _chain_for(f, cfg):
            if len(remaining) <= 1:
                break
            before = set(remaining)
            if rule == "source_priority":
                # Configuration lookup: source_system -> configured priority.
                # Fall back to a legacy dataset_id key only when the record's
                # source system is unknown. NO source name is hardcoded here.
                def _prio(o):
                    if o[0] is not None and o[0] in prio:
                        return prio[o[0]]
                    return prio.get(o[1], 0)
                best = {v: max(_prio(o) for o in occ) for v, occ in remaining.items()}
                top = max(best.values())
                if len({*best.values()}) > 1:
                    remaining = {v: occ for v, occ in remaining.items() if best[v] == top}
            elif rule == "recency":
                latest = {v: max((o[3] for o in occ if o[3] is not None), default=None) for v, occ in remaining.items()}
                dated = {v: t for v, t in latest.items() if t is not None}
                if dated and len(set(dated.values())) > 1:
                    top = max(dated.values())
                    remaining = {v: occ for v, occ in remaining.items() if dated.get(v) == top}
            elif rule == "validity":
                ok = {v: occ for v, occ in remaining.items() if _valid(f, v)}
                if ok and len(ok) < len(remaining):
                    remaining = ok
            elif rule == "frequency":
                counts = {v: len(occ) for v, occ in remaining.items()}
                top = max(counts.values())
                if list(counts.values()).count(top) == 1:
                    remaining = {v: occ for v, occ in remaining.items() if counts[v] == top}
            elif rule == "completeness":
                lens = {v: len(v) for v in remaining}
                top = max(lens.values())
                if list(lens.values()).count(top) == 1:
                    remaining = {v: occ for v, occ in remaining.items() if lens[v] == top}
            trace.append(f"{rule}:{'decided' if len(remaining) == 1 and len(before) > 1 else ('narrowed' if len(remaining) < len(before) else 'no_effect')}")
            if len(remaining) == 1:
                decided_by = rule
                break

        genuinely_conflicting = len(remaining) > 1
        if genuinely_conflicting:
            # deterministic provisional pick (value order) -- explicitly NOT a resolution
            chosen = sorted(remaining)[0]
            rule_applied = "UNRESOLVED_CONFLICT"
        else:
            chosen = next(iter(remaining))
            rule_applied = decided_by or "single_value"

        occ = sorted(cands[chosen], key=lambda o: (o[3] is None, -(o[3].timestamp() if o[3] else 0), str(o[1]), str(o[2])))
        n_distinct = len(cands)
        support = len(cands[chosen]) / n_members
        if n_distinct == 1:
            conf = round(min(1.0, 0.6 + 0.4 * support), 4)
        elif genuinely_conflicting:
            conf = round(0.3 * support, 4)
        else:
            conf = round(min(0.95, 0.4 + 0.5 * support), 4)
        golden[f] = chosen
        lineage[f] = {
            "value": chosen, "source_system": occ[0][0], "dataset_id": occ[0][1],
            "record_id": occ[0][2], "source_updated_at": occ[0][3].isoformat(sep=" ") if occ[0][3] else None,
            "support_count": len(cands[chosen]),
            "rule": rule_applied, "rule_trace": trace,
            "status": "NEEDS_REVIEW" if genuinely_conflicting else "AUTO_SELECTED" if decided_by else "RESOLVED",
            "reason": (f"rule={rule_applied}; support={len(cands[chosen])}/{n_members}; distinct_values={n_distinct}"
                       + (f"; source_system={occ[0][0]}" if occ[0][0] else "")),
            "confidence": conf, "selected_at": now,
        }
        confidence[f] = conf
        conflicts[f] = [
            {"value": v, "source_system": _first(o)[0], "dataset_id": _first(o)[1], "record_id": _first(o)[2],
             "source_updated_at": _first(o)[3].isoformat(sep=" ") if _first(o)[3] else None, "support": len(o)}
            for v, o in sorted(cands.items()) if v != chosen
        ]
        if genuinely_conflicting:
            needs_review.append({
                "field": f, "message": f"Conflicting {f} values require review.",
                "candidates": [{"value": v, "source_system": _first(o)[0], "dataset_id": _first(o)[1],
                                "record_id": _first(o)[2],
                                "source_updated_at": _first(o)[3].isoformat(sep=" ") if _first(o)[3] else None,
                                "support": len(o)}
                               for v, o in sorted(remaining.items())],
            })

    return GoldenRecordResult(golden, lineage, conflicts, confidence, needs_review, cfg.get("version", 1))
