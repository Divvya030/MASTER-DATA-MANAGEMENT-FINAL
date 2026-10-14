"""
Deterministic, reproducible training-data preparation for Dedupe.

Real customer/supplier datasets in this project have no ground-truth
duplicate labels, so MATCH examples are bootstrapped via SYNTHETIC
CORRUPTION of real records: each sampled base record becomes one "entity",
and 1-2 conservatively corrupted clones of it are generated. DISTINCT
examples come from genuinely different entities: some fully random, some
"hard negatives" (different entities whose primary MATCH field shares a
leading token, e.g. "ABC Industries" vs "ABC Trading Co" -- lexically
similar but not the same thing).

ENTITY-LEVEL SPLITTING: every record derived from the same base entity
(the original + all of its corrupted clones) is assigned to exactly ONE of
train/validation/test. Pairs are only ever generated between two records in
the SAME split. This is what prevents leakage -- a model could otherwise
"cheat" by memorizing a corrupted variant it already saw during training.

Everything here is domain-agnostic: it operates purely on whichever fields
the ACTIVE ER configuration marks as MATCH, never on a hardcoded field name.
"""
from __future__ import annotations
import hashlib
import random
from dataclasses import dataclass


@dataclass
class SplitDatasets:
    train: dict
    validation: dict
    test: dict
    train_match_pairs: list
    train_distinct_pairs: list
    validation_match_pairs: list
    validation_distinct_pairs: list
    test_match_pairs: list
    test_distinct_pairs: list
    entity_count: int
    record_count: int
    pair_counts: dict
    entity_members: dict
    entity_split: dict


# ---------------------------------------------------------------------------
# SYNTHETIC AUGMENTATION (label_source = SYNTHETIC).
#
# The existing STANDARDIZATION layer (app/modules/standardization.py) already
# normalizes whitespace, all-upper/all-lower case, phone/email/identifier
# formatting and missing tokens BEFORE Dedupe sees any data -- so corrupting
# case, spacing or separators here would teach the model about noise it will
# never see in production. Only variation that SURVIVES standardization is
# generated here:
#   * realistic typos (transposition / dropped letter / doubled letter)
#   * legal-form and abbreviation representation differences
#     ("Pvt Ltd" vs "Private Limited", "Street" vs "St") -- standardization
#     deliberately does NOT expand/contract these (no safe semantic rule)
#   * partial names (a descriptive token dropped)
#   * missing OPTIONAL values (a non-primary field absent in one source)
#   * multi-field corruption (two different fields perturbed at once)
# The equivalence tables are configuration (SyntheticConfig), not code, so
# domain knowledge lives in config and nothing here names a dataset/customer.
# ---------------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    # Groups of interchangeable surface forms (matched as whole tokens,
    # case-insensitively). Any member may be swapped for another member.
    legal_form_groups: tuple = (
        ("pvt ltd", "private limited"), ("ltd", "limited"), ("inc", "incorporated"),
        ("corp", "corporation"), ("co", "company"), ("pvt", "private"),
    )
    abbreviation_groups: tuple = (
        ("st", "street"), ("rd", "road"), ("ave", "avenue"), ("blvd", "boulevard"),
        ("dept", "department"), ("intl", "international"), ("mfg", "manufacturing"),
    )
    max_fields_multi: int = 2


DEFAULT_SYNTHETIC_CONFIG = SyntheticConfig()


def _swap_group_form(s: str, groups, rng: random.Random) -> str:
    """Replace one token (or two-token phrase) that belongs to an equivalence
    group with a different member of that group. Returns s unchanged if no
    group member is present."""
    words = s.split(" ")
    low = [w.lower().strip(".,") for w in words]
    candidates = []
    for group in groups:
        for form in group:
            form_words = form.split(" ")
            n = len(form_words)
            for i in range(len(words) - n + 1):
                if low[i:i + n] == form_words:
                    candidates.append((i, n, group, form))
    if not candidates:
        return s
    i, n, group, form = rng.choice(candidates)
    replacement = rng.choice([g for g in group if g != form])
    rep_words = [w.capitalize() for w in replacement.split(" ")]
    return " ".join(words[:i] + rep_words + words[i + n:])


def _corrupt_typo(s: str, rng: random.Random, cfg=None) -> str:
    if len(s) < 5:
        return s
    i = rng.randint(1, len(s) - 3)   # never the first character (people rarely mistype it)
    chars = list(s)
    kind = rng.choice(["transpose", "drop", "double"])
    if chars[i] == " " or chars[i + 1] == " ":
        return s
    if kind == "transpose":
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif kind == "drop":
        del chars[i]
    else:
        chars.insert(i, chars[i])
    return "".join(chars)


def _corrupt_legal_form(s: str, rng: random.Random, cfg=None) -> str:
    cfg = cfg or DEFAULT_SYNTHETIC_CONFIG
    out = _swap_group_form(s, cfg.legal_form_groups, rng)
    if out != s:
        return out
    words = s.split(" ")
    # otherwise: drop a trailing legal suffix (source that omits it)
    for group in cfg.legal_form_groups:
        for form in group:
            fw = form.split(" ")
            if [w.lower().strip(".,") for w in words[-len(fw):]] == fw and len(words) > len(fw):
                return " ".join(words[:-len(fw)])
    return s


def _corrupt_abbreviation(s: str, rng: random.Random, cfg=None) -> str:
    cfg = cfg or DEFAULT_SYNTHETIC_CONFIG
    return _swap_group_form(s, cfg.abbreviation_groups, rng)


def _corrupt_token_drop(s: str, rng: random.Random, cfg=None) -> str:
    words = s.split(" ")
    if len(words) < 3:
        return s
    i = rng.randint(1, len(words) - 1)     # keep the leading token (the most identifying one)
    return " ".join(words[:i] + words[i + 1:])


_CORRUPTIONS = [_corrupt_typo, _corrupt_legal_form, _corrupt_abbreviation, _corrupt_token_drop]


def _apply_one(value: str, rng: random.Random, cfg) -> str:
    """Try corruption kinds in random order until one actually changes the
    value (some kinds don't apply to some strings)."""
    fns = list(_CORRUPTIONS)
    rng.shuffle(fns)
    for fn in fns:
        out = fn(value, rng, cfg)
        if out != value and out.strip():
            return out
    return value


def corrupt_record(record: dict, rng: random.Random, cfg: SyntheticConfig | None = None) -> dict:
    """Produces ONE plausible residual-variation clone of a STANDARDIZED
    record. Modes: single-field corruption (most common), multi-field
    corruption, or a missing optional value. Never touches non-string
    values, never invents a field, never blanks the primary (first) field."""
    cfg = cfg or DEFAULT_SYNTHETIC_CONFIG
    out = dict(record)
    string_fields = [k for k, v in record.items() if isinstance(v, str) and v.strip()]
    if not string_fields:
        return out
    roll = rng.random()
    if roll < 0.20 and len(string_fields) >= 2:
        # missing OPTIONAL value: blank a non-primary field
        optional = [k for k in string_fields if k != list(record.keys())[0]]
        if optional:
            out[rng.choice(optional)] = None
            return out
    if roll < 0.45 and len(string_fields) >= 2:
        # multi-field corruption
        for f in rng.sample(string_fields, min(cfg.max_fields_multi, len(string_fields))):
            out[f] = _apply_one(out[f], rng, cfg)
        return out
    f = rng.choice(string_fields)
    out[f] = _apply_one(out[f], rng, cfg)
    return out


# ---------------------------------------------------------------------------
# Entity-level split
# ---------------------------------------------------------------------------

def _split_bucket(entity_key: str, seed: int) -> str:
    """Deterministic hash-based split assignment -- same entity always lands
    in the same split for a given seed, with no shared mutable state."""
    h = hashlib.sha256(f"{seed}:{entity_key}".encode()).hexdigest()
    bucket = int(h[:8], 16) % 100
    if bucket < 70:
        return "train"
    elif bucket < 85:
        return "validation"
    return "test"


def _first_token(s: str) -> str:
    return s.strip().split(" ")[0].lower() if s and s.strip() else ""


def prepare_training_data(
    base_records: dict,
    match_fields: list,
    clones_per_entity: int = 2,
    max_entities: int = 400,
    hard_negative_ratio: float = 0.4,
    seed: int = 42,
) -> SplitDatasets:
    """
    base_records: {row_id: record_dict} from the STANDARDIZED layer, already
                  restricted to MATCH-role fields only.
    Returns entity-level train/validation/test splits with generated MATCH
    and DISTINCT pairs, ready to feed into Dedupe's prepare_training/
    mark_pairs. Deterministic and reproducible for a fixed seed.
    """
    rng = random.Random(seed)

    base_ids = sorted(base_records.keys())
    if len(base_ids) > max_entities:
        rng.shuffle(base_ids)
        base_ids = sorted(base_ids[:max_entities])

    next_id = (max(base_records.keys()) + 1) if base_records else 1
    all_records = {}
    entity_members = {}  # entity_key (=base_id) -> [record ids]
    entity_split = {}

    for base_id in base_ids:
        base_record = base_records[base_id]
        split = _split_bucket(str(base_id), seed)
        entity_split[base_id] = split
        members = [base_id]
        all_records[base_id] = base_record
        for _ in range(clones_per_entity):
            clone = corrupt_record(base_record, rng)
            clone_id = next_id
            next_id += 1
            all_records[clone_id] = clone
            members.append(clone_id)
        entity_members[base_id] = members

    def _records_for_split(split_name):
        return {rid: all_records[rid] for eid, members in entity_members.items()
                if entity_split[eid] == split_name for rid in members}

    def _match_pairs_for_split(split_name, with_ids):
        pairs = []
        for eid, members in entity_members.items():
            if entity_split[eid] != split_name or len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    if with_ids:
                        pairs.append(((a, all_records[a]), (b, all_records[b])))
                    else:
                        pairs.append((all_records[a], all_records[b]))
        return pairs

    def _distinct_pairs_for_split(split_name, with_ids):
        entities_in_split = [eid for eid, s in entity_split.items() if s == split_name]
        primary_field = match_fields[0] if match_fields else None
        pairs = []
        n_target = max(len(entities_in_split), 5)

        if primary_field:
            by_first_token = {}
            for eid in entities_in_split:
                tok = _first_token(all_records[eid].get(primary_field, ""))
                if tok:
                    by_first_token.setdefault(tok, []).append(eid)
            hard_candidates = []
            for tok, eids in by_first_token.items():
                if len(eids) > 1:
                    for i in range(len(eids)):
                        for j in range(i + 1, len(eids)):
                            hard_candidates.append((eids[i], eids[j]))
            rng.shuffle(hard_candidates)
            # Coverage, not just a small fixed slice: with many entities sharing a
            # first token (e.g. a dozen companies all starting "Nova ...", each
            # genuinely distinct), a tiny fixed sample of hard_candidates left the
            # classifier untrained on most of the confusable pairs, exactly the
            # "same-looking but distinct" case hard negatives exist for. Take at
            # least half of what was found (capped, so this stays bounded on large
            # datasets), or all of it if the pool is small.
            n_hard = int(n_target * hard_negative_ratio)
            for eid_a, eid_b in hard_candidates[:n_hard]:
                a, b = entity_members[eid_a][0], entity_members[eid_b][0]
                pairs.append(((a, all_records[a]), (b, all_records[b])) if with_ids
                             else (all_records[a], all_records[b]))

        attempts = 0
        while len(pairs) < n_target and len(entities_in_split) >= 2 and attempts < n_target * 20:
            attempts += 1
            eid_a, eid_b = rng.sample(entities_in_split, 2)
            a, b = entity_members[eid_a][0], entity_members[eid_b][0]
            pairs.append(((a, all_records[a]), (b, all_records[b])) if with_ids
                         else (all_records[a], all_records[b]))
        return pairs

    train_records = _records_for_split("train")
    validation_records = _records_for_split("validation")
    test_records = _records_for_split("test")

    train_match = _match_pairs_for_split("train", with_ids=False)
    train_distinct = _distinct_pairs_for_split("train", with_ids=False)
    val_match = _match_pairs_for_split("validation", with_ids=True)
    val_distinct = _distinct_pairs_for_split("validation", with_ids=True)
    test_match = _match_pairs_for_split("test", with_ids=True)
    test_distinct = _distinct_pairs_for_split("test", with_ids=True)

    return SplitDatasets(
        train=train_records, validation=validation_records, test=test_records,
        train_match_pairs=train_match, train_distinct_pairs=train_distinct,
        validation_match_pairs=val_match, validation_distinct_pairs=val_distinct,
        test_match_pairs=test_match, test_distinct_pairs=test_distinct,
        entity_count=len(base_ids), record_count=len(all_records),
        pair_counts={
            "train": {"match": len(train_match), "distinct": len(train_distinct)},
            "validation": {"match": len(val_match), "distinct": len(val_distinct)},
            "test": {"match": len(test_match), "distinct": len(test_distinct)},
        },
        entity_members=entity_members, entity_split=entity_split,
    )
