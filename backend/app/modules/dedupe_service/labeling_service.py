"""
Candidate-pair generation for HUMAN labelling. The steward is never asked to
label random pairs out of millions: candidates come from Dedupe itself.

BOOTSTRAP (no model yet): Dedupe's own active learner. prepare_training()
builds a bounded sample of candidate pairs from its learned/learnable blocking
predicates (most from blocked = plausibly-similar records, some random), and
uncertain_pairs() returns the pair the learner is currently most curious
about. This is Dedupe's documented interactive-labelling mechanism.

MODEL-ASSISTED (a trained model exists): StaticDedupe.pairs() (its learned
blocking) streamed in bounded CHUNKS, each chunk scored immediately, and
selection maintained incrementally with fixed-size heaps (one per band):
  MODEL_UNCERTAIN       scores closest to the decision threshold
  MODEL_STRONG          highest-scoring pairs (likely matches to confirm)
  MODEL_HARD_NEGATIVE   the highest scores BELOW the uncertain band
                        (look similar, probably distinct -- the costly errors)

MEMORY SAFETY: `dedupe`'s `score()` consumes its pairs iterable entirely and
returns one array for all pairs, and `list(pairs)` would materialise every
blocked pair (up to ~50M pairs for 10k records). We therefore never hold more
than CHUNK_SIZE pair objects in memory at once; only the best ~n candidates
per band (kept in fixed-size heaps) survive. Peak memory is
O(CHUNK_SIZE + n), independent of dataset size.

IDENTIFIER evidence (same / different non-null identifier values) is attached
as a HINT for the steward. It never sets a label.

No custom fuzzy matching is written here; all similarity/blocking comes from
dedupe. The amount of labelling needed is dataset-dependent and is not
claimed anywhere.
"""
from __future__ import annotations
import heapq
import io
import itertools
import random

from app.modules.dedupe_service.field_definitions import (
    ERFieldRoles, build_dedupe_variables, coerce_record, coerce_for_dedupe,
)

# Streaming chunk size for MODEL-assisted candidate generation. Sized so a
# chunk of pair objects (each holding two small record dicts) stays in the
# low tens of MB even at 10k+ records, while amortising the per-chunk numpy
# overhead of dedupe's scorer. Neither this nor any other bound below grows
# with dataset size.
MODEL_PAIR_CHUNK_SIZE = 20_000
# Upper bound on blocked pairs examined per request; with 15-200 candidates
# requested this only triggers on pathologically permissive blocking, and
# guarantees generation time/memory stay bounded.
MODEL_PAIR_SCAN_LIMIT = 2_000_000

# Band quotas (of requested n): half uncertain, a fifth strong, rest hard-negatives.
_UNCERTAIN_QUOTA = lambda n: max(1, n // 2)        # noqa: E731
_STRONG_QUOTA = lambda n: max(1, n // 5)           # noqa: E731


def identifier_evidence(rec_a: dict, rec_b: dict, identifier_fields: list) -> dict:
    ev = {}
    for f in identifier_fields:
        a, b = coerce_for_dedupe(rec_a.get(f)), coerce_for_dedupe(rec_b.get(f))
        if a is None or b is None:
            ev[f] = "UNKNOWN"
        else:
            ev[f] = "SAME" if a.strip().lower() == b.strip().lower() else "DIFFERENT"
    return ev


def _key(rec: dict, fields: list):
    return tuple((f, rec.get(f)) for f in fields)


def _index_by_value(data: dict, fields: list) -> dict:
    idx: dict = {}
    for rid in sorted(data, key=str):
        idx.setdefault(_key(data[rid], fields), []).append(rid)
    return idx


def _resolve_ids(pair, idx, fields, used: set):
    """Map a pair of record dicts back to two DISTINCT record ids."""
    ka, kb = _key(pair[0], fields), _key(pair[1], fields)
    ids_a, ids_b = idx.get(ka, []), idx.get(kb, [])
    for a in ids_a:
        for b in ids_b:
            if a != b and (min(str(a), str(b)), max(str(a), str(b))) not in used:
                return a, b
    return None


def _band_key(a: str, b: str) -> tuple:
    return (min(a, b), max(a, b))


def _process_scored_chunk(static, chunk_pairs, threshold, used, heaps, seen):
    """Score one bounded chunk of (record_a, record_b) pairs and fold results
    into the fixed-size band heaps. Returns number of NEW (unused) pairs seen."""
    if not chunk_pairs:
        return 0
    scored = static.score(chunk_pairs)
    new_pairs = 0
    for r in scored:
        a, b = str(r["pairs"][0]), str(r["pairs"][1])
        k = _band_key(a, b)
        if k in used or k in seen:
            continue
        seen.add(k)
        new_pairs += 1
        sc = float(r["score"])
        distance = abs(sc - threshold)
        # uncertain: keep the n smallest |score - threshold| (evict farthest)
        if len(heaps["uncertain"]) < heaps["cap"][0]:
            heapq.heappush(heaps["uncertain"], (distance, k, sc))
        elif distance < heaps["uncertain"][0][0]:
            heapq.heapreplace(heaps["uncertain"], (distance, k, sc))
        # strong: keep the n largest scores -> min-heap on score
        if len(heaps["strong"]) < heaps["cap"][1]:
            heapq.heappush(heaps["strong"], (sc, k, sc))
        elif sc > heaps["strong"][0][0]:
            heapq.heapreplace(heaps["strong"], (sc, k, sc))
        # hard negatives: keep the n largest scores BELOW (threshold - 0.10)
        if sc < threshold - 0.10:
            if len(heaps["hard_negative"]) < heaps["cap"][2]:
                heapq.heappush(heaps["hard_negative"], (sc, k, sc))
            elif sc > heaps["hard_negative"][0][0]:
                heapq.heapreplace(heaps["hard_negative"], (sc, k, sc))
    return new_pairs


def _model_assisted_candidates(static, data: dict, n: int, threshold: float,
                               used: set) -> list:
    """Stream static.pairs() in bounded chunks; select per-band best via
    fixed-size heaps. Never materialises the full blocked-pair collection."""
    str_data = {str(k): v for k, v in data.items()}
    heaps = {
        "uncertain": [],       # max-heap by distance (store negated)
        "strong": [],          # min-heap by score
        "hard_negative": [],   # min-heap by score
        "cap": (_UNCERTAIN_QUOTA(n), _STRONG_QUOTA(n), max(1, n)),
    }
    seen: set = set()
    scanned = 0
    pairs_iter = static.pairs(str_data)
    while scanned < MODEL_PAIR_SCAN_LIMIT:
        chunk = list(itertools.islice(pairs_iter, MODEL_PAIR_CHUNK_SIZE))
        if not chunk:
            break
        scanned += len(chunk)
        _process_scored_chunk(static, chunk, threshold, used, heaps, seen)

    uncertain = sorted(heaps["uncertain"], key=lambda t: t[0])
    strong = sorted(heaps["strong"], key=lambda t: -t[0])
    hard_neg = sorted(heaps["hard_negative"], key=lambda t: -t[0])
    quotas = [("MODEL_UNCERTAIN", uncertain, heaps["cap"][0]),
              ("MODEL_STRONG", strong, heaps["cap"][1]),
              ("MODEL_HARD_NEGATIVE", hard_neg, max(1, n - _UNCERTAIN_QUOTA(n) - _STRONG_QUOTA(n)))]
    out, taken = [], set(used)
    for strategy, source, quota in quotas:
        c = 0
        for _dist_or_score, k, sc in source:
            if k in taken:
                continue
            taken.add(k)
            a, b = k
            out.append({"record_id_a": a, "record_id_b": b,
                        "strategy": strategy, "model_score": round(sc, 6)})
            c += 1
            if c >= quota:
                break
    return out[:n]


def generate_label_candidates(records: dict, er_roles: ERFieldRoles, n: int, seed: int = 42,
                              existing_pair_keys: set | None = None, model_settings_bytes: bytes | None = None,
                              threshold: float = 0.5, max_records: int = 3000) -> list:
    """records: {row_id: {field: value}} STANDARDIZED, containing at least the
    MATCH fields (identifier fields optional, for hints).
    Returns [{record_id_a, record_id_b, strategy, model_score}] (ids as given).
    Bounded in memory regardless of dataset size."""
    variables = build_dedupe_variables(er_roles)
    fields = [v.field for v in variables]
    used = set()
    for k in (existing_pair_keys or set()):
        used.add(_band_key(str(k[0]), str(k[1])))

    rng = random.Random(seed)
    ids = sorted(records.keys(), key=str)
    if len(ids) > max_records:
        rng.shuffle(ids)
        ids = sorted(ids[:max_records], key=str)
    data = {rid: coerce_record(records[rid], fields) for rid in ids}

    from app.modules.dedupe_service.dedupe_factory import load_static, new_dedupe
    if model_settings_bytes is not None:
        static = load_static(model_settings_bytes)
        return _model_assisted_candidates(static, data, n, threshold, used)

    idx = _index_by_value(data, fields)
    out: list = []
    deduper = new_dedupe(variables)
    deduper.prepare_training(data, sample_size=1500)
    attempts = 0
    while len(out) < n and attempts < n * 6:
        attempts += 1
        try:
            pair = deduper.uncertain_pairs()[0]
        except (IndexError, AssertionError):
            break
        resolved = _resolve_ids(pair, idx, fields, used)
        if resolved is None:
            continue
        a, b = resolved
        used.add((min(str(a), str(b)), max(str(a), str(b))))
        out.append({"record_id_a": str(a), "record_id_b": str(b),
                    "strategy": "DEDUPE_ACTIVE_LEARNING", "model_score": None})
    return out
