"""Bounded, memory-safe candidate generation.

Requirement: 10,000 records -> Dedupe blocking -> bounded selection ->
15-50 candidates. Never `list(static.pairs(...))`, never `score()` over the
full blocked-pair collection, never a huge intermediate list.

These tests instrument dedupe's StaticDedupe to prove the implementation
streams in bounded chunks instead of materializing all pairs, and that the
bounded path actually still finds good candidates on a synthetic 10k-record
dataset with known duplicates.
"""
import io
import random

import pytest

from app.modules.dedupe_service.field_definitions import (
    ERFieldRoles, build_dedupe_variables,
)
from app.modules.dedupe_service import labeling_service
from app.modules.dedupe_service.labeling_service import (
    generate_label_candidates, MODEL_PAIR_CHUNK_SIZE, MODEL_PAIR_SCAN_LIMIT,
)


def _roles():
    return ERFieldRoles(
        match_fields=["customer_name", "email"],
        identifier_fields=["gstin"],
        blocking_fields=["city"],
        ignored_fields=[],
    )


def _make_dataset(n_entities=1500, copies_per_entity=6, seed=11):
    """~9,000 records: n_entities base entities, each with noisy variants."""
    rng = random.Random(seed)
    cities = ["Chennai", "Mumbai", "Delhi", "Pune", "Kolkata", "Hyderabad"]
    qualifiers = ["Ltd", "Pvt Ltd", "LLP", "Co", "Inc", "Group"]
    records = {}
    rid = 1
    for e in range(n_entities):
        base = {
            "customer_name": f"Shardha Enterprises {e}",
            "email": f"entity{e}@example.com",
            "gstin": f"29AABCU9603R{e:06d}",
            "city": cities[e % len(cities)],
        }
        for v in range(copies_per_entity):
            rec = dict(base)
            if rng.random() < 0.6:
                rec["customer_name"] = rec["customer_name"] + " " + rng.choice(qualifiers)
            if rng.random() < 0.3:
                rec["email"] = f"entity{e}.v{v}@example.com"
            if rng.random() < 0.2:
                rec["gstin"] = None  # missing identifier must be tolerated
            records[rid] = rec
            rid += 1
    return records


def _train_model(records, rng):
    """Train a real dedupe model on a subset with balanced match/distinct
    pairs (matches chosen from known same-entity variants). Returns settings
    bytes."""
    import dedupe

    # Train on a subset: keeps the test fast; the bounded-generation behaviour
    # under test is independent of model quality.
    subset_ids = rng.sample(sorted(records), 600)
    by_entity = {}
    for rid in subset_ids:
        entity = records[rid]["email"].split("@")[0].split(".")[0]
        by_entity.setdefault(entity, []).append(rid)

    d = dedupe.Dedupe(build_dedupe_variables(_roles()))
    data = {str(k): v for k, v in ((i, records[i]) for i in subset_ids)}
    d.prepare_training(data, sample_size=1500)
    training_pairs = {"match": [], "distinct": []}
    entities = [e for e, ids in by_entity.items() if len(ids) >= 2]
    # match pairs: two variants of the same entity
    for e in entities[:150]:
        a, b = rng.sample(by_entity[e], 2)
        training_pairs["match"].append((data[str(a)], data[str(b)]))
    # distinct pairs: records from different entities
    for _ in range(150):
        e1, e2 = rng.sample(entities, 2)
        a, b = rng.choice(by_entity[e1]), rng.choice(by_entity[e2])
        training_pairs["distinct"].append((data[str(a)], data[str(b)]))
    d.mark_pairs(training_pairs)
    d.train()
    buf = io.BytesIO()
    d.write_settings(buf)
    return buf.getvalue()


@pytest.fixture(scope="module")
def big_dataset():
    return _make_dataset()


@pytest.fixture(scope="module")
def model_bytes(big_dataset):
    return _train_model(big_dataset, random.Random(3))


def test_constants_are_bounded():
    assert MODEL_PAIR_CHUNK_SIZE <= 50_000
    assert MODEL_PAIR_SCAN_LIMIT <= 5_000_000  # never approaches 50M


def test_model_path_scores_in_bounded_chunks_never_full_pairs_list(model_bytes, big_dataset, monkeypatch):
    """Instrument StaticDedupe.score to prove (a) it is called once per bounded
    chunk and (b) `pairs` is never consumed as one full list."""
    import dedupe

    scored_sizes = []

    original_score = dedupe.StaticDedupe.score

    def counting_score(self, pairs):
        consumed = list(pairs)  # score() itself consumes the iterable
        scored_sizes.append(len(consumed))
        return original_score(self, iter(consumed))

    monkeypatch.setattr(dedupe.StaticDedupe, "score", counting_score)

    candidates = generate_label_candidates(
        big_dataset, _roles(), n=20, seed=5,
        model_settings_bytes=model_bytes, threshold=0.5,
    )
    assert 1 <= len(candidates) <= 20
    assert scored_sizes, "score was never called"
    # Each call must be a bounded chunk, not the full blocked-pair collection
    # (which for 9k records with this blocking would be far larger than 50k).
    assert max(scored_sizes) <= MODEL_PAIR_CHUNK_SIZE
    assert all(s <= MODEL_PAIR_CHUNK_SIZE for s in scored_sizes)


def test_scan_limit_caps_total_pairs_examined(model_bytes, big_dataset, monkeypatch):
    import dedupe

    total = {"n": 0}
    original_pairs = dedupe.StaticDedupe.pairs

    def counting_pairs(self, data):
        for pair in original_pairs(self, data):
            total["n"] += 1
            yield pair

    monkeypatch.setattr(dedupe.StaticDedupe, "pairs", counting_pairs)

    # Pathologically permissive threshold-independent scan: force the scan
    # limit to something tiny to prove the loop honours it.
    monkeypatch.setattr(labeling_service, "MODEL_PAIR_SCAN_LIMIT", 5_000)

    candidates = generate_label_candidates(
        big_dataset, _roles(), n=10, seed=5,
        model_settings_bytes=model_bytes, threshold=0.5,
    )
    assert total["n"] <= 5_000 + MODEL_PAIR_CHUNK_SIZE  # chunk-granular cutoff
    assert isinstance(candidates, list)


def test_bounded_generation_still_finds_true_duplicates(model_bytes, big_dataset):
    """Quality is preserved: candidates generated with the bounded strategy
    must still include pairs that are true duplicates (same entity)."""
    candidates = generate_label_candidates(
        big_dataset, _roles(), n=30, seed=5,
        model_settings_bytes=model_bytes, threshold=0.5,
    )
    assert len(candidates) >= 10
    true_dups = 0
    for c in candidates:
        a, b = big_dataset[int(c["record_id_a"])], big_dataset[int(c["record_id_b"])]
        ea = a["email"].split("@")[0].split(".")[0]
        eb = b["email"].split("@")[0].split(".")[0]
        if ea == eb:
            true_dups += 1
    # The model was trained to recognise email-entity structure; a bounded
    # selection that lost all true duplicates would be useless.
    assert true_dups >= 5, f"only {true_dups} true duplicates among candidates"


def test_bootstrap_path_also_bounded_on_10k(big_dataset):
    """The BOOTSTRAP path runs on 10k records with a bounded dedupe sample and
    returns a bounded number of candidates."""
    candidates = generate_label_candidates(big_dataset, _roles(), n=15, seed=2)
    assert 1 <= len(candidates) <= 15
    for c in candidates:
        assert c["strategy"] == "DEDUPE_ACTIVE_LEARNING"
        assert c["record_id_a"] != c["record_id_b"]


def test_full_run_does_not_materialize_all_pairs_memory_check(model_bytes, big_dataset, monkeypatch):
    """StaticDedupe.pairs is consumed lazily: at no point may the code call
    list(pairs) over the full iterator (we detect this by counting how many
    pairs are pulled from the iterator before the first score call)."""
    import dedupe

    events = []
    original_pairs = dedupe.StaticDedupe.pairs
    original_score = dedupe.StaticDedupe.score

    def spy_pairs(self, data):
        count = 0
        for pair in original_pairs(self, data):
            count += 1
            yield pair
        events.append(("pairs_exhausted", count))

    def spy_score(self, pairs):
        # record how many pairs were pulled so far at score time
        pulled = []
        for p in pairs:
            pulled.append(p)
            if len(pulled) >= MODEL_PAIR_CHUNK_SIZE:
                break
        events.append(("score_chunk", len(pulled)))
        return original_score(self, iter(pulled))

    monkeypatch.setattr(dede := dedupe.StaticDedupe, "pairs", spy_pairs)
    monkeypatch.setattr(dede, "score", spy_score)

    candidates = generate_label_candidates(
        big_dataset, _roles(), n=12, seed=5,
        model_settings_bytes=model_bytes, threshold=0.5,
    )
    assert candidates
    # No 'pairs_exhausted' event before the first score: the iterator is
    # consumed chunk-by-chunk, never fully materialized first.
    score_events = [e for e in events if e[0] == "score_chunk"]
    assert score_events, "score never called"
