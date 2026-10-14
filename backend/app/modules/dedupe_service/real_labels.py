"""
Real steward-label handling: entity-level (connected-component) splitting,
snapshot coercion, and metric computation. Pure functions -- no DB, no
FastAPI -- so they are unit-testable in isolation.

HOW THE REAL-LABEL SPLIT IS CONSTRUCTED (documented, not implied):
There is no ground-truth entity id for real, unlabeled data, and none is
invented. Instead the labelled pairs themselves define a graph: every record
is a node, and every labelled pair (MATCH *or* DISTINCT) is an edge. Each
connected component is one indivisible unit and is assigned to exactly ONE of
train/validation/test by a deterministic hash. Consequently no record ever
appears in two splits, and a record that was confirmed to be the same entity
as another (MATCH) always travels with it -- which is what prevents the
leakage the brief warns about. Components can be uneven in size with small
label sets, so validation/test may be empty when very few labels exist; that
is reported honestly rather than papered over.
"""
from __future__ import annotations
import hashlib

from app.modules.dedupe_service.field_definitions import coerce_record


def node_key(dataset_id, record_id) -> str:
    return f"{dataset_id or '-'}::{record_id}"


def _bucket(component_key: str, seed: int) -> str:
    h = int(hashlib.sha256(f"{seed}:{component_key}".encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else ("validation" if h < 85 else "test")


def split_real_labels(labels: list, seed: int = 42) -> dict:
    """labels: [{pair_id, dataset_id, record_id_a, record_id_b, label, ...}]
    (only MATCH / DISTINCT are considered; UNSURE and pending are excluded).
    Returns {"train": [...], "validation": [...], "test": [...]}."""
    usable = [l for l in labels if l.get("label") in ("MATCH", "DISTINCT")]
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for l in usable:
        union(node_key(l.get("dataset_id"), l["record_id_a"]), node_key(l.get("dataset_id"), l["record_id_b"]))

    out = {"train": [], "validation": [], "test": []}
    for l in usable:
        comp = find(node_key(l.get("dataset_id"), l["record_id_a"]))
        out[_bucket(comp, seed)].append(l)
    return out


def snapshot_pairs(labels: list, match_fields: list):
    """-> (match_pairs, distinct_pairs), each [(rec_a, rec_b)] with records
    restricted to the model's MATCH fields and coerced for Dedupe."""
    m, d = [], []
    for l in labels:
        pair = (coerce_record(l["values_a"], match_fields), coerce_record(l["values_b"], match_fields))
        (m if l["label"] == "MATCH" else d).append(pair)
    return m, d


def confusion_metrics(scores, labels, threshold: float) -> dict:
    tp = fp = fn = tn = 0
    for s, y in zip(scores, labels):
        p = 1 if s >= threshold else 0
        tp += p == 1 and y == 1
        fp += p == 1 and y == 0
        fn += p == 0 and y == 1
        tn += p == 0 and y == 0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "threshold": threshold, "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
        "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) else None,
    }


def best_f1_threshold(scores, labels, grid=None):
    """Threshold maximising F1 on the supplied labelled scores; ties broken
    toward the HIGHER threshold (fewer false merges -- the costlier MDM error)."""
    grid = grid or [round(x / 100, 2) for x in range(10, 96, 5)]
    best = None
    for t in grid:
        m = confusion_metrics(scores, labels, t)
        key = (m["f1"], t)
        if best is None or key > best[0]:
            best = (key, t, m)
    return best[1], best[2]


def score_labeled_pairs(scorer, pairs_with_labels: list, match_fields: list):
    """scorer: a dedupe Dedupe/StaticDedupe exposing .score(). pairs_with_labels:
    [(values_a, values_b, y)] -> (scores list, labels list)."""
    if not pairs_with_labels:
        return [], []
    pairs = [((f"a{i}", coerce_record(a, match_fields)), (f"b{i}", coerce_record(b, match_fields)))
             for i, (a, b, _) in enumerate(pairs_with_labels)]
    scored = scorer.score(pairs)
    # Key by the (unique) pair ids rather than trusting output order.
    by_ids = {(str(r["pairs"][0]), str(r["pairs"][1])): float(r["score"]) for r in scored}
    scores, ys = [], []
    for i, (_, _, y) in enumerate(pairs_with_labels):
        sc = by_ids.get((f"a{i}", f"b{i}"))
        if sc is None:            # a pair dedupe declined to score counts as score 0
            sc = 0.0
        scores.append(sc)
        ys.append(y)
    return scores, ys
