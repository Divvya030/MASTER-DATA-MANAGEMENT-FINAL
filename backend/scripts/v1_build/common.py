"""Shared OFFLINE machinery for building the bundled V1 Dedupe models.

The running application never trains V1 -- it only loads the artifacts this tooling
writes. Rules: Dedupe 3.0.3 is the only ER engine (no LLM, no custom fuzzy matcher);
training data goes through the app's OWN ingestion -> mapping -> profiling ->
standardization modules (nothing is re-standardized); splits are entity-level so no
entity leaks across train/validation/test; evaluation runs through the PRODUCTION
`run_matching` / `run_clustering`; every label carries its provenance.
"""
from __future__ import annotations
import hashlib, io, itertools, math, os, random, shutil, sys
from collections import defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def onboard(domain, path, mapping, workdir: Path):
    """Run `path` through the real app pipeline (upload -> confirm mapping -> profile ->
    standardize) in an isolated scratch workspace.
    Returns (raw_df, {row_idx: {canonical_field: standardized value}})."""
    import app.db.session as session_mod
    import app.modules.ingestion as ing
    import app.main as main_mod
    from fastapi.testclient import TestClient
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    session_mod.DB_URL = f"sqlite:///{(workdir / 'mdm.db').resolve()}"
    session_mod._engine = None
    session_mod._SessionLocal = None
    ing.DATA_DIR = str((workdir / "data").resolve())
    main_mod.DATA_DIR_FOR_DEDUPE = ing.DATA_DIR
    os.environ["MDM_MODELS_DIR"] = str((workdir / "models").resolve())
    client = TestClient(main_mod.app)
    with open(path, "rb") as f:
        r = client.post("/api/datasets/upload", data={"domain": domain},
                        files={"file": (Path(path).name, f, "application/octet-stream")})
    assert r.status_code == 200, r.text
    ds_id = r.json()["dataset_id"]
    body = {"mappings": [{"canonical_field": k, "source_column": v} for k, v in mapping.items()]}
    assert client.post(f"/api/datasets/{ds_id}/mapping", json=body).status_code == 200
    assert client.post(f"/api/datasets/{ds_id}/profile").status_code == 200
    assert client.post(f"/api/datasets/{ds_id}/standardize").status_code == 200
    from app.db.session import get_session_factory
    from app.modules.dedupe_service.record_loading import records_from_standardized
    db = get_session_factory()()
    try:
        records = records_from_standardized(db, ds_id, list(mapping.keys()))
    finally:
        db.close()
    return ing.load_raw(ds_id), records


def bucket(key, seed, fracs=(0.6, 0.2, 0.2)):
    h = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < fracs[0]:
        return "train"
    return "validation" if h < fracs[0] + fracs[1] else "test"


def mask_fields(rec, fields, rng, p_each):
    out = dict(rec)
    for f in fields:
        if p_each is None or rng.random() < p_each:
            out[f] = None
    return out


def train_dedupe(variables, labelled_match, labelled_distinct, background_records, seed, recall,
                 sample_size=3000, max_sampled_records=500, index_predicates=False):
    """Bulk-labelled Dedupe training. Labelled pairs go in through dedupe's documented
    `training_file` route so every labelled record is indexed up front (no prune/retry)."""
    import dedupe
    from dedupe import serializer
    from app.modules.dedupe_service.training_limits import bounded_sample_records
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    data = {f"B:{k}": v for k, v in background_records.items()}
    for i, (a, b) in enumerate(labelled_match):
        data[f"M:{i}:a"], data[f"M:{i}:b"] = a, b
    for i, (a, b) in enumerate(labelled_distinct):
        data[f"D:{i}:a"], data[f"D:{i}:b"] = a, b
    buf = io.StringIO()
    serializer.write_training({"match": list(labelled_match), "distinct": list(labelled_distinct)}, buf)
    buf.seek(0)
    deduper = dedupe.Dedupe(variables)
    with bounded_sample_records(max_sampled_records):
        deduper.prepare_training(data, training_file=buf, sample_size=sample_size)
        deduper.train(recall=recall, index_predicates=index_predicates)
    return deduper


def settings_bytes_of(d):
    b = io.BytesIO(); d.write_settings(b); return b.getvalue()


def training_bytes_of(d):
    b = io.StringIO(); d.write_training(b); return b.getvalue().encode("utf-8")


def confusion(tp, fp, fn, tn=None):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    out = {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
           "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
    if tn is not None:
        out["true_negatives"] = tn
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def pairwise_end_to_end(settings, records, truth, threshold=0.5):
    """Production-path evaluation: candidate generation + scoring via `run_matching`; true
    pairs the blocking never produced count as false negatives."""
    from app.modules.dedupe_service.matching_service import run_matching
    res = run_matching(settings, records, threshold=threshold, review_threshold=max(0.0, threshold - 0.15))
    assert res.status == "COMPLETED", res.error
    by = defaultdict(list)
    for rid, e in truth.items():
        by[e].append(rid)
    true_pairs = {pr for ids in by.values() for pr in itertools.combinations(sorted(ids), 2)}
    cand = {tuple(sorted((p["record_id_a"], p["record_id_b"]))): p["score"] for p in res.pairs}
    return res, true_pairs, cand


def threshold_grid():
    return [round(x / 100, 2) for x in range(30, 100, 5)] + [0.97, 0.99]


def score_grid_metrics(true_pairs, cand, grid):
    out = {}
    cand_true = sum(1 for k in cand if k in true_pairs)
    for t in grid:
        sel = [k for k, s in cand.items() if s >= t]
        tp = sum(1 for k in sel if k in true_pairs)
        out[t] = confusion(tp, len(sel) - tp, len(true_pairs) - tp)
    return out, cand_true


def bcubed(clusters, truth):
    size = defaultdict(int)
    for r, e in truth.items():
        size[e] += 1
    ps = rs = 0.0; n = 0
    for c in clusters:
        cnt = defaultdict(int)
        for r in c:
            cnt[truth[r]] += 1
        for r in c:
            ps += cnt[truth[r]] / len(c); rs += cnt[truth[r]] / size[truth[r]]; n += 1
    p, r = ps / n, rs / n
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def cluster_metrics(res, records, truth, threshold):
    from app.modules.dedupe_service.clustering_service import run_clustering
    cr = run_clustering(res.raw_scores, sorted(records), threshold=threshold)
    assert cr.status == "COMPLETED", cr.error
    cl = [c["record_ids"] for c in cr.clusters]
    pred = {pr for c in cl for pr in itertools.combinations(sorted(c), 2)}
    by = defaultdict(list)
    for r, e in truth.items():
        by[e].append(r)
    tp_set = {pr for ids in by.values() for pr in itertools.combinations(sorted(ids), 2)}
    tp = len(pred & tp_set)
    return {"cluster_count": cr.cluster_count, "true_entity_count": len(by), "singleton_count": cr.singleton_count,
            "largest_cluster_size": cr.largest_cluster_size,
            "pairwise": confusion(tp, len(pred) - tp, len(tp_set) - tp), "bcubed": bcubed(cl, truth),
            "over_merged_clusters": sum(1 for c in cl if len({truth[r] for r in c}) > 1)}
