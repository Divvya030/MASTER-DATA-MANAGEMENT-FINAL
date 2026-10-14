"""Real-data end-to-end verification for one domain.

Drives the FULL product workflow against a real file:
  upload -> mapping suggestions -> confirm -> profile -> standardize -> reprofile
  -> quality -> er-config -> readiness (is_ready) -> bootstrap candidate
  generation -> steward labels (MATCH/DISTINCT/UNSURE) -> training (real labels)
  -> validate -> activate -> MATCH (bounded memory) -> cluster -> golden records
  -> monitoring summary.

Usage:
  .venv/Scripts/python.exe scripts/verify_real_data.py customer \
      "C:/Users/divycons529/Documents/Datasets/customer.csv" customer_name,email,tax_identifier
  .venv/Scripts/python.exe scripts/verify_real_data.py supplier \
      "C:/Users/divycons529/Documents/Datasets/supplier.xlsx" supplier_name,duns,tax_identifier
"""
import json
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402


def main(domain: str, path: str, match_fields_csv: str, n_candidates: int = 80):
    import app.db.session as session_mod
    import app.modules.ingestion as ing
    import app.main as main_mod

    workdir = Path(f"../_verify_{domain}")
    if workdir.exists():
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    session_mod.DB_URL = f"sqlite:///{(workdir / 'mdm.db').resolve()}"
    session_mod._engine = None
    session_mod._SessionLocal = None
    ing.DATA_DIR = str((workdir / "data").resolve())
    main_mod.DATA_DIR_FOR_DEDUPE = str((workdir / "data").resolve())
    import os
    os.environ["MDM_MODELS_DIR"] = str((workdir / "models").resolve())

    client = TestClient(main_mod.app)
    t0 = time.time()
    out = {"domain": domain, "file": path}

    def step(name, fn):
        t = time.time()
        r = fn()
        dt = time.time() - t
        print(f"  [{dt:7.2f}s] {name}")
        return r, dt

    print(f"== {domain}: {path} ==")
    # 1. upload
    with open(path, "rb") as f:
        r, _ = step("upload", lambda: client.post(
            "/api/datasets/upload",
            data={"domain": domain},
            files={"file": (Path(path).name, f, "application/octet-stream")}))
    assert r.status_code == 200, r.text
    body = r.json()
    ds_id = body["dataset_id"]
    out["rows"] = body.get("row_count")

    # 2. mapping
    r, _ = step("mapping suggestions", lambda: client.get(f"/api/datasets/{ds_id}/mapping/suggestions"))
    sugg = r.json()
    mappings = [{"canonical_field": cf, "source_column": c[0]["source_column"]}
                for cf, c in sugg["suggestions"].items() if c]
    out["mapped_fields"] = len(mappings)
    r, _ = step("confirm mapping", lambda: client.post(f"/api/datasets/{ds_id}/mapping", json={"mappings": mappings}))
    assert r.status_code == 200, r.text

    # 3. profile + standardize + reprofile + quality
    r, _ = step("profile", lambda: client.post(f"/api/datasets/{ds_id}/profile"))
    assert r.status_code == 200, r.text
    r, _ = step("standardize", lambda: client.post(f"/api/datasets/{ds_id}/standardize"))
    assert r.status_code == 200, r.text
    r, _ = step("reprofile", lambda: client.post(f"/api/datasets/{ds_id}/reprofile"))
    assert r.status_code == 200, r.text
    r, _ = step("quality comparison", lambda: client.get(f"/api/datasets/{ds_id}/quality-comparison"))
    assert r.status_code == 200, r.text

    # 4. ER config + readiness (MATCH on chosen fields, IDENTIFIER on id-like fields)
    id_like = [m_["canonical_field"] for m_ in mappings
               if (m_["canonical_field"].endswith("_id") or "duns" in m_["canonical_field"]
                   or "tax" in m_["canonical_field"]) and m_["canonical_field"] not in match_fields_csv.split(",")]
    assignments = [{"canonical_field": f, "role": "MATCH"} for f in match_fields_csv.split(",")]
    assignments += [{"canonical_field": f, "role": "IDENTIFIER"} for f in id_like]
    r, _ = step("er-config", lambda: client.post(f"/api/datasets/{ds_id}/er-config", json={"assignments": assignments}))
    assert r.status_code == 200, r.text
    r, _ = step("readiness", lambda: client.get(f"/api/datasets/{ds_id}/readiness"))
    assert r.status_code == 200, r.text
    ready = r.json()["readiness"]
    out["is_ready"] = ready["is_ready"]
    out["failed_checks"] = [c["name"] for c in ready["checks"] if not c["passed"]]
    if not ready["is_ready"]:
        print(f"  !! NOT READY: {out['failed_checks']}")
        print(json.dumps(out, indent=2))
        return out

    # 5. labeling: bootstrap candidates, steward labels
    tracemalloc.start()
    r, _ = step("generate bootstrap candidates", lambda: client.post(
        f"/api/datasets/{ds_id}/er/labeling/candidates", json={"n": n_candidates, "strategy": "BOOTSTRAP"}))
    assert r.status_code == 200, r.text
    created = r.json()["created"]
    out["bootstrap_candidates"] = created
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out["bootstrap_peak_mb"] = round(peak / 1e6, 1)

    # steward labels from identifier agreement + name similarity: automated
    # but deterministic; 60/40 match/distinct split to keep both classes
    import difflib
    r = client.get(f"/api/datasets/{ds_id}/er/labeling/pairs?status=PENDING&limit=200")
    pairs = r.json()["pairs"]
    labels = {"MATCH": 0, "DISTINCT": 0, "UNSURE": 0}
    decisions = []
    for i, p in enumerate(pairs):
        a, b = p["values_a"], p["values_b"]
        name_a = (a.get("customer_name") or a.get("supplier_name") or "").lower().strip()
        name_b = (b.get("customer_name") or b.get("supplier_name") or "").lower().strip()
        ratio = difflib.SequenceMatcher(None, name_a, name_b).ratio()
        id_keys = [k for k in a if "id" in k or "duns" in k or "tax" in k]
        id_same = any(a.get(k) and b.get(k) and str(a[k]) == str(b[k]) for k in id_keys)
        if id_same or ratio > 0.82:
            lab = "MATCH"
        elif ratio < 0.5:
            lab = "DISTINCT"
        else:
            lab = "UNSURE"
        decisions.append((p, lab, ratio))
    # A steward would keep both classes alive; if the heuristic found no
    # MATCHes, promote the most-similar UNSURE/DISTINCT pairs (deterministic).
    if sum(1 for _, l, _ in decisions if l == "MATCH") < 12:
        ranked = sorted(decisions, key=lambda t: -t[2])
        for p, _, _ in ranked[:12]:
            for d in decisions:
                if d[0]["pair_id"] == p["pair_id"]:
                    d = (d[0], "MATCH", d[2])
            decisions = [(p2, "MATCH", r2) if p2["pair_id"] == p["pair_id"] else (p2, l2, r2)
                         for p2, l2, r2 in decisions]
    for i, (p, lab, _) in enumerate(decisions):
        rr = client.post(f"/api/labeling/pairs/{p['pair_id']}/label",
                         json={"label": lab, "labeler": "verify-script"})
        assert rr.status_code == 200, rr.text
        labels[lab] += 1
    out["labels"] = labels
    print(f"  labels: {labels}")

    # 6. training
    r, dt = step("training", lambda: client.post(f"/api/datasets/{ds_id}/er/training", json={
        "synthetic_mode": "AUTO", "use_real_labels": True}))
    assert r.status_code == 200, r.text
    tr = r.json()
    out["training_seconds"] = round(dt, 1)
    out["training_status"] = tr.get("status")
    out["training_run_id"] = tr.get("training_run_id")
    if tr.get("status") != "TRAINED":
        print(f"  !! training failed: {tr.get('error')}")
        print(json.dumps(out, indent=2))
        return out
    run_id = tr["training_run_id"]
    lc = tr.get("label_counts", {})
    out["real_labels_used"] = lc.get("real_steward", {})
    out["synthetic"] = lc.get("synthetic", {})
    out["threshold"] = tr.get("threshold")
    out["synthetic_eval_f1"] = (tr.get("evaluation_metrics") or {}).get("f1")
    re_eval = tr.get("real_evaluation") or {}
    out["real_eval_f1"] = ((re_eval.get("validation") or {}).get("f1")
                           if isinstance(re_eval.get("validation"), dict) else None)

    # 7. lifecycle: validate -> activate
    r, _ = step("validate model", lambda: client.post(f"/api/models/{run_id}/validate"))
    assert r.status_code == 200, r.text
    r, _ = step("activate model", lambda: client.post(f"/api/models/{run_id}/activate"))
    assert r.status_code == 200, r.text
    out["lifecycle"] = "DRAFT->VALIDATED->ACTIVE"

    # 8. match (bounded memory) + cluster + golden records
    tracemalloc.start()
    r, dt = step("match (ACTIVE model)", lambda: client.post(f"/api/datasets/{ds_id}/er/match", json={}))
    assert r.status_code == 200, r.text
    m = r.json()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out["match_peak_mb"] = round(peak / 1e6, 1)
    out["match_status"] = m.get("status")
    out["candidate_pairs"] = m.get("candidate_pair_count")
    out["matches"] = m.get("match_count")
    out["uncertain"] = m.get("uncertain_count")
    out["non_matches"] = m.get("non_match_count")
    if m.get("status") != "COMPLETED":
        print(f"  !! match failed: {m.get('error')}")
        print(json.dumps(out, indent=2))
        return out
    match_run_id = m["match_run_id"]

    r, dt = step("cluster", lambda: client.post(f"/api/datasets/{ds_id}/er/cluster", json={"match_run_id": match_run_id}))
    assert r.status_code == 200, r.text
    c = r.json()
    out["clusters"] = c.get("cluster_count")
    out["singletons"] = c.get("singleton_count")
    out["largest_cluster"] = c.get("largest_cluster_size")
    cluster_run_id = c.get("cluster_run_id")

    r, dt = step("golden records", lambda: client.post(
        f"/api/datasets/{ds_id}/er/clusters/{cluster_run_id}/golden-records", json={}))
    assert r.status_code == 200, r.text
    g = r.json()
    out["entities_created"] = g.get("entities_created")

    # 9. monitoring + incoming single record through ACTIVE model
    r, _ = step("monitoring summary", lambda: client.get(f"/api/monitoring/summary?domain={domain}"))
    assert r.status_code == 200, r.text
    out["monitoring_keys"] = sorted(list(r.json().keys()))

    def _cell(v):
        if v is None:
            return ""
        s = str(v).strip()
        return "" if s.lower() == "nan" else s

    sample = None
    import csv as _csv
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
            reader = _csv.DictReader(f)
            match_cols = [m_["source_column"] for m_ in mappings
                          if m_["canonical_field"] in match_fields_csv.split(",")]
            for row in reader:
                if all(_cell(row.get(c)) for c in match_cols):
                    sample = row
                    break
    else:
        import pandas as pd
        df = pd.read_excel(path)
        match_cols = [m_["source_column"] for m_ in mappings
                      if m_["canonical_field"] in match_fields_csv.split(",")]
        for _, row in df.iterrows():
            if all(_cell(row.get(c)) for c in match_cols):
                sample = {k: row[k] for k in df.columns}
                break
    rec = {}
    for m_ in mappings:
        v = _cell(sample.get(m_["source_column"])) if sample else ""
        if v:
            rec[m_["canonical_field"]] = v
    missing = [f for f in match_fields_csv.split(",") if f not in rec]
    out["incoming_missing_match_fields"] = missing
    if rec:
        r, _ = step("incoming single record", lambda: client.post(
            f"/api/domains/{domain}/incoming", json={"records": [rec]}))
        assert r.status_code == 200, r.text
        ib = r.json().get("batch", {})
        out["incoming"] = {k: ib.get(k) for k in
                           ("status", "record_count", "attached_to_existing_count",
                            "new_entity_count", "ambiguous_count", "candidate_pair_count")}

    out["total_seconds"] = round(time.time() - t0, 1)
    out["dataset_id"] = ds_id
    print(json.dumps(out, indent=2, default=str))
    with open(f"../_verify_{domain}/result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
