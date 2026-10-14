"""
Phase: real-label training workflow, model lifecycle (DRAFT -> VALIDATED ->
ACTIVE -> RETIRED), survivorship configuration, and the JSON incoming-record
API. Reuses the fixture/helper pattern from test_golden_record_registry_review.py.
"""
import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app.db.session as session_mod
    import app.modules.ingestion as ing
    import app.main as main_mod

    monkeypatch.setattr(session_mod, "DB_URL", f"sqlite:///{tmp_path}/mdm.db")
    monkeypatch.setattr(session_mod, "_engine", None)
    monkeypatch.setattr(session_mod, "_SessionLocal", None)
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(main_mod, "DATA_DIR_FOR_DEDUPE", str(tmp_path / "data"))
    monkeypatch.setenv("MDM_MODELS_DIR", str(tmp_path / "models"))
    os.makedirs(str(tmp_path / "data"), exist_ok=True)
    return TestClient(main_mod.app)


_counter = {"n": 0}


def _diverse_csv(n=120, extra_rows=None):
    _counter["n"] += 1
    marker = _counter["n"]
    names = ["Acme Trading", "Beta Industries", "Gamma Corp", "Zenith Traders",
             "Omega Supplies", "Delta Systems", "Epsilon Corp", "Theta Logistics"]
    qualifiers = ["North", "South", "East", "West", "Global", "Regional", "Coastal", "Metro", "Summit", "Valley", "Pacific"]
    cities = ["Chennai", "Mumbai", "Delhi", "Pune", "Kolkata", "Hyderabad", "Bengaluru"]
    rows = ["customer_name,city,email,_batch"]
    for i in range(n):
        rows.append(f"{names[i % len(names)]} {qualifiers[i % len(qualifiers)]},{cities[i % len(cities)]},"
                     f"c{i}@example.com,{marker}")
    if extra_rows:
        rows.extend(f"{r},{marker}" for r in extra_rows)
    return "\n".join(rows).encode()


def _setup_dataset(client, content=None, domain="customer", match_fields=("customer_name",)):
    content = content or _diverse_csv()
    r = client.post("/api/datasets/upload", data={"domain": domain}, files={"file": ("d.csv", content, "text/csv")})
    assert r.status_code == 200, r.text
    ds_id = r.json()["dataset_id"]
    sugg = client.get(f"/api/datasets/{ds_id}/mapping/suggestions").json()
    mappings = [{"canonical_field": cf, "source_column": c[0]["source_column"]} for cf, c in sugg["suggestions"].items() if c]
    assert client.post(f"/api/datasets/{ds_id}/mapping", json={"mappings": mappings}).status_code == 200
    assert client.post(f"/api/datasets/{ds_id}/profile").status_code == 200
    assert client.post(f"/api/datasets/{ds_id}/standardize").status_code == 200
    r = client.post(f"/api/datasets/{ds_id}/er-config",
                     json={"assignments": [{"canonical_field": f, "role": "MATCH"} for f in match_fields]})
    assert r.status_code == 200, r.text
    return ds_id


# =============================================================================
# Real-label workflow
# =============================================================================

def test_bootstrap_candidates_use_dedupe_active_learning(client):
    ds_id = _setup_dataset(client)
    r = client.post(f"/api/datasets/{ds_id}/er/labeling/candidates", json={"n": 8, "strategy": "BOOTSTRAP"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["strategy_used"] == "DEDUPE_ACTIVE_LEARNING_BOOTSTRAP"
    assert 0 < len(body["pairs"]) <= 8
    for p in body["pairs"]:
        assert p["strategy"] == "DEDUPE_ACTIVE_LEARNING"
        assert p["record_id_a"] != p["record_id_b"]


def test_candidate_pairs_never_repeat_and_can_be_labeled(client):
    ds_id = _setup_dataset(client)
    r1 = client.post(f"/api/datasets/{ds_id}/er/labeling/candidates", json={"n": 6, "strategy": "BOOTSTRAP"})
    pairs1 = {(p["record_id_a"], p["record_id_b"]) for p in r1.json()["pairs"]}
    r2 = client.post(f"/api/datasets/{ds_id}/er/labeling/candidates", json={"n": 6, "strategy": "BOOTSTRAP"})
    pairs2 = {(p["record_id_a"], p["record_id_b"]) for p in r2.json()["pairs"]}
    assert pairs1.isdisjoint(pairs2), "candidate generation must not re-offer already-created pairs"

    pending = client.get(f"/api/datasets/{ds_id}/er/labeling/pairs", params={"status": "PENDING"}).json()["pairs"]
    assert len(pending) == len(pairs1) + len(pairs2)
    p = pending[0]
    r = client.post(f"/api/labeling/pairs/{p['pair_id']}/label",
                     json={"label": "MATCH", "labeler": "steward_1", "reason": "same company"})
    assert r.status_code == 200
    assert r.json()["label"] == "MATCH"
    assert r.json()["label_source"] == "REAL_STEWARD"
    assert r.json()["labeled_at"] is not None


def test_label_rejects_missing_labeler_and_invalid_label(client):
    ds_id = _setup_dataset(client)
    cands = client.post(f"/api/datasets/{ds_id}/er/labeling/candidates", json={"n": 3, "strategy": "BOOTSTRAP"}).json()["pairs"]
    pid = cands[0]["pair_id"]
    assert client.post(f"/api/labeling/pairs/{pid}/label", json={"label": "MATCH", "labeler": ""}).status_code == 422
    assert client.post(f"/api/labeling/pairs/{pid}/label", json={"label": "MAYBE", "labeler": "s"}).status_code == 422


def test_manual_label_and_duplicate_pair_rejected(client):
    ds_id = _setup_dataset(client)
    r = client.post(f"/api/datasets/{ds_id}/er/labeling/manual",
                     json={"record_id_a": 0, "record_id_b": 1, "label": "DISTINCT", "labeler": "steward_x"})
    assert r.status_code == 200, r.text
    assert r.json()["label"] == "DISTINCT"
    r2 = client.post(f"/api/datasets/{ds_id}/er/labeling/manual",
                      json={"record_id_a": 0, "record_id_b": 1, "label": "MATCH", "labeler": "steward_x"})
    assert r2.status_code == 409


def test_unsure_labels_excluded_from_summary_usable_count(client):
    ds_id = _setup_dataset(client)
    cands = client.post(f"/api/datasets/{ds_id}/er/labeling/candidates", json={"n": 4, "strategy": "BOOTSTRAP"}).json()["pairs"]
    client.post(f"/api/labeling/pairs/{cands[0]['pair_id']}/label", json={"label": "MATCH", "labeler": "s"})
    client.post(f"/api/labeling/pairs/{cands[1]['pair_id']}/label", json={"label": "DISTINCT", "labeler": "s"})
    client.post(f"/api/labeling/pairs/{cands[2]['pair_id']}/label", json={"label": "UNSURE", "labeler": "s"})
    summary = client.get("/api/domains/customer/labels/summary").json()
    assert summary["MATCH"] == 1 and summary["DISTINCT"] == 1 and summary["UNSURE"] == 1
    assert summary["usable_for_training"] == 2  # UNSURE excluded


def test_training_uses_real_labels_and_reports_provenance_separately(client):
    ds_id = _setup_dataset(client)
    # supply enough real labels to make training usable purely from bootstrap candidates
    for _ in range(3):
        cands = client.post(f"/api/datasets/{ds_id}/er/labeling/candidates", json={"n": 10, "strategy": "BOOTSTRAP"}).json()["pairs"]
        for i, p in enumerate(cands):
            label = "MATCH" if i % 3 == 0 else "DISTINCT"
            client.post(f"/api/labeling/pairs/{p['pair_id']}/label", json={"label": label, "labeler": "steward_1"})

    r = client.post(f"/api/datasets/{ds_id}/er/training", json={"seed": 1, "synthetic_mode": "AUTO"})
    assert r.status_code == 200, r.text
    tr = r.json()
    assert tr["status"] in ("TRAINED", "INSUFFICIENT_TRAINING_DATA")
    if tr["status"] == "TRAINED":
        lc = tr["label_counts"]
        assert lc["real_steward"]["match"] >= 1 and lc["real_steward"]["distinct"] >= 1
        assert "synthetic" in lc
        # SYNTHETIC and REAL evaluations must be reported separately, never merged
        assert tr["evaluation_metrics"] is None or tr["evaluation_metrics"]["kind"] == "SYNTHETIC_VALIDATION"
        if tr["real_evaluation"]:
            assert tr["real_evaluation"]["kind"] == "REAL_STEWARD_LABELLED_EVALUATION"
            assert tr["real_evaluation"]["label_source"] == "REAL_STEWARD"


def test_training_never_mode_fails_without_enough_real_labels(client):
    ds_id = _setup_dataset(client)
    r = client.post(f"/api/datasets/{ds_id}/er/training", json={"seed": 1, "synthetic_mode": "NEVER"})
    assert r.status_code == 200
    assert r.json()["status"] == "INSUFFICIENT_TRAINING_DATA"
    assert "labelled" in r.json()["error"].lower() or "label" in r.json()["error"].lower()


def test_entity_level_split_never_leaks_a_record_across_splits(client):
    from app.modules.dedupe_service.real_labels import split_real_labels
    labels = []
    for i in range(0, 60, 2):
        labels.append({"pair_id": f"p{i}", "dataset_id": "D1", "record_id_a": i, "record_id_b": i + 1, "label": "MATCH"})
    for i in range(100, 140, 2):
        labels.append({"pair_id": f"p{i}", "dataset_id": "D1", "record_id_a": i, "record_id_b": i + 1, "label": "DISTINCT"})
    split = split_real_labels(labels, seed=7)
    seen = {}
    for split_name, items in split.items():
        for l in items:
            for rid in (l["record_id_a"], l["record_id_b"]):
                assert seen.get(rid, split_name) == split_name, f"record {rid} leaked across splits"
                seen[rid] = split_name
    assert sum(len(v) for v in split.values()) == len(labels)


# =============================================================================
# Model lifecycle: DRAFT -> VALIDATED -> ACTIVE -> RETIRED
# =============================================================================

def _train(client, ds_id, seed=1):
    r = client.post(f"/api/datasets/{ds_id}/er/training", json={"seed": seed})
    assert r.status_code == 200, r.text
    tr = r.json()
    assert tr["status"] == "TRAINED", tr
    return tr["training_run_id"]


def test_cannot_activate_without_validate(client):
    ds_id = _setup_dataset(client)
    run_id = _train(client, ds_id)
    assert client.get(f"/api/models/{run_id}").json()["lifecycle_status"] == "DRAFT"
    assert client.post(f"/api/models/{run_id}/activate").status_code == 422
    assert client.post(f"/api/models/{run_id}/validate").status_code == 200
    assert client.get(f"/api/models/{run_id}").json()["lifecycle_status"] == "VALIDATED"
    assert client.post(f"/api/models/{run_id}/activate").status_code == 200
    assert client.get(f"/api/models/{run_id}").json()["lifecycle_status"] == "ACTIVE"


def test_activation_refuses_stale_er_config_version(client):
    ds_id = _setup_dataset(client)
    run_id = _train(client, ds_id)
    client.post(f"/api/models/{run_id}/validate")
    # change the dataset's active ER configuration after training
    r = client.post(f"/api/datasets/{ds_id}/er-config",
                     json={"assignments": [{"canonical_field": "customer_name", "role": "MATCH"},
                                           {"canonical_field": "city", "role": "MATCH"}]})
    assert r.status_code == 200
    r = client.post(f"/api/models/{run_id}/activate")
    assert r.status_code == 422
    assert "configuration" in r.json()["detail"].lower()


def test_threshold_is_stored_on_model_and_used_by_default_in_matching(client):
    ds_id = _setup_dataset(client)
    r = client.post(f"/api/datasets/{ds_id}/er/training", json={"seed": 1, "threshold": 0.65})
    tr = r.json()
    assert tr["threshold"] == 0.65
    assert tr["threshold_calibration"] == "MANUAL"
    run_id = tr["training_run_id"]
    client.post(f"/api/models/{run_id}/validate")
    client.post(f"/api/models/{run_id}/activate")
    r = client.post(f"/api/datasets/{ds_id}/er/match", json={})   # no threshold given -> model's stored one used
    assert r.status_code == 200
    assert r.json()["threshold"] == 0.65


def test_match_endpoint_uses_active_model_when_training_run_not_named(client):
    ds_id = _setup_dataset(client)
    run_id = _train(client, ds_id)
    client.post(f"/api/models/{run_id}/validate")
    client.post(f"/api/models/{run_id}/activate")
    r = client.post(f"/api/datasets/{ds_id}/er/match", json={})
    assert r.status_code == 200
    assert r.json()["training_run_id"] == run_id


def test_no_active_model_is_a_clear_422_not_a_500(client):
    ds_id = _setup_dataset(client)
    r = client.post(f"/api/datasets/{ds_id}/er/match", json={})
    assert r.status_code == 422


# =============================================================================
# Survivorship configuration
# =============================================================================

def test_survivorship_config_default_and_update(client):
    d = client.get("/api/domains/customer/survivorship").json()
    assert d["is_default"] is True
    assert "rule_order" in d["config"]

    # Saving creates a DRAFT: visible via GET, but NOT yet driving production
    # Golden Records (only an ACTIVE configuration does).
    r = client.put("/api/domains/customer/survivorship",
                    json={"source_priority": {"DS-1": 10}, "updated_by": "admin"})
    assert r.status_code == 200
    d2 = r.json()
    assert d2["is_default"] is False
    assert d2["config"]["source_priority"] == {"DS-1": 10}
    assert d2["updated_by"] == "admin"
    assert d2["lifecycle_status"] == "DRAFT"
    assert d2["generates_golden_records"] is False

    # DRAFT -> STEWARD_CONFIRMED -> ACTIVE; only ACTIVE generates records.
    s = client.post("/api/domains/customer/survivorship/submit").json()
    assert s["lifecycle_status"] == "STEWARD_CONFIRMED"
    assert s["generates_golden_records"] is False
    a = client.post("/api/domains/customer/survivorship/confirm").json()
    assert a["lifecycle_status"] == "ACTIVE"
    assert a["generates_golden_records"] is True
    assert a["config"]["source_priority"] == {"DS-1": 10}


def test_survivorship_config_rejects_unknown_rule(client):
    r = client.put("/api/domains/customer/survivorship", json={"rule_order": ["not_a_real_rule"]})
    assert r.status_code == 422


def test_golden_record_conflict_flagged_for_review_and_resolvable(client):
    from app.modules.golden_record_service import select_golden_values
    members = [
        {"dataset_id": "D1", "record_id": "1", "values": {"phone": "+911111111111"}},
        {"dataset_id": "D1", "record_id": "2", "values": {"phone": "+922222222222"}},
    ]
    result = select_golden_values(members, ["phone"])
    assert len(result.needs_review) == 1
    assert result.needs_review[0]["field"] == "phone"
    assert "review" in result.needs_review[0]["message"].lower()
    assert len(result.conflicts["phone"]) == 1


# =============================================================================
# Incoming-record API (single record + batch, same underlying service)
# =============================================================================

def _activate_trained_model(client, ds_id):
    run_id = _train(client, ds_id)
    client.post(f"/api/models/{run_id}/validate")
    client.post(f"/api/models/{run_id}/activate")
    return run_id


def test_single_incoming_record_via_json_api(client):
    ds_id = _setup_dataset(client)
    training_run_id = _activate_trained_model(client, ds_id)
    r = client.post(f"/api/datasets/{ds_id}/er/match", json={})
    mr = r.json()["match_run_id"]
    cr = client.post(f"/api/datasets/{ds_id}/er/cluster", json={"match_run_id": mr}).json()["cluster_run_id"]
    client.post(f"/api/datasets/{ds_id}/er/clusters/{cr}/golden-records")

    r = client.post("/api/domains/customer/incoming",
                     json={"records": [{"customer_name": "Acme Trading North", "city": "Chennai"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["batch"]["status"] == "COMPLETED"
    assert body["batch"]["record_count"] == 1

    models = client.get("/api/models", params={"domain": "customer"}).json()["models"]
    assert len(models) == 1, "a single incoming record must not trigger retraining"


def test_incoming_batch_via_json_api_uses_same_service_as_file(client):
    ds_id = _setup_dataset(client)
    _activate_trained_model(client, ds_id)
    r = client.post("/api/domains/customer/incoming", json={"records": [
        {"customer_name": "New Company Alpha", "city": "Pune"},
        {"customer_name": "New Company Beta", "city": "Kolkata"},
    ]})
    assert r.status_code == 200, r.text
    assert r.json()["batch"]["record_count"] == 2


def test_incoming_records_requires_active_model(client):
    ds_id = _setup_dataset(client)
    r = client.post("/api/domains/customer/incoming", json={"records": [{"customer_name": "X"}]})
    assert r.status_code == 422


def test_incoming_records_rejects_empty_batch(client):
    r = client.post("/api/domains/customer/incoming", json={"records": []})
    assert r.status_code == 422


# =============================================================================
# Jobs
# =============================================================================

def test_training_and_matching_create_job_records(client):
    ds_id = _setup_dataset(client)
    _train(client, ds_id)
    jobs = client.get("/api/jobs", params={"domain": "customer", "job_type": "TRAINING"}).json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["status"] == "DONE"
    assert jobs[0]["started_at"] is not None and jobs[0]["completed_at"] is not None


# =============================================================================
# Monitoring categories
# =============================================================================

def test_monitoring_reports_three_separate_categories(client):
    ds_id = _setup_dataset(client)
    _activate_trained_model(client, ds_id)
    summary = client.get("/api/monitoring/summary", params={"domain": "customer"}).json()
    assert "data_quality" in summary and "model_quality" in summary and "operational_performance" in summary
    assert "note" in summary["data_quality"]
    assert "note" in summary["model_quality"]
    assert "note" in summary["operational_performance"]
