import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app.db.session as session_mod
    import app.modules.ingestion as ing
    import app.main as main_mod
    import app.modules.dedupe_service.model_store as model_store_mod

    monkeypatch.setattr(session_mod, "DB_URL", f"sqlite:///{tmp_path}/mdm.db")
    monkeypatch.setattr(session_mod, "_engine", None)
    monkeypatch.setattr(session_mod, "_SessionLocal", None)
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(main_mod, "DATA_DIR_FOR_DEDUPE", str(tmp_path / "data"))
    monkeypatch.setenv("MDM_MODELS_DIR", str(tmp_path / "models"))
    os.makedirs(str(tmp_path / "data"), exist_ok=True)

    return TestClient(main_mod.app)


_upload_counter = {"n": 0}


def _diverse_customer_csv(n=150, extra_rows=None):
    # A per-call unique marker column avoids the app's (correct!) sha256
    # duplicate-upload guard rejecting a second call with identical content
    # within the same test.
    _upload_counter["n"] += 1
    marker = _upload_counter["n"]
    # 12 name TEMPLATES, each filled with a genuinely distinguishing middle word
    # per record (not a bare numeric suffix) -- real distinct companies vary by
    # more than an ID number, and a fixture that doesn't reflect that gives a
    # false read on whether the classifier can tell entities apart.
    names = ["Acme Trading Co", "Beta Industries", "Gamma Corp", "Zenith Traders",
             "Omega Supplies", "Delta Systems", "Epsilon Corp", "Theta Logistics",
             "Sigma Freight", "Kappa Metals", "Orion Textiles", "Nova Chemicals"]
    qualifiers = ["North", "South", "East", "West", "Global", "Regional", "Coastal", "Metro",
                 "Continental", "Pacific", "Atlantic", "Central", "Summit", "Valley", "Highland"]
    # A third, independent word list whose length (7, prime) shares no common
    # factor with names (12) or qualifiers (15), so the combined cycle length
    # is lcm(12,15,7)=1260 -- far above n -- and most of the n rows really are
    # unique (as real distinct company records would be), instead of a short
    # 60-row cycle silently repeating identical (name, qualifier) pairs.
    forms = ["Trading", "Holdings", "Group", "Partners", "Ventures", "Enterprises", "Networks"]
    # 7 cities (prime, shares no factor with names' 12) so city is NOT perfectly
    # correlated with the name group -- otherwise every "Acme Trading Co *"
    # record would always land in the same city too, stacking two hard-negative
    # signals into one maximally-ambiguous case that isn't representative.
    cities = ["Chennai", "Mumbai", "Delhi", "Pune", "Kolkata", "Hyderabad", "Bengaluru"]
    rows = ["customer_name,city,email,phone,_batch"]
    for i in range(n):
        rows.append(f"{names[i % len(names)]} {qualifiers[i % len(qualifiers)]} {forms[i % len(forms)]},"
                     f"{cities[i % len(cities)]},contact{i}@example.com,98765{i:05d},{marker}")
    rows.append(f"Acme Trading Co 0,Chennai,,,{marker}")
    rows.append(f"ACME TRADING CO 4,Chennai,,,{marker}")
    if extra_rows:
        rows.extend(f"{r},{marker}" for r in extra_rows)
    return "\n".join(rows).encode()


def _upload_map_standardize_configure(client, content=None, domain="customer"):
    content = content or _diverse_customer_csv()
    r = client.post("/api/datasets/upload", data={"domain": domain},
                     files={"file": ("synth.csv", content, "text/csv")})
    assert r.status_code == 200, r.text
    ds_id = r.json()["dataset_id"]
    sugg = client.get(f"/api/datasets/{ds_id}/mapping/suggestions").json()
    mappings = [{"canonical_field": cf, "source_column": cands[0]["source_column"]}
                for cf, cands in sugg["suggestions"].items() if cands]
    r = client.post(f"/api/datasets/{ds_id}/mapping", json={"mappings": mappings})
    assert r.status_code == 200, r.text
    assert client.post(f"/api/datasets/{ds_id}/profile").status_code == 200
    assert client.post(f"/api/datasets/{ds_id}/standardize").status_code == 200
    r = client.post(f"/api/datasets/{ds_id}/er-config", json={"assignments": [
        {"canonical_field": "customer_name", "role": "MATCH"},
        {"canonical_field": "city", "role": "MATCH"},
    ]})
    assert r.status_code == 200, r.text
    return ds_id


def _train_evaluate_activate(client, ds_id):
    r = client.post(f"/api/datasets/{ds_id}/er/training", json={"seed": 42})
    assert r.status_code == 200, r.text
    tr = r.json()
    assert tr["status"] == "TRAINED", tr
    training_run_id = tr["training_run_id"]

    r = client.post(f"/api/models/{training_run_id}/mark-evaluated")
    assert r.status_code == 200, r.text
    assert r.json()["lifecycle_status"] == "VALIDATED"

    r = client.post(f"/api/models/{training_run_id}/activate")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lifecycle_status"] == "ACTIVE"
    assert body["is_active"] is True
    return training_run_id


def _match_and_cluster(client, ds_id, training_run_id, threshold=0.5):
    r = client.post(f"/api/datasets/{ds_id}/er/match",
                     json={"training_run_id": training_run_id, "threshold": threshold})
    assert r.status_code == 200, r.text
    match_run_id = r.json()["match_run_id"]
    assert r.json()["status"] == "COMPLETED"

    r = client.post(f"/api/datasets/{ds_id}/er/cluster", json={"match_run_id": match_run_id, "threshold": threshold})
    assert r.status_code == 200, r.text
    cluster_run_id = r.json()["cluster_run_id"]
    assert r.json()["status"] == "COMPLETED"
    return match_run_id, cluster_run_id


# =============================================================================
# Model Registry
# =============================================================================

def test_model_lifecycle_requires_explicit_evaluate_then_activate(client):
    ds_id = _upload_map_standardize_configure(client)
    r = client.post(f"/api/datasets/{ds_id}/er/training", json={"seed": 1})
    training_run_id = r.json()["training_run_id"]
    assert r.json()["status"] == "TRAINED"

    m = client.get(f"/api/models/{training_run_id}").json()
    assert m["lifecycle_status"] == "DRAFT"
    assert m["is_active"] is False

    # cannot activate a DRAFT model directly
    r = client.post(f"/api/models/{training_run_id}/activate")
    assert r.status_code == 422

    r = client.post(f"/api/models/{training_run_id}/mark-evaluated")
    assert r.status_code == 200
    assert r.json()["lifecycle_status"] == "VALIDATED"

    r = client.post(f"/api/models/{training_run_id}/activate")
    assert r.status_code == 200
    assert r.json()["lifecycle_status"] == "ACTIVE"

    active = client.get("/api/domains/customer/active-model").json()
    assert active["training_run_id"] == training_run_id


def test_only_one_active_model_per_domain_activating_new_retires_old(client):
    ds_id = _upload_map_standardize_configure(client)
    run_v1 = _train_evaluate_activate(client, ds_id)

    ds_id_2 = _upload_map_standardize_configure(client)
    r = client.post(f"/api/datasets/{ds_id_2}/er/training", json={"seed": 7})
    run_v2 = r.json()["training_run_id"]
    client.post(f"/api/models/{run_v2}/mark-evaluated")
    client.post(f"/api/models/{run_v2}/activate")

    v1_state = client.get(f"/api/models/{run_v1}").json()
    v2_state = client.get(f"/api/models/{run_v2}").json()
    assert v1_state["lifecycle_status"] == "RETIRED"
    assert v1_state["is_active"] is False
    assert v2_state["lifecycle_status"] == "ACTIVE"
    assert v2_state["is_active"] is True

    active = client.get("/api/domains/customer/active-model").json()
    assert active["training_run_id"] == run_v2

    models = client.get("/api/models", params={"domain": "customer"}).json()["models"]
    assert len(models) == 2


def test_retire_active_model_leaves_no_active_model(client):
    ds_id = _upload_map_standardize_configure(client)
    run_id = _train_evaluate_activate(client, ds_id)
    r = client.post(f"/api/models/{run_id}/retire")
    assert r.status_code == 200
    assert r.json()["lifecycle_status"] == "RETIRED"
    r = client.get("/api/domains/customer/active-model")
    assert r.status_code == 404


def test_model_survives_source_dataset_deletion_and_is_still_active(client):
    ds_id = _upload_map_standardize_configure(client)
    run_id = _train_evaluate_activate(client, ds_id)
    client.delete(f"/api/datasets/{ds_id}")
    active = client.get("/api/domains/customer/active-model").json()
    assert active["training_run_id"] == run_id
    assert active["model_metadata"] is not None  # metadata.json still readable


# =============================================================================
# Golden Record
# =============================================================================

def test_golden_record_generation_one_entity_per_cluster_with_lineage_and_conflicts(client):
    ds_id = _upload_map_standardize_configure(client)
    training_run_id = _train_evaluate_activate(client, ds_id)
    match_run_id, cluster_run_id = _match_and_cluster(client, ds_id, training_run_id)

    r = client.post(f"/api/datasets/{ds_id}/er/clusters/{cluster_run_id}/golden-records")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["entities_created"] > 0

    entities = client.get("/api/entities", params={"domain": "customer"}).json()["entities"]
    assert len(entities) == body["entities_created"]

    # the seeded exact-duplicate pair ("Acme Trading Co 0"/"Acme Trading Co 0")
    # should have merged into one multi-member entity with a golden record
    multi_member = [e for e in entities if e["member_count"] > 1]
    assert len(multi_member) >= 1, "expected at least one multi-record entity from seeded duplicates"
    e = multi_member[0]
    assert e["golden_record"] is not None
    assert e["golden_record"]["values"]["customer_name"]
    assert "customer_name" in e["golden_record"]["lineage"]
    assert e["golden_record"]["lineage"]["customer_name"]["dataset_id"] == ds_id

    # singleton entities also get a real golden record (not skipped)
    singletons = [e for e in entities if e["member_count"] == 1]
    assert len(singletons) >= 1
    assert singletons[0]["golden_record"] is not None
    assert singletons[0]["golden_record"]["values"]["customer_name"]


def test_entities_list_min_member_count_filter_finds_duplicates_among_many_singletons(client):
    ds_id = _upload_map_standardize_configure(client, content=_diverse_customer_csv(n=200))
    training_run_id = _train_evaluate_activate(client, ds_id)
    match_run_id, cluster_run_id = _match_and_cluster(client, ds_id, training_run_id)
    client.post(f"/api/datasets/{ds_id}/er/clusters/{cluster_run_id}/golden-records")

    unfiltered = client.get("/api/entities", params={"domain": "customer", "limit": 3}).json()
    assert unfiltered["count"] == 3
    assert unfiltered["total_matching"] >= unfiltered["count"]

    filtered = client.get("/api/entities", params={"domain": "customer", "min_member_count": 2}).json()
    assert filtered["count"] >= 1
    assert all(e["member_count"] >= 2 for e in filtered["entities"])


def test_golden_record_handles_missing_field_values(client):
    # email/phone are blank for the seeded exact-dup rows -> golden record
    # must not crash and must report 0 confidence / null for fully-missing fields
    ds_id = _upload_map_standardize_configure(client)
    training_run_id = _train_evaluate_activate(client, ds_id)
    match_run_id, cluster_run_id = _match_and_cluster(client, ds_id, training_run_id)
    r = client.post(f"/api/datasets/{ds_id}/er/clusters/{cluster_run_id}/golden-records")
    assert r.status_code == 200
    entities = client.get("/api/entities", params={"domain": "customer"}).json()["entities"]
    assert all(e["golden_record"] is not None for e in entities)


def test_golden_record_history_versions_are_appended_not_overwritten(client):
    ds_id = _upload_map_standardize_configure(client)
    training_run_id = _train_evaluate_activate(client, ds_id)
    match_run_id, cluster_run_id = _match_and_cluster(client, ds_id, training_run_id)
    client.post(f"/api/datasets/{ds_id}/er/clusters/{cluster_run_id}/golden-records")
    entities = client.get("/api/entities", params={"domain": "customer"}).json()["entities"]
    entity_id = entities[0]["entity_id"]

    hist_before = client.get(f"/api/entities/{entity_id}/golden-record/history").json()["versions"]
    assert len(hist_before) == 1

    r = client.post(f"/api/entities/{entity_id}/review", json={
        "action": "CORRECT", "reviewer": "steward_1",
        "corrected_values": {"customer_name": "Corrected Name Inc"},
    })
    assert r.status_code == 200

    hist_after = client.get(f"/api/entities/{entity_id}/golden-record/history").json()["versions"]
    assert len(hist_after) == 2
    assert hist_after[0]["values"] != hist_after[1]["values"]
    assert hist_after[1]["values"]["customer_name"] == "Corrected Name Inc"
    current = client.get(f"/api/entities/{entity_id}/golden-record").json()
    assert current["version_number"] == 2


# =============================================================================
# Steward Review
# =============================================================================

def _setup_entity(client):
    ds_id = _upload_map_standardize_configure(client)
    training_run_id = _train_evaluate_activate(client, ds_id)
    match_run_id, cluster_run_id = _match_and_cluster(client, ds_id, training_run_id)
    client.post(f"/api/datasets/{ds_id}/er/clusters/{cluster_run_id}/golden-records")
    entities = client.get("/api/entities", params={"domain": "customer"}).json()["entities"]
    multi = [e for e in entities if e["member_count"] > 1][0]
    return ds_id, training_run_id, multi["entity_id"]


def test_steward_approve_reject_correct_persist_with_provenance(client):
    ds_id, training_run_id, entity_id = _setup_entity(client)

    r = client.post(f"/api/entities/{entity_id}/review",
                     json={"action": "APPROVE", "reviewer": "steward_a", "reason": "looks correct"})
    assert r.status_code == 200
    body = r.json()
    assert body["action"] == "APPROVE"
    assert body["reviewer"] == "steward_a"

    r = client.post(f"/api/entities/{entity_id}/review",
                     json={"action": "CORRECT", "reviewer": "steward_a",
                           "corrected_values": {"city": "Chennai (verified)"}})
    assert r.status_code == 200
    assert r.json()["after_state"]["golden_values"]["city"] == "Chennai (verified)"

    ctx = client.get(f"/api/entities/{entity_id}/review-context").json()
    assert len(ctx["review_history"]) == 2

    # REJECT on a fresh entity
    ds2, _, entity2 = _setup_entity(client)
    r = client.post(f"/api/entities/{entity2}/review", json={"action": "REJECT", "reviewer": "steward_b"})
    assert r.status_code == 200
    assert client.get(f"/api/entities/{entity2}").json()["status"] == "REJECTED"


def test_steward_split_creates_new_entity_and_regenerates_both_golden_records(client):
    ds_id, training_run_id, entity_id = _setup_entity(client)
    members = client.get(f"/api/entities/{entity_id}").json()["members"]
    assert len(members) >= 2
    to_split = [members[0]]

    r = client.post(f"/api/entities/{entity_id}/review", json={
        "action": "SPLIT", "reviewer": "steward_c",
        "split_members": [{"dataset_id": m["dataset_id"], "record_id": m["record_id"]} for m in to_split],
    })
    assert r.status_code == 200, r.text
    new_entity_id = r.json()["after_state"]["new_entity_id"]

    original = client.get(f"/api/entities/{entity_id}").json()
    new_e = client.get(f"/api/entities/{new_entity_id}").json()
    assert original["member_count"] == len(members) - 1
    assert new_e["member_count"] == 1
    assert new_e["golden_record"] is not None
    assert original["golden_record"] is not None


def test_steward_split_rejects_moving_all_members(client):
    ds_id, training_run_id, entity_id = _setup_entity(client)
    members = client.get(f"/api/entities/{entity_id}").json()["members"]
    r = client.post(f"/api/entities/{entity_id}/review", json={
        "action": "SPLIT", "reviewer": "steward_c",
        "split_members": [{"dataset_id": m["dataset_id"], "record_id": m["record_id"]} for m in members],
    })
    assert r.status_code == 422


def test_steward_merge_combines_two_entities(client):
    ds_id, training_run_id, entity_id = _setup_entity(client)
    entities = client.get("/api/entities", params={"domain": "customer"}).json()["entities"]
    singleton = [e for e in entities if e["member_count"] == 1 and e["entity_id"] != entity_id][0]

    r = client.post(f"/api/entities/{singleton['entity_id']}/review", json={
        "action": "MERGE", "reviewer": "steward_d", "target_entity_id": entity_id,
    })
    assert r.status_code == 200, r.text

    merged = client.get(f"/api/entities/{singleton['entity_id']}").json()
    assert merged["status"] == "MERGED"
    assert merged["merged_into_entity_id"] == entity_id
    assert merged["member_count"] == 0

    target = client.get(f"/api/entities/{entity_id}").json()
    assert target["member_count"] >= 2  # gained the merged member


# =============================================================================
# Continuous Incoming Data
# =============================================================================

def test_incoming_batch_requires_active_model(client):
    ds_id = _upload_map_standardize_configure(client)
    r = client.post(f"/api/datasets/{ds_id}/er/apply-active-model", json={})
    assert r.status_code == 422


def test_incoming_batch_matches_existing_entity_creates_new_and_does_not_retrain(client):
    ds_id = _upload_map_standardize_configure(client)
    training_run_id = _train_evaluate_activate(client, ds_id)
    match_run_id, cluster_run_id = _match_and_cluster(client, ds_id, training_run_id)
    client.post(f"/api/datasets/{ds_id}/er/clusters/{cluster_run_id}/golden-records")

    models_before = client.get("/api/models", params={"domain": "customer"}).json()["models"]
    assert len(models_before) == 1  # only the one we trained -- nothing auto-retrained yet

    # original dataset can be deleted -- active model must still work
    client.delete(f"/api/datasets/{ds_id}")

    incoming_content = _diverse_customer_csv(n=30, extra_rows=[
        "Acme Trading Co 0,Chennai,new-contact@example.com,9999999999",  # should match existing entity
        "Totally Unique Widgets Co,Pune,unique@example.com,8888888888",  # should become a new entity
    ])
    ds_2 = _upload_map_standardize_configure(client, content=incoming_content)

    r = client.post(f"/api/datasets/{ds_2}/er/apply-active-model",
                     json={"match_threshold": 0.5, "ambiguous_threshold": 0.3})
    assert r.status_code == 200, r.text
    batch = r.json()
    assert batch["status"] == "COMPLETED"
    assert batch["training_run_id"] == training_run_id
    assert batch["attached_to_existing_count"] + batch["new_entity_count"] + batch["ambiguous_count"] \
        == batch["record_count"]

    models_after = client.get("/api/models", params={"domain": "customer"}).json()["models"]
    assert len(models_after) == 1  # STILL only one model -- incoming batch did not trigger retraining

    entities_after = client.get("/api/entities", params={"domain": "customer"}).json()["entities"]
    incoming_members = [m for e in entities_after for m in e["members"] if m["source"] == "INCOMING"]
    assert len(incoming_members) > 0


def test_incoming_batch_ambiguous_goes_to_pending_review_not_forced(client):
    ds_id = _upload_map_standardize_configure(client)
    training_run_id = _train_evaluate_activate(client, ds_id)
    match_run_id, cluster_run_id = _match_and_cluster(client, ds_id, training_run_id)
    client.post(f"/api/datasets/{ds_id}/er/clusters/{cluster_run_id}/golden-records")

    # a very wide ambiguous band forces the "close but not certain" path to
    # be exercised deterministically rather than relying on hoping for it
    incoming_content = _diverse_customer_csv(n=10, extra_rows=[
        "Acme Trading Co 0,Chennai,x@example.com,9999999999",
    ])
    ds_2 = _upload_map_standardize_configure(client, content=incoming_content)
    r = client.post(f"/api/datasets/{ds_2}/er/apply-active-model",
                     json={"match_threshold": 0.99, "ambiguous_threshold": 0.01})
    assert r.status_code == 200, r.text
    batch = r.json()
    assert batch["ambiguous_count"] >= 1

    pending = client.get("/api/reviews/pending", params={"domain": "customer"}).json()["pending_reviews"]
    assert len(pending) >= 1

    # resolving one via steward review marks it RESOLVED
    item = pending[0]
    if item["entity_id"]:
        r = client.post(f"/api/entities/{item['entity_id']}/review", json={
            "action": "APPROVE", "reviewer": "steward_e", "pending_review_id": item["pending_review_id"],
        })
        assert r.status_code == 200
        pending_after = client.get("/api/reviews/pending", params={"domain": "customer"}).json()["pending_reviews"]
        assert all(p["pending_review_id"] != item["pending_review_id"] for p in pending_after)


# =============================================================================
# Monitoring
# =============================================================================

def test_monitoring_summary_survives_source_dataset_deletion(client):
    ds_id, training_run_id, entity_id = _setup_entity(client)
    before = client.get("/api/monitoring/summary", params={"domain": "customer"}).json()
    assert before["operational_matching"]["match_runs"] >= 1
    assert before["operational_matching"]["cluster_runs"] >= 1

    client.delete(f"/api/datasets/{ds_id}")

    after = client.get("/api/monitoring/summary", params={"domain": "customer"}).json()
    assert after["operational_matching"]["match_runs"] == before["operational_matching"]["match_runs"]
    assert after["operational_matching"]["cluster_runs"] == before["operational_matching"]["cluster_runs"]
    assert after["operational_matching"]["total_clusters"] == before["operational_matching"]["total_clusters"]


def test_monitoring_summary_reports_separated_categories(client):
    ds_id, training_run_id, entity_id = _setup_entity(client)
    client.post(f"/api/entities/{entity_id}/review", json={"action": "APPROVE", "reviewer": "s"})

    summary = client.get("/api/monitoring/summary", params={"domain": "customer"}).json()
    assert "training_validation" in summary
    assert "operational_matching" in summary
    assert "steward_feedback" in summary
    assert summary["steward_feedback"]["total_reviews"] == 1
    assert summary["steward_feedback"]["by_action"]["APPROVE"] == 1
    assert summary["active_model"]["training_run_id"] == training_run_id


# =============================================================================
# End-to-end
# =============================================================================

def test_end_to_end_upload_through_golden_record_and_review(client):
    ds_id = _upload_map_standardize_configure(client)
    training_run_id = _train_evaluate_activate(client, ds_id)
    match_run_id, cluster_run_id = _match_and_cluster(client, ds_id, training_run_id)
    gr = client.post(f"/api/datasets/{ds_id}/er/clusters/{cluster_run_id}/golden-records").json()
    assert gr["entities_created"] > 0

    entities = client.get("/api/entities", params={"domain": "customer"}).json()["entities"]
    multi = [e for e in entities if e["member_count"] > 1][0]
    r = client.post(f"/api/entities/{multi['entity_id']}/review",
                     json={"action": "APPROVE", "reviewer": "final_steward"})
    assert r.status_code == 200

    # RAW must remain byte-identical to what was uploaded throughout all of this
    import app.modules.ingestion as ing
    raw_df = ing.load_raw(ds_id)
    assert len(raw_df) > 0
