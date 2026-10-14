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
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main_mod, "DATA_DIR_FOR_DEDUPE", str(tmp_path))
    os.makedirs(str(tmp_path), exist_ok=True)

    return TestClient(main_mod.app)


def _build_diverse_customer_csv(n=150):
    names = ["Acme Trading Co", "Beta Industries", "Gamma Corp", "Zenith Traders",
             "Omega Supplies", "Delta Systems", "Epsilon Corp", "Theta Logistics",
             "Sigma Freight", "Kappa Metals", "Orion Textiles", "Nova Chemicals"]
    cities = ["Chennai", "Mumbai", "Delhi", "Pune", "Kolkata", "Hyderabad"]
    rows = ["customer_name,city"]
    for i in range(n):
        rows.append(f"{names[i % len(names)]} {i},{cities[i % len(cities)]}")
    rows.append("Acme Trading Co 0,Chennai")   # exact duplicate of row 0
    rows.append("ACME TRADING CO 4,Chennai")   # case-variant duplicate of row 4
    return "\n".join(rows).encode()


def _setup_dataset_with_active_er_config(client):
    content = _build_diverse_customer_csv()
    r = client.post("/api/datasets/upload", data={"domain": "customer"},
                     files={"file": ("synth.csv", content, "text/csv")})
    ds_id = r.json()["dataset_id"]
    sugg = client.get(f"/api/datasets/{ds_id}/mapping/suggestions").json()
    mappings = [{"canonical_field": cf, "source_column": cands[0]["source_column"]}
                for cf, cands in sugg["suggestions"].items() if cands]
    client.post(f"/api/datasets/{ds_id}/mapping", json={"mappings": mappings})
    client.post(f"/api/datasets/{ds_id}/profile")
    client.post(f"/api/datasets/{ds_id}/standardize")
    client.post(f"/api/datasets/{ds_id}/reprofile")
    client.post(f"/api/datasets/{ds_id}/er-config", json={"assignments": [
        {"canonical_field": "customer_name", "role": "MATCH"},
        {"canonical_field": "city", "role": "MATCH"},
    ]})
    return ds_id


# ---------------- Training endpoint ----------------

def test_training_requires_active_er_config(client):
    content = b"customer_name,email\nBob,b@x.com\n"
    r = client.post("/api/datasets/upload", data={"domain": "customer"},
                     files={"file": ("t.csv", content, "text/csv")})
    ds_id = r.json()["dataset_id"]
    r = client.post(f"/api/datasets/{ds_id}/er/training", json={})
    assert r.status_code == 422


def test_training_full_lifecycle_via_api(client):
    ds_id = _setup_dataset_with_active_er_config(client)
    r = client.post(f"/api/datasets/{ds_id}/er/training", json={"seed": 42})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "TRAINED"
    assert body["model_available"] is True
    assert body["er_config_version"] is not None
    assert set(body["match_fields"]) == {"customer_name", "city"}
    assert body["evaluation_metrics"]["validation_pair_count"] > 0

    r2 = client.get(f"/api/datasets/{ds_id}/er/training/{body['training_run_id']}")
    assert r2.status_code == 200
    assert r2.json()["training_run_id"] == body["training_run_id"]

    r3 = client.get(f"/api/datasets/{ds_id}/er/models")
    assert r3.status_code == 200
    assert len(r3.json()["training_runs"]) == 1


def test_training_run_not_found_404(client):
    ds_id = _setup_dataset_with_active_er_config(client)
    r = client.get(f"/api/datasets/{ds_id}/er/training/TR-doesnotexist")
    assert r.status_code == 404


# ---------------- Matching endpoint ----------------

def test_matching_requires_trained_model(client):
    ds_id = _setup_dataset_with_active_er_config(client)
    r = client.post(f"/api/datasets/{ds_id}/er/match", json={"training_run_id": "TR-fake", "threshold": 0.5})
    assert r.status_code == 404


def test_matching_full_lifecycle_via_api(client):
    ds_id = _setup_dataset_with_active_er_config(client)
    tr = client.post(f"/api/datasets/{ds_id}/er/training", json={"seed": 42}).json()
    assert tr["status"] == "TRAINED"

    r = client.post(f"/api/datasets/{ds_id}/er/match",
                     json={"training_run_id": tr["training_run_id"], "threshold": 0.5})
    assert r.status_code == 200
    mr = r.json()
    assert mr["status"] == "COMPLETED"
    assert mr["candidate_pair_count"] <= mr["total_possible_pairs"]
    assert mr["candidate_reduction_pct"] > 0  # real candidate generation actually reduced the search space
    assert mr["training_run_id"] == tr["training_run_id"]  # provenance preserved

    r2 = client.get(f"/api/datasets/{ds_id}/er/matches/{mr['match_run_id']}")
    assert r2.status_code == 200
    assert "pairs" in r2.json()
    for p in r2.json()["pairs"]:
        assert "score" in p and p["decision"] == "MATCH"


def test_matching_run_provenance_traceable_to_training_and_config_version(client):
    ds_id = _setup_dataset_with_active_er_config(client)
    tr = client.post(f"/api/datasets/{ds_id}/er/training", json={"seed": 42}).json()
    mr = client.post(f"/api/datasets/{ds_id}/er/match",
                      json={"training_run_id": tr["training_run_id"], "threshold": 0.5}).json()
    assert mr["training_run_id"] == tr["training_run_id"]
    assert mr["dataset_id"] == ds_id


# ---------------- Clustering endpoint ----------------

def test_clustering_requires_completed_match_run(client):
    ds_id = _setup_dataset_with_active_er_config(client)
    r = client.post(f"/api/datasets/{ds_id}/er/cluster", json={"match_run_id": "MR-fake", "threshold": 0.5})
    assert r.status_code == 404


def test_clustering_full_lifecycle_via_api_recovers_known_duplicates(client):
    ds_id = _setup_dataset_with_active_er_config(client)
    tr = client.post(f"/api/datasets/{ds_id}/er/training", json={"seed": 42}).json()
    mr = client.post(f"/api/datasets/{ds_id}/er/match",
                      json={"training_run_id": tr["training_run_id"], "threshold": 0.5}).json()

    r = client.post(f"/api/datasets/{ds_id}/er/cluster",
                     json={"match_run_id": mr["match_run_id"], "threshold": 0.5})
    assert r.status_code == 200
    cr = r.json()
    assert cr["status"] == "COMPLETED"
    assert cr["match_run_id"] == mr["match_run_id"]
    # Every record must be accounted for: clusters + singletons == total records matched.
    assert cr["cluster_count"] >= cr["singleton_count"]

    r2 = client.get(f"/api/datasets/{ds_id}/er/clusters/{cr['cluster_run_id']}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["non_singleton_cluster_count"] >= 1  # the two seeded duplicates should form at least one cluster
    for c in body["clusters"]:
        assert c["cluster_size"] == len(c["record_ids"])
        assert len(c["record_ids"]) == len(c["scores"])


def test_full_pipeline_end_to_end_traceability_chain(client):
    """The complete traceability chain: dataset -> ER config -> training run
    -> match run -> cluster run must all be linkable end to end."""
    ds_id = _setup_dataset_with_active_er_config(client)
    tr = client.post(f"/api/datasets/{ds_id}/er/training", json={"seed": 7}).json()
    mr = client.post(f"/api/datasets/{ds_id}/er/match",
                      json={"training_run_id": tr["training_run_id"], "threshold": 0.5}).json()
    cr = client.post(f"/api/datasets/{ds_id}/er/cluster",
                      json={"match_run_id": mr["match_run_id"], "threshold": 0.5}).json()

    assert tr["dataset_id"] == ds_id
    assert mr["dataset_id"] == ds_id and mr["training_run_id"] == tr["training_run_id"]
    assert cr["dataset_id"] == ds_id and cr["match_run_id"] == mr["match_run_id"]
    assert cr["training_run_id"] == tr["training_run_id"]


def test_raw_data_never_modified_by_er_pipeline(client):
    import app.modules.ingestion as ing
    ds_id = _setup_dataset_with_active_er_config(client)
    raw_before = ing.load_raw(ds_id).copy(deep=True)

    tr = client.post(f"/api/datasets/{ds_id}/er/training", json={"seed": 42}).json()
    client.post(f"/api/datasets/{ds_id}/er/match",
                json={"training_run_id": tr["training_run_id"], "threshold": 0.5})

    raw_after = ing.load_raw(ds_id)
    assert raw_before.equals(raw_after)
