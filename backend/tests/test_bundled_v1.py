"""Bundled V1 models: seeded + activated at start-up, used by find-duplicates, never trained."""
import io
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app.db.session as S
    import app.modules.ingestion as ing
    import app.main as M
    monkeypatch.setattr(S, "DB_URL", f"sqlite:///{tmp_path / 'mdm.db'}")
    monkeypatch.setattr(S, "_engine", None)
    monkeypatch.setattr(S, "_SessionLocal", None)
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(M, "DATA_DIR_FOR_DEDUPE", str(tmp_path / "data"))
    monkeypatch.setenv("MDM_MODELS_DIR", str(tmp_path / "models"))
    with TestClient(M.app) as c:
        yield c


def test_v1_models_active_after_startup(client):
    for domain, rid in (("customer", "TR-V1-CUSTOMER"), ("supplier", "TR-V1-SUPPLIER")):
        r = client.get(f"/api/domains/{domain}/active-model")
        assert r.status_code == 200, r.text
        assert r.json()["training_run_id"] == rid


def test_seeding_is_idempotent(client):
    from app.db.session import get_session_factory
    from app.db.models import TrainingRun, ActiveModel
    from app.modules.dedupe_service.bundled_models import seed_bundled_models
    db = get_session_factory()()
    try:
        seed_bundled_models(db)
        seed_bundled_models(db)
        assert db.query(TrainingRun).filter(TrainingRun.dataset_id == "BUNDLED-V1").count() == 2
        assert db.query(ActiveModel).count() == 2
    finally:
        db.close()


def test_seeding_never_overrides_a_later_active_model(client):
    from app.db.session import get_session_factory
    from app.db.models import ActiveModel
    from app.modules.dedupe_service.bundled_models import seed_bundled_models
    db = get_session_factory()()
    try:
        row = db.query(ActiveModel).filter(ActiveModel.domain == "customer").first()
        row.training_run_id = "TR-SOME-V2"
        db.commit()
        seed_bundled_models(db)
        assert db.query(ActiveModel).filter(ActiveModel.domain == "customer").first().training_run_id == "TR-SOME-V2"
    finally:
        db.close()


def test_find_duplicates_customer_without_training(client):
    csv = ("customer_name,address,city,postal_code,country\n"
           "Acme Industries Inc,12 Elm Street,Dallas,75001,US\n"
           "ACME Industries Inc.,12 Elm St,Dallas,75001,US\n"
           "Zenith Foods Ltd,5 Lake Road,Leeds,LS1 4AB,GB\n"
           "Borealis Tools LLC,900 Pine Avenue,Denver,80201,US\n")
    r = client.post("/api/domains/customer/find-duplicates", files={"file": ("c.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model"]["training_run_id"] == "TR-V1-CUSTOMER"
    assert j["summary"]["duplicate_groups"] == 1 and j["summary"]["records_in_duplicate_groups"] == 2
    g = client.get(f"/api/find-duplicates/{j['dataset_id']}/{j['cluster_run']['cluster_run_id']}/groups").json()
    assert {rec["row_number"] for rec in g["groups"][0]["records"]} == {1, 2}
    e = client.get(f"/api/find-duplicates/{j['dataset_id']}/{j['cluster_run']['cluster_run_id']}/export.csv")
    assert e.status_code == 200 and "duplicate_group_id" in e.text.splitlines()[0]


def test_file_without_name_column_is_rejected_clearly(client):
    r = client.post("/api/domains/customer/find-duplicates",
                    files={"file": ("x.csv", io.BytesIO(b"foo,bar\n1,2\n3,4\n"), "text/csv")})
    assert r.status_code == 422


def test_cluster_groups_return_raw_and_standardized_values(client):
    csv = ("customer_name,address,city,postal_code,country\n"
           "  ACME industries INC ,12 Elm Street,dallas,75001,us\n"
           "Acme Industries Inc.,12 Elm St,Dallas,75001,US\n"
           "Zenith Foods Ltd,5 Lake Road,Leeds,LS1 4AB,GB\n")
    r = client.post("/api/domains/customer/find-duplicates", files={"file": ("c.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 200, r.text
    j = r.json()
    g = client.get(f"/api/find-duplicates/{j['dataset_id']}/{j['cluster_run']['cluster_run_id']}/groups").json()
    assert g["groups"], g
    first = g["groups"][0]["records"][0]
    assert "customer_name" in first["_std"] and "customer_name" in g["standardized_columns"]
    assert g["groups"][0]["match_score"] == g["groups"][0]["confidence"]


def test_retraining_eligibility_counts_only_real_steward_labels(client):
    r = client.get("/api/domains/customer/retraining/eligibility")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["eligible"] is False and j["checks"]["real_steward_labels"]["have"] == 0
    assert "never triggered automatically" in j["note"]


def test_model_comparison_gate_blocks_more_false_merges():
    from app.modules.dedupe_service.retraining import compare_models
    old = {"precision": 0.90, "recall": 0.60, "false_positives": 10}
    assert compare_models(old, {"precision": 0.95, "recall": 0.70, "false_positives": 6})["approved"]
    worse = compare_models(old, {"precision": 0.85, "recall": 0.90, "false_positives": 25})
    assert not worse["approved"] and len(worse["reasons"]) >= 2   # higher recall alone is not enough


def test_customer_threshold_comes_from_metadata(client):
    import json
    from app.modules.dedupe_service.bundled_models import BUNDLED_ROOT
    md = json.loads((BUNDLED_ROOT / "customer" / "v1" / "metadata.json").read_text())
    active = client.get("/api/domains/customer/active-model").json()
    assert active["threshold"] == md["threshold"] and active["review_threshold"] == md["review_threshold"]
