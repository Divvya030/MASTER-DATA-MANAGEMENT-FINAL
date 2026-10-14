import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Reset module-level singletons/state in place rather than importlib.reload(),
    # which would create new Exception classes and break isinstance checks against
    # references imported earlier in other test modules.
    import app.db.session as session_mod
    import app.modules.ingestion as ing
    import app.main as main_mod

    monkeypatch.setattr(session_mod, "DB_URL", f"sqlite:///{tmp_path}/mdm.db")
    monkeypatch.setattr(session_mod, "_engine", None)
    monkeypatch.setattr(session_mod, "_SessionLocal", None)
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    os.makedirs(str(tmp_path), exist_ok=True)

    return TestClient(main_mod.app)


CUSTOMER_CSV = "/mnt/user-data/uploads/customer.csv"
SUPPLIER_XLSX = "/mnt/user-data/uploads/supplier.xlsx"
SUPPLIER_CSV_1 = "/mnt/user-data/uploads/Supplier_1_.csv"


def _run_full_pipeline(client, filepath, domain, filename):
    with open(filepath, "rb") as f:
        r = client.post("/api/datasets/upload", data={"domain": domain},
                         files={"file": (filename, f, "application/octet-stream")})
    assert r.status_code == 200, r.text
    ds_id = r.json()["dataset_id"]

    r = client.get(f"/api/datasets/{ds_id}/mapping/suggestions")
    assert r.status_code == 200
    sugg = r.json()

    mappings = [{"canonical_field": cf, "source_column": cands[0]["source_column"]}
                for cf, cands in sugg["suggestions"].items() if cands]
    r = client.post(f"/api/datasets/{ds_id}/mapping", json={"mappings": mappings})
    assert r.status_code == 200, r.text

    r = client.post(f"/api/datasets/{ds_id}/profile")
    assert r.status_code == 200, r.text

    r = client.post(f"/api/datasets/{ds_id}/standardize")
    assert r.status_code == 200, r.text

    r = client.post(f"/api/datasets/{ds_id}/reprofile")
    assert r.status_code == 200, r.text

    r = client.get(f"/api/datasets/{ds_id}/er-config/suggestions")
    assert r.status_code == 200, r.text
    assignments = [{"canonical_field": cf, "role": role}
                   for cf, v in r.json()["fields"].items() for role in v["suggested_roles"]]
    r = client.post(f"/api/datasets/{ds_id}/er-config", json={"assignments": assignments})
    assert r.status_code == 200, r.text

    r = client.get(f"/api/datasets/{ds_id}/readiness")
    assert r.status_code == 200, r.text
    return ds_id, r.json()


@pytest.mark.skipif(not os.path.exists(CUSTOMER_CSV), reason="real customer.csv not available")
def test_full_pipeline_real_customer_data(client):
    ds_id, result = _run_full_pipeline(client, CUSTOMER_CSV, "customer", "customer.csv")
    assert result["readiness"]["row_count"] == 9996
    assert "dedupe_handoff_configuration" in result

    r = client.get(f"/api/datasets/{ds_id}")
    assert r.json()["status"] in ("READY_FOR_DEDUPE", "ER_CONFIGURED")


@pytest.mark.skipif(not os.path.exists(SUPPLIER_XLSX), reason="real supplier.xlsx not available")
def test_full_pipeline_real_supplier_data(client):
    ds_id, result = _run_full_pipeline(client, SUPPLIER_XLSX, "supplier", "supplier.xlsx")
    assert result["readiness"]["row_count"] == 417


def test_raw_remains_byte_for_byte_unchanged_through_full_pipeline(client):
    content = b"customer_name,email\n  ACME LTD  ,JOHN@EXAMPLE.COM\nBOB INC,bob@x.com\n"
    r = client.post("/api/datasets/upload", data={"domain": "customer"},
                     files={"file": ("t.csv", content, "text/csv")})
    ds_id = r.json()["dataset_id"]

    import app.modules.ingestion as ing
    raw_before = ing.load_raw(ds_id).copy(deep=True)

    mappings = [{"canonical_field": "customer_name", "source_column": "customer_name"},
                {"canonical_field": "email", "source_column": "email"}]
    client.post(f"/api/datasets/{ds_id}/mapping", json={"mappings": mappings})
    client.post(f"/api/datasets/{ds_id}/profile")
    client.post(f"/api/datasets/{ds_id}/standardize")
    client.post(f"/api/datasets/{ds_id}/reprofile")

    raw_after = ing.load_raw(ds_id)
    assert raw_before.equals(raw_after)  # byte-for-byte (value-for-value) unchanged

    std = ing.load_layer(ds_id, "standardized")
    assert std["email"].iloc[0] == "john@example.com"  # standardization actually happened
    assert std["customer_name"].iloc[0] == "Acme Ltd"


def test_quality_comparison_includes_signal_audit_report(client):
    content = b"supplier_name,email\nhorizon llp,JOHN@EXAMPLE.COM\nHorizon LLP,jane@example.com\n"
    r = client.post("/api/datasets/upload", data={"domain": "supplier"},
                     files={"file": ("t.csv", content, "text/csv")})
    ds_id = r.json()["dataset_id"]
    mappings = [{"canonical_field": "supplier_name", "source_column": "supplier_name"},
                {"canonical_field": "email", "source_column": "email"}]
    client.post(f"/api/datasets/{ds_id}/mapping", json={"mappings": mappings})
    client.post(f"/api/datasets/{ds_id}/profile")
    client.post(f"/api/datasets/{ds_id}/standardize")
    client.post(f"/api/datasets/{ds_id}/reprofile")

    r = client.get(f"/api/datasets/{ds_id}/quality-comparison")
    assert r.status_code == 200
    body = r.json()
    assert "signal_audit_report" in body
    assert "comparisons" in body  # existing field preserved for backward compatibility
    entries = body["signal_audit_report"]
    assert any(e["field"] == "supplier_name" and e["signal"] == "CASE_VARIATION" for e in entries)
    for e in entries:
        for key in ("field", "signal", "detected_before", "changed",
                    "fixed_count", "remaining_count", "status", "explanation"):
            assert key in e


def test_reject_unknown_domain(client):
    r = client.post("/api/datasets/upload", data={"domain": "not_a_domain"},
                     files={"file": ("t.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400


def test_mapping_rejects_bad_payload(client):
    r = client.post("/api/datasets/upload", data={"domain": "customer"},
                     files={"file": ("t.csv", b"customer_name,email\nBob,b@x.com\n", "text/csv")})
    ds_id = r.json()["dataset_id"]
    r = client.post(f"/api/datasets/{ds_id}/mapping",
                     json={"mappings": [{"canonical_field": "customer_name", "source_column": "nope"}]})
    assert r.status_code == 422


def test_profile_requires_mapping_first(client):
    r = client.post("/api/datasets/upload", data={"domain": "customer"},
                     files={"file": ("t.csv", b"customer_name,email\nBob,b@x.com\n", "text/csv")})
    ds_id = r.json()["dataset_id"]
    r = client.post(f"/api/datasets/{ds_id}/profile")
    assert r.status_code == 422


def test_dataset_not_found_returns_404(client):
    r = client.get("/api/datasets/DS-9999-9999")
    assert r.status_code == 404


def test_low_cardinality_geo_fields_get_match_not_ignore_on_real_data(client):
    # Item-3 real-data regression: city/state/country/postal_code/website
    # must be suggested as MATCH, not excluded as LOW_VALUE_FOR_ER, purely
    # because they have few distinct categories relative to row count.
    if not os.path.exists(SUPPLIER_CSV_1):
        pytest.skip("real Supplier_1_.csv not available")
    with open(SUPPLIER_CSV_1, "rb") as f:
        r = client.post("/api/datasets/upload", data={"domain": "supplier"},
                         files={"file": ("Supplier_1_.csv", f, "text/csv")})
    ds_id = r.json()["dataset_id"]
    sugg = client.get(f"/api/datasets/{ds_id}/mapping/suggestions").json()
    mappings = [{"canonical_field": cf, "source_column": cands[0]["source_column"]}
                for cf, cands in sugg["suggestions"].items() if cands]
    client.post(f"/api/datasets/{ds_id}/mapping", json={"mappings": mappings})
    client.post(f"/api/datasets/{ds_id}/profile")
    roles = client.get(f"/api/datasets/{ds_id}/er-config/suggestions").json()["fields"]
    for f in ("city", "state", "country", "postal_code", "website"):
        assert "MATCH" in roles[f]["suggested_roles"], f"{f} was excluded from MATCH: {roles[f]}"


# ---------------- Dataset deletion ----------------

def _upload_and_fully_configure(client, content=b"customer_name,email\nAcme Ltd,a@x.com\nBob Inc,b@x.com\n"):
    r = client.post("/api/datasets/upload", data={"domain": "customer"},
                     files={"file": ("t.csv", content, "text/csv")})
    ds_id = r.json()["dataset_id"]
    mappings = [{"canonical_field": "customer_name", "source_column": "customer_name"},
                {"canonical_field": "email", "source_column": "email"}]
    client.post(f"/api/datasets/{ds_id}/mapping", json={"mappings": mappings})
    client.post(f"/api/datasets/{ds_id}/profile")
    client.post(f"/api/datasets/{ds_id}/standardize")
    client.post(f"/api/datasets/{ds_id}/reprofile")
    client.post(f"/api/datasets/{ds_id}/er-config",
                json={"assignments": [{"canonical_field": "email", "role": "MATCH"}]})
    return ds_id


def test_delete_dataset_success(client):
    ds_id = _upload_and_fully_configure(client)
    r = client.delete(f"/api/datasets/{ds_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == ds_id
    assert body["database_status"] == "deleted"
    assert body["filesystem_status"] == "removed"

    # Dataset is genuinely gone -- not just marked, actually absent.
    r2 = client.get(f"/api/datasets/{ds_id}")
    assert r2.status_code == 404
    all_ds = client.get("/api/datasets").json()["datasets"]
    assert ds_id not in [d["dataset_id"] for d in all_ds]


def test_delete_dataset_does_not_exist_returns_404(client):
    r = client.delete("/api/datasets/DS-9999-9999")
    assert r.status_code == 404


def test_delete_removes_raw_file_from_disk(client, tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    ds_id = _upload_and_fully_configure(client)
    raw_dir = os.path.join(ing.DATA_DIR, ds_id)
    assert os.path.isdir(raw_dir)
    assert os.path.exists(os.path.join(raw_dir, "raw.parquet"))

    client.delete(f"/api/datasets/{ds_id}")
    assert not os.path.isdir(raw_dir)


def test_delete_removes_standardized_file_from_disk(client):
    import app.modules.ingestion as ing
    ds_id = _upload_and_fully_configure(client)
    std_path = os.path.join(ing.DATA_DIR, ds_id, "standardized.parquet")
    assert os.path.exists(std_path)  # confirm it was actually created first

    client.delete(f"/api/datasets/{ds_id}")
    assert not os.path.exists(std_path)


def test_delete_cleans_up_related_database_records(client):
    from app.db.session import get_session_factory
    from app.db.models import FieldMapping, Profile, ERFieldConfig

    ds_id = _upload_and_fully_configure(client)
    db = get_session_factory()()
    try:
        assert db.query(FieldMapping).filter(FieldMapping.dataset_id == ds_id).count() > 0
        assert db.query(Profile).filter(Profile.dataset_id == ds_id).count() > 0
        assert db.query(ERFieldConfig).filter(ERFieldConfig.dataset_id == ds_id).count() > 0
    finally:
        db.close()

    client.delete(f"/api/datasets/{ds_id}")

    db = get_session_factory()()
    try:
        assert db.query(FieldMapping).filter(FieldMapping.dataset_id == ds_id).count() == 0
        assert db.query(Profile).filter(Profile.dataset_id == ds_id).count() == 0
        assert db.query(ERFieldConfig).filter(ERFieldConfig.dataset_id == ds_id).count() == 0
    finally:
        db.close()


def test_delete_is_idempotent_second_call_returns_404_not_crash(client):
    ds_id = _upload_and_fully_configure(client)
    r1 = client.delete(f"/api/datasets/{ds_id}")
    assert r1.status_code == 200
    r2 = client.delete(f"/api/datasets/{ds_id}")
    assert r2.status_code == 404  # genuinely gone -- not a crash, not a fake second success


def test_delete_reports_already_absent_filesystem_gracefully(client):
    # Simulates a dataset whose DB row exists but files were already removed
    # out-of-band (e.g. manual cleanup) -- deletion must not crash on this.
    import app.modules.ingestion as ing
    ds_id = _upload_and_fully_configure(client)
    raw_dir = os.path.join(ing.DATA_DIR, ds_id)
    import shutil
    shutil.rmtree(raw_dir)  # simulate files already gone

    r = client.delete(f"/api/datasets/{ds_id}")
    assert r.status_code == 200
    assert r.json()["filesystem_status"] == "already_absent"
    assert r.json()["database_status"] == "deleted"


def test_delete_is_based_on_dataset_id_not_filename(client):
    # Two datasets uploaded with the SAME filename but different content must
    # be independently deletable -- deletion must never match by filename.
    r1 = client.post("/api/datasets/upload", data={"domain": "customer"},
                      files={"file": ("shared_name.csv", b"customer_name,email\nBob,b@x.com\n", "text/csv")})
    r2 = client.post("/api/datasets/upload", data={"domain": "customer", "on_duplicate": "allow_duplicate"},
                      files={"file": ("shared_name.csv", b"customer_name,email\nAlice,a@x.com\n", "text/csv")})
    id1, id2 = r1.json()["dataset_id"], r2.json()["dataset_id"]
    assert id1 != id2

    client.delete(f"/api/datasets/{id1}")
    assert client.get(f"/api/datasets/{id1}").status_code == 404
    assert client.get(f"/api/datasets/{id2}").status_code == 200  # untouched


# ---------------- Item #1: SHA-256 duplicate-upload API behavior ----------------

def test_upload_default_ask_returns_409_on_exact_duplicate(client):
    content = b"customer_name,email\nBob,b@x.com\n"
    r1 = client.post("/api/datasets/upload", data={"domain": "customer"},
                      files={"file": ("first.csv", content, "text/csv")})
    assert r1.status_code == 200
    first_id = r1.json()["dataset_id"]

    # Same bytes, different filename, default on_duplicate="ask"
    r2 = client.post("/api/datasets/upload", data={"domain": "customer"},
                      files={"file": ("second.csv", content, "text/csv")})
    assert r2.status_code == 409
    body = r2.json()["detail"]
    assert body["error"] == "duplicate_file"
    existing_ids = [d["dataset_id"] for d in body["existing_datasets"]]
    assert first_id in existing_ids

    # No second dataset should have been created
    all_ds = client.get("/api/datasets").json()["datasets"]
    assert len([d for d in all_ds if d["dataset_id"] == first_id]) == 1


def test_upload_use_existing_returns_existing_dataset_without_creating_new(client):
    content = b"customer_name,email\nBob,b@x.com\n"
    r1 = client.post("/api/datasets/upload", data={"domain": "customer"},
                      files={"file": ("first.csv", content, "text/csv")})
    first_id = r1.json()["dataset_id"]

    r2 = client.post("/api/datasets/upload", data={"domain": "customer", "on_duplicate": "use_existing"},
                      files={"file": ("second.csv", content, "text/csv")})
    assert r2.status_code == 200
    assert r2.json()["dataset_id"] == first_id

    all_ds = client.get("/api/datasets").json()["datasets"]
    assert len(all_ds) == 1  # still only one dataset


def test_upload_allow_duplicate_creates_new_dataset(client):
    content = b"customer_name,email\nBob,b@x.com\n"
    r1 = client.post("/api/datasets/upload", data={"domain": "customer"},
                      files={"file": ("first.csv", content, "text/csv")})
    first_id = r1.json()["dataset_id"]

    r2 = client.post("/api/datasets/upload", data={"domain": "customer", "on_duplicate": "allow_duplicate"},
                      files={"file": ("second.csv", content, "text/csv")})
    assert r2.status_code == 200
    second_id = r2.json()["dataset_id"]
    assert second_id != first_id
    assert r2.json()["sha256"] == r1.json()["sha256"]  # same fingerprint, both datasets kept

    all_ds = client.get("/api/datasets").json()["datasets"]
    assert len(all_ds) == 2


def test_upload_replace_reports_unsupported_rather_than_faking_it(client):
    content = b"customer_name,email\nBob,b@x.com\n"
    client.post("/api/datasets/upload", data={"domain": "customer"},
                files={"file": ("first.csv", content, "text/csv")})
    r2 = client.post("/api/datasets/upload", data={"domain": "customer", "on_duplicate": "replace"},
                      files={"file": ("second.csv", content, "text/csv")})
    assert r2.status_code == 501
    assert r2.json()["detail"]["error"] == "replace_not_supported"


def test_upload_different_content_same_filename_is_not_a_duplicate(client):
    r1 = client.post("/api/datasets/upload", data={"domain": "customer"},
                      files={"file": ("t.csv", b"customer_name,email\nBob,b@x.com\n", "text/csv")})
    r2 = client.post("/api/datasets/upload", data={"domain": "customer"},
                      files={"file": ("t.csv", b"customer_name,email\nAlice,a@x.com\n", "text/csv")})
    assert r1.status_code == 200
    assert r2.status_code == 200  # not flagged as duplicate -- different content
    assert r1.json()["sha256"] != r2.json()["sha256"]


def test_upload_response_includes_sha256(client):
    r = client.post("/api/datasets/upload", data={"domain": "customer"},
                     files={"file": ("t.csv", b"customer_name,email\nBob,b@x.com\n", "text/csv")})
    assert r.json()["sha256"] is not None
    assert len(r.json()["sha256"]) == 64  # SHA-256 hex digest length
