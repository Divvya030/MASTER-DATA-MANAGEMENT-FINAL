"""
Survivorship inputs: source_system + last_updated.

* both are canonical (metadata) fields, so the mapping step carries them through;
* profiling never recommends them as ER match fields;
* the recency rule reads the mapped timestamp (any common date format);
* source_priority reads the mapped source system;
* the winning value's lineage records the source system and its timestamp.

Uses the bundled 500-row sample files in backend/sample_data when present.
"""
import os

import pandas as pd
import pytest

from app.modules.golden_record_service import select_golden_values, normalize_config, validate_config, _parse_ts
from tests.test_final_mdm_flow import client, _cleanup_dummy_models, _db  # noqa: F401
from tests.test_master_data_and_review_paging import _entity_with_members

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "sample_data")
CUSTOMER_FILE = os.path.join(SAMPLE, "customer_500_realistic_mdm.csv")
SUPPLIER_FILE = os.path.join(SAMPLE, "supplier_500_realistic_mdm.csv")
needs_samples = pytest.mark.skipif(not (os.path.exists(CUSTOMER_FILE) and os.path.exists(SUPPLIER_FILE)),
                                   reason="sample datasets not bundled")


def _rec(ds, rid, **values):
    return {"dataset_id": ds, "record_id": rid, "values": values}


# ------------------------------------------------------------------ timestamps

@pytest.mark.parametrize("raw", ["2026-06-02 11:03:18", "2026-06-02T11:03:18", "2026-06-02T11:03:18Z", "2026-06-02"])
def test_iso_timestamps_parse(raw):
    assert _parse_ts(raw).year == 2026


def test_regional_formats_parse_and_dayfirst_is_configurable():
    assert _parse_ts("05/01/2024").month == 1                    # day first (default): 5 January
    assert _parse_ts("05/01/2024", dayfirst=False).month == 5    # month first: 1 May
    assert _parse_ts("01-May-2024").month == 5
    assert _parse_ts("13/02/2024").month == 2                    # unambiguous even month-first
    assert _parse_ts("not a date") is None and _parse_ts(None) is None and _parse_ts("") is None


# ------------------------------------------------------------------ rules

def test_recency_decides_when_no_source_priority_configured():
    out = select_golden_values([
        _rec("d1", "1", city="Mumbai", source_system="Oracle CX", last_updated="2026-01-01 10:00:00"),
        _rec("d2", "2", city="Navi Mumbai", source_system="Salesforce CRM", last_updated="2026-06-01 10:00:00"),
    ], ["city"])
    lin = out.lineage["city"]
    assert out.golden_values["city"] == "Navi Mumbai" and lin["rule"] == "recency"
    assert lin["source_system"] == "Salesforce CRM" and lin["source_updated_at"] == "2026-06-01 10:00:00"
    assert out.needs_review == []


def test_source_priority_outranks_recency_by_default_order():
    out = select_golden_values([
        _rec("d1", "1", city="Mumbai", source_system="Oracle CX", last_updated="2026-01-01 10:00:00"),
        _rec("d2", "2", city="Navi Mumbai", source_system="Salesforce CRM", last_updated="2026-06-01 10:00:00"),
    ], ["city"], config={"source_priority": {"Oracle CX": 5, "Salesforce CRM": 1}})
    assert out.golden_values["city"] == "Mumbai" and out.lineage["city"]["rule"] == "source_priority"


def test_unparseable_timestamps_never_guess_a_winner():
    out = select_golden_values([
        _rec("d1", "1", city="A", last_updated="garbage"),
        _rec("d2", "2", city="B", last_updated="also garbage"),
    ], ["city"])
    assert out.lineage["city"]["rule"] == "UNRESOLVED_CONFLICT" and out.needs_review


def test_stored_none_recency_field_falls_back_to_default():
    assert normalize_config({"recency_field": None})["recency_field"] == "last_updated"
    assert normalize_config({"recency_field": "modified"})["recency_field"] == "modified"
    assert validate_config(normalize_config({"timestamp_dayfirst": "yes"}))   # must be boolean


# ------------------------------------------------------------------ schema / mapping / profiling on the real files

@needs_samples
@pytest.mark.parametrize("domain,path", [("customer", CUSTOMER_FILE), ("supplier", SUPPLIER_FILE)])
def test_real_files_map_and_profile_source_and_timestamp_as_metadata(client, domain, path):
    with open(path, "rb") as fh:
        r = client.post("/api/datasets/upload", data={"domain": domain}, files={"file": ("d.csv", fh.read(), "text/csv")})
    assert r.status_code == 200, r.text
    ds_id = r.json()["dataset_id"]
    sugg = client.get(f"/api/datasets/{ds_id}/mapping/suggestions").json()["suggestions"]
    assert sugg["source_system"][0]["source_column"] == "source_system"
    assert sugg["last_updated"][0]["source_column"] == "last_updated"
    mappings = [{"canonical_field": cf, "source_column": c[0]["source_column"]} for cf, c in sugg.items() if c]
    assert client.post(f"/api/datasets/{ds_id}/mapping", json={"mappings": mappings}).status_code == 200
    assert client.post(f"/api/datasets/{ds_id}/profile").status_code == 200

    prof = client.get(f"/api/datasets/{ds_id}/profile").json()
    rows = prof["profiles"] if "profiles" in prof else prof
    by_field = {p["canonical_field"]: p for p in (rows if isinstance(rows, list) else rows.values())}
    for f in ("source_system", "last_updated"):
        assert by_field[f]["er_recommendation"] == "LOW_VALUE_FOR_ER", f      # never an ER match field

    assert client.post(f"/api/datasets/{ds_id}/standardize").status_code == 200
    det = client.get(f"/api/domains/{domain}/survivorship/detected").json()
    assert len(det["source_systems"]) == 3 and sum(s["records"] for s in det["source_systems"]) == 500
    assert det["datasets"][0]["source_system_mapped"] and det["datasets"][0]["recency_mapped"]
    assert det["datasets"][0]["timestamps_parsed_pct"] == 100.0

    schema = client.get(f"/api/domains/{domain}/schema").json()["fields"]
    assert {f["name"] for f in schema if f["metadata"]} == {"source_system", "last_updated"}


def test_recency_field_must_be_a_canonical_field(client):
    r = client.put("/api/domains/customer/survivorship", json={"recency_field": "not_a_field"})
    assert r.status_code == 422
    ok = client.put("/api/domains/customer/survivorship", json={"recency_field": "last_updated", "timestamp_dayfirst": False})
    assert ok.status_code == 200 and ok.json()["config"]["timestamp_dayfirst"] is False


# ------------------------------------------------------------------ golden record end to end (member snapshots)

def test_golden_record_uses_recency_and_keeps_metadata_out_of_golden_values(client):
    from tests.test_final_mdm_flow import _setup_dataset, CUSTOMER_CSV
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    _, gv = _entity_with_members(db, ds_id, [
        {"customer_name": "Rohan Nair", "city": "Mumbai", "source_system": "Oracle CX", "last_updated": "2026-01-25 09:01:54"},
        {"customer_name": "Rohan Nair", "city": "Pune", "source_system": "Salesforce CRM", "last_updated": "2026-08-30 17:20:00"},
    ])
    db.close()
    assert gv.golden_values["city"] == "Pune"
    assert gv.lineage["city"]["rule"] == "recency" and gv.lineage["city"]["source_system"] == "Salesforce CRM"
    assert gv.lineage["city"]["source_updated_at"] == "2026-08-30 17:20:00"
    assert "source_system" not in gv.golden_values and "last_updated" not in gv.golden_values
    assert gv.conflicts["city"][0]["value"] == "Mumbai" and gv.conflicts["city"][0]["source_system"] == "Oracle CX"
