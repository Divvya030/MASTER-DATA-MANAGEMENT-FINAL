"""
Final Master Data must include DISTINCT / unmatched (single-record) entities,
and the Steward Review endpoints must be paged and free of per-row queries.
"""
import uuid

from sqlalchemy import event

from tests.test_final_mdm_flow import (  # noqa: F401  (fixtures + helpers)
    client, _cleanup_dummy_models, _setup_dataset, _db, _make_pending, CUSTOMER_CSV,
)


def _entity_with_members(db, ds_id, member_values):
    """Entity + stored member snapshots + a generated Golden Record."""
    from app.db.models import Entity, EntityMember
    from app.api.er_workflow import _generate_golden_record
    e = Entity(entity_id=f"ENT-{uuid.uuid4().hex[:10]}", domain="customer", status="ACTIVE")
    db.add(e)
    db.flush()
    for i, vals in enumerate(member_values):
        db.add(EntityMember(entity_id=e.entity_id, dataset_id=ds_id, record_id=str(uuid.uuid4().int % 10**6 + i),
                            source="BATCH", values=vals))
    db.commit()
    db.refresh(e)
    gv = _generate_golden_record(db, e, reason="INITIAL_GENERATION", created_by="system")
    return e.entity_id, gv


# ---------------------------------------------------------------- master data

def test_distinct_records_appear_in_final_master_data_without_manual_approval(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    pid = _make_pending(db, ds_id, 4, 5, values={"customer_name": "Gamma Corp", "city": "Delhi"})
    db.close()
    r = client.post(f"/api/reviews/pending/{pid}/resolve",
                    json={"decision": "REJECT_MATCH", "reviewer": "priya.s", "reason": "different companies"})
    assert r.status_code == 200, r.text

    md = client.get("/api/domains/customer/master-data").json()
    assert md["count"] == 2                      # both DISTINCT records are final master data
    assert md["distinct_record_count"] == 2 and md["merged_record_count"] == 0
    assert md["awaiting_review_count"] == 0
    assert all(rec["member_count"] == 1 for rec in md["records"])
    assert all(rec["approval_basis"] == "SINGLE_RECORD_NO_CONFLICT" for rec in md["records"])


def test_unmatched_singleton_is_in_master_data(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    _entity_with_members(db, ds_id, [{"customer_name": "Solo Ltd", "city": "Pune"}])
    db.close()
    md = client.get("/api/domains/customer/master-data").json()
    assert md["count"] == 1 and md["records"][0]["values"]["customer_name"] == "Solo Ltd"


def test_merged_entity_still_needs_approval_and_conflicts_never_final(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    merged_id, gv = _entity_with_members(db, ds_id, [
        {"customer_name": "Acme Trading", "city": "Chennai"},
        {"customer_name": "Acme Trading", "city": "Chennai"}])
    assert gv.steward_status == "PROPOSED"
    conflict_id, gv2 = _entity_with_members(db, ds_id, [
        {"customer_name": "Beta", "phone": "9000000001"},
        {"customer_name": "Beta", "phone": "9000000002"}])
    assert gv2.steward_status == "NEEDS_REVIEW"
    db.close()

    md = client.get("/api/domains/customer/master-data").json()
    assert md["count"] == 0 and md["awaiting_review_count"] == 2   # 2-record entities wait for a steward

    ok = client.post(f"/api/entities/{merged_id}/review",
                     json={"action": "APPROVE_GOLDEN_RECORD", "reviewer": "priya.s"})
    assert ok.status_code == 200
    md = client.get("/api/domains/customer/master-data").json()
    assert [r["entity_id"] for r in md["records"]] == [merged_id]
    assert md["records"][0]["approval_basis"] == "STEWARD_APPROVED"
    assert md["merged_record_count"] == 1 and md["distinct_record_count"] == 0
    assert conflict_id not in [r["entity_id"] for r in md["records"]]


def test_master_data_is_paged(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    for i in range(5):
        _entity_with_members(db, ds_id, [{"customer_name": f"Solo {i}"}])
    db.close()
    p1 = client.get("/api/domains/customer/master-data?limit=2&offset=0").json()
    p3 = client.get("/api/domains/customer/master-data?limit=2&offset=4").json()
    assert p1["count"] == 5 and p1["returned"] == 2 and p1["has_more"] is True
    assert p3["returned"] == 1 and p3["has_more"] is False
    ids = {r["entity_id"] for r in p1["records"]} | {r["entity_id"] for r in p3["records"]}
    assert len(ids) == 3


# -------------------------------------------------------------- steward review

def test_pending_reviews_are_paged_oldest_first(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    ids = [_make_pending(db, ds_id, i, i + 100, values={"customer_name": f"N{i}"}) for i in range(5)]
    db.close()
    p1 = client.get("/api/reviews/pending?domain=customer&limit=2").json()
    assert p1["count"] == 2 and p1["total"] == 5 and p1["has_more"] is True
    p3 = client.get("/api/reviews/pending?domain=customer&limit=2&offset=4").json()
    assert p3["count"] == 1 and p3["has_more"] is False
    seen = [x["pending_review_id"] for x in p1["pending_reviews"]] + \
           [x["pending_review_id"] for x in client.get("/api/reviews/pending?domain=customer&limit=2&offset=2").json()["pending_reviews"]] + \
           [x["pending_review_id"] for x in p3["pending_reviews"]]
    assert sorted(seen) == sorted(ids)


def test_pending_reviews_do_not_issue_per_row_queries(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    for i in range(20):
        ent_id, _ = _entity_with_members(db, ds_id, [{"customer_name": f"E{i}"}])
        _make_pending(db, ds_id, 1000 + i, None, entity_id=ent_id, values={"customer_name": f"C{i}"})
    db.close()

    from app.db.session import get_engine
    stmts = []
    eng = get_engine()
    listener = lambda conn, cur, stmt, *a: stmts.append(stmt)
    event.listen(eng, "before_cursor_execute", listener)
    try:
        r = client.get("/api/reviews/pending?domain=customer&limit=20").json()
    finally:
        event.remove(eng, "before_cursor_execute", listener)
    assert r["count"] == 20 and all(x["candidate_entity"]["golden_values"] for x in r["pending_reviews"])
    assert len(stmts) <= 6, f"expected a constant number of queries, got {len(stmts)}"


def test_entities_needing_review_is_paged_and_filtered(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    _entity_with_members(db, ds_id, [{"customer_name": "Clean"}])
    for i in range(3):
        _entity_with_members(db, ds_id, [{"customer_name": f"C{i}", "phone": "9000000001"},
                                         {"customer_name": f"C{i}", "phone": "9000000002"}])
    db.close()
    r = client.get("/api/reviews/entities-needing-review?domain=customer&limit=2").json()
    assert r["total"] == 3 and r["count"] == 2 and r["has_more"] is True
    assert all(e["needs_review_fields"] for e in r["entities"])


# ------------------------------------------------------------- domain schema

def test_domain_schema_endpoint_lists_every_configured_field(client):
    from app.domain_schemas import SCHEMAS
    for domain, schema in SCHEMAS.items():
        r = client.get(f"/api/domains/{domain}/schema")
        assert r.status_code == 200
        names = [f["name"] for f in r.json()["fields"]]
        assert names == schema.field_names()          # all of them, in schema order
    cust = {f["name"]: f for f in client.get("/api/domains/customer/schema").json()["fields"]}
    assert cust["email"]["input_type"] == "email" and cust["dob"]["input_type"] == "date"
    assert cust["duns"]["label"] == "D-U-N-S number" and cust["customer_id"]["identifier"] is True
    assert client.get("/api/domains/nope/schema").status_code == 404
