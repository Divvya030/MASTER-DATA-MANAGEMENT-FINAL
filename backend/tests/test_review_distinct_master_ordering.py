"""
Regression tests for:
  * field order of Golden Records (schema order, not alphabetical) + list-card summary fields
  * Final Master Data: a freshly approved record is not buried, 'awaiting' scope explains what is missing
  * steward DISTINCT / MATCH on mirrored (A-vs-B and B-vs-A) queue items and on an existing merged entity
  * what a training run takes from the labelled-pair repository (training_labels.py)
"""
import uuid

from tests.test_final_mdm_flow import (  # noqa: F401  (fixtures + helpers)
    client, _cleanup_dummy_models, _setup_dataset, _db, _make_pending, CUSTOMER_CSV,
)
from tests.test_master_data_and_review_paging import _entity_with_members


# ------------------------------------------------------------------ field order

def test_domain_schema_helpers_order_fields_like_the_schema():
    from app.domain_schemas import ordered_field_names, order_values, summary_items, display_name
    alpha = {"address": "1 Rd", "city": "Pune", "customer_id": "C1", "customer_name": "Acme", "zzz_extra": "x",
             "email": "a@x.com", "phone": ""}
    assert list(order_values("customer", alpha)) == ["customer_id", "customer_name", "email", "phone",
                                                     "address", "city", "zzz_extra"]
    assert ordered_field_names("nope", ["b", "a"]) == ["a", "b"]
    s = summary_items("customer", alpha)
    assert [i["field"] for i in s][:3] == ["customer_id", "customer_name", "email"]     # id, name first
    assert "phone" not in [i["field"] for i in s]                                         # empty skipped
    assert display_name("customer", alpha) == "Acme"
    assert display_name("customer", {"customer_id": "C9"}) == "C9"                        # falls back to the id


def test_golden_record_values_come_back_in_schema_order_with_summary(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    eid, _ = _entity_with_members(db, ds_id, [
        {"address": "1 Main Rd", "city": "Chennai", "customer_id": "C-1", "customer_name": "Acme Trading",
         "email": "acme@example.com"}])
    db.close()
    e = client.get(f"/api/entities/{eid}").json()
    assert list(e["golden_record"]["values"])[:2] == ["customer_id", "customer_name"]
    assert list(e["golden_record"]["values"]).index("address") > list(e["golden_record"]["values"]).index("email")
    assert e["display_name"] == "Acme Trading"
    assert [i["field"] for i in e["summary"]][:2] == ["customer_id", "customer_name"]
    lst = client.get("/api/entities?domain=customer").json()["entities"][0]
    assert lst["summary"] and lst["display_name"] == "Acme Trading"
    gr = client.get(f"/api/entities/{eid}/golden-record").json()
    assert list(gr["values"])[0] == "customer_id"


# ------------------------------------------------------------ final master data

def test_approved_record_is_on_first_page_even_with_many_newer_singletons(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    merged_id, _ = _entity_with_members(db, ds_id, [{"customer_name": "Old Merged"}, {"customer_name": "Old Merged"}])
    for i in range(60):
        _entity_with_members(db, ds_id, [{"customer_name": f"Solo {i}"}])
    db.close()
    assert client.post(f"/api/entities/{merged_id}/review",
                       json={"action": "APPROVE_GOLDEN_RECORD", "reviewer": "priya.s"}).status_code == 200
    page1 = client.get("/api/domains/customer/master-data?limit=50").json()
    assert page1["count"] == 61
    assert page1["records"][0]["entity_id"] == merged_id        # just approved -> first, not buried on page 2
    assert page1["records"][0]["approval_basis"] == "STEWARD_APPROVED"


def test_master_data_filters_kind_and_search(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    merged_id, _ = _entity_with_members(db, ds_id, [{"customer_name": "Acme Trading", "city": "Chennai"},
                                                    {"customer_name": "Acme Trading", "city": "Chennai"}])
    _entity_with_members(db, ds_id, [{"customer_name": "Solo Ltd", "city": "Pune"}])
    db.close()
    client.post(f"/api/entities/{merged_id}/review", json={"action": "APPROVE_GOLDEN_RECORD", "reviewer": "x"})
    base = "/api/domains/customer/master-data"
    assert client.get(f"{base}?kind=merged").json()["matching"] == 1
    assert client.get(f"{base}?kind=distinct").json()["matching"] == 1
    r = client.get(f"{base}?q=pune").json()
    assert r["matching"] == 1 and r["records"][0]["values"]["customer_name"] == "Solo Ltd"
    assert client.get(f"{base}?scope=bogus").status_code == 422


def test_awaiting_scope_lists_unfinished_entities_with_reason(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    pending_id, _ = _entity_with_members(db, ds_id, [{"customer_name": "Acme"}, {"customer_name": "Acme"}])
    conflict_id, _ = _entity_with_members(db, ds_id, [{"customer_name": "Beta", "phone": "9000000001"},
                                                      {"customer_name": "Beta", "phone": "9000000002"}])
    reapprove_id, _ = _entity_with_members(db, ds_id, [{"customer_name": "Gamma"}, {"customer_name": "Gamma"}])
    db.close()
    client.post(f"/api/entities/{reapprove_id}/review", json={"action": "APPROVE_GOLDEN_RECORD", "reviewer": "x"})
    # a steward edit creates a new, un-approved version of an entity that WAS approved
    client.post(f"/api/entities/{reapprove_id}/review",
                json={"action": "EDIT_FIELD", "reviewer": "x", "field": "city", "value": "Delhi"})

    r = client.get("/api/domains/customer/master-data?scope=awaiting").json()
    reasons = {x["entity_id"]: x["awaiting_reason"] for x in r["records"]}
    assert reasons == {pending_id: "PENDING_APPROVAL", conflict_id: "NEEDS_REVIEW", reapprove_id: "RE_APPROVAL"}
    assert r["awaiting_count"] == 3 and r["count"] == 0


# ------------------------------------------------------------ steward decisions

def test_distinct_against_unapproved_merged_entity_reports_where_the_counterpart_is(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    merged_id, _ = _entity_with_members(db, ds_id, [{"customer_name": "Acme Trading", "city": "Chennai"},
                                                    {"customer_name": "Acme Trading", "city": "Chennai"}])
    pid = _make_pending(db, ds_id, 900, None, entity_id=merged_id,
                        values={"customer_name": "Acme Traders", "city": "Chennai"})
    db.close()
    r = client.post(f"/api/reviews/pending/{pid}/resolve", json={"decision": "REJECT_MATCH", "reviewer": "priya.s"}).json()
    assert r["will_appear_in_final_master_data"] is True               # the incoming record is final
    assert r["counterpart_will_appear_in_final_master_data"] is False  # the merged one still needs approval
    assert r["other_entity_id"] == merged_id

    final = client.get("/api/domains/customer/master-data").json()
    assert [x["values"]["customer_name"] for x in final["records"]] == ["Acme Traders"]
    awaiting = client.get("/api/domains/customer/master-data?scope=awaiting").json()
    assert [x["entity_id"] for x in awaiting["records"]] == [merged_id]    # not invisible any more

    client.post(f"/api/entities/{merged_id}/review", json={"action": "APPROVE_GOLDEN_RECORD", "reviewer": "priya.s"})
    final = client.get("/api/domains/customer/master-data").json()
    assert final["count"] == 2                                          # both records are final master data


def test_mirrored_queue_items_are_resolved_together_and_make_no_duplicate_entities(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    a = _make_pending(db, ds_id, 4, 5, values={"customer_name": "Gamma Corp", "city": "Delhi"})
    _make_pending(db, ds_id, 5, 4, values={"customer_name": "Gamma Corporation", "city": "Delhi"})
    db.close()
    assert client.get("/api/reviews/pending?domain=customer").json()["total"] == 2

    r = client.post(f"/api/reviews/pending/{a}/resolve", json={"decision": "REJECT_MATCH", "reviewer": "priya.s"}).json()
    assert r["mirror_items_resolved"] == 1
    assert r["will_appear_in_final_master_data"] is True and r["counterpart_will_appear_in_final_master_data"] is True
    assert client.get("/api/reviews/pending?domain=customer").json()["total"] == 0     # twin closed too

    ents = client.get("/api/entities?domain=customer").json()["entities"]
    assert len(ents) == 2 and all(e["member_count"] == 1 for e in ents)                # no third entity for record 5
    assert client.get("/api/domains/customer/master-data").json()["count"] == 2
    labels = client.get("/api/domains/customer/labels").json()["labels"]
    assert len(labels) == 1 and labels[0]["label"] == "DISTINCT"                       # one decision, one label


def test_confirm_match_uses_survivorship_and_builds_one_entity(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    a = _make_pending(db, ds_id, 4, 5, values={"customer_name": "Gamma Corp", "city": "Delhi"})
    _make_pending(db, ds_id, 5, 4, values={"customer_name": "Gamma Corp", "city": "Delhi"})
    db.close()
    r = client.post(f"/api/reviews/pending/{a}/resolve", json={"decision": "CONFIRM_MATCH", "reviewer": "priya.s"}).json()
    assert r["mirror_items_resolved"] == 1
    ents = client.get("/api/entities?domain=customer").json()["entities"]
    assert len(ents) == 1 and ents[0]["member_count"] == 2                 # one entity holding both records
    gr = ents[0]["golden_record"]
    assert gr["reason"] == "STEWARD_CONFIRMED_MATCH" and gr["lineage"]["customer_name"]["rule"]
    # merged -> a steward still approves it before it is final master data
    assert r["will_appear_in_final_master_data"] is False
    assert client.get("/api/domains/customer/master-data").json()["count"] == 0
    assert client.get("/api/domains/customer/master-data?scope=awaiting").json()["awaiting_count"] == 1


# -------------------------------------------------- labels -> training

def _add_label(db, domain, a, b, label, values_a, values_b, strategy="STEWARD_SUPPLIED", ds="DS-X"):
    from app.modules.dedupe_service import label_repository as repo
    return repo.add_pair(db, domain=domain, dataset_id=ds, record_id_a=a, record_id_b=b, values_a=values_a,
                         values_b=values_b, strategy=strategy, label=label, labeler="priya.s")


def test_training_label_selection_and_readiness(client):
    from app.modules.dedupe_service.training_labels import select_training_labels, label_training_readiness
    _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    full = {"customer_name": "Acme", "city": "Pune"}
    for i in range(6):
        _add_label(db, "customer", f"m{i}a", f"m{i}b", "MATCH", full, full)
    for i in range(4):
        _add_label(db, "customer", f"d{i}a", f"d{i}b", "DISTINCT", full, {"customer_name": "Zed", "city": "Goa"},
                   strategy="REVIEW_DECISION")
    _add_label(db, "customer", "u1", "u2", "UNSURE", full, full)
    _add_label(db, "customer", "old1", "old2", "MATCH", {"customer_name": "Acme"}, {"customer_name": "Acme"})  # no city
    from app.modules.dedupe_service import label_repository as repo
    repo.add_pair(db, domain="customer", dataset_id="DS-X", record_id_a="p1", record_id_b="p2", values_a=full,
                  values_b=full, strategy="BOOTSTRAP")     # never labelled

    usable, excluded = select_training_labels(db, "customer", ["customer_name", "city"])
    assert len(usable) == 10 and {l["label"] for l in usable} == {"MATCH", "DISTINCT"}
    assert [e["missing_fields"] for e in excluded] == [["city"]]

    info = label_training_readiness(db, "customer", ["customer_name", "city"])
    db.close()
    assert info["repository"] == {"labelled": 12, "match": 7, "distinct": 4, "unsure_excluded": 1, "awaiting_label": 1}
    assert info["usable"]["total"] == 10 and info["usable"]["origin"] == {"steward_review": 4, "labeling_screen": 6}
    assert info["excluded_missing_match_fields_count"] == 1
    sp = info["split_preview"]
    assert sum(sp[k]["match"] + sp[k]["distinct"] for k in sp) == 10          # every usable label lands in one split
    assert info["verdict"]["threshold_can_be_calibrated"] is False and info["notes"]


def test_training_readiness_endpoint_needs_an_er_configuration(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    r = client.get(f"/api/datasets/{ds_id}/er/labeling/training-readiness")
    assert r.status_code == 422
    pairs = client.get("/api/domains/customer/labels/pairs").json()
    assert pairs["total"] == 0 and pairs["pairs"] == []


def test_labelled_pairs_browser_filters_by_origin_and_label(client):
    _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    v = {"customer_name": "A"}
    _add_label(db, "customer", "1", "2", "MATCH", v, v, strategy="REVIEW_DECISION")
    _add_label(db, "customer", "3", "4", "DISTINCT", v, v)
    db.close()
    base = "/api/domains/customer/labels/pairs"
    assert client.get(base).json()["total"] == 2
    assert client.get(f"{base}?origin=review").json()["total"] == 1
    assert client.get(f"{base}?origin=labeling").json()["pairs"][0]["label"] == "DISTINCT"
    assert client.get(f"{base}?label=MATCH").json()["total"] == 1
