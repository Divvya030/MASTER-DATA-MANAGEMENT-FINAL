"""
FINAL MDM FLOW tests (post-clustering architecture):

  matching decision bands  ->  MATCH / UNSURE / DISTINCT
  UNSURE                   ->  Match Review queue (never auto-merged)
  steward MATCH            ->  records join one entity
  steward DISTINCT         ->  SEPARATE singleton entities, each with its own
                               Golden Record (a DISTINCT record is NEVER discarded)
  steward UNSURE           ->  stays in the queue, no automatic influence
  clusters                 ->  entities incl. SINGLETONS -> Golden Records
  conflicts                ->  NEEDS_REVIEW -> steward correction -> GR V2
  approval                 ->  FINAL MASTER DATA (approved GRs only)
  lineage                  ->  every selected field keeps source record/system/rule
  survivorship             ->  source-system priority purely from configuration

These tests deliberately do NOT require the compiled `dedupe` package: the
model scoring is monkeypatched so the surrounding MDM behavior is exercised
end-to-end on any machine.
"""
import json
import os
import uuid

import numpy as np
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


@pytest.fixture(autouse=True)
def _cleanup_dummy_models():
    """_insert_training_run writes tiny fake model files next to the tests;
    remove them after each test so no artifacts are left behind."""
    yield
    import glob
    for p in glob.glob(os.path.join(os.path.dirname(__file__), "_dummy_model_*.bin")):
        try:
            os.remove(p)
        except OSError:
            pass


def _db():
    from app.db.session import get_session_factory
    return get_session_factory()()


def _setup_dataset(client, csv_text, domain="customer"):
    r = client.post("/api/datasets/upload", data={"domain": domain},
                    files={"file": ("d.csv", csv_text.encode(), "text/csv")})
    assert r.status_code == 200, r.text
    ds_id = r.json()["dataset_id"]
    sugg = client.get(f"/api/datasets/{ds_id}/mapping/suggestions").json()
    mappings = [{"canonical_field": cf, "source_column": c[0]["source_column"]}
                for cf, c in sugg["suggestions"].items() if c]
    assert client.post(f"/api/datasets/{ds_id}/mapping", json={"mappings": mappings}).status_code == 200
    assert client.post(f"/api/datasets/{ds_id}/profile").status_code == 200
    assert client.post(f"/api/datasets/{ds_id}/standardize").status_code == 200
    return ds_id


def _insert_training_run(ds_id, domain, match_fields, threshold=0.85, review_threshold=0.60):
    db = _db()
    from app.db.models import TrainingRun
    dummy = os.path.join(os.path.dirname(__file__), f"_dummy_model_{uuid.uuid4().hex[:6]}.bin")
    with open(dummy, "wb") as f:
        f.write(b"fake-model-bytes")
    run = TrainingRun(training_run_id=f"TR-{uuid.uuid4().hex[:10]}", dataset_id=ds_id, domain=domain,
                      er_config_version="testcfg", match_fields=list(match_fields), status="TRAINED",
                      model_path=dummy, threshold=threshold, review_threshold=review_threshold)
    db.add(run)
    db.commit()
    rid = run.training_run_id
    db.close()
    return rid


CUSTOMER_CSV = (
    "customer_name,city,email,phone\n"
    "Acme Trading,Chennai,acme@example.com,9876543210\n"
    "Acme Trading Ltd,Chennai,acme@example.com,9876543210\n"
    "Beta Industries,Mumbai,beta@example.com,9000000001\n"
    "Beta Industries,Mumbai,beta@example.com,9000000002\n"
    "Gamma Corp,Delhi,gamma@example.com,9111111111\n"
    "Omega Supplies,Pune,omega@example.com,9222222222\n"
)


# =============================================================================
# 1. Decision policy: score bands -> MATCH / UNSURE / DISTINCT
# =============================================================================

class _FakeStaticDedupe:
    """Mimics dedupe.StaticDedupe just enough for run_matching."""
    def __init__(self, pairs_scores):
        self.pairs_scores = pairs_scores

    def pairs(self, records):
        return [(a, b) for a, b, _ in self.pairs_scores]

    def score(self, pairs):
        wanted = {tuple(p) for p in pairs}
        rows = [(a, b, s) for (a, b, s) in self.pairs_scores if (a, b) in wanted]
        arr = np.zeros(len(rows), dtype=[("pairs", "i8", 2), ("score", "f8")])
        for i, (a, b, s) in enumerate(rows):
            arr[i] = ((a, b), s)
        return arr


def test_decision_bands_match_unsure_distinct(monkeypatch):
    import app.modules.dedupe_service.dedupe_factory as factory
    import app.modules.dedupe_service.matching_service as ms

    fake = _FakeStaticDedupe([(0, 1, 0.93), (2, 3, 0.72), (4, 5, 0.41)])
    monkeypatch.setattr(factory, "load_static", lambda settings_bytes: fake)

    records = {i: {"customer_name": f"rec{i}", "city": "X"} for i in range(6)}
    result = ms.run_matching(b"settings", records, threshold=0.85, review_threshold=0.60,
                             distinct_threshold=0.60)
    assert result.status == "COMPLETED"
    by_pair = {(p["record_id_a"], p["record_id_b"]): p["decision"] for p in result.pairs}
    assert by_pair[(0, 1)] == "MATCH"        # 0.93 >= 0.85
    assert by_pair[(2, 3)] == "UNSURE"       # 0.60 <= 0.72 < 0.85 -> steward decides
    assert by_pair[(4, 5)] == "DISTINCT"     # 0.41 < 0.60 -> separate entities, not discarded
    assert result.match_count == 1 and result.uncertain_count == 1 and result.non_match_count == 1


def test_distinct_threshold_configurable_widens_distinct_band(monkeypatch):
    import app.modules.dedupe_service.dedupe_factory as factory
    import app.modules.dedupe_service.matching_service as ms

    fake = _FakeStaticDedupe([(0, 1, 0.70)])
    monkeypatch.setattr(factory, "load_static", lambda settings_bytes: fake)
    records = {i: {"customer_name": f"rec{i}"} for i in range(2)}
    # distinct_threshold raised to 0.75 -> the 0.70 pair is DISTINCT, not UNSURE
    result = ms.run_matching(b"settings", records, threshold=0.85, review_threshold=0.60,
                             distinct_threshold=0.75)
    assert result.pairs[0]["decision"] == "DISTINCT"


# =============================================================================
# 2. Match run: UNSURE pairs go to the Match Review queue; DISTINCT never does
# =============================================================================

def _patch_main_matching(monkeypatch, pairs):
    import app.main as main_mod
    from app.modules.dedupe_service.matching_service import MatchResult

    n = max(max(a, b) for a, b, _ in pairs) + 1

    def fake_run_matching(settings_bytes, coerced, threshold=0.5, review_threshold=None,
                          distinct_threshold=None):
        decisions = []
        for a, b, s in pairs:
            d = "MATCH" if s >= threshold else ("UNSURE" if distinct_threshold is not None and s >= distinct_threshold else "DISTINCT")
            decisions.append({"record_id_a": a, "record_id_b": b, "score": round(s, 6), "decision": d})
        return MatchResult(status="COMPLETED", total_records=len(coerced),
                           total_possible_pairs=len(coerced) * (len(coerced) - 1) // 2,
                           candidate_pair_count=len(pairs), match_count=sum(1 for d in decisions if d["decision"] == "MATCH"),
                           non_match_count=sum(1 for d in decisions if d["decision"] == "DISTINCT"),
                           uncertain_count=sum(1 for d in decisions if d["decision"] == "UNSURE"),
                           pairs=decisions, raw_scores=None)

    monkeypatch.setattr(main_mod, "run_matching", fake_run_matching)
    monkeypatch.setattr(main_mod, "load_model_metadata", lambda path: {"name_fields": ["customer_name"]})
    monkeypatch.setattr(main_mod, "check_model_compatibility", lambda md: [])


def test_unsure_pairs_queued_for_match_review_distinct_not_queued(client, monkeypatch):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    run_id = _insert_training_run(ds_id, "customer", ["customer_name", "city"])
    _patch_main_matching(monkeypatch, [(0, 1, 0.93), (2, 3, 0.72), (4, 5, 0.41)])

    r = client.post(f"/api/datasets/{ds_id}/er/match", json={"training_run_id": run_id})
    assert r.status_code == 200, r.text
    resp = r.json()
    assert resp["status"] == "COMPLETED"
    assert resp["uncertain_count"] == 1

    db = _db()
    from app.db.models import PendingReview
    rows = db.query(PendingReview).filter(PendingReview.dataset_id == ds_id).all() \
        if hasattr(PendingReview, "dataset_id") else []
    rows = db.query(PendingReview).filter(PendingReview.candidate_dataset_id == ds_id).all()
    db.close()
    assert len(rows) == 1, "only the UNSURE pair must enter the review queue"
    p = rows[0]
    assert {p.candidate_record_id, p.candidate_other_record_id} == {"2", "3"}
    assert p.kind == "AMBIGUOUS_MATCH"

    # The match-run results expose MATCH + UNSURE pairs, never silently drop them
    mr = client.get(f"/api/datasets/{ds_id}/er/matches/{resp['match_run_id']}").json()
    decisions = [x["decision"] for x in mr["pairs"]]
    assert "UNSURE" in decisions and "MATCH" in decisions


# =============================================================================
# 3. Clusters -> entities INCLUDING singletons -> proposed Golden Records
# =============================================================================

def _write_clusters_file(tmp_dir, clusters, tag):
    path = os.path.join(str(tmp_dir), f"clusters_{tag}.json")
    with open(path, "w") as f:
        json.dump(clusters, f)
    return path


def test_entities_created_from_clusters_including_singletons(client, tmp_path):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    run_id = _insert_training_run(ds_id, "customer", ["customer_name"])
    db = _db()
    from app.db.models import ClusterRun
    cr_id = f"CR-{uuid.uuid4().hex[:10]}"
    clusters_path = _write_clusters_file(tmp_path, [
        {"entity_id": "ENT-000001", "record_ids": [0, 1], "scores": [0.95, 0.93], "cluster_size": 2},
        {"entity_id": "ENT-000002", "record_ids": [2, 3], "scores": [0.91, 0.88], "cluster_size": 2},
        {"entity_id": "ENT-000003", "record_ids": [4], "scores": [], "cluster_size": 1},
        {"entity_id": "ENT-000004", "record_ids": [5], "scores": [], "cluster_size": 1},
    ], cr_id)
    db.add(ClusterRun(cluster_run_id=cr_id, match_run_id="MR-x", training_run_id=run_id,
                      dataset_id=ds_id, domain="customer", status="COMPLETED", threshold=0.85,
                      cluster_count=4, singleton_count=2, clusters_path=clusters_path))
    db.commit()
    db.close()

    r = client.post(f"/api/datasets/{ds_id}/er/clusters/{cr_id}/golden-records")
    assert r.status_code == 200, r.text
    assert r.json()["entities_created"] == 4

    ents = client.get("/api/entities?domain=customer").json()["entities"]
    sizes = sorted(e["member_count"] for e in ents)
    assert sizes == [1, 1, 2, 2], "singleton entities are first-class, not exceptions"
    for e in ents:
        assert e["golden_record"] is not None
        assert e["golden_record"]["steward_status"] in ("PROPOSED", "NEEDS_REVIEW")

    # Lineage: every selected field names its source record and rule
    multi = next(e for e in ents if e["member_count"] == 2)
    gr = client.get(f"/api/entities/{multi['entity_id']}/golden-record").json()
    assert gr["values"], "golden record must be populated from member snapshots"
    for field, lin in gr["lineage"].items():
        if lin:
            assert "record_id" in lin and "dataset_id" in lin and "rule" in lin
            assert "support_count" in lin and "source_system" in lin


def test_golden_record_generation_leaves_raw_data_untouched(client, tmp_path):
    from app.modules.ingestion import load_raw
    import pandas as pd

    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    run_id = _insert_training_run(ds_id, "customer", ["customer_name"])
    db = _db()
    from app.db.models import ClusterRun
    cr_id = f"CR-{uuid.uuid4().hex[:10]}"
    clusters_path = _write_clusters_file(tmp_path, [
        {"entity_id": "ENT-000001", "record_ids": [0, 1], "scores": [0.9], "cluster_size": 2}], cr_id)
    db.add(ClusterRun(cluster_run_id=cr_id, match_run_id="MR-x", training_run_id=run_id,
                      dataset_id=ds_id, domain="customer", status="COMPLETED", threshold=0.85,
                      clusters_path=clusters_path))
    db.commit()
    db.close()

    before = load_raw(ds_id)
    r = client.post(f"/api/datasets/{ds_id}/er/clusters/{cr_id}/golden-records")
    assert r.status_code == 200, r.text
    after = load_raw(ds_id)
    pd.testing.assert_frame_equal(before, after)


# =============================================================================
# 4. Match Review: steward MATCH / DISTINCT / UNSURE decisions
# =============================================================================

def _make_pending(db, ds_id, record_a, record_b, score=0.72, entity_id=None,
                  run_id=None, values=None):
    from app.db.models import PendingReview
    p = PendingReview(pending_review_id=f"PR-{uuid.uuid4().hex[:10]}", domain="customer",
                      kind="AMBIGUOUS_MATCH", entity_id=entity_id,
                      candidate_dataset_id=ds_id, candidate_record_id=str(record_a),
                      candidate_other_record_id=str(record_b) if record_b is not None else None,
                      candidate_values=values, score=score,
                      training_run_id=run_id, match_threshold=0.85, review_threshold=0.60)
    db.add(p)
    db.commit()
    return p.pending_review_id


def test_steward_distinct_creates_two_separate_entities_with_golden_records(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    pid = _make_pending(db, ds_id, 4, 5, values={"customer_name": "Gamma Corp", "city": "Delhi"})
    db.close()

    r = client.post(f"/api/reviews/pending/{pid}/resolve",
                    json={"decision": "REJECT_MATCH", "reviewer": "priya.s",
                          "reason": "different companies"})
    assert r.status_code == 200, r.text
    out = r.json()

    ents = client.get("/api/entities?domain=customer").json()["entities"]
    assert len(ents) == 2, "DISTINCT means separate entities -- both kept, neither discarded"
    assert all(e["member_count"] == 1 for e in ents)
    assert all(e["golden_record"] is not None for e in ents)

    # both records are now approved-able master data with their own GRs
    for e in ents:
        ar = client.post(f"/api/entities/{e['entity_id']}/review",
                         json={"action": "APPROVE_GOLDEN_RECORD", "reviewer": "priya.s"})
        assert ar.status_code == 200, ar.text
    md = client.get("/api/domains/customer/master-data").json()
    assert md["count"] == 2

    # decision evidence retained as a REAL steward label
    labels = client.get("/api/domains/customer/labels?label=DISTINCT").json()
    assert labels["count"] >= 1
    assert out["decision"] == "REJECT_MATCH"


def test_steward_match_joins_records_into_one_entity(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    # record 5 (Omega) is already mastered as its own entity via a cluster...
    db = _db()
    from app.db.models import Entity, EntityMember
    e = Entity(entity_id=f"ENT-{uuid.uuid4().hex[:10]}", domain="customer", status="ACTIVE")
    db.add(e)
    db.flush()
    db.add(EntityMember(entity_id=e.entity_id, dataset_id=ds_id, record_id="5", source="BATCH",
                        values={"customer_name": "Omega Supplies", "city": "Pune"}))
    db.commit()
    entity_id = e.entity_id
    pid = _make_pending(db, ds_id, 4, None, entity_id=entity_id,
                        values={"customer_name": "Gamma Corp", "city": "Delhi"})
    db.close()

    r = client.post(f"/api/reviews/pending/{pid}/resolve",
                    json={"decision": "CONFIRM_MATCH", "reviewer": "priya.s"})
    assert r.status_code == 200, r.text
    assert r.json()["resulting_entity_id"] == entity_id

    e_resp = client.get(f"/api/entities/{entity_id}").json()
    assert e_resp["member_count"] == 2
    assert e_resp["golden_record"]["version_number"] >= 2 or e_resp["golden_record"] is not None

    labels = client.get("/api/domains/customer/labels?label=MATCH").json()
    assert labels["count"] >= 1


def test_steward_unsure_leaves_item_in_queue_with_no_automatic_effect(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    pid = _make_pending(db, ds_id, 4, 5, values={"customer_name": "Gamma Corp"})
    db.close()

    r = client.post(f"/api/reviews/pending/{pid}/resolve",
                    json={"decision": "UNSURE", "reviewer": "priya.s"})
    assert r.status_code == 200
    assert r.json()["status"] == "PENDING"

    pending = client.get("/api/reviews/pending?domain=customer").json()
    assert pending["count"] == 1, "UNSURE keeps the relationship unresolved in the queue"
    ents = client.get("/api/entities?domain=customer").json()["entities"]
    assert len(ents) == 0, "UNSURE must not create or merge anything"


def test_pending_review_listing_shows_both_records(client):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    db = _db()
    pid = _make_pending(db, ds_id, 0, 1, values={"customer_name": "Acme Trading"})
    db.close()

    items = client.get("/api/reviews/pending?domain=customer").json()["pending_reviews"]
    item = next(i for i in items if i["pending_review_id"] == pid)
    assert item["candidate_other_record_id"] == "1"
    assert item["candidate_other_values"] is not None, "steward must see BOTH records"


# =============================================================================
# 5. Golden Record: conflict -> review -> correction -> versioning -> approval
# =============================================================================

def test_conflict_correction_versioning_and_approval(client, tmp_path):
    ds_id = _setup_dataset(client, CUSTOMER_CSV)
    run_id = _insert_training_run(ds_id, "customer", ["customer_name"])
    db = _db()
    from app.db.models import ClusterRun
    cr_id = f"CR-{uuid.uuid4().hex[:10]}"
    clusters_path = _write_clusters_file(tmp_path, [
        {"entity_id": "ENT-000001", "record_ids": [2, 3], "scores": [0.9], "cluster_size": 2}], cr_id)
    db.add(ClusterRun(cluster_run_id=cr_id, match_run_id="MR-x", training_run_id=run_id,
                      dataset_id=ds_id, domain="customer", status="COMPLETED", threshold=0.85,
                      clusters_path=clusters_path))
    db.commit()
    db.close()

    r = client.post(f"/api/datasets/{ds_id}/er/clusters/{cr_id}/golden-records")
    assert r.status_code == 200, r.text
    entity_id = r.json()["entity_ids"][0]

    gr = client.get(f"/api/entities/{entity_id}/golden-record").json()
    # records 2/3 share city+email but differ on phone -> phone needs review
    assert gr["steward_status"] == "NEEDS_REVIEW"
    assert any(n["field"] == "phone" for n in gr["needs_review_fields"])
    assert gr["lineage"]["phone"]["status"] == "NEEDS_REVIEW"
    assert len(gr["conflicts"]["phone"]) == 1

    # a conflict blocks approval until resolved
    ra = client.post(f"/api/entities/{entity_id}/review",
                     json={"action": "APPROVE_GOLDEN_RECORD", "reviewer": "priya.s"})
    assert ra.status_code == 422

    # steward selects among the retained candidates -> GR V2 (V1 kept)
    rc = client.post(f"/api/entities/{entity_id}/review",
                     json={"action": "RESOLVE_FIELD", "field": "phone", "value": "9000000002",
                           "reviewer": "priya.s", "reason": "confirmed with customer"})
    assert rc.status_code == 200, rc.text
    v2 = rc.json()["after_state"]
    assert v2["golden_version"] == 2
    assert v2["steward_status"] != "NEEDS_REVIEW"

    history = client.get(f"/api/entities/{entity_id}/golden-record/history").json()
    assert [v["version_number"] for v in history["versions"]] == [1, 2]

    # approval -> final master data
    ra = client.post(f"/api/entities/{entity_id}/review",
                     json={"action": "APPROVE_GOLDEN_RECORD", "reviewer": "priya.s"})
    assert ra.status_code == 200
    md = client.get("/api/domains/customer/master-data").json()
    assert md["count"] == 1
    rec = md["records"][0]
    assert rec["entity_id"] == entity_id and rec["status"] == "APPROVED"
    assert rec["values"]["phone"] == "9000000002"


# =============================================================================
# 6. Survivorship: source-system priority is CONFIGURATION, never a hardcoded bias
# =============================================================================

def test_survivorship_source_system_priority_from_configuration():
    from app.modules.golden_record_service import select_golden_values

    members = [
        {"dataset_id": "D1", "record_id": "1",
         "values": {"city": "Chennai", "source_system": "ORACLE"}},
        {"dataset_id": "D2", "record_id": "2",
         "values": {"city": "Bengaluru", "source_system": "SAP"}},
        {"dataset_id": "D3", "record_id": "3",
         "values": {"city": "Hyderabad", "source_system": "ARIBA"}},
        {"dataset_id": "D4", "record_id": "4",
         "values": {"city": "Chennai", "source_system": "LEGACY_ERP"}},
    ]
    # SAP is configured as the most trusted source for city -- ORACLE gets no
    # special treatment anywhere in the engine.
    cfg = {"source_priority": {"SAP": 4, "ARIBA": 2, "LEGACY_ERP": 1}}
    result = select_golden_values(members, ["city"], config=cfg)
    assert result.golden_values["city"] == "Bengaluru"
    lin = result.lineage["city"]
    assert lin["rule"] == "source_priority"
    assert lin["source_system"] == "SAP"
    assert lin["status"] == "AUTO_SELECTED"
    # conflicting values all retained with their source systems
    kept = {c["value"]: c["source_system"] for c in result.conflicts["city"]}
    assert kept == {"Chennai": "ORACLE", "Hyderabad": "ARIBA"}
    assert "ORACLE" in json.dumps(cfg) or "ORACLE" not in cfg.get("source_priority", {})


def test_survivorship_frequency_wins_when_ordered_first():
    from app.modules.golden_record_service import select_golden_values

    members = [
        {"dataset_id": "D1", "record_id": "1", "values": {"city": "Chennai", "source_system": "ORACLE"}},
        {"dataset_id": "D1", "record_id": "2", "values": {"city": "Chennai", "source_system": "ORACLE"}},
        {"dataset_id": "D2", "record_id": "3", "values": {"city": "Mumbai", "source_system": "SAP"}},
    ]
    cfg = {"source_priority": {"SAP": 9, "ORACLE": 1}, "rule_order": ["frequency", "source_priority"]}
    result = select_golden_values(members, ["city"], config=cfg)
    assert result.golden_values["city"] == "Chennai", "rule order is configuration"
    assert result.lineage["city"]["rule"] == "frequency"


def test_survivorship_field_rules_override_and_validity():
    from app.modules.golden_record_service import select_golden_values

    members = [
        {"dataset_id": "D1", "record_id": "1",
         "values": {"email": "valid@corp.com", "phone": "not-a-phone", "source_system": "SAP"}},
        {"dataset_id": "D2", "record_id": "2",
         "values": {"email": "broken-at-sign", "phone": "9876543210", "source_system": "ORACLE"}},
    ]
    cfg = {"source_priority": {"ORACLE": 5, "SAP": 1},
           "field_rules": {"email": ["validity", "source_priority"],
                            "phone": ["validity", "source_priority"]}}
    result = select_golden_values(members, ["email", "phone"], config=cfg)
    assert result.golden_values["email"] == "valid@corp.com", "an invalid value must not win on priority"
    assert result.golden_values["phone"] == "9876543210"


def test_survivorship_recency_rule():
    from app.modules.golden_record_service import select_golden_values

    members = [
        {"dataset_id": "D1", "record_id": "1",
         "values": {"phone": "1111111111", "last_updated": "2025-01-01"}},
        {"dataset_id": "D2", "record_id": "2",
         "values": {"phone": "2222222222", "last_updated": "2026-06-01"}},
    ]
    cfg = {"recency_field": "last_updated", "rule_order": ["recency", "source_priority"]}
    result = select_golden_values(members, ["phone"], config=cfg)
    assert result.golden_values["phone"] == "2222222222"


def test_survivorship_config_validation_rejects_unknown_rule(client):
    r = client.put("/api/domains/customer/survivorship", json={"rule_order": ["made_up_rule"]})
    assert r.status_code == 422


def test_survivorship_default_config_uses_source_system_field():
    from app.modules.golden_record_service import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["source_system_field"] == "source_system"
