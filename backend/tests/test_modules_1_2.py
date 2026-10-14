import io
import pytest
import pandas as pd

from app.modules.ingestion import ingest_file, IngestionError, load_raw, save_layer, load_layer
from app.modules.mapping import (normalize_name, suggest_mappings, ColumnEvidence,
                                  validate_and_persist_mapping, MappingValidationError)
from app.domain_schemas import get_schema


# ---------------- Module 1: Ingestion ----------------

def test_csv_upload_basic(tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    csv_bytes = b"name,email\nJohn,john@example.com\nJane,jane@example.com\n"
    ds, df = ing.ingest_file("test.csv", csv_bytes, "customer")
    assert ds.row_count == 2
    assert ds.column_count == 2
    assert ds.status == "UPLOADED"
    assert list(df.columns) == ["name", "email"]


def test_excel_upload_basic(tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    df_in = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    buf = io.BytesIO()
    df_in.to_excel(buf, index=False)
    ds, df = ing.ingest_file("test.xlsx", buf.getvalue(), "supplier")
    assert ds.row_count == 2
    assert ds.column_count == 2


def test_invalid_file_type(tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    with pytest.raises(IngestionError):
        ing.ingest_file("test.txt", b"hello", "customer")


def test_bom_and_delimiter_handling(tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    csv_bytes = "\ufeffname;email\nJohn;john@example.com\n".encode("utf-8")
    ds, df = ing.ingest_file("test.csv", csv_bytes, "customer")
    assert "name" in df.columns
    assert df.columns[0] == "name"  # BOM stripped


def test_cp1252_special_characters_decoded_correctly(tmp_path, monkeypatch):
    # Item #7 regression: cp1252 must be tried BEFORE the latin-1 catch-all.
    # latin-1 never raises on any byte sequence, so if it were tried first it
    # would silently "succeed" by mis-decoding cp1252-only bytes (0x80-0x9F
    # range, e.g. em-dash 0x97) into invisible C1 control characters instead
    # of the correct printable character -- corrupting data with no error.
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    text = 'name,note\nAcme "Corp"\u2014Ltd,ok\n'  # \u2014 = em dash
    cp1252_bytes = text.encode("cp1252")
    ds, df = ing.ingest_file("t.csv", cp1252_bytes, "customer")
    assert "\u2014" in df["name"].iloc[0]
    assert "\x97" not in df["name"].iloc[0]  # would indicate the latin-1 mis-decode


def test_utf8_still_preferred_over_cp1252(tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    csv_bytes = "name,city\nJos\u00e9,S\u00e3o Paulo\n".encode("utf-8")
    ds, df = ing.ingest_file("t.csv", csv_bytes, "customer")
    assert df["name"].iloc[0] == "Jos\u00e9"
    assert df["city"].iloc[0] == "S\u00e3o Paulo"


def test_dataset_id_format(tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    ds, _ = ing.ingest_file("t.csv", b"a,b\n1,2\n", "customer")
    assert ds.dataset_id.startswith("DS-")
    parts = ds.dataset_id.split("-")
    assert len(parts) == 3
    assert len(parts[2]) == 4


def test_raw_preservation_across_layers(tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    ds, df = ing.ingest_file("t.csv", b"a,b\n1, hello \n", "customer")
    raw = ing.load_raw(ds.dataset_id)
    assert raw["b"].iloc[0] == " hello "  # untouched

    modified = raw.copy()
    modified["b"] = modified["b"].str.strip()
    ing.save_layer(ds.dataset_id, "standardized", modified)

    raw_again = ing.load_raw(ds.dataset_id)
    assert raw_again["b"].iloc[0] == " hello "  # still untouched after writing standardized layer

    std = ing.load_layer(ds.dataset_id, "standardized")
    assert std["b"].iloc[0] == "hello"


def test_cannot_overwrite_raw_layer(tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    ds, df = ing.ingest_file("t.csv", b"a,b\n1,2\n", "customer")
    with pytest.raises(IngestionError):
        ing.save_layer(ds.dataset_id, "raw", df)


def test_duplicate_headers_rejected(tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    with pytest.raises(IngestionError):
        ing.ingest_file("t.csv", b"a,a\n1,2\n", "customer")


# ---------------- Module 2: Mapping ----------------

@pytest.mark.parametrize("raw,expected", [
    ("First Name", "first name"),
    ("first_name", "first name"),
    ("FIRSTNAME", "firstname"),
    ("firstName", "first name"),
])
def test_normalize_name_variants(raw, expected):
    assert normalize_name(raw) == expected


def test_exact_mapping():
    schema = get_schema("customer")
    suggestions = suggest_mappings(schema, ["customer_name"])
    cands = suggestions["customer_name"]
    assert cands[0].match_type == "EXACT"
    assert cands[0].confidence == 1.0


def test_alias_mapping():
    schema = get_schema("customer")
    suggestions = suggest_mappings(schema, ["account_number"])
    cands = suggestions["customer_id"]
    assert any(c.match_type == "ALIAS" for c in cands)


def test_token_subset_mapping():
    schema = get_schema("customer")
    suggestions = suggest_mappings(schema, ["state_p"])
    cands = suggestions["state"]
    assert any(c.match_type == "TOKEN_SUBSET" for c in cands)


def test_guarded_substring_prevents_false_positive():
    schema = get_schema("customer")
    suggestions = suggest_mappings(schema, ["statement_cycle_name", "send_statements"])
    assert suggestions["state"] == []


def test_guarded_substring_matches_legitimate_case():
    schema = get_schema("customer")
    # 'state_description' -> tokens {state, description}; canonical 'state' is a whole token
    suggestions = suggest_mappings(schema, ["state_description"])
    assert any(c.canonical_field == "state" for c in [
        c for cands in suggestions.values() for c in cands
    ] if False) or len(suggestions["state"]) >= 0  # smoke check, real assertion below
    cands = suggestions["state"]
    assert len(cands) >= 1


def test_ambiguous_mapping_returns_all_candidates():
    schema = get_schema("customer")
    suggestions = suggest_mappings(
        schema, ["address1_p", "address2_p", "address3_p", "address4_p"])
    assert len(suggestions["address"]) == 4


def test_duplicate_canonical_assignment_rejected():
    schema = get_schema("customer")
    with pytest.raises(MappingValidationError):
        validate_and_persist_mapping(
            schema, ["a", "b"],
            [{"canonical_field": "customer_name", "source_column": "a"},
             {"canonical_field": "customer_name", "source_column": "b"}])


def test_unknown_source_column_rejected():
    schema = get_schema("customer")
    with pytest.raises(MappingValidationError):
        validate_and_persist_mapping(
            schema, ["a"],
            [{"canonical_field": "customer_name", "source_column": "does_not_exist"}])


def test_unknown_canonical_field_rejected():
    schema = get_schema("customer")
    with pytest.raises(MappingValidationError):
        validate_and_persist_mapping(
            schema, ["a"],
            [{"canonical_field": "not_a_real_field", "source_column": "a"}])


def test_invalid_mapping_structure_rejected():
    schema = get_schema("customer")
    with pytest.raises(MappingValidationError):
        validate_and_persist_mapping(schema, ["a"], [{"canonical_field": "customer_name"}])


def test_empty_column_excluded_from_candidates():
    schema = get_schema("customer")
    ev = {"registration_number": ColumnEvidence("registration_number", 0, 100, 0, [])}
    suggestions = suggest_mappings(schema, ["registration_number"], ev)
    assert suggestions["registration_number"] == []


def test_candidate_evidence_fields_present():
    ev = ColumnEvidence("address1_p", 82, 100, 6921, ["123 Main St"])
    assert ev.is_empty is False
    assert ev.completeness == 82.0


def test_valid_mapping_persists():
    schema = get_schema("customer")
    confirmed = validate_and_persist_mapping(
        schema, ["account_number", "customer_name"],
        [{"canonical_field": "customer_id", "source_column": "account_number"},
         {"canonical_field": "customer_name", "source_column": "customer_name"}])
    assert len(confirmed) == 2


# ---------------- Item #1: SHA-256 file fingerprinting ----------------

def test_sha256_computed_from_original_bytes_not_parsed_dataframe(tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    raw_bytes = b"a,b\n1, hello \n"
    ds, df = ing.ingest_file("t.csv", raw_bytes, "customer")
    # Computed directly over raw_bytes, independent of any parsing/whitespace
    # normalization that happens afterward -- verify it matches a fresh,
    # out-of-band hash of the exact same bytes.
    import hashlib
    assert ds.sha256 == hashlib.sha256(raw_bytes).hexdigest()


def test_sha256_same_bytes_different_filename_same_hash(tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    raw_bytes = b"a,b\n1,2\n"
    ds1, _ = ing.ingest_file("file_one.csv", raw_bytes, "customer")
    ds2, _ = ing.ingest_file("file_two.csv", raw_bytes, "customer")
    assert ds1.sha256 == ds2.sha256
    assert ds1.original_filename != ds2.original_filename


def test_sha256_same_filename_different_bytes_different_hash(tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    ds1, _ = ing.ingest_file("t.csv", b"a,b\n1,2\n", "customer")
    ds2, _ = ing.ingest_file("t.csv", b"a,b\n3,4\n", "customer")
    assert ds1.sha256 != ds2.sha256


def test_sha256_different_files_different_fingerprints(tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    ds1, _ = ing.ingest_file("a.csv", b"x,y\n1,2\n", "customer")
    ds2, _ = ing.ingest_file("b.csv", b"p,q\nfoo,bar\n", "customer")
    assert ds1.sha256 != ds2.sha256


def test_sha256_raw_file_remains_immutable(tmp_path, monkeypatch):
    import app.modules.ingestion as ing
    monkeypatch.setattr(ing, "DATA_DIR", str(tmp_path))
    raw_bytes = b"a,b\n1, hello \n"
    ds, df = ing.ingest_file("t.csv", raw_bytes, "customer")
    raw_before = ing.load_raw(ds.dataset_id).copy()
    # sha256 computation and any subsequent reads must never mutate the RAW layer
    ing.compute_file_sha256(raw_bytes)
    raw_after = ing.load_raw(ds.dataset_id)
    assert raw_before.equals(raw_after)
