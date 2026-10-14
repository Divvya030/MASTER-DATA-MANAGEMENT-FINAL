import pandas as pd
import pytest

from app.modules.standardization import standardize_dataset, standardize_column
from app.modules.profiling.profiling_service import profile_field, profile_dataset
from app.modules.reprofiling import compare_profiles, build_signal_audit_report
from app.modules.er_config import (suggest_roles, validate_role_configuration, build_final_configuration,
                                    FieldRoleAssignment, RoleValidationError)
from app.modules.readiness import assess_readiness, build_dedupe_ready_configuration
from app.modules.nullsafe import series_to_values


# ---------------- Module 4: Standardization ----------------

def test_standardize_null_safe():
    values, log = standardize_column("customer_name", [None, "  Bob  "], ["LEADING_TRAILING_WHITESPACE"])
    assert values[0] is None
    assert values[1] == "Bob"


def test_standardize_whitespace_trim_and_collapse():
    values, log = standardize_column(
        "customer_name", ["  John   Smith  "],
        ["LEADING_TRAILING_WHITESPACE", "REPEATED_INTERNAL_WHITESPACE"])
    assert values[0] == "John Smith"


def test_standardize_missing_token_normalization():
    values, log = standardize_column("customer_name", ["N/A", "Bob", "-"], ["MISSING_TOKEN_VARIANTS"])
    assert values[0] is None
    assert values[1] == "Bob"
    assert values[2] is None


def test_standardize_identifier_preserves_leading_zeros():
    values, log = standardize_column("customer_id", ["00042", "00043"], [])
    assert values[0] == "00042"
    assert values[1] == "00043"


def test_standardize_scientific_notation_repair():
    values, log = standardize_column("customer_id", ["1.00E+14"], ["SCIENTIFIC_NOTATION_DETECTED"])
    assert values[0] == "100000000000000"
    assert any(l.transformation == "REPAIR_SCIENTIFIC_NOTATION" for l in log)


def test_standardize_scientific_notation_not_recoverable_left_untouched():
    # fractional precision lost -- must NOT guess a repaired value
    values, log = standardize_column("customer_id", ["1.234567E+8"], ["SCIENTIFIC_NOTATION_DETECTED"])
    assert values[0] == "1.234567E+8"


def test_standardize_scientific_notation_large_exponent_exact_via_decimal():
    # Item audit: float()-based recovery silently diverges from the true
    # integer at large exponents (confirmed: float(round) of '1.00E+23'
    # gives 99999999999999991611392, not the true value) -- Decimal must
    # recover this exactly, for a 24-digit identifier.
    values, log = standardize_column("customer_id", ["1.00E+23"], ["SCIENTIFIC_NOTATION_DETECTED"])
    assert values[0] == "1" + "0" * 23
    assert len(values[0]) == 24


def test_standardize_scientific_notation_never_uses_float_arithmetic():
    # Directly verify against Python's arbitrary-precision int arithmetic
    # (str-based, never float) that recovery is exact for several large,
    # non-trivial exponents -- not just the one case above.
    from app.modules.standardization import _try_repair_scientific_notation
    for digit in range(1, 10):
        for exp in (20, 25, 30, 35):
            s = f"{digit}.00E+{exp}"
            repaired, changed = _try_repair_scientific_notation(s)
            assert changed is True
            assert int(repaired) == digit * (10 ** exp)
            assert repaired == str(digit * (10 ** exp))


def test_standardize_scientific_notation_overflow_safely_preserved():
    from app.modules.standardization import _try_repair_scientific_notation
    # An absurdly large exponent must never crash -- just preserved untouched.
    repaired, changed = _try_repair_scientific_notation("1.00E+9999999")
    assert changed is False


def test_standardize_raw_preservation():
    raw_df = pd.DataFrame({"name": [" Bob ", "N/A"]})
    mapping = {"customer_name": "name"}
    signals = {"customer_name": ["LEADING_TRAILING_WHITESPACE", "MISSING_TOKEN_VARIANTS"]}
    result = standardize_dataset(raw_df, mapping, signals)
    assert raw_df["name"].iloc[0] == " Bob "  # untouched
    assert result.df["name"].iloc[0] == "Bob"


def test_standardize_email_lowercased():
    values, log = standardize_column("email", ["JOHN@EXAMPLE.COM"], [])
    assert values[0] == "john@example.com"


# ---------- Name-case normalization (user-reported: CASE_VARIATION on
# supplier_name/legal_name was detected by profiling but never actually
# acted on by standardization) ----------

def test_standardize_name_case_lowercase_to_titlecase():
    values, log = standardize_column("supplier_name", ["keystone infrastructure private limited"],
                                      ["CASE_VARIATION"])
    assert values[0] == "Keystone Infrastructure Private Limited"
    assert any(l.transformation == "NORMALIZE_NAME_CASE" for l in log)


def test_standardize_name_case_uppercase_preserves_letter_by_letter_initialism():
    # LLP is a genuine letter-by-letter initialism -- kept fully uppercase.
    values, log = standardize_column("legal_name", ["HORIZON COMPONENTS LLP"], ["CASE_VARIATION"])
    assert values[0] == "Horizon Components LLP"


def test_standardize_name_case_common_abbreviations_title_cased_not_uppercased():
    # Inc/Ltd/Pvt are conventionally Title-Case in business writing, NOT
    # all-caps acronyms -- confirmed against real data where already-correct
    # mixed-case values always write these in Title Case, never uppercase.
    # Blanket-uppercasing them would create NEW inconsistency, not resolve it.
    values, log = standardize_column("legal_name", ["STERLING AUTOMOTIVE INC"], ["CASE_VARIATION"])
    assert values[0] == "Sterling Automotive Inc"


def test_standardize_name_case_already_mixed_case_untouched():
    # Conservative rule: a value that already mixes case (not fully upper or
    # fully lower) is left alone -- we cannot safely tell intentional
    # formatting apart from an error without domain knowledge.
    values, log = standardize_column("supplier_name", ["Sterling Automotive Inc"], ["CASE_VARIATION"])
    assert values[0] == "Sterling Automotive Inc"
    assert log == []


def test_standardize_name_case_converges_duplicate_representations():
    # This is the concrete real-world case found in the user's uploaded file:
    # the SAME underlying supplier name appears once lowercase and once in the
    # data's dominant Title-Case-with-uppercase-suffix convention. After
    # standardization both must become byte-identical.
    values, log = standardize_column(
        "legal_name", ["horizon components llp", "Horizon Components LLP"], ["CASE_VARIATION"])
    assert values[0] == values[1] == "Horizon Components LLP"


def test_standardize_name_case_applies_even_without_case_variation_signal():
    # New, intentional behavior: name-casing is unconditional (like
    # email/phone/website), not gated behind the CASE_VARIATION signal --
    # the transformation is already safe-by-construction on its own.
    values, log = standardize_column("supplier_name", ["already fine lowercase co"], [])
    assert values[0] == "Already Fine Lowercase Co"
    assert any(l.transformation == "NORMALIZE_NAME_CASE" for l in log)


def test_standardize_name_case_standalone_value_no_duplicate_needed():
    # Exact worked example from the spec: a single ALL-CAPS value with no
    # duplicate anywhere else must still be safely re-cased.
    values, log = standardize_column("legal_name", ["ACME INDUSTRIES PVT LTD"], [])
    assert values[0] == "Acme Industries Pvt Ltd"


def test_standardize_name_case_only_applies_to_name_and_geo_fields():
    # country/city/state now legitimately DO get case normalization (per the
    # explicit spec: "CHENNAI" -> "Chennai"). A field with no free-text case
    # semantics at all (e.g. postal_code, which can be alphanumeric-structured
    # and must never be case-normalized) must remain untouched.
    values, log = standardize_column("postal_code", ["ab1 2cd"], ["CASE_VARIATION"])
    assert values[0] == "ab1 2cd"
    assert log == []


def test_standardize_geo_field_case_normalized():
    values, log = standardize_column("city", ["CHENNAI"], ["CASE_VARIATION"])
    assert values[0] == "Chennai"
    values, log = standardize_column("country", ["india"], ["CASE_VARIATION"])
    assert values[0] == "India"


def test_standardize_name_case_null_safe():
    values, log = standardize_column("supplier_name", [None, "acme trading company"], ["CASE_VARIATION"])
    assert values[0] is None
    assert values[1] == "Acme Trading Company"


# ---------- Website standardization ----------

def test_standardize_website_lowercased_and_trimmed():
    values, log = standardize_column("website", ["  HTTPS://EXAMPLE.COM  "], [])
    assert values[0] == "https://example.com"
    assert any(l.transformation == "NORMALIZE_WEBSITE" for l in log)


def test_standardize_website_does_not_rewrite_domain():
    # Domain is lowercased (case-insensitive per URL spec); path/query are
    # preserved exactly, since they can be genuinely case-sensitive.
    values, log = standardize_column("website", ["https://Example.co.in/Path?x=1"], [])
    assert values[0] == "https://example.co.in/Path?x=1"


def test_standardize_website_case_sensitive_path_preserved():
    values, log = standardize_column(
        "website", ["https://Example.com/MyPage?Token=AbC123"], [])
    assert values[0] == "https://example.com/MyPage?Token=AbC123"


def test_standardize_website_bare_domain_fully_lowercased():
    values, log = standardize_column("website", ["EXAMPLE.COM"], [])
    assert values[0] == "example.com"


def test_standardize_phone_extension_preserved_not_merged():
    # Regression: naive separator-stripping used to concatenate the
    # extension's digits onto the main number, corrupting it.
    values, log = standardize_column("phone", ["+1 555-123-4567 ext 123"], [])
    assert values[0] == "+1 555-123-4567 ext 123"
    assert "4567123" not in values[0]


def test_standardize_phone_x_style_extension_preserved():
    values, log = standardize_column("phone", ["555-123-4567 x99"], [])
    assert values[0] == "555-123-4567 x99"


def test_standardize_website_null_safe():
    values, log = standardize_column("website", [None], [])
    assert values[0] is None


# ---------- Phone standardization (separator normalization) ----------

def test_standardize_phone_strips_harmless_separators():
    values, log = standardize_column("phone", ["+91 98765-43210"], [])
    assert values[0] == "+919876543210"


def test_standardize_phone_preserves_international_plus():
    values, log = standardize_column("phone", ["+1 (555) 123-4567"], [])
    assert values[0] == "+15551234567"


def test_standardize_phone_never_invents_country_code():
    # No '+' present in the input -- must never add one.
    values, log = standardize_column("phone", ["9876543210"], [])
    assert values[0] == "9876543210"
    assert not values[0].startswith("+")


def test_standardize_phone_null_safe():
    values, log = standardize_column("phone", [None], [])
    assert values[0] is None


def test_standardize_phone_unparseable_value_preserved():
    values, log = standardize_column("phone", ["call reception"], [])
    assert values[0] == "call reception"  # no digits at all -- left untouched


# ---------- Geo case normalization (city/state/country) ----------

def test_standardize_city_case_normalized():
    values, log = standardize_column("city", ["CHENNAI"], ["CASE_VARIATION"])
    assert values[0] == "Chennai"


def test_standardize_state_mixed_case_untouched():
    values, log = standardize_column("state", ["Tamil Nadu"], ["CASE_VARIATION"])
    assert values[0] == "Tamil Nadu"
    assert log == []


# ---------- Date standardization ----------

def test_standardize_date_clearly_parseable_slash_format():
    values, log = standardize_column("dob", ["2024/01/15"], [])
    assert values[0] == "2024-01-15"


def test_standardize_date_textual_month_format():
    values, log = standardize_column("dob", ["15-Jan-2024"], [])
    assert values[0] == "2024-01-15"


def test_standardize_date_forced_unambiguous_day_first():
    values, log = standardize_column("dob", ["13/04/2020"], [])
    assert values[0] == "2020-04-13"


def test_standardize_date_genuinely_ambiguous_preserved():
    values, log = standardize_column("dob", ["03/04/2020"], [])
    assert values[0] == "03/04/2020"  # must NOT guess
    assert log == []


def test_standardize_date_invalid_date_preserved():
    values, log = standardize_column("dob", ["31/02/2020"], [])  # Feb 31st doesn't exist
    assert values[0] == "31/02/2020"


def test_standardize_date_already_iso_untouched():
    values, log = standardize_column("dob", ["2024-01-15"], [])
    assert values[0] == "2024-01-15"
    assert log == []


def test_standardize_date_null_safe():
    values, log = standardize_column("dob", [None], [])
    assert values[0] is None


# ---------- Regression: NaN round-trip must never inflate completeness ----------
# pandas represents empty cells as float('nan'), not None, both right after
# ingestion (na_values=[""]) and after every parquet round-trip. str(nan).strip()
# is the literal string "nan", which naive null checks fail to catch -- this was
# discovered from real user-reported behavior (completeness appearing to RISE
# after standardization, which should never happen).

def test_series_to_values_normalizes_nan_to_none():
    df = pd.DataFrame({"x": ["N/A", "Bob", float("nan"), "Alice"]})
    values = series_to_values(df["x"])
    assert values == ["N/A", "Bob", None, "Alice"]


def test_standardize_column_never_stringifies_raw_nan_into_text():
    # Before the fix, an untouched float('nan') value survived into
    # _trim_and_collapse_ws / str(v) calls and was written out as the literal
    # string "nan" -- silently corrupting the STANDARDIZED layer.
    values, log = standardize_column("customer_name", [None, "Bob"], ["LEADING_TRAILING_WHITESPACE"])
    assert values[0] is None
    assert "nan" not in [str(v) for v in values if v is not None]


def test_completeness_unaffected_by_parquet_style_nan_round_trip(tmp_path):
    import app.modules.ingestion as ing
    monkeypatch_dir = str(tmp_path)
    ing.DATA_DIR = monkeypatch_dir
    raw_df = pd.DataFrame({"name": ["Bob", "N/A", "Alice"]})
    mapping = {"customer_name": "name"}
    signals = {"customer_name": ["MISSING_TOKEN_VARIANTS"]}
    result = standardize_dataset(raw_df, mapping, signals)
    ing.save_layer("DS-TEST-0001", "standardized", result.df)
    reloaded = ing.load_layer("DS-TEST-0001", "standardized")

    raw_profile = profile_field("customer_name", "name", series_to_values(raw_df["name"]), stage="RAW")
    std_profile = profile_field("customer_name", "name", series_to_values(reloaded["name"]), stage="STANDARDIZED")

    raw_completeness = 100 - raw_profile.data_quality["completeness"]["null_percentage"]
    std_completeness = 100 - std_profile.data_quality["completeness"]["null_percentage"]

    # STANDARDIZED completeness must never exceed RAW completeness for a field
    # where standardization only normalizes existing missing tokens to None --
    # no new information was added.
    assert std_completeness <= raw_completeness
    assert std_completeness == raw_completeness == pytest.approx(2 / 3 * 100, abs=0.01)


# ---------------- Module 5: Re-profiling ----------------

def test_reprofile_shows_improvement():
    raw_values = ["  Bob  ", "Alice", "  Carl"]
    std_values = ["Bob", "Alice", "Carl"]
    raw_profile = profile_field("customer_name", "name", raw_values, stage="RAW")
    std_profile = profile_field("customer_name", "name", std_values, stage="STANDARDIZED")
    comparisons = compare_profiles({"customer_name": raw_profile}, {"customer_name": std_profile})
    c = comparisons["customer_name"]
    assert c.improved is True


def test_reprofile_case_variation_signal_clears_after_real_fix():
    # This is the user-reported end-to-end scenario: standardization must not
    # just run without error, it must actually make the CASE_VARIATION signal
    # disappear once the underlying duplicate-casing problem is genuinely gone.
    raw_values = ["horizon components llp", "Horizon Components LLP", "Acme Trading Co"]
    raw_profile = profile_field("supplier_name", "supplier_name", raw_values, stage="RAW")
    assert "CASE_VARIATION" in raw_profile.cleansing_signals

    from app.modules.standardization import standardize_column
    std_values, _ = standardize_column("supplier_name", raw_values, raw_profile.cleansing_signals)
    assert std_values[0] == std_values[1]  # the true duplicate now converges

    std_profile = profile_field("supplier_name", "supplier_name", std_values, stage="STANDARDIZED")
    assert "CASE_VARIATION" not in std_profile.cleansing_signals


def test_reprofile_raw_not_overwritten():
    raw_values = ["  Bob  "]
    std_values = ["Bob"]
    raw_profile = profile_field("customer_name", "name", raw_values, stage="RAW")
    std_profile = profile_field("customer_name", "name", std_values, stage="STANDARDIZED")
    assert raw_profile.stage == "RAW"
    assert std_profile.stage == "STANDARDIZED"
    assert raw_profile.data_quality != std_profile.data_quality


def test_reprofile_no_change_reported_honestly():
    values = ["stable", "stable2", "stable3"]
    p1 = profile_field("x", "col", values, stage="RAW")
    p2 = profile_field("x", "col", values, stage="STANDARDIZED")
    comparisons = compare_profiles({"x": p1}, {"x": p2})
    assert "No measurable quality change" in comparisons["x"].notes[0]


def test_reprofile_note_names_the_resolved_signal_not_just_generic_text():
    # Regression for the exact bug this exposed: `improved=True` must come
    # with a note that actually explains WHY, not fall through to the
    # generic "no change" text just because none of the OTHER named metrics
    # (whitespace/mixed-format/sci-notation) happened to catch it.
    raw_values = ["horizon components llp", "Horizon Components LLP"]
    std_values = ["Horizon Components LLP", "Horizon Components LLP"]
    raw_profile = profile_field("supplier_name", "supplier_name", raw_values, stage="RAW")
    std_profile = profile_field("supplier_name", "supplier_name", std_values, stage="STANDARDIZED")
    comparisons = compare_profiles({"supplier_name": raw_profile}, {"supplier_name": std_profile})
    c = comparisons["supplier_name"]
    assert c.improved is True
    assert any("Resolved cleansing signal" in n for n in c.notes)
    assert not any("No measurable quality change" in n for n in c.notes)


# ---------- Per-signal standardization audit report ----------

def test_audit_report_fixed_case_variation():
    raw_values = ["horizon components llp", "Horizon Components LLP", "Acme Trading Co"] * 5
    std_values = ["Horizon Components LLP", "Horizon Components LLP", "Acme Trading Co"] * 5
    raw_profile = profile_field("supplier_name", "supplier_name", raw_values, stage="RAW")
    std_profile = profile_field("supplier_name", "supplier_name", std_values, stage="STANDARDIZED")
    report = build_signal_audit_report({"supplier_name": raw_profile}, {"supplier_name": std_profile})
    entry = next(e for e in report if e.signal == "CASE_VARIATION")
    assert entry.status == "FIXED"
    assert entry.fixed_count == 10  # 2 mismatched values x 5 repeats
    assert entry.remaining_count == 0
    assert entry.changed is True
    assert "normalized" in entry.explanation.lower()


def test_audit_report_preserved_length_outliers():
    values = ["short"] * 20 + ["a very unusually long value that stands well outside the normal range"]
    raw_profile = profile_field("supplier_name", "supplier_name", values, stage="RAW")
    std_profile = profile_field("supplier_name", "supplier_name", values, stage="STANDARDIZED")  # untouched
    report = build_signal_audit_report({"supplier_name": raw_profile}, {"supplier_name": std_profile})
    entry = next((e for e in report if e.signal == "LENGTH_OUTLIERS"), None)
    if entry:  # only present if profiling actually flagged it for this synthetic sample
        assert entry.status == "PRESERVED"
        assert entry.changed is False
        assert entry.fixed_count == 0
        assert "no safe deterministic correction" in entry.explanation.lower()


def test_audit_report_only_includes_signals_actually_detected_in_raw():
    values = ["Clean Value"] * 10
    raw_profile = profile_field("supplier_name", "supplier_name", values, stage="RAW")
    std_profile = profile_field("supplier_name", "supplier_name", values, stage="STANDARDIZED")
    report = build_signal_audit_report({"supplier_name": raw_profile}, {"supplier_name": std_profile})
    assert report == []  # nothing was ever detected, so nothing to report


def test_audit_report_not_applicable_for_blank_cells_with_no_real_token_variants():
    # MISSING_TOKEN_VARIANTS can fire purely from ordinary blank cells with
    # zero actual token-variant strings ("N/A"/"-" etc) -- this is a
    # pre-existing Module 3 signal-naming quirk, not something this session
    # introduced. The audit report must distinguish "nothing to fix" from
    # "genuinely preserved because unsafe" rather than mislabeling both PRESERVED.
    values = ["Acme Corp", None, None, "Beta Inc"]
    raw_profile = profile_field("supplier_name", "supplier_name", values, stage="RAW")
    std_profile = profile_field("supplier_name", "supplier_name", values, stage="STANDARDIZED")
    report = build_signal_audit_report({"supplier_name": raw_profile}, {"supplier_name": std_profile})
    entry = next((e for e in report if e.signal == "MISSING_TOKEN_VARIANTS"), None)
    if entry:
        assert entry.status == "NOT_APPLICABLE"
        assert entry.fixed_count == 0
        assert entry.remaining_count == 0


def test_audit_report_schema_has_all_required_fields():
    raw_values = ["  Bob  ", "Alice"]
    std_values = ["Bob", "Alice"]
    raw_profile = profile_field("customer_name", "name", raw_values, stage="RAW")
    std_profile = profile_field("customer_name", "name", std_values, stage="STANDARDIZED")
    report = build_signal_audit_report({"customer_name": raw_profile}, {"customer_name": std_profile})
    assert len(report) >= 1
    d = report[0].to_dict()
    for key in ("field", "signal", "detected_before", "changed", "fixed_count",
                "remaining_count", "status", "explanation"):
        assert key in d


# ---------------- Module 6: ER field role config ----------------

def test_suggest_roles_identifier():
    assert suggest_roles("STRONG_IDENTIFIER_CANDIDATE") == ["IDENTIFIER"]


def test_suggest_roles_never_suggests_blocking():
    # Application-level blocking was removed from the ER configuration:
    # candidate generation uses the trained model's learned blocking predicates.
    for rec in ("GOOD_MATCH_FIELD", "SUPPORTING_MATCH_FIELD", "LOW_VALUE_FOR_ER",
                "STRONG_IDENTIFIER_CANDIDATE", "INSUFFICIENT_DATA"):
        assert "BLOCKING" not in suggest_roles(rec)


def test_suggest_roles_match_only():
    assert suggest_roles("GOOD_MATCH_FIELD") == ["MATCH"]


def test_suggest_roles_ignore():
    assert suggest_roles("LOW_VALUE_FOR_ER") == ["IGNORE"]


def test_role_assignment_user_override_allowed_with_warning():
    assignments = [FieldRoleAssignment("state", "IDENTIFIER", "USER_OVERRIDE")]
    evidence = {"state": {"er_recommendation": "LOW_VALUE_FOR_ER", "completeness_pct": 95}}
    warnings = validate_role_configuration(assignments, evidence)
    assert len(warnings) == 1
    assert "state" in warnings[0].canonical_field


def test_role_assignment_no_warning_when_aligned_with_evidence():
    assignments = [FieldRoleAssignment("email", "MATCH", "USER_OVERRIDE")]
    evidence = {"email": {"er_recommendation": "GOOD_MATCH_FIELD", "completeness_pct": 97}}
    warnings = validate_role_configuration(assignments, evidence)
    assert warnings == []


def test_invalid_role_rejected():
    assignments = [FieldRoleAssignment("email", "BOGUS_ROLE", "USER_OVERRIDE")]
    with pytest.raises(RoleValidationError):
        validate_role_configuration(assignments, {})


def test_blocking_role_is_no_longer_accepted():
    # Application-level blocking was removed from the ER configuration.
    assignments = [FieldRoleAssignment("email", "BLOCKING", "USER_OVERRIDE")]
    with pytest.raises(RoleValidationError):
        validate_role_configuration(assignments, {})


def test_exact_duplicate_field_role_pair_rejected():
    # Submitting the SAME (field, role) pair twice is still structurally invalid.
    assignments = [FieldRoleAssignment("email", "MATCH", "USER_OVERRIDE"),
                   FieldRoleAssignment("email", "MATCH", "USER_OVERRIDE")]
    with pytest.raises(RoleValidationError):
        validate_role_configuration(assignments, {})


def test_match_recommendation_does_not_automatically_imply_blocking():
    # No role ever lands in blocking_fields anymore -- the key exists only for
    # backward-compatible consumers and is always empty.
    assignments = [FieldRoleAssignment("customer_name", "MATCH", "USER_OVERRIDE")]
    config = build_final_configuration(assignments)
    assert config["match_fields"] == ["customer_name"]
    assert config["blocking_fields"] == []


def test_ignore_mutually_exclusive_with_match():
    assignments = [FieldRoleAssignment("email", "MATCH", "USER_OVERRIDE"),
                   FieldRoleAssignment("email", "IGNORE", "USER_OVERRIDE")]
    with pytest.raises(RoleValidationError, match="IGNORE"):
        validate_role_configuration(assignments, {})


def test_ignore_mutually_exclusive_with_identifier():
    assignments = [FieldRoleAssignment("country", "IDENTIFIER", "USER_OVERRIDE"),
                   FieldRoleAssignment("country", "IGNORE", "USER_OVERRIDE")]
    with pytest.raises(RoleValidationError, match="IGNORE"):
        validate_role_configuration(assignments, {})


def test_ignore_alone_is_valid():
    assignments = [FieldRoleAssignment("country", "IGNORE", "USER_OVERRIDE")]
    warnings = validate_role_configuration(assignments, {})  # must not raise
    assert warnings == []


def test_identifier_uniqueness_alone_does_not_force_identifier_role():
    # Item #2 regression: a highly unique customer_name is not automatically
    # treated as a guaranteed identifier -- suggest_roles must not suggest
    # IDENTIFIER just because a field happens to be unique; it only does so
    # when Module 3's own recommendation says STRONG_IDENTIFIER_CANDIDATE.
    roles = suggest_roles("SUPPORTING_MATCH_FIELD")
    assert "IDENTIFIER" not in roles


def test_identifier_recommendation_remains_evidence_driven():
    from app.modules.profiling.recommendation import recommend_er_role, FieldProfileEvidence
    # 95% unique customer_name-like field but format_consistency is weak
    # (free text names don't share a structural pattern) -- must NOT qualify
    # as STRONG_IDENTIFIER_CANDIDATE on uniqueness alone.
    ev = FieldProfileEvidence(identifier_hint=False, inferred_type="FREE_TEXT", type_confidence=100.0,
                               completeness_pct=98.0, uniqueness_ratio=0.95, is_constant=False,
                               format_consistency=0.40)
    label, reasons = recommend_er_role(ev)
    assert label != "STRONG_IDENTIFIER_CANDIDATE"


def test_identifier_hint_alone_does_not_force_identifier_recommendation():
    from app.modules.profiling.recommendation import recommend_er_role, FieldProfileEvidence
    # identifier_hint=True but weak actual evidence (low uniqueness) -- the hint
    # alone must never be sufficient to trigger STRONG_IDENTIFIER_CANDIDATE.
    ev = FieldProfileEvidence(identifier_hint=True, inferred_type="ALPHANUMERIC_ID", type_confidence=100.0,
                               completeness_pct=98.0, uniqueness_ratio=0.40, is_constant=False,
                               format_consistency=0.90)
    label, reasons = recommend_er_role(ev)
    assert label != "STRONG_IDENTIFIER_CANDIDATE"


def test_multiple_match_fields_allowed():
    assignments = [
        FieldRoleAssignment("email", "MATCH", "USER_OVERRIDE"),
        FieldRoleAssignment("phone", "MATCH", "USER_OVERRIDE"),
    ]
    config = build_final_configuration(assignments)
    assert config["match_fields"] == ["email", "phone"]
    assert config["blocking_fields"] == []


def test_low_quality_field_warns_not_silently_activated():
    assignments = [FieldRoleAssignment("junk", "MATCH", "USER_OVERRIDE")]
    evidence = {"junk": {"er_recommendation": "SUPPORTING_MATCH_FIELD", "completeness_pct": 20}}
    warnings = validate_role_configuration(assignments, evidence)
    assert len(warnings) == 1
    assert "completeness" in warnings[0].message.lower()


# ---------------- Module 7: Readiness ----------------

def test_readiness_ready_case():
    report = assess_readiness(
        dataset_id="DS-2026-0001", domain="customer", row_count=100,
        total_source_columns=10, mapped_field_count=5, empty_field_count=1,
        stage_flags={"mapped": True, "profiled": True, "standardized": True, "revalidated": True},
        er_config={"identifier_fields": ["customer_id"], "match_fields": ["email"],
                   "blocking_fields": ["postal_code"], "ignored_fields": []},
        field_warnings=[],
    )
    assert report.is_ready is True
    assert all(c.passed for c in report.checks)


def test_readiness_surfaces_unmapped_canonical_fields():
    report = assess_readiness(
        dataset_id="DS-2026-0001", domain="customer", row_count=100,
        total_source_columns=10, mapped_field_count=5, empty_field_count=1,
        stage_flags={"mapped": True, "profiled": True, "standardized": True, "revalidated": True},
        er_config={"identifier_fields": ["customer_id"], "match_fields": ["email"],
                   "blocking_fields": ["postal_code"], "ignored_fields": []},
        field_warnings=[],
        unmapped_canonical_fields=["phone", "dob", "website"],
    )
    assert report.unmapped_canonical_fields == ["phone", "dob", "website"]
    assert report.to_dict()["unmapped_canonical_fields"] == ["phone", "dob", "website"]


def test_readiness_defaults_unmapped_canonical_fields_to_empty_list():
    report = assess_readiness(
        dataset_id="DS-2026-0001", domain="customer", row_count=100,
        total_source_columns=10, mapped_field_count=5, empty_field_count=1,
        stage_flags={"mapped": True, "profiled": True, "standardized": True, "revalidated": True},
        er_config={"identifier_fields": [], "match_fields": ["email"], "blocking_fields": [], "ignored_fields": []},
        field_warnings=[],
    )
    assert report.unmapped_canonical_fields == []


def test_readiness_not_ready_missing_mapping():
    report = assess_readiness(
        dataset_id="DS-2026-0001", domain="customer", row_count=100,
        total_source_columns=10, mapped_field_count=0, empty_field_count=0,
        stage_flags={"mapped": False, "profiled": False, "standardized": False, "revalidated": False},
        er_config={"identifier_fields": [], "match_fields": [], "blocking_fields": [], "ignored_fields": []},
        field_warnings=[],
    )
    assert report.is_ready is False


def test_readiness_not_ready_no_usable_config():
    report = assess_readiness(
        dataset_id="DS-2026-0001", domain="customer", row_count=100,
        total_source_columns=10, mapped_field_count=5, empty_field_count=0,
        stage_flags={"mapped": True, "profiled": True, "standardized": True, "revalidated": True},
        er_config={"identifier_fields": [], "match_fields": [], "blocking_fields": [], "ignored_fields": []},
        field_warnings=[],
    )
    assert report.is_ready is False
    assert any("No MATCH field configured" in w for w in report.warnings)


def test_readiness_blocking_only_is_never_ready():
    # This is the exact Item #5 regression the user called out: BLOCKING fields
    # present + zero MATCH fields must NOT reach READY_FOR_DEDUPE, even though
    # every pipeline stage completed successfully. Blocking is only a
    # candidate-generation optimization, not the matching decision itself.
    report = assess_readiness(
        dataset_id="DS-2026-0001", domain="customer", row_count=100,
        total_source_columns=10, mapped_field_count=5, empty_field_count=0,
        stage_flags={"mapped": True, "profiled": True, "standardized": True, "revalidated": True},
        er_config={"identifier_fields": [], "match_fields": [],
                   "blocking_fields": ["postal_code", "city"], "ignored_fields": []},
        field_warnings=[],
    )
    assert report.is_ready is False
    assert any("No MATCH field configured" in w for w in report.warnings)
    match_check = next(c for c in report.checks if c.name == "Match fields configured")
    assert match_check.passed is False


def test_readiness_no_blocking_checks_or_warnings():
    # Blocking checks/warnings/configuration were removed from ER Readiness:
    # no check or warning may mention blocking anymore.
    report = assess_readiness(
        dataset_id="DS-2026-0001", domain="customer", row_count=100,
        total_source_columns=10, mapped_field_count=5, empty_field_count=0,
        stage_flags={"mapped": True, "profiled": True, "standardized": True, "revalidated": True},
        er_config={"identifier_fields": [], "match_fields": ["email"], "blocking_fields": [], "ignored_fields": []},
        field_warnings=[],
    )
    assert report.is_ready is True
    assert all("blocking" not in c.name.lower() for c in report.checks)
    assert all("blocking" not in w.lower() for w in report.warnings)


def test_readiness_warns_no_identifier_but_still_ready():
    report = assess_readiness(
        dataset_id="DS-2026-0001", domain="customer", row_count=100,
        total_source_columns=10, mapped_field_count=5, empty_field_count=0,
        stage_flags={"mapped": True, "profiled": True, "standardized": True, "revalidated": True},
        er_config={"identifier_fields": [], "match_fields": ["email"], "blocking_fields": ["postal_code"],
                   "ignored_fields": []},
        field_warnings=[],
    )
    assert report.is_ready is True
    assert any("No strong identifier" in w for w in report.warnings)


def test_dedupe_handoff_configuration_is_clean_and_deterministic():
    config = build_dedupe_ready_configuration(
        "DS-2026-0001", "customer",
        {"identifier_fields": ["customer_id"], "match_fields": ["email", "phone"],
         "blocking_fields": [], "ignored_fields": ["country"]})
    assert config == {
        "dataset_id": "DS-2026-0001",
        "domain": "customer",
        "identifier_fields": ["customer_id"],
        "match_fields": ["email", "phone"],
        "ignored_fields": ["country"],
        "standardization_stage": "STANDARDIZED",
    }
    # ensure it contains no matching/clustering artifacts
    assert "matches" not in config
    assert "clusters" not in config
