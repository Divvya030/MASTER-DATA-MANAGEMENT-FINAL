import pytest

from app.modules.profiling import metrics as m
from app.modules.profiling import type_inference as ti
from app.modules.profiling import recommendation as rec
from app.modules.profiling.profiling_service import profile_field


def test_completeness_basic():
    values = ["a", None, "", "N/A", "b"]
    r = m.compute_completeness(values)
    assert r.total_rows == 5
    assert r.non_null_count == 2
    assert r.null_count == 3


def test_missing_token_breakdown():
    values = ["N/A", "-", "", "x", "N/A"]
    r = m.compute_completeness(values)
    assert r.missing_token_breakdown["N/A"] == 2
    assert r.missing_token_breakdown["-"] == 1
    assert r.missing_token_breakdown[""] == 1


def test_cardinality_basic():
    values = ["a", "a", "b", "c", None]
    r = m.compute_cardinality(values)
    assert r.distinct_count == 3
    assert r.duplicate_value_count == 1


def test_cardinality_duplicate_value_vs_duplicate_row_count_are_different_questions():
    # One value appearing 3 times: duplicate_row_count counts the 2 EXTRA row
    # occurrences beyond the first; duplicate_value_count counts that it is
    # ONE distinct value which happens to repeat. These must differ here.
    values = ["a", "a", "a", "b", "c"]
    r = m.compute_cardinality(values)
    assert r.duplicate_row_count == 2   # "a" contributes 2 extra occurrences
    assert r.duplicate_value_count == 1  # only ONE distinct value ("a") repeats


def test_cardinality_constant_field():
    values = ["US", "US", "US", "US"]
    r = m.compute_cardinality(values)
    assert r.is_constant is True


def test_cardinality_block_distribution_evidence():
    # Item #6: block-distribution stats derived from the same Counter, no
    # second pass over the data. "Block" = the set of rows sharing one value.
    values = ["US", "US", "US", "CA", "CA", "MX"]
    r = m.compute_cardinality(values)
    assert r.singleton_value_count == 1        # only "MX" appears once
    assert r.largest_block_size == 3            # "US" appears 3 times
    assert r.avg_block_size == 2.0               # 6 values / 3 distinct


def test_cardinality_block_distribution_empty_input():
    r = m.compute_cardinality([])
    assert r.singleton_value_count == 0
    assert r.largest_block_size == 0
    assert r.avg_block_size == 0.0


def test_type_inference_email():
    result = ti.infer_type(["a@b.com", "c@d.com"])
    assert result.inferred_type == "EMAIL"
    assert result.type_confidence == 100.0


def test_type_inference_mixed_dominant():
    result = ti.infer_type(["12345", "12346", "abc-not-a-number"])
    assert result.inferred_type == "NUMERIC_ID"


def test_type_inference_from_values_not_names():
    # field named 'email' but values are free text -- must infer from VALUES
    result = ti.infer_type(["just some text", "more free text here"])
    assert result.inferred_type == "FREE_TEXT"


# ---------- ALPHANUMERIC_ID fix: ordinary words must not be misclassified ----------

def test_alphanumeric_id_ordinary_words_are_free_text():
    for word in ["Chennai", "Mumbai", "India", "Supplier", "Active"]:
        result = ti.infer_type([word] * 5)
        assert result.inferred_type == "FREE_TEXT", f"{word!r} was misclassified as {result.inferred_type}"


def test_alphanumeric_id_categorical_alphabetic_column_is_free_text():
    values = ["Chennai", "Mumbai", "Delhi", "Pune", "Chennai", "Mumbai"] 
    result = ti.infer_type(values)
    assert result.inferred_type == "FREE_TEXT"


def test_alphanumeric_id_mixed_alphanumeric_still_detected():
    for val in ["ABC12345", "A12B34"]:
        result = ti.infer_type([val] * 5)
        assert result.inferred_type == "ALPHANUMERIC_ID", f"{val!r} was classified as {result.inferred_type}"


def test_alphanumeric_id_structured_identifier_with_separators_still_detected():
    for val in ["SUP-10293", "IN_98231"]:
        result = ti.infer_type([val] * 5)
        assert result.inferred_type == "ALPHANUMERIC_ID", f"{val!r} was classified as {result.inferred_type}"


def test_numeric_id_still_detected():
    result = ti.infer_type(["10293", "10294", "10295"])
    assert result.inferred_type == "NUMERIC_ID"


def test_scientific_notation_numeric_id_still_detected():
    result = ti.infer_type(["1.00E+14", "2.00E+14", "3.00E+14"])
    assert result.inferred_type == "NUMERIC_ID"


def test_format_patterns_detect_mixed():
    r = m.compute_format_patterns(["TX-00231", "AB12345", "TX-00232"])
    assert r.is_mixed_format is True


def test_format_patterns_consistent():
    r = m.compute_format_patterns(["TX-001", "AB-002", "CD-003"])
    assert r.is_mixed_format is False


def test_case_variation_detected_same_value_different_case():
    r = m.compute_string_quality(["John Smith", "JOHN SMITH", "john smith"])
    assert r.case_variation_detected is True


def test_case_variation_detected_across_conventions():
    r = m.compute_string_quality(["JOHN SMITH", "jane doe", "Bob Jones"])
    assert r.case_variation_detected is True


# ---------- Split of case_variation into same-value vs convention-mixed
# (user-reported: standardization fixed real duplicate-casing problems but
# the CASE_VARIATION signal never cleared, because it conflated a genuinely
# fixable problem with benign diversity across different values) ----------

def test_case_variation_same_value_true_for_real_duplicates():
    r = m.compute_string_quality(["Acme Corp", "ACME CORP", "acme corp"])
    assert r.case_variation_same_value is True


def test_case_variation_same_value_false_for_genuinely_different_names():
    # Different real companies with different casing conventions -- this is
    # normal diversity, NOT the same value represented inconsistently.
    r = m.compute_string_quality(["Horizon Components LLP", "Acme Trading Company"])
    assert r.case_variation_same_value is False


def test_case_variation_convention_mixed_detected_separately():
    r = m.compute_string_quality(["Horizon Components LLP", "Acme Trading Company"])
    assert r.case_variation_convention_mixed is True  # LLP-style vs plain Title Case


def test_case_variation_combined_flag_is_or_of_both():
    r = m.compute_string_quality(["Acme Corp", "ACME CORP"])
    assert r.case_variation_same_value is True
    assert r.case_variation_detected is True  # combined flag still True via OR


def test_whitespace_detection():
    r = m.compute_string_quality([" trimme ", "no_ws", "double  space"])
    assert r.leading_trailing_whitespace_count == 1
    assert r.repeated_internal_whitespace_count == 1


def test_length_outliers():
    values = ["abc"] * 20 + ["a" * 200]
    r = m.compute_string_quality(values)
    assert r.length_outlier_count >= 1


def test_identifier_quality_scientific_notation():
    r = m.compute_identifier_quality(["12345678901234", "1.00E+14", "98765432109876"])
    assert r.scientific_notation_count == 1


def test_date_ambiguity_forced_dd_mm():
    r = m.compute_date_ambiguity(["13/04/2020", "01/05/2021"])
    assert r.dd_mm_evidence_count >= 1
    assert r.is_ambiguous is False


def test_date_ambiguity_genuinely_ambiguous():
    r = m.compute_date_ambiguity(["03/04/2020", "01/02/2021"])
    assert r.ambiguous_count >= 1
    assert r.is_ambiguous is True


def test_recommendation_insufficient_data():
    ev = rec.FieldProfileEvidence(False, "FREE_TEXT", 50.0, 20.0, 0.5, False, None)
    label, reasons = rec.recommend_er_role(ev)
    assert label == "INSUFFICIENT_DATA"


def test_recommendation_strong_identifier():
    ev = rec.FieldProfileEvidence(True, "ALPHANUMERIC_ID", 95.0, 99.0, 0.98, False, 0.95)
    label, reasons = rec.recommend_er_role(ev)
    assert label == "STRONG_IDENTIFIER_CANDIDATE"


def test_recommendation_low_value_constant():
    ev = rec.FieldProfileEvidence(False, "FREE_TEXT", 100.0, 99.0, 0.01, True, None)
    label, reasons = rec.recommend_er_role(ev)
    assert label == "LOW_VALUE_FOR_ER"


# ---------- Item-3 MATCH-field correction: low uniqueness alone must NOT
# exclude a genuinely useful categorical field from MATCH candidacy ----------

def test_low_uniqueness_categorical_field_is_not_low_value():
    # city-like field: 28 real categories, no single value dominates -- must
    # NOT be treated as low-value merely because uniqueness_ratio is tiny.
    ev = rec.FieldProfileEvidence(
        identifier_hint=False, inferred_type="FREE_TEXT", type_confidence=100.0,
        completeness_pct=100.0, uniqueness_ratio=0.0028, is_constant=False,
        format_consistency=None, dominant_value_share=0.0713,
    )
    label, reasons = rec.recommend_er_role(ev)
    assert label != "LOW_VALUE_FOR_ER"
    assert label == "SUPPORTING_MATCH_FIELD"


def test_genuinely_dominant_single_value_still_excluded():
    # A field where ONE value accounts for 99% of everything IS genuinely
    # low-value, regardless of category count -- this must still be caught.
    ev = rec.FieldProfileEvidence(
        identifier_hint=False, inferred_type="FREE_TEXT", type_confidence=100.0,
        completeness_pct=100.0, uniqueness_ratio=0.0003, is_constant=False,
        format_consistency=None, dominant_value_share=0.99,
    )
    label, reasons = rec.recommend_er_role(ev)
    assert label == "LOW_VALUE_FOR_ER"


def test_moderately_concentrated_field_still_usable_for_match():
    # country-like field: one dominant value (~72%) but real remaining
    # variance -- still usable as supporting MATCH evidence, not excluded.
    ev = rec.FieldProfileEvidence(
        identifier_hint=False, inferred_type="FREE_TEXT", type_confidence=100.0,
        completeness_pct=100.0, uniqueness_ratio=0.0005, is_constant=False,
        format_consistency=None, dominant_value_share=0.7228,
    )
    label, reasons = rec.recommend_er_role(ev)
    assert label != "LOW_VALUE_FOR_ER"


# ---------- Exact audit test scenarios: LOW UNIQUENESS != LOW ER VALUE ----------

def test_1000_rows_10_distributed_categories_not_low_value():
    from app.modules.profiling.metrics import compute_cardinality
    # 1000 rows, 10 categories, ~100 each -> largest share ~10%, well under 98%.
    values = [f"cat_{i % 10}" for i in range(1000)]
    card = compute_cardinality(values)
    dominant_share = card.top_values[0][1] / 1000
    assert dominant_share < 0.15
    ev = rec.FieldProfileEvidence(
        identifier_hint=False, inferred_type="FREE_TEXT", type_confidence=100.0,
        completeness_pct=100.0, uniqueness_ratio=card.uniqueness_ratio, is_constant=False,
        format_consistency=None, dominant_value_share=dominant_share,
    )
    label, reasons = rec.recommend_er_role(ev)
    assert label != "LOW_VALUE_FOR_ER"


def test_1000_rows_one_value_99_percent_is_low_value():
    from app.modules.profiling.metrics import compute_cardinality
    values = ["dominant"] * 990 + [f"other_{i}" for i in range(10)]
    card = compute_cardinality(values)
    dominant_share = card.top_values[0][1] / 1000
    assert dominant_share >= 0.98
    ev = rec.FieldProfileEvidence(
        identifier_hint=False, inferred_type="FREE_TEXT", type_confidence=100.0,
        completeness_pct=100.0, uniqueness_ratio=card.uniqueness_ratio, is_constant=False,
        format_consistency=None, dominant_value_share=dominant_share,
    )
    label, reasons = rec.recommend_er_role(ev)
    assert label == "LOW_VALUE_FOR_ER"


def test_constant_column_is_low_value():
    ev = rec.FieldProfileEvidence(
        identifier_hint=False, inferred_type="FREE_TEXT", type_confidence=100.0,
        completeness_pct=100.0, uniqueness_ratio=0.001, is_constant=True,
        format_consistency=None, dominant_value_share=1.0,
    )
    label, reasons = rec.recommend_er_role(ev)
    assert label == "LOW_VALUE_FOR_ER"


# ---------- Date validity: date-like structure is not calendar validity ----------

def test_date_structure_invalid_calendar_month_and_day_flagged():
    r = m.compute_date_ambiguity(["2025-99-99"])
    assert r.invalid_date_count == 1
    assert r.parseable_rate == 0.0  # must NOT be counted as parseable


def test_date_slash_format_invalid_under_both_readings():
    # Neither DD/MM (Feb 31) nor MM/DD (month 31) is a real calendar date.
    r = m.compute_date_ambiguity(["31/02/2020"])
    assert r.invalid_date_count == 1
    assert r.ambiguous_count == 0
    assert r.dd_mm_evidence_count == 0
    assert r.mm_dd_evidence_count == 0


def test_date_evidence_preserved_across_mixed_values():
    # 31/01/2025 forces DD/MM (no month 31); 12/02/2025 and 03/04/2025 are
    # each independently ambiguous (both readings valid) -- evidence from one
    # row must not overwrite or resolve a DIFFERENT row's ambiguity.
    r = m.compute_date_ambiguity(["31/01/2025", "12/02/2025", "03/04/2025"])
    assert r.dd_mm_evidence_count == 1
    assert r.ambiguous_count == 2


def test_valid_iso_date_still_parseable():
    r = m.compute_date_ambiguity(["2024-01-15"])
    assert r.invalid_date_count == 0
    assert r.parseable_rate == 100.0


def test_valid_slash_date_forced_unambiguous_still_works():
    r = m.compute_date_ambiguity(["31/01/2024"])
    assert r.dd_mm_evidence_count == 1
    assert r.invalid_date_count == 0


def test_recommendation_good_match_field():
    ev = rec.FieldProfileEvidence(False, "EMAIL", 100.0, 97.0, 0.94, False, None)
    label, reasons = rec.recommend_er_role(ev)
    assert label == "GOOD_MATCH_FIELD"


def test_blocking_assessment_removed_from_profiling():
    # Application-level blocking was removed from Profiling: no assess_blocking
    # exists, and profiles carry no blocking_assessment field -- the steward
    # sees human-readable explanations plus the raw statistics instead.
    assert not hasattr(rec, "assess_blocking")
    ev = rec.FieldProfileEvidence(False, "EMAIL", 100.0, 97.0, 0.94, False, None)
    label, reasons = rec.recommend_er_role(ev)
    assert label == "GOOD_MATCH_FIELD"
    assert "blocking" not in " ".join(reasons).lower()


def test_profile_field_produces_steward_explanation():
    profile = profile_field("email", "email_col", ["a@b.com", "c@d.com"] * 10)
    assert profile.steward_explanation
    assert "blocking" not in profile.steward_explanation.lower()
    # raw statistics are still retained
    assert "completeness" in profile.data_quality
    assert "missing_token_breakdown" in profile.data_quality["completeness"]


def test_profile_field_end_to_end_email():
    values = ["a@b.com", "c@d.com", " e@f.com ", "a@b.com"] * 10
    profile = profile_field("email", "email_col", values)
    assert profile.data_quality["type_inference"]["inferred_type"] == "EMAIL"
    assert "LEADING_TRAILING_WHITESPACE" in profile.cleansing_signals


# ---------- MIXED_FORMAT_PATTERNS false-positive fix (user-reported: company
# names of different word counts were flagged as a "mixed format" quality
# problem, which is a false positive for FREE_TEXT fields) ----------

def test_mixed_format_signal_suppressed_for_free_text_names():
    # "Summit Global Ltd" (3 words) vs "Quantum Systems Private Limited"
    # (4 words) legitimately produce different structural skeletons -- that's
    # normal for company names, not a cleansing problem.
    values = ["Summit Global Ltd", "Quantum Systems Private Limited",
              "Unity Supplies Pvt Ltd", "Acme Corp"] * 10
    profile = profile_field("supplier_name", "supplier_name", values)
    assert profile.data_quality["type_inference"]["inferred_type"] == "FREE_TEXT"
    assert "MIXED_FORMAT_PATTERNS" not in profile.cleansing_signals
    # the underlying evidence is still visible for transparency, just not
    # raised as an actionable cleansing signal
    assert profile.data_quality["format_patterns"]["is_mixed_format"] is True


def test_mixed_format_signal_still_raised_for_structured_fields():
    # A genuinely structured/identifier-like field mixing formats (e.g.
    # "TX-00231" vs "AB12345") IS a real quality signal and must still fire.
    values = ["TX-00231", "AB12345", "TX-00232", "CD99999"] * 10
    profile = profile_field("customer_id", "customer_id", values, identifier_hint=True)
    assert profile.data_quality["format_patterns"]["is_mixed_format"] is True
    assert "MIXED_FORMAT_PATTERNS" in profile.cleansing_signals


def test_profiling_engine_is_generic_no_domain_branching():
    """Genericity check: profiling a column called 'mdm_hdr_id' or any dataset-specific
    name produces output based purely on VALUE evidence, proving there is no hidden
    branching on field/dataset identity."""
    values = [str(i) for i in range(1, 101)]
    profile_a = profile_field("customer_id", "mdm_hdr_id", values)
    profile_b = profile_field("customer_id", "some_other_random_column_name", values)
    # same values -> same data_quality output regardless of source column name
    assert profile_a.data_quality == profile_b.data_quality
    assert profile_a.er_recommendation == profile_b.er_recommendation
