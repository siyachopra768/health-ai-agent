import pytest
from verifier import (
    normalize_test_name,
    parse_reference_range,
    verify_value_against_source,
    has_direction_mismatch,
    is_likely_health_related,)

class TestNormalizeTestName:
    def test_maps_known_alias(self):
        assert normalize_test_name("T. Cholesterol") == "Total Cholesterol"

    def test_case_insensitive(self):
        assert normalize_test_name("hb")=="Hemoglobin"

    def test_returns_unchanged_if_no_alias(self):
        assert normalize_test_name("Urea") == "Urea"


class TestParseReferenceRange:
    def test_two_sided_range(self):
        assert parse_reference_range("70-100") == (70.0,100.0)

    def test_one_sided_greater_than(self):
        low,high = parse_reference_range("> 40")
        assert low == 40.0
        assert high == float("inf")

    def test_one_sided_less_than(self):
        low,high = parse_reference_range("<150")
        assert low == 0.0 
        assert high == 150.0

    def test_raises_on_unparseable_range(self):
        with pytest.raises(ValueError):
            parse_reference_range("not a range")


class TestVerifyValueAgainstSource:
    def test_accepts_real_value_present_in_text(self):
        assert verify_value_against_source(190.0,"Total Cholesterol 190 mg/dL") is True

    def test_rejects_fabricated_value_not_in_text(self):
        assert verify_value_against_source(777.0,"Total Cholesterol 190 mg/dL") is False

    def test_rejects_degenerate_zero(self):
        assert verify_value_against_source(0.0,"Triglycerides Not Done") is False

class TestDirectionMismatch:
    def test_detects_low_query_high_chunk_mismatch(self):
        assert has_direction_mismatch("Creatinine low","Creatinine-High") is True

    def test_no_mismatch_when_direction_align(self):
        assert has_direction_mismatch("Creatinine high","Creatinine - High") is False 

class TestHealthRelatedFilter:
     def test_flags_health_keyword_as_relevant(self):
        assert is_likely_health_related("what does my cholesterol mean") is True

     def test_flags_gibberish_as_not_relevant(self):
        assert is_likely_health_related("dsjhdsykwsyrjrgtjnh") is False

           

                                      