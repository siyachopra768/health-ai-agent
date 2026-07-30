import pytest
from unittest.mock import patch,MagicMock

from parser import LabValueExtractor
from utils import analyze_values,calculate_risk_score


@pytest.fixture
def extractor():
    return LabValueExtractor(groq_api_key="test-key-not-used")



SAMPLE_REPORT_TEXT = """
METRO CITY DIAGNOSTIC CENTER
Total Cholesterol 190 mg/dL 0-200
HDL Cholesterol 42 mg/dL 40-100
LDL Cholesterol 128 mg/dL 0-130
Creatinine 1.2 mg/dL 0.6-1.3
Urea 24.0 mg/dL 15-40
"""


class TestRegexExtraction:
    def test_extracts_known_values(self,extractor):
        result = extractor._extract_regex(SAMPLE_REPORT_TEXT)
        assert "Total Cholesterol" in result
        assert result["Total Cholesterol"]["value"] == 190.0
        assert result["Total Cholesterol"]["ref_low"] == 0.0
        assert result["Total Cholesterol"]["ref_high"] == 200.0


    def test_returns_empty_dict_on_unparseable_text(self,extractor):
        result = extractor._extract_regex("This text has no lab values at all")
        assert  result == {}


    def test_handles_multiple_tests_in_one_report(self,extractor):
        result = extractor._extract_regex(SAMPLE_REPORT_TEXT) 
        assert len(result) >= 4   




# class TestVerifyAgainstSource:
#     def test_accepts_value_present_in_source_text(self,extractor):
#         extracted = {
#             "Total Cholesterol": {"value": 190.0,"unit":"mg/dL","ref_low":0.0,"ref_high":200.0}
#         }
#         raw_text = "Total Cholesterol 190 mg/dL 0-200"
#         result = extractor._verify_against_source(extracted,raw_text)
#         assert result["Total Cholesterol"] is not None
#         assert result["Total Cholesterol"]["value"] == 190.0

#     def test_discards_value_not_present_in_source_text(self,extractor):
#         extracted = {
#             "Phantom Test": {"value": 777.0,"unit": "mg/dL","ref_low":0.0,"ref_high":100.0}}
#         raw_text = "Total Cholesterol 190 mg/dL 0-200"
#         result = extractor._verify_against_source(extracted,raw_text)
#         assert result["Phantom Test"] is  None

#     def test_discards_degenerate_zero_values(self,extractor):
#         extracted = {
#             "Triglycerides": {"value": 0.0,"unit":"mg/dL","ref_low":0.0,"ref_high":150.0}}
#         raw_text = "Triglycerides Not Done mg/dL <150"
#         result = extractor._verify_against_source(extracted,raw_text)
#         assert result["Triglycerides"] is None


#     def test_passes_through_none_values_unchanged(self,extractor):
#         extracted = {"Some Test": None}
#         result = extractor._verify_against_source(extracted,"irrelevant text")
#         assert result["Some Test"] is None

class TestLLMExtraction:
    @patch.object(LabValueExtractor,"_extract_llm")
    def test_extract_falls_back_to_llm_when_regex_finds_nothing(self,mock_llm,extractor):
        mock_llm.return_value = {
            "Total Cholesterol": {"value": 190.0, "unit": "mg/dL", "ref_low": 0.0, "ref_high": 200.0}
        }
        result = extractor.extract("some unparseable scanned text")
        mock_llm.assert_called_once()
        assert "Total Cholesterol" in result

    @patch.object(LabValueExtractor,"_extract_llm")
    def test_extract_skips_llm_when_regex_succeeds(self,mock_llm,extractor):
        extractor.extract(SAMPLE_REPORT_TEXT)
        mock_llm.assert_not_called()



class TestAnalyzeValues:
    def test_flags_high_value_correctly(self):
        parsed = {"Total Cholesterol":{"value":220.0,"ref_low":0.0,"ref_high":200.0}}
        result = analyze_values(parsed)
        assert result["Total Cholesterol"]["status"] == "high"            

    def test_flags_low_value_correctly(self):
            parsed = {"HDL Cholesterol":{"value":30.0,"ref_low":40.0,"ref_high":100.0}}
            result = analyze_values(parsed)
            assert result["HDL Cholesterol"]["status"] == "low"     

    def test_flags_normal_value_correctly(self):
            parsed = {"Creatinine":{"value":1.0,"ref_low":0.6,"ref_high":1.3}}
            result = analyze_values(parsed)
            assert result["Creatinine"]["status"] == "normal"   

    def test_does_not_crash_on_none_value(self):
            parsed = {"Triglycerides": None}
            result = analyze_values(parsed)
            assert result["Triglycerides"]["status"] == "not_available"            
                     

class TestCalculateRiskScore:
    def test_returns_safe_for_all_normal_values(self):
        analysis = {
            "Test A": {"value": 10,"status": "normal","severity":0},
            "Test B":{"value": 20,"status": "normal","severity":0},

          } 
        score,triage = calculate_risk_score(analysis)
        assert score == 0
        assert "Safe" in triage

    def test_excludes_not_available_from_score_calculation(self):
        analysis = {
            "Test A": {"value": 300, "status": "high", "severity": 1.0},
            "Test B": {"value": None, "status": "not_available", "severity": 0},
        }
        score,triage = calculate_risk_score(analysis)
        assert score == 100


    def  test_handles_all_values_missing_without_crashing(self):
        analysis = {
            "Test A": {"value": None, "status": "not_available", "severity": 0},
        }
        score, triage = calculate_risk_score(analysis)
        assert score == 0
        assert "Insufficient" in triage or "insufficient" in triage.lower()  

              
            






                          
              

                  

                  
