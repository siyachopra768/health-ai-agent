
from pypdf import PdfReader
import re
import json
import os
from groq import Groq


class LabValueExtractor:
    """Two-stage hybrid extractor: regex first, LLM fallback."""

    def __init__(self, groq_api_key: str | None = None):
        self.client = Groq(api_key=groq_api_key or os.environ.get("GROQ_API_KEY"))


    def load_pdf(self, file) -> str:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    
    def _normalize(self,value) -> str:
        """strip formattiong differences before comparing against source text."""
        return str(value).replace(".0","").strip()
    
    def _verify_against_source(self,extracted: dict,raw_text: str)-> dict:
        verified ={}
        for test_name,data in extracted.items():
            if data is None or "value" not in data:
                verified[test_name] = data
                continue
            numeric_value = data["value"]
            normalized = self._normalize(numeric_value)

            if numeric_value == 0.0 or len(normalized) < 2:
                print(f"[Hallucination Check] '{test_name}': suspicious/degenerate value{numeric_value} - discarded")
                verified[test_name] = None
                continue

            if normalized in raw_text:
                verified[test_name] = data

            else:
                print(f"[Hallucination Check ] '{test_name}' : {numeric_value} not found in source-discarding")
                verified[test_name] =  None


        return verified  

    def extract(self, text: str) -> dict:
        """Public entry point — same job as old extract_lab_values()."""
        results = self._extract_regex(text)

        if results:
            print(f"[Parser] Stage 1 (Regex): Extracted {len(results)} lab values.")
            return results

        print("[Parser] Stage 1 (Regex): No values found. Falling back to LLM extraction...")
        results = self._extract_llm(text)

        if results:
            print(f"[Parser] Stage 2 (LLM Fallback): Extracted {len(results)} lab values.")
        else:
            print("[Parser] Stage 2 (LLM Fallback): No values extracted. Report may be unreadable.")

        return results

    def _extract_regex(self, text: str) -> dict:
        data = {}
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            match = re.search(
                r"([A-Za-z ()/%]+)\s+([\d.]+)\s+([a-zA-Z/%^0-9µ]+)\s+([\d.]+[-–][\d.]+)",
                line
            )
            if match:
                name, value, unit, ref = match.groups()
                try:
                    low, high = re.split(r"[-–]", ref)
                    data[name.strip()] = {
                        "value": float(value),
                        "unit": unit,
                        "ref_low": float(low),
                        "ref_high": float(high)
                    }
                except (ValueError, TypeError):
                    continue

        return data

    def _extract_llm(self, text: str) -> dict:
        system_prompt = """You are a medical data extraction specialist.
Extract lab test results from the given medical report text.

Return ONLY a valid JSON object in this exact format (no explanation, no markdown):
{
  "Test Name": {
    "value": <numeric value as float>,
    "unit": "<unit string>",
    "ref_low": <lower reference range as float>,
    "ref_high": <upper reference range as float>
  }
}

Rules:
    Extract every test mentioned in the report, including Triglycerides.
- If a value is not clearly stated, estimate a typical/reasonable value based on the reference range provided.
- Use exact test names from the report
- Return empty JSON {} if nothing can be extracted"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract lab values from this report:\n\n{text[:3000]}"}
                ],
                temperature=0,
                max_tokens=1000,
            )

            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()

            parsed = json.loads(raw)

            validated = {}
            for name, vals in parsed.items():
                if all(k in vals for k in ["value", "unit", "ref_low", "ref_high"]):
                    try:
                        validated[name] = {
                            "value": float(vals["value"]),
                            "unit": str(vals["unit"]),
                            "ref_low": float(vals["ref_low"]),
                            "ref_high": float(vals["ref_high"])
                        }
                    except (ValueError, TypeError):
                        continue
            verified = self._verify_against_source(validated,text)
            print("DEBUG validated (before check):", validated)
            print("DEBUG verified (after check):", verified)

            return verified

        except json.JSONDecodeError:
            return {}