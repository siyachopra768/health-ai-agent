"""
parser.py — Hybrid Lab Value Extraction Pipeline
-------------------------------------------------
Stage 1: Regex-based extraction (fast, deterministic, zero cost)
Stage 2: LLM-based extraction (fallback when regex yields nothing)

Why this matters:
- Regex works perfectly on standard lab report formats
- LLM fallback handles non-standard layouts, scanned PDFs, varied formatting
- This is a classic reliability vs. cost tradeoff in production AI systems
"""

from pypdf import PdfReader
import re
import json
import os
from groq import Groq

# ── PDF Loading ───────────────────────────────────────────────────────────────

def load_pdf(file) -> str:
    """Extract raw text from a PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


# ── Stage 1: Regex Extraction ─────────────────────────────────────────────────

def extract_lab_values_regex(text: str) -> dict:
    """
    Fast regex-based extraction for standard lab report formats.
    
    Matches patterns like:
      Hemoglobin   13.5   g/dL   12.0-17.0
      WBC Count    7.2    10^3/µL  4.0-11.0
    
    Returns dict of { test_name: { value, unit, ref_low, ref_high } }
    """
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


# ── Stage 2: LLM Fallback Extraction ─────────────────────────────────────────

def extract_lab_values_llm(text: str) -> dict:
    """
    LLM-based extraction for non-standard or complex lab report formats.
    
    Used when regex returns zero results — handles:
    - Varied column layouts
    - Verbose narrative reports
    - Inconsistent spacing or formatting
    
    Returns same dict format as regex extractor for seamless integration.
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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
- Only include tests where you can find a numeric value AND a reference range
- If reference range is missing, skip that test
- Use exact test names from the report
- Return empty JSON {} if nothing can be extracted"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract lab values from this report:\n\n{text[:3000]}"}
            ],
            temperature=0,       # deterministic output
            max_tokens=1000,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"```json|```", "", raw).strip()

        parsed = json.loads(raw)

        # Validate structure — ensure each entry has required fields
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

        return validated

    except (json.JSONDecodeError, Exception):
        # LLM also failed — return empty, caller handles it
        return {}


# ── Hybrid Pipeline (Main Entry Point) ───────────────────────────────────────

def extract_lab_values(text: str) -> dict:
    """
    Two-stage hybrid extraction pipeline:
    
    Stage 1 → Regex (fast, free, deterministic)
    Stage 2 → LLM fallback (if regex finds nothing)
    
    This design gives us:
    - Speed and zero cost on standard reports
    - Robustness on edge cases without over-engineering
    - A clear, explainable architecture decision
    
    Returns: dict of lab values in unified format
    """
    # Stage 1: Try regex first
    results = extract_lab_values_regex(text)

    if results:
        # Regex succeeded — log which stage was used (useful in production)
        print(f"[Parser] Stage 1 (Regex): Extracted {len(results)} lab values.")
        return results

    # Stage 2: Regex found nothing — fall back to LLM
    print("[Parser] Stage 1 (Regex): No values found. Falling back to LLM extraction...")
    results = extract_lab_values_llm(text)

    if results:
        print(f"[Parser] Stage 2 (LLM Fallback): Extracted {len(results)} lab values.")
    else:
        print("[Parser] Stage 2 (LLM Fallback): No values extracted. Report may be unreadable.")

    return results
