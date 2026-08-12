from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import logging
# -----------------------------
# 🤖 LLM
# -----------------------------
def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )


# -----------------------------
# 🧪 ANALYSIS
# -----------------------------
def analyze_values(parsed):
    results = {}

    for test, info in parsed.items():
        if info is None:
            results[test] = { 
                "value":None,
                "status" : "not_available",
                "severity":0,
                "note":"value could not be reliably extracted"
                
            }
            continue

        value = info["value"]
        low = info["ref_low"]
        high = info["ref_high"]

        if value < low:
            status = "low"
            severity = (low - value) / (high - low + 1e-6)
        elif value > high:
            status = "high"
            severity = (value - high) / (high - low + 1e-6)
        else:
            status = "normal"
            severity = 0

        logging.info(f"RISK_CHECK test = {test} status={status} low={low} high={high} value={value} -> severity={round(severity,3)}")

        results[test] = {
            "value": value,
            "status": status,
            "severity": round(severity, 2)
        }

    return results


# -----------------------------
# 📊 RISK SCORE
# -----------------------------
def calculate_risk_score(analysis):
    total_severity = 0
    count = 0

    for test, info in analysis.items():
        if info.get("status") == "not_available":
            continue
        total_severity += info["severity"]
        count += 1

    if count == 0:
        logging.warning("RISK_SCORE count=0 — no lab values to score, returning 0")
        return 0,"⚠️ Insufficient data to assess risk"
    
    avg_severity = total_severity / count
    score = int(min(avg_severity * 100, 100))
    logging.info(f"RISK_SCORE avg_severity={round(avg_severity,3)} count={count} -> score={score}")

    if score > 70:
        triage = "🚨 Emergency"
    elif score > 30:
        triage = "⚠️ Moderate"
    else:
        triage = "✅ Safe"

    return score, triage