from langchain_groq import ChatGroq
import os

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
        total_severity += info["severity"]
        count += 1

    avg_severity = total_severity / (count + 1e-6)
    score = int(min(avg_severity * 100, 100))

    if score > 70:
        triage = "🚨 Emergency"
    elif score > 30:
        triage = "⚠️ Moderate"
    else:
        triage = "✅ Safe"

    return score, triage