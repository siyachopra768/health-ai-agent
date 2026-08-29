# 🩺 Health AI Agent

An end-to-end, multi-turn AI healthcare assistant designed with deterministic-first guardrails, hybrid RAG, and lab report parsing. Built with Python, FastAPI, Streamlit, and Groq LLM.

---

## 🌟 Key Features

* **Deterministic-First Intent Routing:** Uses a stateful intent router for structured workflows (appointment booking, specialty lookup, hospital selection) with automatic fallback to a LangGraph LLM agent for open-ended queries.
* **Hybrid RAG for Medical Guidelines:** Combines dense and sparse embeddings with cross-encoder reranking to retrieve clinical guidelines with high precision, bypassing LLM generation when confidence is high.
* **Structured Lab Report Ingestion:** Parses patient PDF test reports, extracts laboratory values, and validates metrics against reference ranges using an integrated verifier pipeline.
* **FastAPI Backend & Interactive UI:** Exposes structured endpoints for multi-turn processing while offering a user-friendly Streamlit dashboard for real-time interaction.

---

## 🏗️ Project Architecture
health-ai-agent/
├── app.py                   # Streamlit frontend UI
├── api.py                   # FastAPI backend services
├── parser.py                # PDF lab report parsing logic
├── verifier.py              # Reference range validation & extraction verifier
├── utils.py                 # Utility & helper functions
├── booking_appointment/     # Stateful intent router & booking workflow
├── rag/                     # Hybrid search, vector indexing, & retrieval logic
├── data/                    # Clinical guidelines & sample data
├── test_parser.py           # Unit tests for lab report parsing
└── test_verifier.py         # Unit tests for output verification


---

## 🚀 Quick Start

### 1. Prerequisites
* Python 3.10+
* Groq API Key

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone [https://github.com/siyachopra768/health-ai-agent.git](https://github.com/siyachopra768/health-ai-agent.git)
cd health-ai-agent
pip install -r requirements.txt
Set your environment variables (create a .env file in the root directory):

Code snippet
GROQ_API_KEY=your_groq_api_key_here
3. Running the Application
Start the FastAPI backend:

Bash
uvicorn api:app --reload
Launch the Streamlit frontend:

Bash
streamlit run app.py
🧪 Testing
Run the test suite to verify the parser and verifier pipeline:

Bash
pytest test_parser.py test_verifier.py


'''

## 🧪 Evaluation Suite

Comprehensive LLM eval framework covering all four core components. Run the full suite or individual modules:

```bash
# Full evaluation suite (all 4 modules → JSON + markdown report)
python -m evals.eval_suite

# Individual modules
python -m evals.eval_lab_parser       # Lab report extraction (P/R/F1, hallucination rate)
python -m evals.eval_intent_router    # Intent classification + entity F1
python -m evals.eval_rag            # RAG retrieval (Precision@k, Recall@k, MRR@k)
python -m evals.eval_hybrid_router   # End-to-end routing accuracy

# pytest (all modules, all cases)
pytest evals/ -v
```

### Outputs
| File | Description |
|------|-------------|
| `eval_suite_results.json` | Full structured results from all modules |
| `eval_suite_report.md` | Human-readable markdown summary with grades |

### Metrics by Module

| Module | Metrics |
|--------|---------|
| **Lab Parser** | Precision, Recall, F1, Hallucination Rate, Verifier Filter Rate, Regex Coverage |
| **Intent Router** | Intent Accuracy, Entity F1 (per field), Missing Detection Rate, Actionable Accuracy |
| **RAG Retrieval** | Precision@3, Recall@3, MRR@3, Coverage, Stage Comparison (dense/sparse/hybrid) |
| **Hybrid Router** | Route Accuracy, Outcome Accuracy, Multi-turn Success Rate, Latency |

### Test Data
All test cases are in `evals/eval_data/` as JSON:
- `lab_reports.json` — Lab report text + expected extractions
- `intent_cases.json` — Intent classification cases with entities
- `rag_queries.json` — RAG query + relevant chunk ground truth
- `router_cases.json` — End-to-end routing scenarios (single & multi-turn)

---
🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.
