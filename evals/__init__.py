"""
evals/ — LLM evaluation framework for AI Health Agent.

Modules:
  eval_lab_parser.py      Lab report extraction (precision, recall, hallucination rate)
  eval_intent_router.py   Deterministic intent classification + entity F1
  eval_rag.py            Hybrid retrieval (Precision@k, Recall@k, MRR@k)
  eval_hybrid_router.py   End-to-end routing (path accuracy, multi-turn booking)
  eval_suite.py          Unified runner → JSON + markdown report

Run everything:  python -m evals.eval_suite
Run individual:  python -m evals.eval_lab_parser
                python -m evals.eval_intent_router
                python -m evals.eval_rag
                python -m evals.eval_hybrid_router
Run pytest:      pytest evals/ -v
"""
