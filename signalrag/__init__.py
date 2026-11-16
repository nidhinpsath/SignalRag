"""
SignalRAG — black-box hallucination detection for RAG systems.

Public API
----------
SignalExtractor  — extract all 17 signals from a query/passage/answer triple
EnsembleJudge    — 3-judge LLM ensemble for faithfulness labelling
SignalRAGClassifier — supervised classifier wrapping an sklearn estimator + scaler
load_ragtruth    — load and filter the RAGTruth QA subset
"""

from .signals    import SignalExtractor
from .labeling   import EnsembleJudge
from .classifier import SignalRAGClassifier
from .data       import load_ragtruth

__all__ = [
    "SignalExtractor",
    "EnsembleJudge",
    "SignalRAGClassifier",
    "load_ragtruth",
]
