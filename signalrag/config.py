"""
config.py — global constants for SignalRAG.

All tunable parameters live here. No other module hardcodes values.
"""

import os

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
RAGTRUTH_ID  = "wandb/RAGTruth-processed"
TASK_FILTER  = "QA"
N_TRAIN      = 200
N_TEST       = 100
SEED         = 42

# ---------------------------------------------------------------------------
# Retrieval / generation
# ---------------------------------------------------------------------------
TOP_K        = 3
MAX_NEW_TOK  = 100

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
LLM_ID   = "Qwen/Qwen2.5-7B-Instruct"
EMBED_ID = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Judge models (served via Groq)
# ---------------------------------------------------------------------------
JUDGE_SPECS = [
    ("GPT-oss-120b (OpenAI)",  "GROQ_API_KEY_1", "openai/gpt-oss-120b"),
    ("GPT-oss-20b (OpenAI)",   "GROQ_API_KEY_2", "openai/gpt-oss-20b"),
    ("Qwen3.8-27b (Alibaba)",  "GROQ_API_KEY_3", "qwen/qwen3.8-27b"),
]

# ---------------------------------------------------------------------------
# Cache paths  (absolute, so notebooks run from any working directory)
# ---------------------------------------------------------------------------
_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(_ROOT, "data", "ragtruth_train_signals.csv")
TEST_CSV  = os.path.join(_ROOT, "data", "ragtruth_test_signals.csv")

# ---------------------------------------------------------------------------
# Feature columns (order must match extraction output)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # Retrieval (6)
    "retriever_score",
    "doc_similarity",
    "ret_score_std",
    "second_passage_sim",
    "n_docs_retrieved",
    "context_length",
    # Generation (8)
    "entropy",
    "avg_token_prob",
    "min_token_prob",
    "var_token_prob",
    "top2_margin",
    "tail_prob",
    "answer_length",
    "answer_is_letter_only",
    # Alignment (3)
    "context_answer_overlap",
    "embed_sim",
    "rouge_l",
]

LABEL_COL = "reliability_label"
