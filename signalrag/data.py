"""
data.py — dataset loading and splitting for SignalRAG.

Provides load_ragtruth(), which loads the RAGTruth QA subset from
HuggingFace, filters to good-quality non-duplicate examples, and
returns a list of records ready for the pipeline.
"""

import numpy as np
from datasets import load_dataset

from .config import RAGTRUTH_ID, TASK_FILTER, TOP_K, N_TRAIN, SEED


def load_ragtruth(n_samples: int, split: str = "train") -> list[dict]:
    """
    Load n_samples records from the RAGTruth QA subset.

    Parameters
    ----------
    n_samples : int
        Number of examples to return.
    split : str
        "train" returns the first n_samples after shuffling.
        "test"  returns the next N_TRAIN:N_TRAIN+n_samples slice,
                ensuring no overlap with the training split.

    Returns
    -------
    list of dict with keys: query, passages (list[str]), context (str).
    """
    print(f"  Loading RAGTruth {split} split ({TASK_FILTER} task)...")
    ds = load_dataset(RAGTRUTH_ID, split="train")
    df = ds.to_pandas()

    df = df[df["task_type"].str.upper() == TASK_FILTER.upper()]
    df = df[df["quality"] == "good"].reset_index(drop=True)
    df = df.drop_duplicates(subset=["query"]).reset_index(drop=True)

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    if split == "train":
        df = df.iloc[:n_samples]
    else:
        df = df.iloc[N_TRAIN : N_TRAIN + n_samples]

    records = []
    for _, row in df.iterrows():
        context = str(row.get("context", ""))
        if not context.strip():
            continue
        passages = [p.strip() for p in context.split("\n\n") if p.strip()]
        if not passages:
            passages = [context]

        records.append({
            "query":    str(row.get("query", "")),
            "passages": passages[:TOP_K],
            "context":  context,
        })

    print(f"    Loaded {len(records)} records")
    return records
