"""
utils.py — shared stateless helpers for SignalRAG.
"""

import re


def normalise_text(s: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    s = str(s).lower().strip()
    s = re.sub(r"[\W_]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()
