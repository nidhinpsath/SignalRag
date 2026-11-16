"""
labeling.py — EnsembleJudge class for SignalRAG.

Three Groq-hosted LLM judges vote independently on whether a generated
answer is faithful to the retrieved context. The final label is the
majority vote (1 = faithful, 0 = hallucinated).

Judge models
------------
GPT-oss-120b  (OpenAI  via Groq)  — GROQ_API_KEY_1
GPT-oss-20b   (OpenAI  via Groq)  — GROQ_API_KEY_2
Qwen3.8-27b   (Alibaba via Groq)  — GROQ_API_KEY_3

API keys are read from environment variables. Store them in a .env file
and load with python-dotenv (see notebooks/01_data_preparation.ipynb).
"""

import os
import re
import time

from groq import Groq

from .config import JUDGE_SPECS


class EnsembleJudge:
    """
    Three-judge LLM ensemble for faithfulness labelling.

    Instantiating this class reads the three Groq API keys from the
    environment and initialises one Groq client per judge.

    Parameters
    ----------
    max_retries : int
        Number of times to retry a single judge call before returning -1.
    """

    _PROMPT_TEMPLATE = (
        "Retrieved context:\n{context}\n\n"
        "Generated answer: {answer}\n\n"
        "Is the generated answer faithful to the retrieved context? "
        "Faithful means the answer only contains information supported "
        "by the context and does not add unsupported claims.\n"
        "Reply with one word only: FAITHFUL or HALLUCINATED"
    )

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._judges = []
        for name, env_key, model_id in JUDGE_SPECS:
            api_key = os.environ.get(env_key)
            if not api_key:
                raise EnvironmentError(
                    f"Environment variable '{env_key}' not set for judge '{name}'. "
                    f"Add it to your .env file."
                )
            self._judges.append((name, Groq(api_key=api_key), model_id))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_one_judge(self, client: Groq, model: str, prompt: str) -> int:
        """
        Call a single judge. Returns 1 (faithful), 0 (hallucinated), or -1 (failed).
        """
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=1000,
                )
                raw = response.choices[0].message.content.strip().upper()
                if "FAITHFUL" in raw:
                    return 1
                if "HALLUCINATED" in raw:
                    return 0
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                return -1

            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    wait_match = re.search(r"try again in ([\d.]+)s", err)
                    wait_time  = float(wait_match.group(1)) + 1.0 if wait_match else 10.0
                    print(f"  Rate limit ({model[:20]}) — waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                if "day" in err.lower() and "limit" in err.lower():
                    print(f"  Daily limit reached: {model[:30]}")
                    return -1
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  Judge failed ({model[:30]}): {err[:60]}")
                    return -1
        return -1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def judge(self, generated_answer: str, context: str) -> int:
        """
        Run all three judges and return the majority vote label.

        Returns
        -------
        1  — faithful (majority vote)
        0  — hallucinated (majority vote, or conservative tie-break)
        -1 — all judges failed; caller should discard this example
        """
        prompt = self._PROMPT_TEMPLATE.format(
            context=context[:600],
            answer=generated_answer,
        )

        votes = [
            self._call_one_judge(client, model, prompt)
            for _, client, model in self._judges
        ]

        valid = [v for v in votes if v != -1]
        if not valid:
            return -1

        # Majority vote; when two valid votes tie (one judge failed),
        # apply the conservative label HALLUCINATED.
        faithful_votes = sum(valid)
        return 1 if faithful_votes > len(valid) / 2 else 0

    def verify(self) -> None:
        """
        Quick sanity check: send a known-faithful example to all judges
        and print each response. Useful at the start of a notebook run.
        """
        test_answer  = "Keep bagels in a sealed container."
        test_context = "Bagels stay fresh in an airtight container at room temperature."
        prompt = self._PROMPT_TEMPLATE.format(
            context=test_context, answer=test_answer
        )
        print("Verifying 3 judges...")
        for name, client, model in self._judges:
            v = self._call_one_judge(client, model, prompt)
            status = "FAITHFUL ✓" if v == 1 else "HALLUCINATED ✗" if v == 0 else "FAILED ✗"
            print(f"  {name}: {status}")
