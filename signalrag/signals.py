"""
signals.py — SignalExtractor class for SignalRAG.

Loads the sentence embedder and the 4-bit quantised LLM once and exposes
three extraction methods that together produce the 17 externally observable
signals used by the supervised classifier.

Signal groups
-------------
Retrieval  (6): retriever_score, doc_similarity, ret_score_std,
                second_passage_sim, n_docs_retrieved, context_length
Generation (8): entropy, avg_token_prob, min_token_prob, var_token_prob,
                top2_margin, tail_prob, answer_length, answer_is_letter_only
Alignment  (3): context_answer_overlap, embed_sim, rouge_l
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rouge_score import rouge_scorer as rouge_scorer_lib

from .config import LLM_ID, EMBED_ID, MAX_NEW_TOK
from .utils import normalise_text

_RETRIEVAL_KEYS = [
    "retriever_score", "doc_similarity", "ret_score_std",
    "second_passage_sim", "n_docs_retrieved", "context_length",
]


class SignalExtractor:
    """
    Loads the sentence embedder and the quantised LLM once at construction
    time and provides methods to extract all three signal groups for a given
    query / passage / generated-answer triple.

    Parameters
    ----------
    device : str, optional
        "cuda" or "cpu". Defaults to auto-detect.
    load_llm : bool, optional
        Set False when signal CSVs already exist and generation is not needed.
    """

    def __init__(self, device: str | None = None, load_llm: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print("[1] Loading sentence embedder (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer(EMBED_ID)

        self.tokenizer = None
        self.llm = None

        if load_llm:
            print(f"[2] Loading LLM: {LLM_ID} (4-bit NF4 quantisation)...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
            self.llm = AutoModelForCausalLM.from_pretrained(
                LLM_ID,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.llm.eval()
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"    VRAM: ~5 GB (4-bit) | Device: {self.device}")

        print("SignalExtractor ready.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, query: str, passages: list[str]) -> tuple[str, object, list]:
        """
        Generate an answer for query given passages.

        Returns
        -------
        (generated_text, hf_output, gen_ids)
            gen_ids is the list of generated token ids (excluding prompt).
        """
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("LLM not loaded. Construct with load_llm=True.")

        context = "\n\n".join(
            [f"Passage {i+1}: {p}" for i, p in enumerate(passages)]
        )[:3000]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer the question based ONLY on the provided passages. "
                    "Be concise and factual. "
                    "Do not add information not present in the passages."
                ),
            },
            {
                "role": "user",
                "content": f"Passages:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            },
        ]

        prompt_str = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(
            prompt_str, return_tensors="pt", truncation=True, max_length=2048
        )
        input_ids = inputs["input_ids"].to(self.device)
        prompt_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = self.llm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOK,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = output.sequences[0][prompt_len:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return gen_text, output, gen_ids.tolist()

    def extract_generation_signals(
        self, output, gen_ids: list, generated_text: str
    ) -> dict:
        """
        Compute the 8 generation signals from the LLM output object.

        Parameters
        ----------
        output : HuggingFace GenerateOutput
        gen_ids : list of int — generated token ids
        generated_text : str — decoded answer (used for answer_is_letter_only)
        """
        token_log_probs = []
        step_entropies  = []

        for i, step_logits in enumerate(output.scores):
            lp    = torch.nn.functional.log_softmax(step_logits[0], dim=-1)
            probs = torch.exp(lp)
            cid   = gen_ids[i] if i < len(gen_ids) else 0
            token_log_probs.append(lp[cid].item())
            valid = probs > 1e-10
            step_entropies.append(-(probs[valid] * lp[valid]).sum().item())

        pa  = np.exp(token_log_probs) if token_log_probs else np.array([0.5])

        # top-2 margin at first generation step
        t2m = 0.0
        if output.scores:
            fp = torch.softmax(output.scores[0][0], dim=-1)
            t2 = torch.topk(fp, 2).values
            t2m = float((t2[0] - t2[1]).item())

        return {
            "entropy":              float(np.mean(step_entropies)) if step_entropies else 0.0,
            "avg_token_prob":       float(np.mean(pa)),
            "min_token_prob":       float(np.min(pa)),
            "var_token_prob":       float(np.var(pa)) if len(pa) > 1 else 0.0,
            "top2_margin":          t2m,
            "tail_prob":            float(np.mean(pa[-3:])) if len(pa) >= 3 else float(np.mean(pa)),
            "answer_length":        max(len(token_log_probs), 1),
            "answer_is_letter_only": int(len(normalise_text(generated_text).split()) <= 1),
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def extract_retrieval_signals(self, passages: list[str], query: str) -> dict:
        """
        Compute the 6 retrieval signals from query--passage cosine similarities.
        All embeddings are L2-normalised so cosine similarity == inner product.
        """
        if not passages:
            return {k: 0.0 for k in _RETRIEVAL_KEYS}

        q_emb  = self.embedder.encode([query], normalize_embeddings=True)
        p_embs = self.embedder.encode(passages, normalize_embeddings=True)
        scores = (p_embs @ q_emb.T).flatten()

        sorted_scores = np.sort(scores)
        second = float(sorted_scores[-2]) if len(sorted_scores) >= 2 else float(sorted_scores[-1])

        return {
            "retriever_score":    float(scores.max()),
            "doc_similarity":     float(scores.mean()),
            "ret_score_std":      float(scores.std()),
            "second_passage_sim": second,
            "n_docs_retrieved":   len(passages),
            "context_length":     len(" ".join(passages).split()),
        }

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def extract_alignment_signals(
        self, generated_answer: str, context_text: str
    ) -> dict:
        """
        Compute the 3 alignment signals between the generated answer and
        the retrieved context.
        """
        # Lexical overlap
        ctx_toks = set(context_text.lower().split())
        ans_toks = set(generated_answer.lower().split())
        overlap  = len(ctx_toks & ans_toks) / len(ans_toks) if ans_toks else 0.0

        # Embedding cosine similarity (whole-text)
        try:
            ans_emb = self.embedder.encode([generated_answer], normalize_embeddings=True)
            ctx_emb = self.embedder.encode([context_text[:512]], normalize_embeddings=True)
            embed_sim = float((ans_emb * ctx_emb).sum())
        except Exception:
            embed_sim = 0.0

        # ROUGE-L F1
        try:
            scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)
            rouge_l = scorer.score(context_text[:512], generated_answer)["rougeL"].fmeasure
        except Exception:
            rouge_l = 0.0

        return {
            "context_answer_overlap": overlap,
            "embed_sim":              embed_sim,
            "rouge_l":                rouge_l,
        }

    # ------------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------------

    def extract_all_signals(
        self, query: str, passages: list[str], context: str
    ) -> tuple[str, dict]:
        """
        Run generation and extract all 17 signals in one call.

        Returns
        -------
        (generated_text, signals_dict)
        """
        ret_sigs  = self.extract_retrieval_signals(passages, query)
        gen_text, output, gen_ids = self.generate(query, passages)
        gen_sigs  = self.extract_generation_signals(output, gen_ids, gen_text)
        ali_sigs  = self.extract_alignment_signals(gen_text, context)
        return gen_text, {**ret_sigs, **gen_sigs, **ali_sigs}
