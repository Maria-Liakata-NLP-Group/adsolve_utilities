import bert_score as bertscore
from nltk import sent_tokenize
import numpy as np
import torch



class MHIC:
    def __init__(self):
        self.bs_scorer = bertscore.BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            device="cuda" if torch.cuda.is_available() else "cpu",
            lang="en"
        )
        # deberta-xlarge-mnli's tokenizer config leaves model_max_length at HF's
        # unset-sentinel value (~1e30); the model itself caps at 512 tokens, but
        # newer transformers can't pass that sentinel into its truncation setter.
        self.bs_scorer._tokenizer.model_max_length = 512

    def calculate_metric(self, candidate: str, references: list[str]) -> float:
        results = []
        candidate_sentences = sent_tokenize(candidate)

        for reference in references:
            if not reference.strip():
                # bert_score's empty-string code path calls a tokenizer method
                # that newer transformers versions removed; skip blank references
                # since there's nothing meaningful to compare a summary against.
                continue
            _, R, _ = self.bs_scorer.score(
                cands=candidate_sentences,
                refs=[reference]*len(candidate_sentences)
            )

            results.append(R.max().item())

        return np.mean(results)
