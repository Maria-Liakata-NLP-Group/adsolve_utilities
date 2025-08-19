import argparse
import json
import numpy as np
from nltk import sent_tokenize
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from tqdm import tqdm

class NLIScorer:
    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)

    def _compute_nli_scores(self, premise: str, hypothesis: str):
        input = self.tokenizer(
            premise, hypothesis, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input["input_ids"].to(self.device))
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        prediction = {name: float(pred) for pred, name in zip(prediction, label_names)}
        return prediction

    def compute_entailment_score(self, premise: str, hypothesis: str):
        return self._compute_nli_scores(premise, hypothesis)["entailment"]

    def compute_contradiction_score(self, premise: str, hypothesis: str):
        return self._compute_nli_scores(premise, hypothesis)["contradiction"]

    def compute_nli_scores(
        self,
        source_sents: List[str],
        predicted_sents: List[str],
        task: str,
        prefix: str,
        source_name: str,
    ):
        """
        Args:
            source_sents: List of source sentences to compare against
            predicted_sents: List of predicted sentences
            task: Task ID ('B' or 'C')
            prefix: Prefix for metric names ('post' or 'timeline')
            source_name: Name of the source ('gold' or 'post' or 'timeline')

        Returns:
            Dictionary with computed NLI metrics
        """
        entail_scores, contradict_scores = [], []
        if predicted_sents:
            for source_sent in source_sents:
                for predicted_sent in predicted_sents:
                    scores = self._compute_nli_scores(source_sent, predicted_sent)
                    entail_scores.append(scores["entailment"])
                    contradict_scores.append(scores["contradiction"])
            entail_scores, contradict_scores = np.array(entail_scores), np.array(
                contradict_scores
            )
            mean_consistency = 1 - contradict_scores.mean()
            max_entailment = entail_scores.max()
            max_contradiction = contradict_scores.max()
        else:
            mean_consistency = 0.0
            max_entailment = 0.0
            max_contradiction = 1.0
        
        consistency_scores = []
        # consisteny scores for each predict sentence against all source sentences
        for i, predicted_sent in enumerate(predicted_sents):
            sentence_level_consistency = [
                1 - contradict_scores[j * len(predicted_sents) + i]
                for j in range(len(source_sents))
            ]
            consistency_scores.append(np.mean(sentence_level_consistency))
        return {"mean_consistency": mean_consistency, 
                "sentence_level_consistencies": consistency_scores,
                "sentences": predicted_sents,
                }

    def compute_post_nli_gold(self, gold_sents: List[str], predicted_sents: List[str]):
        """Compute NLI scores for post-level summary against gold summary."""
        return self.compute_nli_scores(
            source_sents=gold_sents,
            predicted_sents=predicted_sents,
            task="B",
            prefix="post",
            source_name="gold",
        )

    def compute_post_nli_post(self, post_sents: List[str], predicted_sents: List[str]):
        """Compute NLI scores for post-level summary against post content."""
        return self.compute_nli_scores(
            source_sents=post_sents,
            predicted_sents=predicted_sents,
            task="B",
            prefix="post",
            source_name="post",
        )

    def compute_timeline_nli_gold(
        self, gold_sents: List[str], predicted_sents: List[str]
    ):
        """Compute NLI scores for timeline-level summary against timeline summary."""
        return self.compute_nli_scores(
            source_sents=gold_sents,
            predicted_sents=predicted_sents,
            task="C",
            prefix="timeline",
            source_name="gold",
        )

    def compute_timeline_nli_timeline(
        self, timeline_sents: List[str], predicted_sents: List[str]
    ):
        """Compute NLI scores for timeline-level summary against timeline content."""
        return self.compute_nli_scores(
            source_sents=timeline_sents,
            predicted_sents=predicted_sents,
            task="C",
            prefix="timeline",
            source_name="timeline",
        )



class FactualConsistency:
    def __init__(self):
        self.ns = NLIScorer()
    
    
    def _calculate_fc_expert_metric(self, llm_text: str, gold_text: str) -> float:
        llm_sentences = sent_tokenize(llm_text)
        gold_sentences = sent_tokenize(gold_text)
        # Compute NLI scores
        result = self.ns.compute_timeline_nli_gold(gold_sentences, llm_sentences)

        return result
    
    def _calculate_fc_timeline_metric(self, llm_text: str, timeline_text: str) -> float:
        llm_sentences = sent_tokenize(llm_text)
        timeline_sentences = sent_tokenize(timeline_text)
        # Compute NLI scores
        result = self.ns.compute_timeline_nli_timeline(timeline_sentences, llm_sentences)

        return result
    
    def calculate_metric(self, llm_text: str, reference_text: str | List[str]) -> float:

        if isinstance(reference_text, str):
            # If reference_text is a single string it is treated as the gold summary
            result = self._calculate_fc_expert_metric(llm_text, reference_text)
            # split into right format
            return result["mean_consistency"], {"scores": result["sentence_level_consistencies"], "sentences": result["sentences"]}
        elif isinstance(reference_text, list):
            # If reference_text is a list, it is treated as the timeline posts
            # convert timeline posts to a single string
            reference_text = " ".join(reference_text)
            return self._calculate_fc_timeline_metric(llm_text, reference_text)
        else:
            raise ValueError("reference_text must be a string or a list of strings")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate factual consisteny.")
    parser.add_argument('--llm_summaries', type=str, help=' Path to the LLM summaries JSON file.')
    parser.add_argument('--gold_summaries', type=str, help='Path to the gold summaries JSON file.')
    parser.add_argument('--combined_summaries', type=str, default=None, help='Path to the combined summaries JSON file (optional). If provided, it will be used instead of LLM summaries and gold summaries.')
    parser.add_argument('--output_file', type=str, default='factual_consisteny_evaluation_results.json', help='Path to save the evaluation results JSON file.')
    args = parser.parse_args()

    if args.combined_summaries:
        # Load combined summaries
        print(f"Loading combined summaries from {args.combined_summaries}")
        with open(args.combined_summaries, 'r') as f:
            combined_summaries = json.load(f)
        llm_summaries = {key: value["summary"] for key, value in combined_summaries.items()}    
        gold_summaries = {key: value["reference"] for key, value in combined_summaries.items()}
    elif args.llm_summaries and args.gold_summaries:
        # Load LLM summaries
        print(f"Loading LLM summaries from {args.llm_summaries}")
        with open(args.llm_summaries, 'r') as f:
            llm_summaries = json.load(f)    
        # Load gold summaries
        print(f"Loading gold summaries from {args.gold_summaries}")
        with open(args.gold_summaries, 'r') as f:
            gold_summaries = json.load(f) 
    else:
        raise ValueError("Either --combined_summaries or both --llm_summaries and --gold_summaries must be provided.") 

    fc_expert = FactualConsistency()

    results = {
                'fc_expert': {
                "document_level": [],
                "mean": None,
                "detail": [],
                },
                'document_ids': list(llm_summaries.keys())
            }
    
    for document_id in tqdm(results['document_ids'], desc="Evaluating Factual Consistency"):
        llm_summary = llm_summaries[document_id]
        gold_summary = gold_summaries[document_id]

        # Calculate factual consistency expert score
        fc_expert_score, fc_expert_detail = fc_expert.calculate_metric(llm_summary, gold_summary)
        results['fc_expert']['document_level'].append(fc_expert_score)
        results['fc_expert']['detail'].append(fc_expert_detail)
    
    # Calculate mean scores
    results['fc_expert']['mean'] = sum(results['fc_expert']['document_level']) / len(results['fc_expert']['document_level'])
    
    print(f"Saving evaluation results to {args.output_file}")
    output_file = args.output_file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
