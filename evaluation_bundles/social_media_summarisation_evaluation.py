# add metrics folder to paths from which to import classes
import argparse
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/metrics")

from mhic import MHIC
from intra_nli import IntraNLI
from fc import FactualConsistency
from style_roberta import StyleSimilarity
from bertscore import BERTScore

import numpy as np


class SocialMediaSummarisationEvaluationBundle:
    def __init__(self):
        self.mhic = MHIC()
        self.intra_nli = IntraNLI()
        self.fc = FactualConsistency()
        self.style_similarity = StyleSimilarity()
        self.bert_score = BERTScore()

    def evaluate(self, posts: dict, llm_summaries: dict, gold_summaries: dict) -> dict:
        # extract document ids from posts dict
        document_ids = list(posts.keys())

        # Dictionary to store results
        results = {
            'mhic': {
                "document_level": [],
                "mean": None,
            },
            'intra_nli': {
                "document_level": [],
                "mean": None,
            },
            'fc_expert': {
                "document_level": [],
                "mean": None,
            },
            'fc_document': {
                "document_level": [],
                "mean": None,
            },
            'style_similarity': {
                "document_level": [],
                "mean": None,
            },
            'bert_score': {
                "document_level": [],
                "mean": None,
            },
            'document_ids': document_ids
        }

        # iterate over each document
        for document_id in document_ids:
            document_posts = posts[document_id]
            llm_summary = llm_summaries[document_id]
            gold_summary = gold_summaries[document_id]

            # Evaluate MHIC
            mhic_score = self.mhic.calculate_metric(llm_summary, document_posts)
            results['mhic']['document_level'].append(mhic_score)

            # Evaluate IntraNLI
            intra_nli_score = self.intra_nli.calculate_metric(llm_summary)
            results['intra_nli']['document_level'].append(intra_nli_score)

            # Evaluate FCExpert
            fc_expert_score = self.fc.calculate_metric(llm_summary, gold_summary)
            results['fc_expert']['document_level'].append(fc_expert_score)

            # Evaluate FCdocument 
            fc_document_score = self.fc.calculate_metric(llm_summary, document_posts)
            results['fc_document']['document_level'].append(fc_document_score)

            # Evaluate Style Similarity
            style_similarity_score = self.style_similarity.calculate_metric(llm_summary, gold_summary)
            results['style_similarity']['document_level'].append(style_similarity_score)

            # Evaluate BERT Score
            bert_score = self.bert_score.calculate_metric(llm_summary, gold_summary)
            results['bert_score']['document_level'].append(bert_score)


        for metric in ['mhic', 'intra_nli', 'fc_expert', 'fc_document', 'style_similarity', 'bert_score']:
            # Calculate mean for each metric
            results[metric]['mean'] = np.mean(results[metric]['document_level'])
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate social media summarisation.")
    parser.add_argument('--posts', type=str, required=True, help='Path to the posts JSON file.')
    parser.add_argument('--llm_summaries', type=str, required=True, help='Path to the LLM summaries JSON file.')
    parser.add_argument('--gold_summaries', type=str, required=True, help='Path to the gold summaries JSON file.')
    parser.add_argument('--output_file', type=str, default='social_media_summarisation_evaluation_results.json', help='Path to save the evaluation results JSON file.')
    args = parser.parse_args()

    # Load posts
    print(f"Loading posts from {args.posts}")
    with open(args.posts, 'r') as f:
        posts = json.load(f)

    # Load LLM summaries
    print(f"Loading LLM summaries from {args.llm_summaries}")
    with open(args.llm_summaries, 'r') as f:
        llm_summaries = json.load(f)

    # Load gold summaries
    print(f"Loading gold summaries from {args.gold_summaries}")
    with open(args.gold_summaries, 'r') as f:
        gold_summaries = json.load(f)

    # Create evaluation bundle
    print("Creating evaluation bundle for social media summarisation.")
    evaluation_bundle = SocialMediaSummarisationEvaluationBundle()

    # Evaluate
    print("Evaluating LLM.")
    results = evaluation_bundle.evaluate(posts, llm_summaries, gold_summaries)

    # Save results
    print(f"Saving evaluation results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)