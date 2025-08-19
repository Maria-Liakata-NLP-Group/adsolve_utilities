import argparse
import sys
import os
import json
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metrics.intra_nli import IntraNLI
from metrics.readability_metric import ReadabilityMetric
from metrics.fact import FactScorer
from metrics.fc import FactualConsistency


class CourtCaseSummarisationEvaluationBundle:
    def __init__(self):
        self.readability_metric = ReadabilityMetric("flesch_kincaid")
        self.intra_nli = IntraNLI()
        self.fc_expert = FactualConsistency()
        self.fact_scorer = FactScorer(llm_text=llm_summaries, reference=gold_summaries, min_claim=1, max_claim=30)

    def evaluate(self, llm_summaries: dict, gold_summaries: dict) -> dict:
        # Dictionary to store results
        results = {
            'readability': {
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
                "detail": [],
            },
            'conciseness': {
                "document_level": [],
                "mean": None,
                "detail": [],
            },
            'document_ids': list(llm_summaries.keys())
        }

        print("Generating claims for conciseness evaluation.")
        llm_claims = self.fact_scorer.get_claims(llm_summaries)

        # iterate over each document
        for document_id in tqdm(results['document_ids']):
            llm_summary = llm_summaries[document_id]
            gold_summary = gold_summaries[document_id]

            # Calculate readability score
            readability_score = self.readability_metric.calculate_metric(llm_summary)
            results['readability']['document_level'].append(readability_score)

            # Calculate intra-NLI score
            intra_nli_score = self.intra_nli.calculate_metric(llm_summary)
            results['intra_nli']['document_level'].append(intra_nli_score)

            # Calculate factual consistency expert score
            fc_expert_score, fc_expert_detail = self.fc_expert.calculate_metric(llm_summary, gold_summary)
            results['fc_expert']['document_level'].append(fc_expert_score)
            results['fc_expert']['detail'].append(fc_expert_detail)

            # Calculate conciseness score
            conciseness_score, conciseness_detail = self.fact_scorer.calculate_metric(
                type="recall",
                claims=llm_claims[document_id],
                reference=gold_summary
            )
            results['conciseness']['document_level'].append(conciseness_score)
            results['conciseness']['detail'].append(conciseness_detail)

        # Calculate mean scores
        results['readability']['mean'] = sum(results['readability']['document_level']) / len(results['readability']['document_level'])
        results['intra_nli']['mean'] = sum(results['intra_nli']['document_level']) / len(results['intra_nli']['document_level'])
        results['fc_expert']['mean'] = sum(results['fc_expert']['document_level']) / len(results['fc_expert']['document_level'])
        results['conciseness']['mean'] = sum(results['conciseness']['document_level']) / len(results['conciseness']['document_level'])

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate court case summarisation.")
    parser.add_argument('--llm_summaries', type=str, help=' Path to the LLM summaries JSON file.')
    parser.add_argument('--gold_summaries', type=str, help='Path to the gold summaries JSON file.')
    parser.add_argument('--combined_summaries', type=str, default=None, help='Path to the combined summaries JSON file (optional). If provided, it will be used instead of LLM summaries and gold summaries.')
    parser.add_argument('--output_file', type=str, default='court_case_summarisation_evaluation_results.json', help='Path to save the evaluation results JSON file.')
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
    
    # Create evaluation bundle
    print("Creating evaluation bundle for court case summarisation.")
    evaluation_bundle = CourtCaseSummarisationEvaluationBundle()
    # Evaluate
    print("Evaluating LLM.")
    results = evaluation_bundle.evaluate(llm_summaries, gold_summaries)
    
    # Save results
    print(f"Saving evaluation results to {args.output_file}")
    output_file = args.output_file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)