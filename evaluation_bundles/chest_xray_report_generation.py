import argparse
import sys
import os
import json
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adsolve_utils.evaluation_bundles.metrics.evidence_appropriateness import EA
from adsolve_utils.evaluation_bundles.metrics.rouge import ROUGE

class ChestXRayReportsEvaluationBundle:
    def __init__(self):
        self.ea = EA(hg_model_hub_name="pritamdeka/PubMedBERT-MNLI-MedNLI")
        self.rouge_1 = ROUGE(configuration="1", metric="p")
        self.rouge_l = ROUGE(configuration="l", metric="p")

    def evaluate(self, llm_summaries: dict, gold_summaries: dict) -> dict:
        # Dictionary to store results
        results = {
            'evidence_appropriateness': {
                "document_level": [],
                "mean": None,
            },
            'rouge-1': {
                "document_level": [],
                "mean": None,
            },
            'rouge-L': {
                "document_level": [],
                "mean": None,   
            },
            'document_ids': list(llm_summaries.keys())
        }

        # iterate over each document
        for document_id in tqdm(results['document_ids']):
            llm_summary = llm_summaries[document_id]
            gold_summary = gold_summaries[document_id]

            # Calculate evidence_appropriateness score
            evidence_score = self.ea.calculate_metric(llm_summary, gold_summary)
            results['evidence_appropriateness']['document_level'].append(evidence_score)

            # Calculate rouge score
            rouge_1_score = self.rouge_1.calculate_metric(llm_summary, gold_summary)
            results['rouge-1']['document_level'].append(rouge_1_score)

            rouge_l_score = self.rouge_l.calculate_metric(llm_summary, gold_summary)
            results['rouge-L']['document_level'].append(rouge_l_score)


        # Calculate mean scores
        results['evidence_appropriateness']['mean'] = sum(results['evidence_appropriateness']['document_level']) / len(results['evidence_appropriateness']['document_level'])
        results['rouge-1']['mean'] = sum(results['rouge-1']['document_level']) / len(results['rouge-1']['document_level'])
        results['rouge-L']['mean'] = sum(results['rouge-L']['document_level']) / len(results['rouge-L']['document_level'])

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate chest X-ray report generation.")
    parser.add_argument('--llm_summaries', type=str, help=' Path to the LLM summaries JSON file.')
    parser.add_argument('--gold_summaries', type=str, help='Path to the gold summaries JSON file.')
    parser.add_argument('--combined_summaries', type=str, default=None, help='Path to the combined summaries JSON file (optional). If provided, it will be used instead of LLM summaries and gold summaries.')
    parser.add_argument('--output_file', type=str, default='chest_xray_reports_evaluation_results.json', help='Path to save the evaluation results JSON file.')
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
    print("Creating evaluation bundle for chest xray report generation.")
    evaluation_bundle = ChestXRayReportsEvaluationBundle()
    # Evaluate
    print("Evaluating LLM.")
    results = evaluation_bundle.evaluate(llm_summaries, gold_summaries)
    
    # Save results
    print(f"Saving evaluation results to {args.output_file}")
    output_file = args.output_file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)