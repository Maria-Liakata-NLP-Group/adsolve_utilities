import argparse
import os
import torch
import json
import numpy as np
import re
from tqdm import tqdm
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    GenerationConfig,
)
import logging

'''
From Dimitris' implementation: https://github.com/gkoumasd/cross_nli/tree/main 
'''


# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("cross_nli.log"),  # Logs to a file
        logging.StreamHandler()  # Logs to console
    ],
)


class Cross_NLI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nli_id = "cross-encoder/nli-deberta-base"
        self.llm_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        self.tokenizer_nli = AutoTokenizer.from_pretrained(self.nli_id)
        self.model_nli = AutoModelForSequenceClassification.from_pretrained(self.nli_id).to(self.device)

        self.tokenizer_llm = AutoTokenizer.from_pretrained(self.llm_id, trust_remote_code=True)

        PAD_TOKEN = "<|pad|>"
        # 1) Make sure the pad token is in the vocab (if not, add it):
        if PAD_TOKEN not in self.tokenizer_llm.get_vocab():
            self.tokenizer_llm.add_special_tokens({"pad_token": PAD_TOKEN})

        # 2) Assign pad_token_id distinct from eos_token_id
        self.tokenizer_llm.pad_token = PAD_TOKEN
        self.tokenizer_llm.pad_token_id = self.tokenizer_llm.convert_tokens_to_ids(PAD_TOKEN)

        # 3) Ensure padding happens on the right:
        self.tokenizer_llm.padding_side = "right"
        # self.tokenizer_llm.pad_token = self.tokenizer_llm.eos_token
        # self.tokenizer_llm.padding_side = "right"

        self.model_llm = AutoModelForCausalLM.from_pretrained(
            self.llm_id,
            device_map=self.device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        self.generation_config = GenerationConfig.from_pretrained(
            self.llm_id,
            best_of=1,
            presence_penalty=0.0,
            frequency_penalty=1.0,
            top_p=0.9,
            temperature=1e-10,
            do_sample=True,
            stop=["###", self.tokenizer_llm.eos_token, self.tokenizer_llm.pad_token],
            use_beam_search=False,
            max_new_tokens=600,
            logprobs=5,
            pad_token_id=self.tokenizer_llm.eos_token_id,
        )

    def atomic_unit_extraction(self, datum):
        """Extracts atomic text units from input text using LLM."""
        prompt = """
                **
                IMPORTANT: 
                 - Extract only factual truths from the text.
                 - Do not include any prior knowledge or interpretations.
                 - Take the text at face value when extracting facts.
                 - Ensure each unit is concise and represents the smallest possible factual statement.
                 - Do not include any introductory or explanatory text. Only output the numbered list of atomic units.
                **
        """
    
        messages = [
            {"role": "user", "content": f"Break down the following text into independent atomic text units: {prompt} {datum}"},
        ]
        
        inputs = self.tokenizer_llm.apply_chat_template(messages, return_tensors="pt", padding=True,       # now makes real pad tokens
                truncation=True).to(self.device)

        generated_ids = self.model_llm.generate(inputs, self.generation_config)
        decoded = self.tokenizer_llm.batch_decode(generated_ids[:, inputs.shape[1]:], skip_special_tokens=True)

        return decoded

    def extract_numbered_text(self, text):
        """Extracts numbered list items from a text string."""
        return re.findall(r"\d+\.\s(.*)", text)

    def calculate_metric(self, predicted, gold):
       

        """Evaluates hallucination and coverage using NLI model."""
        # data = [json.loads(line) for line in open(self.path, "r", encoding="utf-8")]
        # total_coverage = 0
        # total_hallucination = 0


        # logging.info(f"Evaluating {i+1} of {len(data)}")
        golden = self.extract_numbered_text(self.atomic_unit_extraction(gold)[0])
        generated = self.extract_numbered_text(self.atomic_unit_extraction(predicted)[0])

        halucination = np.zeros((len(generated), len(golden)))
        caverage = np.zeros((len(golden), len(generated)))

        # Calculate Hallucination
        for i in range(len(generated)):
            for j in range(len(golden)):
                input_ids = self.tokenizer_nli(golden[j], generated[i], truncation=True, return_tensors="pt")["input_ids"]
                output = self.model_nli(input_ids.to(self.device))
                logits = torch.softmax(output["logits"][0], -1).tolist()
                halucination[i][j] = 1 - logits[0]

        halucination_score = float(np.sum(np.max(halucination, axis=1)) / len(halucination))
        return halucination_score, {"scores": list(np.max(halucination, axis=1)), "sents": generated}


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

    cross_nli = Cross_NLI()

    results = {
                'cross_nli': {
                "document_level": [],
                "mean": None,
                "detail": [],
                },
                'document_ids': list(llm_summaries.keys())
            }
    
    for document_id in tqdm(results['document_ids'], desc="Evaluating Cross NLI"):
        llm_summary = llm_summaries[document_id]
        gold_summary = gold_summaries[document_id]

        # Calculate factual consistency expert score
        cross_nli_score, cross_nli_detail = cross_nli.calculate_metric(llm_summary, gold_summary)
        results['cross_nli']['document_level'].append(cross_nli_score)
        results['cross_nli']['detail'].append(cross_nli_detail)
    
    # Calculate mean scores
    results['cross_nli']['mean'] = sum(results['cross_nli']['document_level']) / len(results['cross_nli']['document_level'])
    
    print(f"Saving evaluation results to {args.output_file}")
    output_file = args.output_file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)