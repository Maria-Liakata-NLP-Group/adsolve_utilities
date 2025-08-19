import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import json
from typing import List, Tuple, Union
# import datasets
import numpy as np

CACHE_DIR = "/import/nlp/jchim/hf-cache/"
os.environ["HF_HOME"] = CACHE_DIR

# from accelerate import infer_auto_device_map


class TRUEScorer:
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_name="google/t5_xxl_true_nli_mixture",
    ):
        self.model_name = model_name
        # Load model first on CPU to calculate device map
        """
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            # torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # Custom max_memory based on lichfield...
        max_memory = {
            0: "36GiB",  # A100
            1: "36GiB",  # A40
            2: "36GiB",  # A100
            3: "36GiB",  # A40
        }

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["T5Block"],  # Don't split transformer blocks
            # dtype="float16"
        )
        """
        # Now load with device map
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            device_map="balanced_low_0"
            # torch_dtype=torch.float16
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.device = device
        print(f"Using device: {device}")
        # self.model.to(self.device)

    def predict(
        self, premise: Union[str, List[str]], hypothesis: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        """Predict whether hypothesis follows from premise(s).

        Args:
            premise: Single premise string or list of premise strings
            hypothesis: Single hypothesis string or list of hypothesis strings

        Returns:
            Single prediction (0 or 1) or list of predictions
        """
        # Handle single input case
        if isinstance(premise, str) and isinstance(hypothesis, str):
            input_text = f"premise: {premise} hypothesis: {hypothesis}"
            input_ids = self.tokenizer.encode(
                input_text, return_tensors="pt", truncation=True, max_length=500
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(input_ids, max_new_tokens=3)
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if self.model_name == "google/t5_xxl_true_nmixture":
                result = 1 if result == "1" else 0
            return result

        # Handle batch input case
        elif isinstance(premise, list) and isinstance(hypothesis, list):
            if len(premise) != len(hypothesis):
                raise ValueError("Number of premises and hypotheses must be equal")

            # Prepare batch inputs
            input_texts = [
                f"premise: {p} hypothesis: {h}" for p, h in zip(premise, hypothesis)
            ]
            inputs = self.tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                max_length=500,
                return_tensors="pt",
            ).to(self.device)

            # Generate predictions
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=3,
                )

            # Decode results
            results = []
            for output in outputs:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                decoded = decoded.strip()
                if len(decoded) > 1:
                    decoded = decoded[0]
                if self.model_name == "google/t5_xxl_true_nli_mixture":
                    decoded = 1 if decoded == "1" else 0
                results.append(decoded)
            return results

        else:
            raise TypeError(
                "Both premise and hypothesis must be either strings or lists of strings"
            )

    def test(self):
        # Example single input
        premise = "The cat sat on the mat."
        hypothesis = "The cat is on the mat."
        print(f"Single input prediction:")
        print(f"Premise: {premise}")
        print(f"Hypothesis: {hypothesis}")
        result = self.predict(premise, hypothesis)
        print(f"Model prediction: {result}\n")

        # Example batch input
        premises = [
            "The cat sat on the mat.",
            "It's raining outside.",
            "The team won the championship.",
        ]
        hypotheses = [
            "The cat is on the mat.",
            "The weather is sunny.",
            "They performed well in the competition.",
        ]
        print(f"Batch input predictions:")
        for p, h in zip(premises, hypotheses):
            print(f"Premise: {p}")
            print(f"Hypothesis: {h}")
        results = self.predict(premises, hypotheses)
        print(f"Model predictions: {results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TRUEScorer")
    parser.add_argument("--test", type=bool, default=False, help="Run test predictions")
    parser.add_argument("--claims_file", type=str, help="Path to claims file")
    parser.add_argument("--text_file", type=str, help="Path to text file")
    args = parser.parse_args()
    scorer = TRUEScorer()
    print(scorer.model.hf_device_map)

    if args.test:
        print(scorer.test())
    else:

        claims_file = args.claims_file
        text_file = args.text_file

        # Load claims and text examples
        with open(claims_file, "r") as f:
            claims_data = json.load(f)
        with open(text_file, "r") as f:
            text_data = json.load(f)
        
        text_data = {id: value["summary"] for id, value in text_data.items()}

        def process_recall_example(claim_example: list, text_example: str, batch_size=8):
            all_recalls = []
            # build a repeated text list
            texts = [text_example] * len(claim_example)

            for idx in range(0, len(claim_example), batch_size):
                batch_claims = claim_example[idx : idx + batch_size]
                batch_texts  = texts[idx : idx + batch_size]

                # run the smaller batch
                batch_recalls = scorer.predict(
                    premise=batch_texts,
                    hypothesis=batch_claims,
                )
                all_recalls.extend(batch_recalls)

                # clear any cached, unused memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {"claim_recall": all_recalls}

        # process example
        # extract first key

        recall_results = {}
        for id in claims_data.keys():
            recall_result = process_recall_example(
                claims_data[id],
                text_data[id],
            )
            recall_results[id] = {
                "claims": claims_data[id],
                "text": text_data[id],
                "recall_example": recall_result,
            } 


        # Save example
        output_file = f"iqra_summaries_recall_example.json"
        save_data = recall_results,
        with open(output_file, "w") as f:
            json.dump(save_data, f, indent=4)
        



        def process_example(example):
            reference_note, reference_claims = (
                example["reference_note"],
                example["reference_claims"],
            )
            predicted_note, predicted_claims = (
                example["predicted_note"],
                example["predicted_claims"],
            )

            num_reference_claims, num_predicted_claims = len(reference_claims), len(
                predicted_claims
            )

            recall_raw = scorer.predict(
                premise=[predicted_note] * num_reference_claims, hypothesis=reference_claims
            )
            precision_raw = scorer.predict(
                premise=[reference_note] * num_predicted_claims, hypothesis=predicted_claims
            )

            return {"claim_recall": recall_raw, "claim_precision": precision_raw}


        def process_batch(batch):
            # Initialize lists to store all claims and notes
            all_premises_recall = []
            all_hypotheses_recall = []
            all_premises_precision = []
            all_hypotheses_precision = []
            example_indices = []  # To keep track of which claims belong to which example

            # Prepare all claims for batch processing
            for i, (ref_note, ref_claims, pred_note, pred_claims) in enumerate(
                zip(
                    batch["reference_note"],
                    batch["reference_claims"],
                    batch["predicted_note"],
                    batch["predicted_claims"],
                )
            ):
                # For recall: predicted_note vs reference_claims
                all_premises_recall.extend([pred_note] * len(ref_claims))
                all_hypotheses_recall.extend(ref_claims)

                # For precision: reference_note vs predicted_claims
                all_premises_precision.extend([ref_note] * len(pred_claims))
                all_hypotheses_precision.extend(pred_claims)

                # Track which claims belong to which example
                example_indices.append((i, len(ref_claims), len(pred_claims)))

            # Process all recall predictions in one batch
            recall_raw = []
            if all_hypotheses_recall:
                recall_raw = scorer.predict(all_premises_recall, all_hypotheses_recall)

            # Process all precision predictions in one batch
            precision_raw = []
            if all_hypotheses_precision:
                precision_raw = scorer.predict(all_premises_precision, all_hypotheses_precision)

            # Now reconstruct the results per example
            claim_recalls = []
            claim_precisions = []

            recall_idx = 0
            precision_idx = 0

            for i, num_ref_claims, num_pred_claims in example_indices:
                # Get recall scores for this example
                example_recall = recall_raw[recall_idx : recall_idx + num_ref_claims]
                recall_idx += num_ref_claims

                # Get precision scores for this example
                example_precision = precision_raw[
                    precision_idx : precision_idx + num_pred_claims
                ]
                precision_idx += num_pred_claims

                claim_recalls.append(example_recall)
                claim_precisions.append(example_precision)

            return {"claim_recall": claim_recalls, "claim_precision": claim_precisions}