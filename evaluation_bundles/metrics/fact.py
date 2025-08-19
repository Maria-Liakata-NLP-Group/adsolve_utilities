import argparse
import os
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import json
from typing import List, Tuple, Union
import numpy as np

class Mushroom:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-8B",
        padding_side: str = "left",
        device_map: str = "sequential",
        torch_dtype: str = "auto",
        cache_dir: str = os.environ.get("HF_HOME"),
        min_claim: int = 1,
        max_claim: int = 30,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device_map, cache_dir=cache_dir
        )
        if self.device and not device_map:
            self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=False, padding_side=padding_side, cache_dir=cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.min_claim = min_claim
        self.max_claim = max_claim

    def make_prompt(self, text: str):
        return [
            {
                "role": "system",
                "content": f"Instruction: You are a helpful assistant. Read the document and generate at least {self.min_claim} at most {self.max_claim} short claims that are supported by the document. Each short claim should contain only one fact. The generated claims should cover all facts in the clinical report.",
            },
            {
                "role": "system",
                "name": "example_user",
                "content": "content of the document ...",
            },
            {
                "role": "system",
                "name": "example_assistant",
                "content": "Claim 1: ...\nClaim 2: ... ",
            },
            {"role": "user", "content": text},
        ]
    
    def make_chunk(self, lst: list, n: int):
    # Adapted from: https://stackoverflow.com/questions/312443/
        for i in tqdm(range(0, len(lst), n)):
            yield lst[i : i + n]

    def generate_batches(
        self,
        data,
        batch_size=4,
        do_sample=True,
        temperature=0.7,  # 0.9,
        top_k=20,  # 50,
        top_p=0.8,  # 0.9,
        max_new_tokens=2000,
        repetition_penalty=1.0,
    ):
        # qwen - "For non-thinking mode, we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0"

        for chunk in self.make_chunk(data, batch_size):

            templated_chunk = self.tokenizer.apply_chat_template(
                chunk,
                tokenize=False,
                add_generation_prompt=True,  # Note: doesn't apply to LLaMA
                enable_thinking=False,  # Note: qwen
            )

            inputs = self.tokenizer(templated_chunk, return_tensors="pt", padding=True)

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )

            for output in self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            ):
                yield output

    def decompose(self, texts, batch_size=24, max_new_tokens=500):
        """
        Decompose texts with automatic batch size reduction on OOM errors.
        Starts with specified batch_size (default 8) and halves it on failure until batch_size=1.

        Args:
            texts: List of texts to process
            batch_size: Starting batch size (will auto-reduce if OOM occurs)
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            List of decomposed outputs

        Raises:
            RuntimeError: If all batch sizes down to 1 fail
        """
        current_batch_size = batch_size
        last_exception = None

        while current_batch_size >= 1:
            try:
                return list(
                    self.generate_batches(
                        data=[self.make_prompt(text) for text in texts],
                        batch_size=current_batch_size,
                        max_new_tokens=max_new_tokens,
                    )
                )
            except torch.cuda.OutOfMemoryError as e:
                last_exception = e
                torch.cuda.empty_cache()
                print(
                    f"Out of Memory with batch size {current_batch_size}, trying with {current_batch_size//2}"
                )
                current_batch_size = current_batch_size // 2
            except Exception as e:
                # For non-OOM errors, raise immediately
                raise RuntimeError(
                    f"Error during decomposition with batch size {current_batch_size}"
                ) from e

        # If we get here, all batch sizes failed
        raise RuntimeError(
            f"Failed to process texts with batch sizes down to 1. "
            f"Last error: {str(last_exception)}"
        )


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

class FactScorer:
    def __init__(self, llm_text, reference, min_claim: int = 1, max_claim: int = 10):
        self.mushroom = Mushroom(min_claim=min_claim, max_claim=max_claim)
        self.true_scorer = TRUEScorer()
        self.llm_text = llm_text
        self.reference = reference
    
    def get_claims(self, text: dict, cache_file_path: str = None) -> List[str]:
        if cache_file_path and os.path.exists(cache_file_path):
            print(f"Loading cached claims from {cache_file_path}")
            with open(cache_file_path, "r") as f:
                return json.load(f)
        
        # If cache file does not exist, generate claims
        print("Generating claims...")
        ids = []
        references = []
        for key, value in text.items():
            references.append(value)
            ids.append(key)

        decomposed_references = self.mushroom.decompose(texts=references, batch_size=12, max_new_tokens=2000)

        with open("decomposed_references.txt", "w") as f:
            for ref in decomposed_references:
                f.write(ref + "\n\n")


        claims_by_id = {}
        for i, ref in enumerate(decomposed_references):
            # Only take assistant response
            # Find the last occurence of Claim 1 and split just before it
            idx = ref.rfind("Claim 1:")
            claims_full = ref[idx:]
            # split by "\nClaim" to get individual claims
            claim_parts = claims_full.split("\nClaim ")
            # Split by claims, each claim starts with "Claim"
            claims = []
            for claim in claim_parts:
                claim = claim.split(":", 1)[-1].strip()
                claims.append(claim)
            claims_by_id[ids[i]] = claims
        
        # Cache claims if cache_file_path is provided
        if cache_file_path:
            print(f"Caching claims to {cache_file_path}")
            os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
            with open(cache_file_path, "w") as f:
                json.dump(claims_by_id, f)

        return claims_by_id
    
    def verify_claims(self, claims: list, reference: str, batch_size: int=8):
        all_verifications = []
        # build a repeated text list
        texts = [reference] * len(claims)

        for idx in range(0, len(claims), batch_size):
            batch_claims = claims[idx : idx + batch_size]
            batch_texts  = texts[idx : idx + batch_size]

            # run the smaller batch
            batch_verifications = self.true_scorer.predict(
                premise=batch_texts,
                hypothesis=batch_claims,
            )
            all_verifications.extend(batch_verifications)

            # clear any cached, unused memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return all_verifications
    
    def calculate_metric(self, type: str, claims: list[str], reference: str) -> dict:
        results = self.verify_claims(claims, reference)
        return np.mean(results), {
            f"claims": claims,
            f"results": results,
        }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate fact recall and precision.")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSON file containing llm summaries and gold summaries."
    )
    parser.add_argument(
        "--llm_cache_file", type=str, default=None, help="Path to the cache file for claims."
    )
    parser.add_argument(
        "--cache_file", type=str, default=None, help="Path to the cache file for reference claims."
    )
    parser.add_argument(
        "--type", type=str, choices=["recall", "precision"], default="recall",
        help="Type of metric to calculate (recall or precision)."
    )
    
    args = parser.parse_args()
    
    with open(args.input_file, "r") as f:
        data = json.load(f)
    
    # split input data into llm_summaries and gold_summaries
    llm_summaries = {id: value["summary"] for id, value in data.items()}
    gold_summaries = {id: value["reference"] for id, value in data.items()}
    
    # Initialize the FactScorer
    fact_scorer = FactScorer(llm_text=llm_summaries, reference=gold_summaries, min_claim=1, max_claim=30)

    if args.type == "recall":
        print("Calculating recall...")
        # Get claims
        gold_claims = fact_scorer.get_claims(gold_summaries, cache_file_path=args.cache_file)

        # Calculate recall for each reference
        all_recall_results = {}
        for id, llm_text in tqdm(llm_summaries.items(), desc="Processing references"):
            recall = fact_scorer.calculate_metric(
                type="recall",
                claims=gold_claims[id],
                reference=llm_text
            )
            all_recall_results[id] = recall

        output_file = "fact_recall_results.json"
        with open(output_file, "w") as f:
            json.dump(all_recall_results, f, indent=4)
        print(f"Recall results saved to {output_file}")

    if args.type == "precision":
        print("Calculating precision...")
        # Get claims
        llm_claims = fact_scorer.get_claims(llm_summaries, cache_file_path=args.cache_file)

        # Calculate recall for each reference
        all_precision_results = {}
        for id, gold_text in tqdm(gold_summaries.items(), desc="Processing references"):
            recall = fact_scorer.calculate_metric(
                type="precision",
                claims=llm_claims[id],
                reference=gold_text
            )
            all_precision_results[id] = recall

        output_file = "fact_precision_results.json"
        with open(output_file, "w") as f:
            json.dump(all_precision_results, f, indent=4)
        print(f"Precision results saved to {output_file}")
