import os
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# CACHE_DIR = "/import/nlp/jchim/hf-cache/"
CACHE_DIR = "/import/nlp-datasets/LLMs/"
os.environ["HF_HOME"] = CACHE_DIR


def make_chunk(lst, n):
    # Adapted from: https://stackoverflow.com/questions/312443/
    for i in tqdm(range(0, len(lst), n)):
        yield lst[i : i + n]


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
                "content": f"Instruction: You are a helpful medical assistant. Read the clinical report and generate at least {self.min_claim} at most {self.max_claim} short claims that are supported by the clinical report. Each short claim should contain only one fact. The generated claims should cover all facts in the clinical report.",
            },
            {
                "role": "system",
                "name": "example_user",
                "content": "content of the clinical report ...",
            },
            {
                "role": "system",
                "name": "example_assistant",
                "content": "Claim 1: ...\nClaim 2: ... ",
            },
            {"role": "user", "content": text},
        ]

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

        for chunk in make_chunk(data, batch_size):

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
