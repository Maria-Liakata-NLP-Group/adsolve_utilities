import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from typing import List

logger = logging.getLogger(__name__)

class LLMGenerator:
    def __init__(self, model_name: str, cache_dir: str ="/import/nlp-datasets/LLMs/"): 
        self.model_name = model_name
        if model_name == "tulu":
            # TODO: if we want to distribute this work we have to think about where to upload weights etc. (probably huggingface)
            ADAPTER_PATH = "/import/nlp/jsong/tulu_k1_pkt/"
            BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

            base_model = AutoModelForCausalLM.from_pretrained(
                            BASE_MODEL_ID,
                            torch_dtype="auto",    
                            device_map="auto",     
                        )
            self.model = PeftModel.from_pretrained(
                    base_model,
                    ADAPTER_PATH,
                    is_trainable=False
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_ID,
                cache_dir=cache_dir
            )
        else:
            self.model = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16
        )

        self.terminators = [
            self.pipe.tokenizer.eos_token_id,
            self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    def create_prompt_message(self, prompt: str, text: str):
        """
        Create a message structure for the prompt.
        """
        return [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": text
            }
        ]

    def generate(self, prompt: str, text: str, max_tokens: int = 5000, temperature: float = 0.0):
        sampling = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "eos_token_id": self.terminators,
            "do_sample": sampling,
        }

        if sampling:
            gen_kwargs.update({
                "temperature": temperature,
                "top_p": 0.9,             
            })
        with torch.no_grad():
            outputs = self.pipe(
                        self.create_prompt_message(prompt, text),
                        **gen_kwargs
                    )
            return outputs[0]["generated_text"][-1]["content"]
    
    def run_summary(self, prompt: List[str], text: List[str], max_tokens: List[int], temperatures: List[float]):
        """
           Generate summaries in n number of rounds. The list of prompts is interpreted as how many rounds of summaries to generate.
           For example, it could be that the content of posts will be summarised first and then these summaries will be summarised
           again to create a timeline summary.

           prompt, max_tokens and temperatures should be lists of the same length, where each element corresponds to a round of summarisation.
           list is passed as a list in case the original material is presented as multiple documents. 
           However, the output of the generation will be concatenated into a single string. So from the 2nd round onwards,
           the summarisation will be applied to one string rather than a list of strings.
        """

        for p, max, temperature in zip(prompt, max_tokens, temperatures):
            summary = []
            for t in text:
                current_summary = self.generate(
                    prompt=p,
                    text=t,  
                    max_tokens=max,  
                    temperature=temperature
                )
                summary.append(current_summary)
            # redefine text
            text = [''.join(summary)]

        # return the final iteration of summarisation
        return text[0]