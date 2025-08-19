import itertools
from nltk import sent_tokenize
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class IntraNLI:
    def __init__(self, hf_cache_dir="/import/nlp-datasets/LLMs", hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"):
        self.model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name, cache_dir=hf_cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name, cache_dir=hf_cache_dir)
        self.model.to("cuda")
    
    def score_nli(self, premise, hypothesis, max_length=250, do_return_all=False):
        tokenized_input_seq_pair = self.tokenizer.encode_plus(premise, hypothesis,
                                                            max_length=max_length,
                                                            return_token_type_ids=True, truncation=True)

        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
        # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_ids.to(self.model.device),
                            attention_mask=attention_mask.to(self.model.device),
                            token_type_ids=token_type_ids.to(self.model.device),
                            labels=None)

        entail, neutral, contradict = torch.softmax(outputs[0], dim=1)[0].tolist()
        if do_return_all:
            return entail, neutral, contradict
        return entail 
    
    def calculate_metric(self, text: str) -> float:
        sentences = sent_tokenize(text)
        cs = []
        for sent1, sent2 in itertools.combinations(sentences, 2):
            _, _, c = self.score_nli(sent1, sent2, do_return_all=True)
            cs.append(c)
            _, _, c = self.score_nli(sent2, sent1, do_return_all=True)
            cs.append(c)
        
        return 1 - np.mean(cs)