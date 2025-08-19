import torch
from nltk import sent_tokenize, word_tokenize
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class EA:
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

    def chunk_sents(self, sents, chunk_size=60):
        """
        sents: list of sent_tokenized str
        chunk_size: maximum token count of each chunk
        """
        chunks = []
        curr_chunk = []
        curr_chunk_len = 0
        for sent in sents:
            curr_len = len(word_tokenize(sent))
            if curr_len + curr_chunk_len < chunk_size:
                curr_chunk.append(sent)
                curr_chunk_len += curr_len
            else:
                chunks.append(curr_chunk)
                curr_chunk = [sent]
                curr_chunk_len = curr_len
        return chunks

    def avg_helper(self, summary_type, scores_dict, return_idx=0):
        return np.mean(
            [
                score_tuple[return_idx]
                for tlid, scores in scores_dict.items()
                for score_tuple in scores[summary_type]
            ]
        )
    
    def calculate_metric(self, final_summary, intermediate_summary):
        curr_nli_factual_consistency_contradict_mean = []

        intermediate_sentences = sent_tokenize(intermediate_summary)
        final_sentences = sent_tokenize(final_summary)

        for final_sent in final_sentences:
            cs, cs_binary = [], []
            for intermediate_sent in intermediate_sentences:
                _, _, c = self.score_nli(
                    f"The individual wrote: {intermediate_sent}.",
                    final_sent,
                    do_return_all=True
                )
                cs.append(1 - c)
                cs_binary.append(1 if 1 - c > .5 else 0)
            curr_nli_factual_consistency_contradict_mean.append(
                (np.mean(cs), np.mean(cs_binary))
            )
        
        return curr_nli_factual_consistency_contradict_mean