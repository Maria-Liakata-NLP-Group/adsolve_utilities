from collections import Counter
import numpy as np
import spacy
import torch
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

nlp = spacy.load("en_core_web_sm")

# https://github.com/lingjzhu/idiolect
class AttentionPooling(torch.nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.W = torch.nn.Linear(input_dim, 1)
        self.softmax = torch.nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
        N: batch size, T: sequence length, H: Hidden dimension
        input:
            batch_rep : size (N, T, H)
        attention_weight:
            att_w : size (N, T, 1)
        return:
            utter_rep: size (N, H)
        """
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class DNNSelfAttention(torch.nn.Module):
    def __init__(self, hidden_dim, **kwargs):
        super(DNNSelfAttention, self).__init__()
        self.pooling = AttentionPooling(hidden_dim)
        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features, att_mask):
        out = self.pooling(features, att_mask).squeeze(-1)
        predicted = self.out_layer(out)
        return predicted


class SRoberta(torch.nn.Module):
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        config = RobertaConfig.from_pretrained(model_name, return_dict=True)
        config.output_hidden_states = True
        self.roberta = RobertaModel.from_pretrained(model_name, config=config)

        self.pooler = DNNSelfAttention(768)

    def forward(self, input_ids, att_mask=None):
        out = self.roberta(input_ids, att_mask)
        out = out.last_hidden_state
        out = self.pooler(out, att_mask)
        return out




class IdiolectEmbeddings:
    def __init__(self, state_dict_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.model = SRoberta()
        state_dict = torch.load(state_dict_path)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.cos = torch.nn.CosineSimilarity(dim=0)

    def extract_emb(self, text):
        tokenized = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = self.model(
                tokenized["input_ids"].to(self.device),
                tokenized["attention_mask"].to(self.device),
            )
            hidden = F.normalize(hidden, dim=-1)
            hidden = hidden.cpu().detach()
            return hidden

    def get_style_sim(self, source, target):
        return self.cos(
            self.extract_emb(source).squeeze(), self.extract_emb(target).squeeze()
        ).item()

    def get_style_similarities(self, src, targets, **kwargs):
        return np.mean([self.get_style_sim(t, src) for t in targets]).item()


class POSStyleSimilarityScorer:
    def __init__(self):
        # Ireland and Pennebaker, 2010 captures writing styles
        # by examining POS tag occurences across categories:
        # 0) adv, 1) adj, 2) conj, 3) det, 4) noun, 5) pron, 6) preposition, 7) punct
        self._VALID_UPOS = {
            "ADV",
            "ADJ",
            # NO AUX,
            "CCONJ",
            "SCONJ",
            "DET",
            # NO INTJ
            "NOUN",
            "PROPN",
            # NO NUM
            "PRON",
            "ADP",
            "PART",
            "PUNCT",
            # NO SYMB
            # NO VERB,
            # NO X
        }
        self.VALID_UPOS = sorted(self.map_tag(t) for t in self._VALID_UPOS)

    def map_tag(self, tag):
        # collapse UPOS tagset to categories
        mapper = {
            "CCONJ": "CONJ",
            "SCONJ": "CONJ",
            "PROPN": "NOUN",
            "ADP": "PREP",
            "PART": "PREP",
            # BNC2014
            "SUBST": "NOUN",
            "ART": "DET",
            "INTERJ": "INTJ",
        }
        return mapper.get(tag, tag)

    def compute_jaccard_similarity(self, list1, list2):
        set1, set2 = set(list1), set(list2)
        intersection = list(set1.intersection(set2))
        intersection_length = len(list(set1.intersection(set2)))
        union_length = (len(set1) + len(set2)) - intersection_length
        if union_length == 0:
            return union_length
        return float(intersection_length) / union_length

    def tag_and_filter(self, text):
        doc = nlp(text)
        return [self.map_tag(t.pos_) for t in doc if t.pos_ in self._VALID_UPOS], len(
            doc
        )

    def word_pos_score(self, pos1, pos2, len1, len2):
        """
        Calculate POS similarity (Ireland and Pennbaker 2010) over UPOS tags.
            1. for each POS category, get its count in proportion to total sentence length
            2. calculate similarity score wrt each category
            3. average to get total POS similarity score
        """
        pos_counts1 = Counter(pos1)
        pos_counts2 = Counter(pos2)

        category_scores = []
        for t in self.VALID_UPOS:
            cat1 = pos_counts1.get(t, 0) / len1
            cat2 = pos_counts2.get(t, 0) / len2
            if cat1 == 0 and cat2 == 0:
                score = 1
            else:
                score = 1 - (abs(cat1 - cat2) / (cat1 + cat2))
            category_scores.append(score)

        return np.mean(category_scores)

    def trigram_pos_score(self, pos1, pos2):
        # note that this will return 0 for shorter texts
        pos1 = self._make_ngrams(pos1, n=3)
        pos2 = self._make_ngrams(pos2, n=3)
        return self.compute_jaccard_similarity(pos1, pos2)

    def get_trigram_pos_score(self, text1, text2):
        pos1, _ = self.tag_and_filter(text1)
        pos2, _ = self.tag_and_filter(text2)
        return self.trigram_pos_score(pos1, pos2)

    def get_mean_trigram_pos_scores(self, src, targets, **kwargs):
        return np.mean([self.get_trigram_pos_score(t, src) for t in targets]).item()

    def _make_ngrams(self, l, n=3):
        return ["".join(l[i : i + n]) for i in range(len(l) - n + 1)]


class StyleSimilarity:
    def __init__(self):
        self.scorer = POSStyleSimilarityScorer()
    
    def calculate_metric(self, llm_text: str, reference_text: str) -> float:
        return self.scorer.trigram_pos_score(reference_text, llm_text)