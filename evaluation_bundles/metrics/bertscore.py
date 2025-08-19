from bert_score import BERTScorer
from nltk import sent_tokenize

class BERTScore:
    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli", lang: str = "en"):
        """
        uses implmentation from https://github.com/Tiiiger/bert_score?tab=readme-ov-file

        Args:
            - :param: `model_type` (str): The BERT model type to use for scoring. 
                        Check https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit?gid=0#gid=0
                        for available models.
            - :param: `lang` (str): The language of the text to be scored.
        """

        self.scorer = BERTScorer(model_type=model_type, lang="en", rescale_with_baseline=True)
        
    def calculate_metric(self, llm_text: str, reference_text: str) -> float:
        # llm_sentences = sent_tokenize(llm_text)
        # reference_sentences = sent_tokenize(reference_text)
        P, R, F1 = self.scorer.score([llm_text], [reference_text])
        return F1[0].item()