from rouge import Rouge 
from typing import Literal

# Define typing so that configuration can only be "1", "2", or "l" and metric "f", "p" or "r"
ROUGEConfiguration = Literal["1", "2", "l"]
ROUGEMetric = Literal["f", "p", "r"]

class ROUGE:
    def __init__(self, configuration: ROUGEConfiguration = "1", metric: ROUGEMetric = "p"):
        self.configuration = configuration
        self.metric = metric
        self.rouge = Rouge()

    def calculate_metric(self, llm_text: str, reference_text: str) -> float:
        scores = self.rouge.get_scores(llm_text, reference_text)
        rouge_key = f"rouge-{self.configuration}"
        return scores[0][rouge_key][self.metric] 