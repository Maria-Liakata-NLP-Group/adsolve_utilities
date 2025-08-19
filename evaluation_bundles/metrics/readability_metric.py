from enum import Enum
from readability import Readability
class ReadabilityType(str, Enum):
    FLESCH_KINCAID            = "flesch_kincaid"
    FLESCH                    = "flesch"
    GUNNING_FOG               = "gunning_fog"
    COLEMAN_LIAU              = "coleman_liau"
    AUTOMATED_READABILITY_IDX = "automated_readability_index"
    DALE_CHALL                = "dale_chall"
    LINSEAR_WRTIE             = "linsear_wrtie"
    SPACHE                    = "spache"
    SMOG                      = "smog"

class ReadabilityMetric:
    def __init__(
        self,
        readability_type: ReadabilityType = ReadabilityType.FLESCH_KINCAID
    ) -> None:
        self.type = readability_type
  

    def calculate_metric(self, text: str) -> float:
        # check that text has at least 100 words. If not, add sentences from the beginning until it has at least 100 words.
        if len(text.split()) < 100:
            sentences = text.split('. ')
            while len(text.split()) < 100 and sentences:
                text += ' ' + sentences.pop(0)
        # if text is still less than 100 words, return 0
        if len(text.split()) < 100:
            return 0.0
        r = Readability(text)
        if self.type == ReadabilityType.FLESCH_KINCAID:
            return r.flesch_kincaid().score
        elif self.type == ReadabilityType.FLESCH:
            return r.flesch().score
        elif self.type == ReadabilityType.GUNNING_FOG:
            return r.gunning_fog().score
        elif self.type == ReadabilityType.COLEMAN_LIAU:
            return r.coleman_liau().score
        elif self.type == ReadabilityType.AUTOMATED_READABILITY_IDX:
            return r.ari().score
        elif self.type == ReadabilityType.DALE_CHALL:
            return r.dale_chall().score
        elif self.type == ReadabilityType.LINSEAR_WRTIE:
            return r.linsear_write().score
        elif self.type == ReadabilityType.SPACHE:
            return r.spache().score
        elif self.type == ReadabilityType.SMOG:
            return r.smog().score
        else:
            raise ValueError(f"Unknown readability type: {self.type}")
