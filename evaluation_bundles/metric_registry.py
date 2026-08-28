"""Static metadata describing the metric wrapper classes in evaluation_bundles/metrics/.

Used by generate_bundle.py to validate bundle specs and render bundle scripts without
needing to import the (heavy, GPU-dependent) metric classes themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet


class InputKind(str, Enum):
    NONE = "none"
    SINGLE = "single"
    LIST = "list"
    DUAL = "dual"
    PRECOMPUTE_CLAIMS = "precompute_claims"
    BATCH = "batch"


@dataclass(frozen=True)
class MetricInfo:
    key: str
    module: str
    class_name: str
    kind: InputKind
    returns_detail: bool
    default_params: dict = field(default_factory=dict)
    allowed_params: FrozenSet[str] = field(default_factory=frozenset)
    fixed_params: dict = field(default_factory=dict)


METRIC_REGISTRY: Dict[str, MetricInfo] = {
    "rouge": MetricInfo(
        key="rouge",
        module="metrics.rouge",
        class_name="ROUGE",
        kind=InputKind.SINGLE,
        returns_detail=False,
        default_params={"configuration": "1", "metric": "p"},
        allowed_params=frozenset({"configuration", "metric"}),
    ),
    "bertscore": MetricInfo(
        key="bertscore",
        module="metrics.bertscore",
        class_name="BERTScore",
        kind=InputKind.SINGLE,
        returns_detail=False,
        default_params={"model_type": "microsoft/deberta-xlarge-mnli", "lang": "en"},
        allowed_params=frozenset({"model_type", "lang"}),
    ),
    "style_roberta": MetricInfo(
        key="style_roberta",
        module="metrics.style_roberta",
        class_name="StyleSimilarity",
        kind=InputKind.SINGLE,
        returns_detail=False,
        default_params={},
        allowed_params=frozenset(),
    ),
    "evidence_appropriateness": MetricInfo(
        key="evidence_appropriateness",
        module="metrics.evidence_appropriateness",
        class_name="EA",
        kind=InputKind.SINGLE,
        returns_detail=False,
        default_params={},
        allowed_params=frozenset({"hf_cache_dir", "hg_model_hub_name"}),
    ),
    "cross_nli": MetricInfo(
        key="cross_nli",
        module="metrics.cross_nli",
        class_name="Cross_NLI",
        kind=InputKind.SINGLE,
        returns_detail=True,
        default_params={},
        allowed_params=frozenset(),
    ),
    "intra_nli": MetricInfo(
        key="intra_nli",
        module="metrics.intra_nli",
        class_name="IntraNLI",
        kind=InputKind.NONE,
        returns_detail=False,
        default_params={},
        allowed_params=frozenset({"hf_cache_dir", "hg_model_hub_name"}),
    ),
    "readability": MetricInfo(
        key="readability",
        module="metrics.readability_metric",
        class_name="ReadabilityMetric",
        kind=InputKind.NONE,
        returns_detail=False,
        default_params={"readability_type": "flesch_kincaid"},
        allowed_params=frozenset({"readability_type"}),
    ),
    "mhic": MetricInfo(
        key="mhic",
        module="metrics.mhic",
        class_name="MHIC",
        kind=InputKind.LIST,
        returns_detail=False,
        default_params={},
        allowed_params=frozenset(),
    ),
    "fc": MetricInfo(
        key="fc",
        module="metrics.fc",
        class_name="FactualConsistency",
        kind=InputKind.DUAL,
        returns_detail=True,
        default_params={},
        allowed_params=frozenset(),
    ),
    "fact": MetricInfo(
        key="fact",
        module="metrics.fact",
        class_name="FactScorer",
        kind=InputKind.PRECOMPUTE_CLAIMS,
        returns_detail=True,
        default_params={"min_claim": 1, "max_claim": 30},
        allowed_params=frozenset({"min_claim", "max_claim"}),
        # FactScorer's constructor requires llm_text/reference positionally, but neither
        # attribute is read anywhere else in the class (get_claims/calculate_metric take
        # their own parameters) -- these are dead placeholders, not tunable inputs.
        fixed_params={"llm_text": {}, "reference": {}},
    ),
    "greenscore": MetricInfo(
        key="greenscore",
        module="metrics.greenscore",
        class_name="GreenScorer",
        kind=InputKind.BATCH,
        returns_detail=False,
        default_params={"model_name": "StanfordAIMI/GREEN-radllama2-7b"},
        allowed_params=frozenset({"model_name"}),
    ),
}
