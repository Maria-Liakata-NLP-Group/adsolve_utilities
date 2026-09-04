"""Static metadata describing the metric wrapper classes in evaluation_bundles/metrics/.

Used by generate_bundle.py to validate bundle specs and render bundle scripts without
needing to import the (heavy, GPU-dependent) metric classes themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Set


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
    # Which container image can run this metric. Metrics only leave "standard"
    # when their dependencies demonstrably conflict -- today, only greenscore.
    environment: str = "standard"


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
        environment="greenscore",
    ),
}


from typing import Optional


@dataclass(frozen=True)
class CatalogEntry:
    """One selectable metric offered over the API.

    METRIC_REGISTRY is keyed by implementation ('fc'); a selectable metric is a
    *combination* of implementation, reference source and parameters ('fc_expert'
    is fc against the gold summary, 'fc_document' the same class against the
    source posts). Which combinations to offer, and what to call them, is an
    editorial decision that cannot be derived from the registry -- `kind` does
    not determine `reference`.

    Only the editorial part lives here. Module, class name, kind, returns_detail
    and allowed_params are always looked up through `metric`, never copied, so
    variants of one metric cannot drift apart.
    """

    metric: str                    # key into METRIC_REGISTRY
    reference: Optional[str]       # "gold" | "posts" | "llm" | None
    label: str
    description: str = ""
    params: dict = field(default_factory=dict)
    # Extra bundle-spec keys some metric kinds need, e.g. fact's mode/claim_source.
    extra: dict = field(default_factory=dict)


METRIC_CATALOG: Dict[str, CatalogEntry] = {
    "rouge": CatalogEntry(
        metric="rouge", reference="gold", label="ROUGE",
        description="N-gram overlap with the reference summary."),
    # Named to match the platform's own metric id, so results ingest unmapped.
    "bertscore": CatalogEntry(
        metric="bertscore", reference="gold", label="BERTScore",
        description="Contextual embedding similarity to the reference summary."),
    "style_similarity": CatalogEntry(
        metric="style_roberta", reference="gold", label="Style similarity",
        description="Stylistic closeness to the reference summary."),
    "evidence_appropriateness": CatalogEntry(
        metric="evidence_appropriateness", reference="gold",
        label="Evidence appropriateness",
        description="Whether cited evidence supports the summary's claims."),
    "cross_nli": CatalogEntry(
        metric="cross_nli", reference="gold", label="Cross-summary entailment",
        description="Entailment between the generated and reference summaries."),
    "intra_nli": CatalogEntry(
        metric="intra_nli", reference=None, label="Intra-summary consistency",
        description="Self-contradiction within the generated summary."),
    "flesch_kincaid_grade_level": CatalogEntry(
        metric="readability", reference=None, label="Flesch-Kincaid grade level",
        description="US school grade level required to read the summary.",
        params={"readability_type": "flesch_kincaid"}),
    "mhic": CatalogEntry(
        metric="mhic", reference="posts", label="MHIC",
        description="Mental-health information coverage against the source."),
    "fc_expert": CatalogEntry(
        metric="fc", reference="gold", label="Factual consistency (vs. expert summary)",
        description="Sentence-level factual support from the reference summary."),
    "fc_document": CatalogEntry(
        metric="fc", reference="posts", label="Factual consistency (vs. source document)",
        description="Sentence-level factual support from the source document."),
    "fact_recall": CatalogEntry(
        metric="fact", reference="gold", label="FActScore recall",
        description="Share of reference claims present in the generated summary.",
        extra={"mode": "recall", "claim_source": "gold"}),
    "fact_precision": CatalogEntry(
        metric="fact", reference="gold", label="FActScore precision",
        description="Share of generated claims supported by the reference.",
        extra={"mode": "precision", "claim_source": "llm"}),
    "green_score": CatalogEntry(
        metric="greenscore", reference="gold", label="GREEN score",
        description="Radiology report generation quality (isolated environment)."),
}


class UnknownMetricError(Exception):
    """Raised when a caller names metric ids that are not in the catalog."""

    def __init__(self, unknown: List[str]) -> None:
        self.unknown = unknown
        super().__init__(f"Unknown metric ids: {', '.join(unknown)}")


def expand_catalog_ids(metric_ids: List[str]) -> List[dict]:
    """Turn catalog ids into bundle-spec metric entries, preserving order.

    The result is exactly what someone would hand-write under bundle_specs/, so
    everything downstream -- validate_spec, render_bundle -- is unchanged.
    """
    unknown = [m for m in metric_ids if m not in METRIC_CATALOG]
    if unknown:
        raise UnknownMetricError(unknown)

    entries: List[dict] = []
    for metric_id in metric_ids:
        catalog_entry = METRIC_CATALOG[metric_id]
        entry: dict = {"id": metric_id, "metric": catalog_entry.metric}
        if catalog_entry.reference is not None:
            entry["reference"] = catalog_entry.reference
        if catalog_entry.params:
            entry["params"] = dict(catalog_entry.params)
        entry.update(catalog_entry.extra)
        entries.append(entry)
    return entries


def build_spec(name: str, metric_ids: List[str]) -> dict:
    """Build a full bundle spec, the same shape load_spec() returns for YAML."""
    return {"name": name, "metrics": expand_catalog_ids(metric_ids)}


def required_references(metric_ids: List[str]) -> Set[str]:
    """Which reference data the run needs: subset of {'gold', 'posts'}."""
    references = {METRIC_CATALOG[m].reference for m in metric_ids}
    return {r for r in references if r in {"gold", "posts"}}
