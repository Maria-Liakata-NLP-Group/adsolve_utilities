import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metric_registry import METRIC_CATALOG, METRIC_REGISTRY, CatalogEntry, InputKind

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

EXPECTED_CATALOG_IDS = {
    "rouge", "bertscore", "style_similarity", "evidence_appropriateness",
    "cross_nli", "intra_nli", "flesch_kincaid_grade_level", "mhic",
    "fc_expert", "fc_document", "fact_recall", "fact_precision", "green_score",
}


def test_catalog_has_all_expected_ids():
    assert set(METRIC_CATALOG) == EXPECTED_CATALOG_IDS


def test_every_catalog_entry_references_a_known_metric():
    for catalog_id, entry in METRIC_CATALOG.items():
        assert entry.metric in METRIC_REGISTRY, f"{catalog_id}: unknown metric '{entry.metric}'"


def test_catalog_ids_are_snake_case():
    for catalog_id in METRIC_CATALOG:
        assert NAME_PATTERN.match(catalog_id), f"{catalog_id} is not snake_case"


def test_none_kind_entries_have_no_reference():
    for catalog_id, entry in METRIC_CATALOG.items():
        if METRIC_REGISTRY[entry.metric].kind == InputKind.NONE:
            assert entry.reference is None, f"{catalog_id} must not set a reference"


def test_reference_taking_entries_declare_a_valid_reference():
    for catalog_id, entry in METRIC_CATALOG.items():
        info = METRIC_REGISTRY[entry.metric]
        if info.kind == InputKind.NONE:
            continue
        if info.kind == InputKind.PRECOMPUTE_CLAIMS:
            assert entry.reference in {"gold", "posts", "llm"}, catalog_id
        else:
            assert entry.reference in {"gold", "posts"}, catalog_id


def test_catalog_params_are_allowed_by_the_registry():
    for catalog_id, entry in METRIC_CATALOG.items():
        allowed = METRIC_REGISTRY[entry.metric].allowed_params
        assert set(entry.params) <= allowed, f"{catalog_id}: {set(entry.params) - allowed}"


def test_precompute_claims_entries_declare_mode_and_claim_source():
    for catalog_id, entry in METRIC_CATALOG.items():
        if METRIC_REGISTRY[entry.metric].kind != InputKind.PRECOMPUTE_CLAIMS:
            continue
        assert entry.extra.get("mode") in {"recall", "precision"}, catalog_id
        assert entry.extra.get("claim_source") in {"llm", "gold"}, catalog_id


def test_labels_are_non_empty():
    for catalog_id, entry in METRIC_CATALOG.items():
        assert entry.label.strip(), f"{catalog_id} has an empty label"


def test_only_greenscore_leaves_the_standard_environment():
    non_standard = {k for k, v in METRIC_REGISTRY.items() if v.environment != "standard"}
    assert non_standard == {"greenscore"}


def test_green_score_catalog_entry_maps_to_greenscore_environment():
    entry = METRIC_CATALOG["green_score"]
    assert METRIC_REGISTRY[entry.metric].environment == "greenscore"


def test_fc_variants_share_one_implementation_and_differ_only_by_reference():
    expert, document = METRIC_CATALOG["fc_expert"], METRIC_CATALOG["fc_document"]
    assert expert.metric == document.metric == "fc"
    assert expert.reference == "gold"
    assert document.reference == "posts"
