import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation_bundles"))

import pytest

import generate_bundle as gb
from evaluation_api import catalog


def test_expand_produces_a_valid_bundle_spec_entry():
    assert catalog.expand(["fc_document"]) == [
        {"id": "fc_document", "metric": "fc", "reference": "posts"}
    ]


def test_expand_omits_reference_for_none_kind_metrics():
    assert catalog.expand(["intra_nli"]) == [{"id": "intra_nli", "metric": "intra_nli"}]


def test_expand_includes_params():
    entry = catalog.expand(["flesch_kincaid_grade_level"])[0]
    assert entry["params"] == {"readability_type": "flesch_kincaid"}


def test_expand_includes_extra_keys_for_fact():
    entry = catalog.expand(["fact_recall"])[0]
    assert entry["mode"] == "recall"
    assert entry["claim_source"] == "gold"


def test_expand_raises_on_unknown_ids_and_names_all_of_them():
    with pytest.raises(catalog.UnknownMetricError) as excinfo:
        catalog.expand(["mhic", "nope", "also_nope"])
    assert excinfo.value.unknown == ["nope", "also_nope"]


def test_expanded_spec_passes_existing_validation():
    spec = catalog.build_spec("my_run", ["mhic", "intra_nli", "fc_expert"])
    gb.validate_spec(spec)  # must not raise


def test_expansion_reproduces_the_hand_written_social_media_spec():
    """The catalog is a pre-registered form of what people hand-write today."""
    spec_path = str(REPO_ROOT / "evaluation_bundles/bundle_specs/social_media_example.yaml")
    hand_written = gb.load_spec(spec_path)
    expanded = catalog.build_spec(
        "social_media_example",
        ["mhic", "intra_nli", "fc_expert", "fc_document", "style_similarity", "bert_score"],
    )
    assert gb.render_bundle(expanded, spec_path) == gb.render_bundle(hand_written, spec_path)


def test_required_references_reports_what_data_the_run_needs():
    assert catalog.required_references(["fc_expert", "mhic"]) == {"gold", "posts"}
    assert catalog.required_references(["intra_nli"]) == set()


def test_environments_used_reports_standard_by_default():
    assert catalog.environments_used(["mhic", "fc_expert"]) == {"standard"}
    assert catalog.environments_used(["green_score"]) == {"greenscore"}


def test_all_ids_lists_every_selectable_metric():
    ids = catalog.all_ids()
    assert "fc_expert" in ids and "green_score" in ids
    assert len(ids) == len(set(ids))


def test_describe_returns_the_api_listing_shape():
    assert catalog.describe("fc_document", available=True) == {
        "id": "fc_document",
        "label": "Factual consistency (vs. source document)",
        "description": "Sentence-level factual support from the source document.",
        "requires": "posts",
        "environment": "standard",
        "available": True,
    }
