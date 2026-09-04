"""Catalog-id expansion, used by the batch job runner.

METRIC_CATALOG lives in metric_registry, so expansion belongs beside it rather
than in evaluation_api -- the batch job must not depend on the API package.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import generate_bundle as gb
from metric_registry import UnknownMetricError, build_spec, expand_catalog_ids


def test_expand_produces_a_valid_bundle_spec_entry():
    assert expand_catalog_ids(["fc_document"]) == [
        {"id": "fc_document", "metric": "fc", "reference": "posts"}
    ]


def test_expand_omits_reference_for_none_kind_metrics():
    assert expand_catalog_ids(["intra_nli"]) == [{"id": "intra_nli", "metric": "intra_nli"}]


def test_expand_includes_params():
    entry = expand_catalog_ids(["flesch_kincaid_grade_level"])[0]
    assert entry["params"] == {"readability_type": "flesch_kincaid"}


def test_expand_includes_extra_keys_for_fact():
    entry = expand_catalog_ids(["fact_recall"])[0]
    assert entry["mode"] == "recall"
    assert entry["claim_source"] == "gold"


def test_expand_preserves_the_requested_order():
    ids = ["mhic", "intra_nli", "fc_expert"]
    assert [e["id"] for e in expand_catalog_ids(ids)] == ids


def test_expand_raises_on_unknown_ids_and_names_all_of_them():
    with pytest.raises(UnknownMetricError) as excinfo:
        expand_catalog_ids(["mhic", "nope", "also_nope"])
    assert excinfo.value.unknown == ["nope", "also_nope"]


def test_build_spec_passes_existing_validation():
    gb.validate_spec(build_spec("my_run", ["mhic", "intra_nli", "fc_expert"]))


def test_build_spec_reproduces_the_hand_written_social_media_spec():
    """Catalog ids are a pre-registered form of what people hand-write today."""
    spec_path = str(Path(__file__).resolve().parents[1] / "bundle_specs/social_media_example.yaml")
    hand_written = gb.load_spec(spec_path)
    expanded = build_spec(
        "social_media_example",
        ["mhic", "intra_nli", "fc_expert", "fc_document", "style_similarity", "bertscore"],
    )
    assert gb.render_bundle(expanded, spec_path) == gb.render_bundle(hand_written, spec_path)
