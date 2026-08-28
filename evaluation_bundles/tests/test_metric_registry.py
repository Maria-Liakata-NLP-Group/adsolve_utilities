import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metric_registry import METRIC_REGISTRY, InputKind


EXPECTED_KEYS = {
    "rouge", "bertscore", "style_roberta", "evidence_appropriateness",
    "cross_nli", "intra_nli", "readability", "mhic", "fc", "fact", "greenscore",
}


def test_registry_has_all_expected_metrics():
    assert set(METRIC_REGISTRY) == EXPECTED_KEYS


def test_registry_keys_match_entry_key_field():
    for key, info in METRIC_REGISTRY.items():
        assert info.key == key


def test_registry_kinds_are_valid_input_kind_members():
    for info in METRIC_REGISTRY.values():
        assert isinstance(info.kind, InputKind)


def test_default_params_are_subset_of_allowed_params():
    for key, info in METRIC_REGISTRY.items():
        assert set(info.default_params) <= info.allowed_params, (
            f"{key}: default_params {set(info.default_params)} not a subset of "
            f"allowed_params {info.allowed_params}"
        )


def test_fact_has_fixed_placeholder_params():
    fact = METRIC_REGISTRY["fact"]
    assert fact.fixed_params == {"llm_text": {}, "reference": {}}


def test_greenscore_is_batch_kind_without_detail():
    greenscore = METRIC_REGISTRY["greenscore"]
    assert greenscore.kind == InputKind.BATCH
    assert greenscore.returns_detail is False


def test_fc_is_dual_kind_with_detail():
    fc = METRIC_REGISTRY["fc"]
    assert fc.kind == InputKind.DUAL
    assert fc.returns_detail is True


def test_intra_nli_and_readability_are_none_kind():
    assert METRIC_REGISTRY["intra_nli"].kind == InputKind.NONE
    assert METRIC_REGISTRY["readability"].kind == InputKind.NONE
