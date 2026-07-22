import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import generate_bundle as gb


MINIMAL_VALID_SPEC = """
name: my_use_case
metrics:
  - id: rouge_1
    metric: rouge
    reference: gold
"""


def _write_spec(tmp_path, content: str) -> str:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(content)
    return str(spec_path)


def test_load_spec_reads_yaml(tmp_path):
    path = _write_spec(tmp_path, MINIMAL_VALID_SPEC)
    spec = gb.load_spec(path)
    assert spec["name"] == "my_use_case"
    assert spec["metrics"][0]["metric"] == "rouge"


def test_validate_spec_accepts_minimal_valid_spec(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, MINIMAL_VALID_SPEC))
    gb.validate_spec(spec)  # should not raise


def test_validate_spec_rejects_unknown_metric_key(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, """
name: my_use_case
metrics:
  - id: made_up
    metric: not_a_real_metric
    reference: gold
"""))
    with pytest.raises(gb.SpecError, match="unknown metric 'not_a_real_metric'"):
        gb.validate_spec(spec)


def test_validate_spec_rejects_duplicate_ids(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, """
name: my_use_case
metrics:
  - id: rouge_1
    metric: rouge
    reference: gold
  - id: rouge_1
    metric: bertscore
    reference: gold
"""))
    with pytest.raises(gb.SpecError, match="duplicate id 'rouge_1'"):
        gb.validate_spec(spec)


def test_validate_spec_rejects_missing_reference_for_single_kind(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, """
name: my_use_case
metrics:
  - id: rouge_1
    metric: rouge
"""))
    with pytest.raises(gb.SpecError, match="'reference' must be one of"):
        gb.validate_spec(spec)


def test_validate_spec_rejects_bad_param_key(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, """
name: my_use_case
metrics:
  - id: rouge_1
    metric: rouge
    reference: gold
    params:
      not_a_real_param: 1
"""))
    with pytest.raises(gb.SpecError, match="unknown params"):
        gb.validate_spec(spec)


def test_validate_spec_rejects_invalid_name_format(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, """
name: "My Use Case"
metrics:
  - id: rouge_1
    metric: rouge
    reference: gold
"""))
    with pytest.raises(gb.SpecError, match="lowercase snake_case"):
        gb.validate_spec(spec)


def test_validate_spec_rejects_fact_missing_mode_and_claim_source(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, """
name: my_use_case
metrics:
  - id: conciseness
    metric: fact
    reference: gold
"""))
    with pytest.raises(gb.SpecError, match="'mode' must be one of"):
        gb.validate_spec(spec)


def test_validate_spec_collects_multiple_errors(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, """
name: my_use_case
metrics:
  - id: a
    metric: rouge
  - id: b
    metric: does_not_exist
"""))
    with pytest.raises(gb.SpecError) as exc_info:
        gb.validate_spec(spec)
    message = str(exc_info.value)
    assert "'reference' must be one of" in message
    assert "unknown metric 'does_not_exist'" in message
