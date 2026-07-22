import subprocess
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


def test_to_pascal_case():
    assert gb.to_pascal_case("my_use_case") == "MyUseCase"
    assert gb.to_pascal_case("court_case") == "CourtCase"
    assert gb.to_pascal_case("x") == "X"


def test_uses_posts_true_when_any_reference_is_posts():
    spec = {"metrics": [
        {"id": "a", "metric": "rouge", "reference": "gold"},
        {"id": "b", "metric": "mhic", "reference": "posts"},
    ]}
    assert gb._uses_posts(spec) is True


def test_uses_posts_false_when_no_reference_is_posts():
    spec = {"metrics": [{"id": "a", "metric": "rouge", "reference": "gold"}]}
    assert gb._uses_posts(spec) is False


def test_instance_groups_dedups_identical_metric_and_params():
    spec = {"metrics": [
        {"id": "fc_expert", "metric": "fc", "reference": "gold"},
        {"id": "fc_document", "metric": "fc", "reference": "posts"},
    ]}
    entry_attr, instances = gb._instance_groups(spec)
    assert entry_attr == {"fc_expert": "fc_expert", "fc_document": "fc_expert"}
    assert instances == [("fc_expert", "fc", {})]


def test_instance_groups_separate_instances_for_different_params():
    spec = {"metrics": [
        {"id": "rouge_1", "metric": "rouge", "reference": "gold", "params": {"configuration": "1"}},
        {"id": "rouge_l", "metric": "rouge", "reference": "gold", "params": {"configuration": "l"}},
    ]}
    entry_attr, instances = gb._instance_groups(spec)
    assert entry_attr == {"rouge_1": "rouge_1", "rouge_l": "rouge_l"}
    assert len(instances) == 2


def test_reference_expr_single_kind_gold():
    entry = {"reference": "gold"}
    info = gb.METRIC_REGISTRY["rouge"]
    assert gb._reference_expr(entry, info) == "gold_summary"


def test_reference_expr_single_kind_posts_joins():
    entry = {"reference": "posts"}
    info = gb.METRIC_REGISTRY["rouge"]
    assert gb._reference_expr(entry, info) == '" ".join(document_posts)'


def test_reference_expr_list_kind_gold_wraps():
    entry = {"reference": "gold"}
    info = gb.METRIC_REGISTRY["mhic"]
    assert gb._reference_expr(entry, info) == "[gold_summary]"


def test_reference_expr_list_kind_posts_passthrough():
    entry = {"reference": "posts"}
    info = gb.METRIC_REGISTRY["mhic"]
    assert gb._reference_expr(entry, info) == "document_posts"


def test_reference_expr_dual_kind_passthrough():
    entry = {"reference": "posts"}
    info = gb.METRIC_REGISTRY["fc"]
    assert gb._reference_expr(entry, info) == "document_posts"


def test_reference_expr_precompute_claims_posts_joins():
    entry = {"reference": "posts"}
    info = gb.METRIC_REGISTRY["fact"]
    assert gb._reference_expr(entry, info) == '" ".join(document_posts)'


def test_format_ctor_call_merges_default_and_entry_params():
    info = gb.METRIC_REGISTRY["rouge"]
    call = gb._format_ctor_call(info, {"metric": "f"})
    assert call == "ROUGE(configuration='1', metric='f')"


def test_format_ctor_call_includes_fact_fixed_params():
    info = gb.METRIC_REGISTRY["fact"]
    call = gb._format_ctor_call(info, {})
    assert call == "FactScorer(llm_text={}, reference={}, min_claim=1, max_claim=30)"


def test_loop_body_lines_none_kind_no_detail():
    entry = {"id": "intra", "metric": "intra_nli"}
    info = gb.METRIC_REGISTRY["intra_nli"]
    lines = gb._loop_body_lines(entry, info, "intra")
    assert lines == [
        "intra_score = self.intra.calculate_metric(llm_summary)",
        "results['intra']['document_level'].append(intra_score)",
    ]


def test_loop_body_lines_single_kind_no_detail():
    entry = {"id": "rouge_1", "metric": "rouge", "reference": "gold"}
    info = gb.METRIC_REGISTRY["rouge"]
    lines = gb._loop_body_lines(entry, info, "rouge_1")
    assert lines == [
        "rouge_1_score = self.rouge_1.calculate_metric(llm_summary, gold_summary)",
        "results['rouge_1']['document_level'].append(rouge_1_score)",
    ]


def test_loop_body_lines_dual_kind_with_detail_uses_shared_attr():
    entry = {"id": "fc_document", "metric": "fc", "reference": "posts"}
    info = gb.METRIC_REGISTRY["fc"]
    lines = gb._loop_body_lines(entry, info, "fc_expert")
    assert lines == [
        "fc_document_score, fc_document_detail = self.fc_expert.calculate_metric(llm_summary, document_posts)",
        "results['fc_document']['document_level'].append(fc_document_score)",
        "results['fc_document']['detail'].append(fc_document_detail)",
    ]


def test_loop_body_lines_precompute_claims_kind():
    entry = {
        "id": "conciseness", "metric": "fact", "mode": "recall",
        "claim_source": "llm", "reference": "gold",
    }
    info = gb.METRIC_REGISTRY["fact"]
    lines = gb._loop_body_lines(entry, info, "conciseness")
    assert lines == [
        'conciseness_score, conciseness_detail = self.conciseness.calculate_metric('
        'type="recall", claims=conciseness_claims[document_id], reference=gold_summary)',
        "results['conciseness']['document_level'].append(conciseness_score)",
        "results['conciseness']['detail'].append(conciseness_detail)",
    ]


def test_claims_block_lines_llm_source():
    entry = {"id": "conciseness", "claim_source": "llm"}
    lines = gb._claims_block_lines(entry, "conciseness")
    assert lines == [
        "print(\"Generating claims for 'conciseness'...\")",
        "conciseness_claims = self.conciseness.get_claims(llm_summaries)",
    ]


def test_claims_block_lines_gold_source():
    entry = {"id": "conciseness", "claim_source": "gold"}
    lines = gb._claims_block_lines(entry, "conciseness")
    assert lines[1] == "conciseness_claims = self.conciseness.get_claims(gold_summaries)"


def test_batch_block_lines_gold_reference():
    entry = {"id": "green_score", "reference": "gold"}
    lines = gb._batch_block_lines(entry, "green_score")
    assert lines == [
        "green_score_references = [gold_summaries[doc_id] for doc_id in results['document_ids']]",
        "green_score_hypotheses = [llm_summaries[doc_id] for doc_id in results['document_ids']]",
        "green_score_mean, green_score_std, green_score_score_list, green_score_summary, _ = "
        "self.green_score.calculate_metric(green_score_references, green_score_hypotheses)",
        "results['green_score']['document_level'] = green_score_score_list",
        "results['green_score']['mean'] = green_score_mean",
        "results['green_score']['std'] = green_score_std",
        "results['green_score']['summary'] = green_score_summary",
    ]


def test_batch_block_lines_posts_reference_joins():
    entry = {"id": "green_score", "reference": "posts"}
    lines = gb._batch_block_lines(entry, "green_score")
    assert lines[0] == (
        'green_score_references = [" ".join(posts[doc_id]) for doc_id in '
        "results['document_ids']]"
    )


FULL_FEATURED_SPEC = """
name: full_featured
metrics:
  - id: intra
    metric: intra_nli
  - id: rouge_1
    metric: rouge
    reference: gold
  - id: mhic
    metric: mhic
    reference: posts
  - id: fc_expert
    metric: fc
    reference: gold
  - id: fc_document
    metric: fc
    reference: posts
  - id: conciseness
    metric: fact
    mode: recall
    claim_source: llm
    reference: gold
  - id: green_score
    metric: greenscore
    reference: gold
"""


def test_render_bundle_is_valid_python_for_full_featured_spec(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, FULL_FEATURED_SPEC))
    gb.validate_spec(spec)
    source = gb.render_bundle(spec, "bundle_specs/full_featured.yaml")
    compile(source, "<generated:full_featured>", "exec")  # raises SyntaxError if invalid


ALL_METRICS_SPEC = """
name: all_metrics
metrics:
  - id: intra
    metric: intra_nli
  - id: readability
    metric: readability
  - id: rouge_1
    metric: rouge
    reference: gold
  - id: bert_score
    metric: bertscore
    reference: gold
  - id: style_similarity
    metric: style_roberta
    reference: gold
  - id: evidence
    metric: evidence_appropriateness
    reference: gold
  - id: cross_nli_score
    metric: cross_nli
    reference: gold
  - id: mhic
    metric: mhic
    reference: posts
  - id: fc_expert
    metric: fc
    reference: gold
  - id: conciseness
    metric: fact
    mode: recall
    claim_source: llm
    reference: gold
  - id: green_score
    metric: greenscore
    reference: gold
"""


def test_render_bundle_is_valid_python_for_all_registry_metrics(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, ALL_METRICS_SPEC))
    gb.validate_spec(spec)
    source = gb.render_bundle(spec, "bundle_specs/all_metrics.yaml")
    compile(source, "<generated:all_metrics>", "exec")


def test_render_bundle_class_name_and_imports():
    spec = {"name": "my_use_case", "metrics": [
        {"id": "rouge_1", "metric": "rouge", "reference": "gold"},
    ]}
    source = gb.render_bundle(spec, "spec.yaml")
    assert "class MyUseCaseEvaluationBundle:" in source
    assert "from metrics.rouge import ROUGE" in source
    assert "self.rouge_1 = ROUGE(configuration='1', metric='p')" in source


def test_render_bundle_dedups_shared_instance_in_init():
    spec = {"name": "social", "metrics": [
        {"id": "fc_expert", "metric": "fc", "reference": "gold"},
        {"id": "fc_document", "metric": "fc", "reference": "posts"},
    ]}
    source = gb.render_bundle(spec, "spec.yaml")
    assert source.count("FactualConsistency()") == 1
    assert "self.fc_expert.calculate_metric(llm_summary, gold_summary)" in source
    assert "self.fc_expert.calculate_metric(llm_summary, document_posts)" in source


def test_render_bundle_omits_posts_cli_arg_when_unused():
    spec = {"name": "simple", "metrics": [
        {"id": "rouge_1", "metric": "rouge", "reference": "gold"},
    ]}
    source = gb.render_bundle(spec, "spec.yaml")
    assert "--posts" not in source
    assert "def evaluate(self, llm_summaries: dict, gold_summaries: dict) -> dict:" in source


def test_render_bundle_includes_posts_cli_arg_when_used():
    spec = {"name": "social", "metrics": [
        {"id": "mhic", "metric": "mhic", "reference": "posts"},
    ]}
    source = gb.render_bundle(spec, "spec.yaml")
    assert "--posts" in source
    assert "posts: dict = None) -> dict:" in source
    assert "document_posts = posts[document_id]" in source


def test_render_bundle_header_references_spec_path():
    spec = {"name": "simple", "metrics": [
        {"id": "rouge_1", "metric": "rouge", "reference": "gold"},
    ]}
    source = gb.render_bundle(spec, "bundle_specs/simple.yaml")
    assert source.startswith("# Generated by generate_bundle.py from bundle_specs/simple.yaml")


def test_render_bundle_final_mean_loop_excludes_batch_metrics():
    spec = {"name": "mixed", "metrics": [
        {"id": "rouge_1", "metric": "rouge", "reference": "gold"},
        {"id": "green_score", "metric": "greenscore", "reference": "gold"},
    ]}
    source = gb.render_bundle(spec, "spec.yaml")
    assert "for metric_id in ['rouge_1']:" in source
    assert "results['green_score']['mean'] = green_score_mean" in source


def test_main_writes_generated_file(tmp_path):
    spec_path = _write_spec(tmp_path, MINIMAL_VALID_SPEC)
    exit_code = gb.main(["--spec", spec_path, "--output-dir", str(tmp_path)])
    assert exit_code == 0
    output_path = tmp_path / "my_use_case_evaluation.py"
    assert output_path.exists()
    content = output_path.read_text()
    assert "class MyUseCaseEvaluationBundle:" in content
    compile(content, "<generated>", "exec")


def test_main_refuses_to_overwrite_non_generated_file(tmp_path, capsys):
    """Verify that main() refuses to overwrite a non-generated file."""
    output_path = tmp_path / "my_use_case_evaluation.py"
    non_generated_content = "# hand-written bundle, not generated\nprint('do not overwrite me')\n"
    output_path.write_text(non_generated_content)

    spec_path = _write_spec(tmp_path, MINIMAL_VALID_SPEC)
    exit_code = gb.main(["--spec", spec_path, "--output-dir", str(tmp_path)])

    assert exit_code == 1
    assert output_path.read_text() == non_generated_content
    captured = capsys.readouterr()
    assert "Refusing to overwrite" in captured.err


def test_main_allows_overwriting_previously_generated_file(tmp_path):
    """Verify that main() allows regenerating a previously-generated file."""
    spec_path = _write_spec(tmp_path, MINIMAL_VALID_SPEC)

    # First call: create the file
    exit_code1 = gb.main(["--spec", spec_path, "--output-dir", str(tmp_path)])
    assert exit_code1 == 0

    # Second call: regenerate the same file
    exit_code2 = gb.main(["--spec", spec_path, "--output-dir", str(tmp_path)])
    assert exit_code2 == 0

    output_path = tmp_path / "my_use_case_evaluation.py"
    content = output_path.read_text()
    assert "# Generated by generate_bundle.py" in content
    compile(content, "<generated>", "exec")


def test_main_returns_nonzero_and_writes_nothing_on_invalid_spec(tmp_path, capsys):
    spec_path = _write_spec(tmp_path, """
name: my_use_case
metrics:
  - id: bad
    metric: not_a_real_metric
""")
    exit_code = gb.main(["--spec", spec_path, "--output-dir", str(tmp_path)])
    assert exit_code == 1
    assert not (tmp_path / "my_use_case_evaluation.py").exists()
    captured = capsys.readouterr()
    assert "unknown metric" in captured.err


def test_cli_subprocess_generates_file(tmp_path):
    spec_path = _write_spec(tmp_path, MINIMAL_VALID_SPEC)
    generator = Path(__file__).resolve().parents[1] / "generate_bundle.py"
    result = subprocess.run(
        [sys.executable, str(generator), "--spec", spec_path, "--output-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "my_use_case_evaluation.py").exists()
