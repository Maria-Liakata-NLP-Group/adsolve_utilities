"""The batch entrypoint: spec -> rendered script -> subprocess -> results.json.

No test here loads a model or starts a real subprocess; the runner is injected.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import run_job

RESULTS = {"flesch_kincaid_grade_level": {"document_level": [12.0], "mean": 12.0},
           "document_ids": ["doc1"]}


def _inputs(tmp_path, posts=False) -> dict:
    """Write input files and return the CLI paths pointing at them."""
    (tmp_path / "llm.json").write_text(json.dumps({"doc1": "a summary"}))
    (tmp_path / "gold.json").write_text(json.dumps({"doc1": "a reference"}))
    paths = {"llm": str(tmp_path / "llm.json"), "gold": str(tmp_path / "gold.json"),
             "out": str(tmp_path / "results.json")}
    if posts:
        (tmp_path / "posts.json").write_text(json.dumps({"doc1": ["a post"]}))
        paths["posts"] = str(tmp_path / "posts.json")
    return paths


def _fake_run(results=RESULTS, returncode=0, stderr=""):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if returncode == 0:
            Path(argv[argv.index("--output_file") + 1]).write_text(json.dumps(results))
        return subprocess.CompletedProcess(argv, returncode, stdout="", stderr=stderr)

    run.calls = calls
    return run


def _argv(paths, *extra) -> list:
    return ["--name", "my_run", "--llm-summaries", paths["llm"],
            "--gold-summaries", paths["gold"], "--output", paths["out"], *extra]


def test_metrics_flag_writes_results_and_exits_zero(tmp_path):
    paths = _inputs(tmp_path)
    run = _fake_run()
    assert run_job.main(_argv(paths, "--metrics", "flesch_kincaid_grade_level"), run=run) == 0
    assert json.loads(Path(paths["out"]).read_text()) == RESULTS


def test_spec_flag_is_accepted_instead_of_metrics(tmp_path):
    paths = _inputs(tmp_path)
    spec = tmp_path / "spec.yaml"
    spec.write_text("name: from_spec\nmetrics:\n  - id: intra_nli\n    metric: intra_nli\n")
    run = _fake_run(results={"intra_nli": {"mean": 1.0}, "document_ids": ["doc1"]})
    assert run_job.main(["--spec", str(spec), "--llm-summaries", paths["llm"],
                         "--gold-summaries", paths["gold"], "--output", paths["out"]],
                        run=run) == 0
    assert Path(paths["out"]).exists()


def test_spec_name_is_used_when_no_name_is_given(tmp_path):
    paths = _inputs(tmp_path)
    spec = tmp_path / "spec.yaml"
    spec.write_text("name: from_spec\nmetrics:\n  - id: intra_nli\n    metric: intra_nli\n")
    run = _fake_run(results={"intra_nli": {"mean": 1.0}})
    run_job.main(["--spec", str(spec), "--llm-summaries", paths["llm"],
                  "--gold-summaries", paths["gold"], "--output", paths["out"]], run=run)
    script = [a for a in run.calls[0][0] if a.endswith(".py")][0]
    assert Path(script).name == "from_spec_evaluation.py"


def test_unknown_metric_id_exits_non_zero_without_running_anything(tmp_path):
    paths = _inputs(tmp_path)
    run = _fake_run()
    assert run_job.main(_argv(paths, "--metrics", "not_a_metric"), run=run) != 0
    assert run.calls == []


def test_requesting_both_spec_and_metrics_is_rejected(tmp_path):
    paths = _inputs(tmp_path)
    with pytest.raises(SystemExit):
        run_job.main(_argv(paths, "--metrics", "intra_nli", "--spec", "x.yaml"),
                     run=_fake_run())


def test_posts_flag_is_passed_only_when_a_metric_needs_posts(tmp_path):
    paths = _inputs(tmp_path, posts=True)
    run = _fake_run(results={"mhic": {"mean": 1.0}})
    run_job.main(_argv(paths, "--metrics", "mhic", "--posts", paths["posts"]), run=run)
    assert "--posts" in run.calls[0][0]

    run = _fake_run()
    run_job.main(_argv(paths, "--metrics", "flesch_kincaid_grade_level"), run=run)
    assert "--posts" not in run.calls[0][0]


def test_a_posts_metric_without_posts_exits_non_zero(tmp_path):
    """Fail before the GPU work, not during it."""
    paths = _inputs(tmp_path)
    run = _fake_run()
    assert run_job.main(_argv(paths, "--metrics", "mhic"), run=run) != 0
    assert run.calls == []


def test_gold_summaries_are_synthesised_when_no_metric_needs_them(tmp_path):
    """render_bundle emits an unconditional gold lookup; validation proves it unread."""
    (tmp_path / "llm.json").write_text(json.dumps({"doc1": "a", "doc2": "b"}))
    out = tmp_path / "results.json"
    seen = {}

    # Read the file during the call: the temp dir holding it is cleaned up on exit.
    def run(argv, **kwargs):
        gold_path = argv[argv.index("--gold_summaries") + 1]
        seen["gold"] = json.loads(Path(gold_path).read_text())
        Path(argv[argv.index("--output_file") + 1]).write_text(json.dumps(RESULTS))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    assert run_job.main(["--name", "my_run", "--metrics", "flesch_kincaid_grade_level",
                         "--llm-summaries", str(tmp_path / "llm.json"),
                         "--output", str(out)], run=run) == 0
    assert seen["gold"] == {"doc1": "", "doc2": ""}


def test_subprocess_gets_pythonpath_pointing_at_evaluation_bundles(tmp_path):
    paths = _inputs(tmp_path)
    run = _fake_run()
    run_job.main(_argv(paths, "--metrics", "flesch_kincaid_grade_level"), run=run)
    assert run.calls[0][1]["env"]["PYTHONPATH"].endswith("evaluation_bundles")


def test_a_failing_bundle_exits_non_zero_and_reports_stderr(tmp_path, capsys):
    paths = _inputs(tmp_path)
    run = _fake_run(returncode=1, stderr="CUDA out of memory")
    assert run_job.main(_argv(paths, "--metrics", "flesch_kincaid_grade_level"), run=run) != 0
    assert "CUDA out of memory" in capsys.readouterr().err


def test_the_generated_script_is_not_written_into_the_repo(tmp_path):
    """Rendering into a temp dir keeps concurrent jobs from colliding."""
    paths = _inputs(tmp_path)
    run = _fake_run()
    run_job.main(_argv(paths, "--metrics", "flesch_kincaid_grade_level"), run=run)
    script = Path([a for a in run.calls[0][0] if a.endswith(".py")][0])
    assert Path(__file__).parents[1] not in script.parents


def test_the_output_directory_is_created_if_absent(tmp_path):
    paths = _inputs(tmp_path)
    paths["out"] = str(tmp_path / "nested" / "dir" / "results.json")
    run = _fake_run()
    assert run_job.main(_argv(paths, "--metrics", "flesch_kincaid_grade_level"), run=run) == 0
    assert Path(paths["out"]).exists()
