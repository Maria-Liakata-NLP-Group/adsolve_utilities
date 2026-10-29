"""One real run: render a script, execute it as a subprocess, read the results.

Uses `flesch_kincaid_grade_level` because readability is InputKind.NONE and needs
only readability-lxml -- no GPU, no HuggingFace download. This is the test that
catches a broken PYTHONPATH, which no mocked test can.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from evaluation_api import runner
from evaluation_api.config import load_settings

SUMMARIES = {
    "doc1": "The patient described persistent low mood over several weeks.",
    "doc2": "Sleep improved after the client began a regular evening routine.",
}


@pytest.mark.slow
def test_a_real_bundle_runs_and_produces_scores(tmp_path):
    import json

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "llm_summaries.json").write_text(json.dumps(SUMMARIES))
    # No metric needs gold here, so empty strings are written and never read.
    (job_dir / "gold_summaries.json").write_text(
        json.dumps({doc_id: "" for doc_id in SUMMARIES}))

    settings = load_settings({"METRIC_JOB_TIMEOUT_SECONDS": "300"})
    results = runner.execute("smoke_test", ["flesch_kincaid_grade_level"],
                             job_dir, settings)

    assert results["document_ids"] == ["doc1", "doc2"]
    assert len(results["flesch_kincaid_grade_level"]["document_level"]) == 2
    assert isinstance(results["flesch_kincaid_grade_level"]["mean"], float)
