import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx

from evaluation_api import worker
from evaluation_api.config import load_settings
from evaluation_api.store import JobStore

RESULTS = {
    "fc_document": {"document_level": [0.6], "mean": 0.6,
                    "detail": [{"scores": [0.6], "sentences": ["private text"]}]},
    "document_ids": ["doc1"],
}


def _setup(tmp_path, sensitive=False, callback_url=None):
    settings = load_settings({"METRIC_JOB_ROOT": str(tmp_path / "jobs"),
                              "METRIC_DB_PATH": str(tmp_path / "jobs.db")})
    store = JobStore(settings.db_path)
    job_dir = settings.job_root / "job-dir"
    job_dir.mkdir(parents=True)
    (job_dir / "llm_summaries.json").write_text(json.dumps({"doc1": "a"}))
    (job_dir / "gold_summaries.json").write_text(json.dumps({"doc1": "b"}))
    (job_dir / "posts.json").write_text(json.dumps({"doc1": ["p"]}))
    store.create(name="my_run", metric_ids=["fc_document"], sensitive=sensitive,
                 job_dir=str(job_dir), metadata={"path_id": "p", "title": "t"},
                 callback_url=callback_url,
                 callback_token="token" if callback_url else None,
                 callback_header="X-Admin-Token" if callback_url else None)
    return settings, store, store.claim_next(), job_dir


def _run_ok(argv, **kwargs):
    Path(argv[argv.index("--output_file") + 1]).write_text(json.dumps(RESULTS))
    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def _client_factory(status_code=201, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(status_code)
    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


def test_successful_job_is_marked_succeeded_with_results(tmp_path):
    settings, store, job, _ = _setup(tmp_path)
    worker.process_job(job, store, settings, run=_run_ok)
    stored = store.get(job.id)
    assert stored.status == "succeeded"
    assert stored.results["fc_document"]["mean"] == 0.6


def test_input_files_are_deleted_when_the_job_finishes(tmp_path):
    settings, store, job, job_dir = _setup(tmp_path)
    worker.process_job(job, store, settings, run=_run_ok)
    assert not (job_dir / "llm_summaries.json").exists()
    assert not (job_dir / "gold_summaries.json").exists()
    assert not (job_dir / "posts.json").exists()


def test_input_files_are_deleted_even_when_the_job_fails(tmp_path):
    settings, store, job, job_dir = _setup(tmp_path)

    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")

    worker.process_job(job, store, settings, run=run)
    assert store.get(job.id).status == "failed"
    assert not (job_dir / "llm_summaries.json").exists()


def test_failure_records_the_stderr_tail(tmp_path):
    settings, store, job, _ = _setup(tmp_path)

    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="CUDA out of memory")

    worker.process_job(job, store, settings, run=run)
    assert "CUDA out of memory" in store.get(job.id).error


def test_sensitive_job_stores_no_document_text(tmp_path):
    settings, store, job, _ = _setup(tmp_path, sensitive=True)
    worker.process_job(job, store, settings, run=_run_ok)
    assert "private text" not in json.dumps(store.get(job.id).results)


def test_callback_body_for_a_non_sensitive_job_carries_the_data(tmp_path):
    seen = []
    settings, store, job, _ = _setup(tmp_path, callback_url="http://platform/ingest")
    worker.process_job(job, store, settings, run=_run_ok,
                       client_factory=_client_factory(seen=seen))
    assert seen[0]["llm_summaries"] == {"doc1": "a"}
    assert seen[0]["inputs"] == {"doc1": ["p"]}


def test_callback_body_for_a_sensitive_job_carries_no_text(tmp_path):
    seen = []
    settings, store, job, _ = _setup(tmp_path, sensitive=True,
                                     callback_url="http://platform/ingest")
    worker.process_job(job, store, settings, run=_run_ok,
                       client_factory=_client_factory(seen=seen))
    assert "llm_summaries" not in seen[0]
    assert "private text" not in json.dumps(seen[0])


def test_successful_delivery_purges_results_but_keeps_the_row(tmp_path):
    settings, store, job, job_dir = _setup(tmp_path, callback_url="http://platform/ingest")
    worker.process_job(job, store, settings, run=_run_ok, client_factory=_client_factory())
    stored = store.get(job.id)
    assert stored is not None
    assert stored.status == "succeeded"
    assert stored.results is None
    assert stored.callback_status == "sent"
    assert stored.purged_at is not None
    assert not job_dir.exists()


def test_failed_delivery_keeps_the_results(tmp_path):
    """Purging here would destroy the only remaining copy."""
    settings, store, job, job_dir = _setup(tmp_path, callback_url="http://platform/ingest")
    worker.process_job(job, store, settings, run=_run_ok,
                       client_factory=_client_factory(status_code=500))
    stored = store.get(job.id)
    assert stored.callback_status == "failed"
    assert stored.results is not None
    assert stored.purged_at is None
    assert job_dir.exists()


def test_a_job_without_a_callback_is_never_purged(tmp_path):
    settings, store, job, _ = _setup(tmp_path)
    worker.process_job(job, store, settings, run=_run_ok)
    stored = store.get(job.id)
    assert stored.results is not None
    assert stored.purged_at is None
    assert stored.callback_status == "not_configured"


def test_a_failed_job_fires_no_callback(tmp_path):
    seen = []
    settings, store, job, _ = _setup(tmp_path, callback_url="http://platform/ingest")

    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")

    worker.process_job(job, store, settings, run=run,
                       client_factory=_client_factory(seen=seen))
    assert seen == []
    assert store.get(job.id).error is not None


def test_sweep_expired_removes_aged_jobs_and_their_directories(tmp_path):
    settings, store, job, job_dir = _setup(tmp_path)
    worker.process_job(job, store, settings, run=_run_ok)
    future = datetime.now(timezone.utc) + timedelta(hours=settings.retention_hours + 1)
    assert worker.sweep_expired(store, settings, now=future) == 1
    assert store.get(job.id) is None
    assert not job_dir.exists()


def test_sweep_expired_leaves_recent_jobs_alone(tmp_path):
    settings, store, job, _ = _setup(tmp_path)
    worker.process_job(job, store, settings, run=_run_ok)
    assert worker.sweep_expired(store, settings) == 0
    assert store.get(job.id) is not None
