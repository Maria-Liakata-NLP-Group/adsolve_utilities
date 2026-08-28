import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation_api.store import JobStore

METADATA = {"path_id": "affiniti_therapy", "title": "Run 1"}


def _store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "jobs.db")


def _create(store, **overrides) -> str:
    kwargs = dict(name="my_run", metric_ids=["mhic"], sensitive=False,
                  job_dir="/tmp/job", metadata=METADATA, callback_url=None)
    kwargs.update(overrides)
    return store.create(**kwargs)


def test_created_job_starts_queued(tmp_path):
    store = _store(tmp_path)
    job = store.get(_create(store))
    assert job.status == "queued"
    assert job.results is None
    assert job.submitted_at is not None


def test_metadata_and_metric_ids_round_trip(tmp_path):
    store = _store(tmp_path)
    job = store.get(_create(store, metric_ids=["mhic", "fc_expert"]))
    assert job.metadata == METADATA
    assert job.metric_ids == ["mhic", "fc_expert"]


def test_get_returns_none_for_unknown_id(tmp_path):
    assert _store(tmp_path).get("does-not-exist") is None


def test_callback_status_is_not_configured_without_a_callback_url(tmp_path):
    store = _store(tmp_path)
    assert store.get(_create(store)).callback_status == "not_configured"


def test_callback_status_is_pending_with_a_callback_url(tmp_path):
    store = _store(tmp_path)
    job_id = _create(store, callback_url="http://platform/api/runs/ingest")
    assert store.get(job_id).callback_status == "pending"


def test_callback_credentials_round_trip(tmp_path):
    store = _store(tmp_path)
    job_id = _create(store, callback_url="http://platform/ingest",
                     callback_token="secret", callback_header="X-Admin-Token")
    job = store.get(job_id)
    assert job.callback_token == "secret"
    assert job.callback_header == "X-Admin-Token"


def test_an_explicit_job_id_is_honoured(tmp_path):
    """The route mints the id first so inputs land on disk before the row exists."""
    store = _store(tmp_path)
    assert _create(store, job_id="chosen-id") == "chosen-id"


def test_claim_next_marks_the_job_running(tmp_path):
    store = _store(tmp_path)
    job_id = _create(store)
    claimed = store.claim_next()
    assert claimed.id == job_id
    assert store.get(job_id).status == "running"
    assert store.get(job_id).started_at is not None


def test_claim_next_returns_none_when_nothing_is_queued(tmp_path):
    assert _store(tmp_path).claim_next() is None


def test_claim_next_takes_the_oldest_job_first(tmp_path):
    store = _store(tmp_path)
    first = _create(store, name="first")
    _create(store, name="second")
    assert store.claim_next().id == first


def test_a_claimed_job_is_not_claimed_twice(tmp_path):
    store = _store(tmp_path)
    _create(store)
    store.claim_next()
    assert store.claim_next() is None


def test_mark_succeeded_stores_results(tmp_path):
    store = _store(tmp_path)
    job_id = _create(store)
    store.claim_next()
    store.mark_succeeded(job_id, {"mhic": {"mean": 0.5}})
    job = store.get(job_id)
    assert job.status == "succeeded"
    assert job.results == {"mhic": {"mean": 0.5}}
    assert job.finished_at is not None


def test_mark_failed_stores_the_error(tmp_path):
    store = _store(tmp_path)
    job_id = _create(store)
    store.claim_next()
    store.mark_failed(job_id, "boom")
    job = store.get(job_id)
    assert job.status == "failed"
    assert job.error == "boom"


def test_purge_clears_results_but_keeps_the_row(tmp_path):
    """A surviving row is what distinguishes 'delivered' from 'never existed'."""
    store = _store(tmp_path)
    job_id = _create(store, callback_url="http://platform/ingest")
    store.claim_next()
    store.mark_succeeded(job_id, {"mhic": {"mean": 0.5}})
    store.mark_delivered(job_id)
    store.purge(job_id)
    job = store.get(job_id)
    assert job is not None
    assert job.status == "succeeded"
    assert job.results is None
    assert job.delivered_at is not None
    assert job.purged_at is not None


def test_sweep_orphans_fails_jobs_left_running_by_a_restart(tmp_path):
    store = _store(tmp_path)
    job_id = _create(store)
    store.claim_next()
    assert store.sweep_orphans() == 1
    job = store.get(job_id)
    assert job.status == "failed"
    assert "restart" in job.error


def test_sweep_orphans_leaves_terminal_jobs_alone(tmp_path):
    store = _store(tmp_path)
    job_id = _create(store)
    store.claim_next()
    store.mark_succeeded(job_id, {})
    assert store.sweep_orphans() == 0
    assert store.get(job_id).status == "succeeded"


def test_expired_lists_only_jobs_finished_before_the_cutoff(tmp_path):
    store = _store(tmp_path)
    old = _create(store, name="old")
    store.claim_next()
    store.mark_succeeded(old, {})
    now = datetime.now(timezone.utc)
    assert [j.id for j in store.expired(now, retention_hours=168)] == []
    assert [j.id for j in store.expired(now + timedelta(hours=169), 168)] == [old]


def test_delete_removes_the_row_entirely(tmp_path):
    store = _store(tmp_path)
    job_id = _create(store)
    store.delete(job_id)
    assert store.get(job_id) is None
