"""Background worker: runs one job at a time, delivers results, purges on delivery."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import httpx

from . import callback as callback_module
from . import runner
from .config import Settings
from .sensitivity import safe_error, strip_results
from .store import Job, JobStore

logger = logging.getLogger(__name__)

INPUT_FILES = ("llm_summaries.json", "gold_summaries.json", "posts.json")
POLL_SECONDS = 1.0


def _read_inputs(job_dir: Path) -> dict:
    """Load the run's data so the callback can carry it. Non-sensitive jobs only."""
    def load(filename: str):
        path = job_dir / filename
        return json.loads(path.read_text()) if path.exists() else {}

    return {
        "llm_summaries": load("llm_summaries.json"),
        "gold_summaries": load("gold_summaries.json"),
        "posts": load("posts.json"),
    }


def _delete_inputs(job_dir: Path) -> None:
    for filename in INPUT_FILES:
        (job_dir / filename).unlink(missing_ok=True)


def process_job(job: Job, store: JobStore, settings: Settings,
                run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                client_factory: Callable[[], httpx.Client] = httpx.Client) -> None:
    """Run one job to a terminal state, then deliver and purge if configured."""
    job_dir = Path(job.job_dir)

    try:
        raw_results = runner.execute(job.name, job.metric_ids, job_dir, settings, run=run)
    except runner.BundleError as exc:
        store.mark_failed(job.id, safe_error(str(exc), job.sensitive))
        _delete_inputs(job_dir)
        return  # no callback on failure: there is nothing to ingest

    results = strip_results(raw_results, job.sensitive)
    store.mark_succeeded(job.id, results)

    # Read the inputs before deleting them -- a non-sensitive callback carries
    # them. Sensitive inputs are never transmitted, so they are simply dropped.
    inputs = None if job.sensitive else _read_inputs(job_dir)
    _delete_inputs(job_dir)

    if not job.callback_url:
        return

    with client_factory() as client:
        body = callback_module.compose_body(job, results, inputs)
        delivered = callback_module.deliver(
            job.callback_url, job.callback_header or "X-Admin-Token",
            job.callback_token or "", body, client,
        )

    if not delivered:
        # Keep the results: this is now the only copy the caller can reach.
        store.set_callback_status(job.id, "failed")
        return

    store.mark_delivered(job.id)
    store.purge(job.id)
    shutil.rmtree(job_dir, ignore_errors=True)


def sweep_expired(store: JobStore, settings: Settings,
                  now: Optional[datetime] = None) -> int:
    """Delete aged job directories and rows. A backstop, not the main mechanism."""
    now = now or datetime.now(timezone.utc)
    swept = 0
    for job in store.expired(now, settings.retention_hours):
        shutil.rmtree(job.job_dir, ignore_errors=True)
        store.delete(job.id)
        swept += 1
    return swept


class Worker:
    """One thread, one job at a time, so two runs never contend for the GPU."""

    def __init__(self, store: JobStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        orphaned = self.store.sweep_orphans()
        if orphaned:
            logger.warning("Failed %d job(s) interrupted by a restart", orphaned)
        sweep_expired(self.store, self.settings)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.store.claim_next()
            if job is None:
                self._stop.wait(POLL_SECONDS)
                continue
            try:
                process_job(job, self.store, self.settings)
            except Exception:  # a worker crash must not kill the thread
                logger.exception("Unhandled error processing job %s", job.id)
                self.store.mark_failed(job.id, "Internal error while processing the job.")
