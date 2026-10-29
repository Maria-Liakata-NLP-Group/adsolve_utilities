"""FastAPI application: auth, metric listing, and evaluation endpoints."""
from __future__ import annotations

import hmac
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List

from fastapi import Depends, FastAPI, Header, HTTPException

from . import catalog
from .config import Settings
from .schemas import EvaluationRequest, validate_request
from .store import JobStore
from .worker import Worker


def _write_inputs(job_dir: Path, request: EvaluationRequest) -> None:
    """Write the run's data to disk. Raw content lives here, never in SQLite.

    gold_summaries is always written, even when no selected metric uses a gold
    reference: render_bundle emits `gold_summary = gold_summaries[document_id]`
    unconditionally and the generated CLI requires the flag. Validation
    guarantees no gold-referencing metric is present in that case, so the empty
    strings are provably unread.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    gold = request.gold_summaries or {doc_id: "" for doc_id in request.llm_summaries}
    (job_dir / "llm_summaries.json").write_text(json.dumps(request.llm_summaries))
    (job_dir / "gold_summaries.json").write_text(json.dumps(gold))
    if request.posts:
        (job_dir / "posts.json").write_text(json.dumps(request.posts))


def create_app(settings: Settings, start_worker: bool = True) -> FastAPI:
    store = JobStore(settings.db_path)
    worker = Worker(store, settings) if start_worker else None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Start the worker with the app; stop it on shutdown.

        Worker.start() also sweeps jobs left 'running' by a previous crash, so a
        restart never leaves the platform polling a job id that can never change.
        """
        if worker:
            worker.start()
        yield
        if worker:
            worker.stop()

    app = FastAPI(title="AdSoLve Metric Calculation API", lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.worker = worker

    def require_token(x_api_token: str = Header(...)) -> None:
        """Reject requests without the shared token.

        Mirrors require_admin in the platform backend so both repos share one
        auth pattern. An unset token is a 503, never an open endpoint.
        """
        if not settings.api_token:
            raise HTTPException(status_code=503, detail="API auth not configured on server.")
        if not hmac.compare_digest(x_api_token, settings.api_token):
            raise HTTPException(status_code=401, detail="Unauthorized")

    app.state.require_token = require_token

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/metrics", dependencies=[Depends(require_token)])
    def list_metrics() -> List[dict]:
        return [
            catalog.describe(
                metric_id,
                available=settings.is_available(
                    catalog.environments_used([metric_id]).pop()
                ),
            )
            for metric_id in catalog.all_ids()
        ]

    @app.post("/evaluations", status_code=202, dependencies=[Depends(require_token)])
    def submit_evaluation(request: EvaluationRequest) -> dict:
        errors = validate_request(request, settings)
        if errors:
            raise HTTPException(status_code=422, detail=errors)

        # Mint the id first so the inputs are on disk before the row exists.
        # The worker polls for queued jobs, so a row must never be visible
        # while its job directory is still being written.
        job_id = str(uuid.uuid4())
        job_dir = Path(settings.job_root) / job_id
        _write_inputs(job_dir, request)
        store.create(
            job_id=job_id,
            name=request.name,
            metric_ids=request.metrics,
            sensitive=request.sensitive,
            job_dir=str(job_dir),
            metadata=request.metadata,
            callback_url=request.callback.url if request.callback else None,
            callback_token=request.callback.token if request.callback else None,
            callback_header=request.callback.header_name if request.callback else None,
        )
        return {"job_id": job_id, "status": "queued"}

    @app.get("/evaluations/{job_id}", dependencies=[Depends(require_token)])
    def get_evaluation(job_id: str) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        return {
            "job_id": job.id,
            "status": job.status,
            "results": job.results,
            "error": job.error,
            "callback_status": job.callback_status,
            "submitted_at": job.submitted_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "delivered_at": job.delivered_at,
            "purged_at": job.purged_at,
        }

    return app


def app() -> FastAPI:
    """Entry point for `uvicorn evaluation_api.app:app --factory`."""
    from .config import load_settings
    return create_app(load_settings())
