"""FastAPI application: auth, metric listing, and evaluation endpoints."""
from __future__ import annotations

import hmac
from typing import List

from fastapi import Depends, FastAPI, Header, HTTPException

from . import catalog
from .config import Settings


def create_app(settings: Settings, start_worker: bool = True) -> FastAPI:
    app = FastAPI(title="AdSoLve Metric Calculation API")
    app.state.settings = settings

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

    return app
