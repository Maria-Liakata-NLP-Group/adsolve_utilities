"""Request and response models for the evaluation endpoints."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from . import catalog
from .config import Settings

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class Callback(BaseModel):
    """Where to push results once the run succeeds."""

    url: str = Field(min_length=1)
    token: str = Field(min_length=1)
    header_name: str = "X-Admin-Token"


class EvaluationRequest(BaseModel):
    name: str = Field(min_length=1)
    metrics: List[str] = Field(min_length=1)
    llm_summaries: Dict[str, str]
    gold_summaries: Dict[str, str] = Field(default_factory=dict)
    posts: Dict[str, List[str]] = Field(default_factory=dict)
    sensitive: bool = False
    callback: Optional[Callback] = None
    # Opaque: carries the platform's path_id/title/dataset/model to the callback.
    metadata: Dict[str, Any] = Field(default_factory=dict)


def validate_request(request: EvaluationRequest, settings: Settings) -> List[str]:
    """Return every problem with the request. Empty list means it is valid.

    Everything checkable is checked here, at submit time, so a malformed request
    fails in milliseconds instead of after a multi-hour GPU run.
    """
    errors: List[str] = []

    if not NAME_PATTERN.match(request.name):
        errors.append(f"'name' must be a lowercase snake_case identifier (got {request.name!r})")

    known = set(catalog.all_ids())
    unknown = [m for m in request.metrics if m not in known]
    if unknown:
        errors.append(f"Unknown metric ids: {', '.join(unknown)}")
        return errors  # later checks would raise on these ids

    if not request.llm_summaries:
        errors.append("'llm_summaries' must not be empty")
        return errors

    document_ids = set(request.llm_summaries)
    required = catalog.required_references(request.metrics)

    if "gold" in required:
        if not request.gold_summaries:
            errors.append("'gold_summaries' is required by the selected metrics")
        elif set(request.gold_summaries) != document_ids:
            errors.append("'gold_summaries' document ids do not match 'llm_summaries'")

    if "posts" in required:
        if not request.posts:
            errors.append("'posts' is required by the selected metrics")
        elif set(request.posts) != document_ids:
            errors.append("'posts' document ids do not match 'llm_summaries'")

    for environment in catalog.environments_used(request.metrics):
        if not settings.is_available(environment):
            unavailable = [
                m for m in request.metrics
                if catalog.environments_used([m]) == {environment}
            ]
            errors.append(
                f"Metrics {unavailable} need the '{environment}' environment, "
                f"which is not configured on this server"
            )

    return errors
