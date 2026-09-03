"""Composes and delivers the results callback.

The body is shaped so it is a valid platform IngestRequest as-is. Two couplings
to the platform live here and nowhere else: `posts` is sent under the key
`inputs`, and the token goes in a header the request names (X-Admin-Token by
default).
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import httpx

from .store import Job

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0


def compose_body(job: Job, results: dict, inputs: Optional[dict]) -> dict:
    """metadata + results, plus the run's data when the dataset is not sensitive.

    Omitting the three data keys for a sensitive run still yields a valid
    IngestRequest -- the platform declares them with default_factory=dict -- so
    the privacy rule needs no separate code path on either side.
    """
    body = dict(job.metadata)
    body["results"] = results
    if inputs is not None:
        body["llm_summaries"] = inputs["llm_summaries"]
        body["gold_summaries"] = inputs["gold_summaries"]
        body["inputs"] = inputs["posts"]  # the platform's name for source documents
    return body


def deliver(url: str, header_name: str, token: str, body: dict,
            client: httpx.Client, max_attempts: int = MAX_ATTEMPTS,
            sleep: Callable[[float], None] = time.sleep) -> bool:
    """POST the body, retrying transient failures. True if it was accepted."""
    for attempt in range(max_attempts):
        try:
            response = client.post(url, json=body, headers={header_name: token},
                                   timeout=30.0)
        except httpx.HTTPError:
            pass  # connection-level failure: worth retrying
        else:
            if response.is_success:
                return True
            if response.status_code < 500:
                return False  # the body is wrong; retrying will not fix it
        if attempt < max_attempts - 1:
            sleep(BACKOFF_SECONDS * (2 ** attempt))
    return False
