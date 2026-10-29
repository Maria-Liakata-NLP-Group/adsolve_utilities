import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx

from evaluation_api import callback
from evaluation_api.store import Job

RESULTS = {"mhic": {"document_level": [0.5], "mean": 0.5}, "document_ids": ["doc1"]}
INPUTS = {"llm_summaries": {"doc1": "a"}, "gold_summaries": {"doc1": "b"},
          "posts": {"doc1": ["p"]}}


def _job(sensitive: bool = False) -> Job:
    return Job(
        id="job-1", status="succeeded", name="my_run", metric_ids=["mhic"],
        sensitive=sensitive, job_dir="/tmp/job",
        metadata={"path_id": "affiniti_therapy", "title": "Run 1",
                  "dataset": {"name": "d", "sensitive": sensitive},
                  "model": {"name": "m"}},
        callback_url="http://platform/api/runs/ingest", callback_token="t",
        callback_header="X-Admin-Token", results=None, error=None,
        callback_status="pending", submitted_at="t0", started_at="t1",
        finished_at="t2", delivered_at=None, purged_at=None,
    )


def test_metadata_is_echoed_verbatim():
    body = callback.compose_body(_job(), RESULTS, INPUTS)
    assert body["path_id"] == "affiniti_therapy"
    assert body["title"] == "Run 1"
    assert body["model"] == {"name": "m"}


def test_results_are_included():
    assert callback.compose_body(_job(), RESULTS, INPUTS)["results"] == RESULTS


def test_non_sensitive_body_carries_the_data_under_the_platform_key_names():
    body = callback.compose_body(_job(), RESULTS, INPUTS)
    assert body["llm_summaries"] == {"doc1": "a"}
    assert body["gold_summaries"] == {"doc1": "b"}
    assert body["inputs"] == {"doc1": ["p"]}  # 'posts' is 'inputs' to the platform


def test_sensitive_body_omits_every_text_field():
    body = callback.compose_body(_job(sensitive=True), RESULTS, None)
    assert "llm_summaries" not in body
    assert "gold_summaries" not in body
    assert "inputs" not in body


def test_deliver_returns_true_on_success():
    transport = httpx.MockTransport(lambda request: httpx.Response(201))
    with httpx.Client(transport=transport) as client:
        assert callback.deliver("http://platform/ingest", "X-Admin-Token", "t",
                                {"a": 1}, client) is True


def test_deliver_sends_the_token_in_the_configured_header():
    seen = {}

    def handler(request):
        seen["token"] = request.headers.get("x-admin-token")
        return httpx.Response(201)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        callback.deliver("http://platform/ingest", "X-Admin-Token", "secret", {}, client)
    assert seen["token"] == "secret"


def test_server_errors_are_retried_three_times():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert callback.deliver("http://p/i", "X-Admin-Token", "t", {}, client,
                                sleep=lambda _: None) is False
    assert len(attempts) == 3


def test_client_errors_are_not_retried():
    """A 422 means the body is wrong; retrying cannot fix it."""
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(422)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert callback.deliver("http://p/i", "X-Admin-Token", "t", {}, client,
                                sleep=lambda _: None) is False
    assert len(attempts) == 1


def test_connection_errors_are_retried():
    attempts = []

    def handler(request):
        attempts.append(1)
        raise httpx.ConnectError("refused")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert callback.deliver("http://p/i", "X-Admin-Token", "t", {}, client,
                                sleep=lambda _: None) is False
    assert len(attempts) == 3
