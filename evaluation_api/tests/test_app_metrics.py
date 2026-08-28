import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from fastapi.testclient import TestClient

from evaluation_api.app import create_app
from evaluation_api.config import load_settings


def _client(tmp_path, **env):
    env.setdefault("METRIC_JOB_ROOT", str(tmp_path / "jobs"))
    env.setdefault("METRIC_DB_PATH", str(tmp_path / "jobs.db"))
    settings = load_settings(env)
    return TestClient(create_app(settings, start_worker=False))


def test_health_needs_no_token(tmp_path):
    response = _client(tmp_path).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_rejects_a_missing_token(tmp_path):
    response = _client(tmp_path, METRIC_API_TOKEN="secret").get("/metrics")
    assert response.status_code == 422  # header is required


def test_metrics_rejects_a_wrong_token(tmp_path):
    client = _client(tmp_path, METRIC_API_TOKEN="secret")
    response = client.get("/metrics", headers={"X-Api-Token": "wrong"})
    assert response.status_code == 401


def test_metrics_returns_503_when_no_token_is_configured(tmp_path):
    """A misconfigured deployment must fail loudly, never serve unauthenticated."""
    client = _client(tmp_path)
    response = client.get("/metrics", headers={"X-Api-Token": "anything"})
    assert response.status_code == 503


def test_metrics_lists_the_catalog(tmp_path):
    client = _client(tmp_path, METRIC_API_TOKEN="secret")
    response = client.get("/metrics", headers={"X-Api-Token": "secret"})
    assert response.status_code == 200
    rows = {row["id"]: row for row in response.json()}
    assert rows["fc_document"]["requires"] == "posts"
    assert rows["intra_nli"]["requires"] is None
    assert rows["mhic"]["available"] is True


def test_unconfigured_environment_is_reported_unavailable(tmp_path):
    client = _client(tmp_path, METRIC_API_TOKEN="secret")
    rows = {r["id"]: r for r in client.get("/metrics", headers={"X-Api-Token": "secret"}).json()}
    assert rows["green_score"]["available"] is False


def test_configured_environment_is_reported_available(tmp_path):
    client = _client(tmp_path, METRIC_API_TOKEN="secret",
                     METRIC_ENV_GREENSCORE_URL="http://green:8000")
    rows = {r["id"]: r for r in client.get("/metrics", headers={"X-Api-Token": "secret"}).json()}
    assert rows["green_score"]["available"] is True
