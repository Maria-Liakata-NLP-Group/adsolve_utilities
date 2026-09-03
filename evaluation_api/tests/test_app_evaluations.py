import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient

from evaluation_api.app import create_app
from evaluation_api.config import load_settings

TOKEN = {"X-Api-Token": "secret"}


def _client(tmp_path, **env):
    env.setdefault("METRIC_API_TOKEN", "secret")
    env.setdefault("METRIC_JOB_ROOT", str(tmp_path / "jobs"))
    env.setdefault("METRIC_DB_PATH", str(tmp_path / "jobs.db"))
    return TestClient(create_app(load_settings(env), start_worker=False))


def _body(**overrides) -> dict:
    body = {
        "name": "my_run",
        "metrics": ["intra_nli"],
        "llm_summaries": {"doc1": "a summary"},
        "metadata": {"path_id": "affiniti_therapy", "title": "Run 1"},
    }
    body.update(overrides)
    return body


def test_submit_returns_202_and_a_queued_job(tmp_path):
    response = _client(tmp_path).post("/evaluations", json=_body(), headers=TOKEN)
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["job_id"]


def test_submit_writes_the_input_files_into_the_job_directory(tmp_path):
    client = _client(tmp_path)
    job_id = client.post("/evaluations", json=_body(), headers=TOKEN).json()["job_id"]
    job_dir = tmp_path / "jobs" / job_id
    assert (job_dir / "llm_summaries.json").exists()
    assert (job_dir / "gold_summaries.json").exists()


def test_status_reports_the_queued_job(tmp_path):
    client = _client(tmp_path)
    job_id = client.post("/evaluations", json=_body(), headers=TOKEN).json()["job_id"]
    response = client.get(f"/evaluations/{job_id}", headers=TOKEN)
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert response.json()["results"] is None


def test_status_returns_404_for_an_unknown_job(tmp_path):
    assert _client(tmp_path).get("/evaluations/nope", headers=TOKEN).status_code == 404


def test_submit_rejects_an_unknown_metric_id(tmp_path):
    response = _client(tmp_path).post(
        "/evaluations", json=_body(metrics=["not_a_metric"]), headers=TOKEN)
    assert response.status_code == 422
    assert "not_a_metric" in str(response.json()["detail"])


def test_submit_rejects_a_gold_metric_without_gold_summaries(tmp_path):
    response = _client(tmp_path).post(
        "/evaluations", json=_body(metrics=["fc_expert"]), headers=TOKEN)
    assert response.status_code == 422
    assert "gold_summaries" in str(response.json()["detail"])


def test_submit_rejects_a_posts_metric_without_posts(tmp_path):
    response = _client(tmp_path).post(
        "/evaluations", json=_body(metrics=["mhic"]), headers=TOKEN)
    assert response.status_code == 422
    assert "posts" in str(response.json()["detail"])


def test_submit_rejects_mismatched_document_ids(tmp_path):
    response = _client(tmp_path).post("/evaluations", json=_body(
        metrics=["fc_expert"],
        llm_summaries={"doc1": "a"},
        gold_summaries={"doc2": "b"},
    ), headers=TOKEN)
    assert response.status_code == 422
    assert "document ids" in str(response.json()["detail"]).lower()


def test_submit_rejects_empty_llm_summaries(tmp_path):
    response = _client(tmp_path).post(
        "/evaluations", json=_body(llm_summaries={}), headers=TOKEN)
    assert response.status_code == 422


def test_submit_rejects_a_name_that_is_not_snake_case(tmp_path):
    response = _client(tmp_path).post(
        "/evaluations", json=_body(name="My Run!"), headers=TOKEN)
    assert response.status_code == 422


def test_submit_rejects_a_metric_whose_environment_is_unavailable(tmp_path):
    """Fail at submit, not forty minutes into a run."""
    response = _client(tmp_path).post("/evaluations", json=_body(
        metrics=["green_score"], gold_summaries={"doc1": "b"}), headers=TOKEN)
    assert response.status_code == 422
    assert "green_score" in str(response.json()["detail"])


def test_submit_accepts_a_run_whose_environment_is_configured(tmp_path):
    client = _client(tmp_path, METRIC_ENV_GREENSCORE_URL="http://green:8000")
    response = client.post("/evaluations", json=_body(
        metrics=["green_score"], gold_summaries={"doc1": "b"}), headers=TOKEN)
    assert response.status_code == 202


def test_submit_requires_a_token(tmp_path):
    assert _client(tmp_path).post("/evaluations", json=_body()).status_code == 422


def test_a_purged_job_returns_200_with_null_results(tmp_path):
    """Purged and never-existed must not look the same, or a retry re-runs the GPU."""
    client = _client(tmp_path)
    job_id = client.post("/evaluations", json=_body(), headers=TOKEN).json()["job_id"]
    store = client.app.state.store
    store.claim_next()
    store.mark_succeeded(job_id, {"intra_nli": {"mean": 1.0}})
    store.mark_delivered(job_id)
    store.purge(job_id)

    response = client.get(f"/evaluations/{job_id}", headers=TOKEN)
    assert response.status_code == 200
    assert response.json()["results"] is None
    assert response.json()["purged_at"] is not None
    assert client.get("/evaluations/unknown-id", headers=TOKEN).status_code == 404
