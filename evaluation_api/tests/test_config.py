import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation_api.config import load_settings


def test_defaults_apply_when_env_is_empty():
    settings = load_settings({})
    assert settings.api_token is None
    assert settings.job_timeout_seconds == 21600
    assert settings.retention_hours == 168
    assert settings.environment_urls == {}


def test_values_are_read_from_env():
    settings = load_settings({
        "METRIC_API_TOKEN": "secret",
        "METRIC_JOB_ROOT": "/var/jobs",
        "METRIC_DB_PATH": "/var/jobs.db",
        "METRIC_JOB_TIMEOUT_SECONDS": "60",
        "METRIC_RETENTION_HOURS": "24",
    })
    assert settings.api_token == "secret"
    assert settings.job_root == Path("/var/jobs")
    assert settings.db_path == Path("/var/jobs.db")
    assert settings.job_timeout_seconds == 60
    assert settings.retention_hours == 24


def test_environment_urls_are_discovered_by_naming_convention():
    settings = load_settings({"METRIC_ENV_GREENSCORE_URL": "http://green:8000"})
    assert settings.environment_urls == {"greenscore": "http://green:8000"}


def test_standard_environment_is_always_available():
    assert load_settings({}).is_available("standard") is True


def test_other_environments_are_available_only_when_configured():
    assert load_settings({}).is_available("greenscore") is False
    configured = load_settings({"METRIC_ENV_GREENSCORE_URL": "http://green:8000"})
    assert configured.is_available("greenscore") is True


def _write_env_file(tmp_path, body: str) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(body)
    return env_file


def test_settings_are_read_from_a_dotenv_file(tmp_path):
    """Running the server should not require exporting anything by hand."""
    env_file = _write_env_file(tmp_path, "\n".join([
        "# the API token the platform sends",
        "METRIC_API_TOKEN=from-dotenv",
        "METRIC_JOB_ROOT=/var/dotenv_jobs",
        "METRIC_RETENTION_HOURS=24",
    ]))
    settings = load_settings(dotenv_path=env_file)
    assert settings.api_token == "from-dotenv"
    assert settings.job_root == Path("/var/dotenv_jobs")
    assert settings.retention_hours == 24


def test_a_real_environment_variable_beats_the_dotenv_file(tmp_path, monkeypatch):
    """So a single value can be overridden for one run without editing .env."""
    env_file = _write_env_file(tmp_path, "METRIC_API_TOKEN=from-dotenv\n")
    monkeypatch.setenv("METRIC_API_TOKEN", "from-real-env")
    assert load_settings(dotenv_path=env_file).api_token == "from-real-env"


def test_an_explicit_env_mapping_ignores_the_dotenv_file(tmp_path):
    """Tests pass an explicit mapping; .env must never leak into them."""
    env_file = _write_env_file(tmp_path, "METRIC_API_TOKEN=from-dotenv\n")
    settings = load_settings({"METRIC_API_TOKEN": "explicit"}, dotenv_path=env_file)
    assert settings.api_token == "explicit"
