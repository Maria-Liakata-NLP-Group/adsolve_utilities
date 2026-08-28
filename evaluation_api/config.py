"""Settings loaded from the process environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional

ENVIRONMENT_URL_PREFIX = "METRIC_ENV_"
ENVIRONMENT_URL_SUFFIX = "_URL"


@dataclass(frozen=True)
class Settings:
    api_token: Optional[str]
    job_root: Path
    db_path: Path
    job_timeout_seconds: int
    retention_hours: int
    environment_urls: Dict[str, str] = field(default_factory=dict)

    def is_available(self, environment: str) -> bool:
        """The standard environment runs in-process; others need a configured URL."""
        if environment == "standard":
            return True
        return environment in self.environment_urls


def _environment_urls(env: Mapping[str, str]) -> Dict[str, str]:
    """Discover METRIC_ENV_<NAME>_URL variables, e.g. METRIC_ENV_GREENSCORE_URL."""
    urls = {}
    for key, value in env.items():
        if key.startswith(ENVIRONMENT_URL_PREFIX) and key.endswith(ENVIRONMENT_URL_SUFFIX):
            name = key[len(ENVIRONMENT_URL_PREFIX):-len(ENVIRONMENT_URL_SUFFIX)].lower()
            if name and value:
                urls[name] = value
    return urls


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    env = os.environ if env is None else env
    return Settings(
        api_token=env.get("METRIC_API_TOKEN") or None,
        job_root=Path(env.get("METRIC_JOB_ROOT", "./job_data")),
        db_path=Path(env.get("METRIC_DB_PATH", "./jobs.db")),
        job_timeout_seconds=int(env.get("METRIC_JOB_TIMEOUT_SECONDS", "21600")),
        retention_hours=int(env.get("METRIC_RETENTION_HOURS", "168")),
        environment_urls=_environment_urls(env),
    )
