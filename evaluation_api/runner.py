"""Renders a bundle script and runs it as a subprocess.

The metrics run in a child process, never in the server: that is what releases
GPU memory when the run finishes, and it keeps a metric crash from taking down
the API.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List

BUNDLES_DIR = Path(__file__).resolve().parent.parent / "evaluation_bundles"
sys.path.insert(0, str(BUNDLES_DIR))

from generate_bundle import render_bundle, validate_spec  # noqa: E402

from . import catalog  # noqa: E402
from .config import Settings  # noqa: E402

STDERR_TAIL_BYTES = 4096


class BundleError(Exception):
    """The evaluation subprocess did not produce results."""


class UnsupportedEnvironmentError(BundleError):
    """A metric needs an isolated environment that is not implemented yet."""


def run_local_bundle(name: str, metric_ids: List[str], job_dir: Path,
                     timeout_seconds: int,
                     run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict:
    """Render the bundle for these metrics and execute it against the job's data."""
    spec = catalog.build_spec(name, metric_ids)
    validate_spec(spec)

    script_path = job_dir / f"{name}_evaluation.py"
    script_path.write_text(render_bundle(spec, f"<api:{name}>"))

    output_path = job_dir / "results.json"
    argv = [
        sys.executable, str(script_path),
        "--llm_summaries", str(job_dir / "llm_summaries.json"),
        "--gold_summaries", str(job_dir / "gold_summaries.json"),
        "--output_file", str(output_path),
    ]
    if "posts" in catalog.required_references(metric_ids):
        argv += ["--posts", str(job_dir / "posts.json")]

    # The generated script does sys.path.append(<its own dir>) then imports
    # `metrics.*`. With the script in a per-job directory that append is useless,
    # so the import path has to come from the environment instead.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BUNDLES_DIR)

    try:
        completed = run(argv, env=env, capture_output=True, text=True,
                        timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        raise BundleError(f"Evaluation timed out after {timeout_seconds}s")

    if completed.returncode != 0:
        raise BundleError((completed.stderr or "")[-STDERR_TAIL_BYTES:] or
                          f"Bundle exited with code {completed.returncode}")

    if not output_path.exists():
        raise BundleError("Bundle exited cleanly but wrote no results file")

    return json.loads(output_path.read_text())


def run_remote_environment(environment: str, metric_ids: List[str],
                           job_dir: Path, settings: Settings) -> dict:
    """Run metrics that need their own container. Not implemented yet.

    The seam exists so adding GreenScorer later is writing this function plus a
    Dockerfile, rather than restructuring the worker.
    """
    raise UnsupportedEnvironmentError(
        f"The '{environment}' environment is not implemented yet "
        f"(requested by: {', '.join(metric_ids)})"
    )


def _group_by_environment(metric_ids: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for metric_id in metric_ids:
        environment = catalog.environments_used([metric_id]).pop()
        groups.setdefault(environment, []).append(metric_id)
    return groups


def execute(name: str, metric_ids: List[str], job_dir: Path, settings: Settings,
            run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict:
    """Run every phase of the job, one environment at a time.

    Each phase exits fully before the next begins, so the GPU is released
    between phases rather than shared.
    """
    results: dict = {}
    for environment, group in _group_by_environment(metric_ids).items():
        if environment == "standard":
            phase = run_local_bundle(name, group, job_dir, settings.job_timeout_seconds, run=run)
        else:
            phase = run_remote_environment(environment, group, job_dir, settings)
        results.update(phase)
    return results
