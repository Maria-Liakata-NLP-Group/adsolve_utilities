"""Run one evaluation to completion, then exit. The container entrypoint.

This is the batch counterpart to the HTTP API: no server, no queue, no job
store. A Kubernetes Job starts it, it renders a bundle script for the requested
metrics, runs that script against mounted input files, writes results.json to a
mounted volume, and exits with a status the scheduler can read.

    python -m evaluation_bundles.run_job \
        --metrics fc_expert,mhic --name affiniti_run \
        --llm-summaries /data/in/llm_summaries.json \
        --gold-summaries /data/in/gold_summaries.json \
        --posts /data/in/posts.json \
        --output /data/out/results.json

Metrics come either from --metrics (catalog ids) or --spec (a bundle_specs YAML).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

BUNDLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLES_DIR))

from generate_bundle import SpecError, load_spec, render_bundle, validate_spec  # noqa: E402
from metric_registry import (  # noqa: E402
    UnknownMetricError,
    build_spec,
    required_references,
)

EXIT_OK = 0
EXIT_BAD_REQUEST = 2
EXIT_BUNDLE_FAILED = 3


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_job",
        description="Render an evaluation bundle and run it against input data.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metrics", type=str,
                        help="Comma-separated catalog metric ids, e.g. fc_expert,mhic")
    source.add_argument("--spec", type=str,
                        help="Path to a bundle_specs YAML file.")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name; becomes the script filename. Defaults to the "
                             "spec's name, or 'evaluation' when using --metrics.")
    parser.add_argument("--llm-summaries", type=str, required=True,
                        help="JSON: {document_id: summary}")
    parser.add_argument("--gold-summaries", type=str, default=None,
                        help="JSON: {document_id: reference summary}. Required only if "
                             "a selected metric uses a gold reference.")
    parser.add_argument("--posts", type=str, default=None,
                        help="JSON: {document_id: [source text, ...]}. Required only if "
                             "a selected metric uses a posts reference.")
    parser.add_argument("--output", type=str, required=True,
                        help="Where to write results.json.")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Seconds before the bundle subprocess is killed. "
                             "Default: no timeout, so the scheduler owns the deadline.")
    return parser.parse_args(argv)


def _resolve_spec(args: argparse.Namespace) -> dict:
    """Build the bundle spec from either --spec or --metrics."""
    if args.spec:
        spec = load_spec(args.spec)
        if args.name:
            spec["name"] = args.name
        return spec

    metric_ids = [m.strip() for m in args.metrics.split(",") if m.strip()]
    return build_spec(args.name or "evaluation", metric_ids)


def _spec_references(spec: dict) -> set:
    """Which reference data this spec needs, for hand-written specs too."""
    return {entry.get("reference") for entry in spec["metrics"]} & {"gold", "posts"}


def main(argv: Optional[List[str]] = None,
         run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> int:
    args = _parse_args(argv)

    try:
        spec = _resolve_spec(args)
        validate_spec(spec)
    except UnknownMetricError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_REQUEST
    except (SpecError, FileNotFoundError) as exc:
        print(f"error: invalid spec: {exc}", file=sys.stderr)
        return EXIT_BAD_REQUEST

    references = _spec_references(spec)
    if "posts" in references and not args.posts:
        print("error: the selected metrics need --posts", file=sys.stderr)
        return EXIT_BAD_REQUEST
    if "gold" in references and not args.gold_summaries:
        print("error: the selected metrics need --gold-summaries", file=sys.stderr)
        return EXIT_BAD_REQUEST

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Render into a temp dir rather than evaluation_bundles/: keeps concurrent
    # jobs from colliding and avoids generate_bundle's overwrite guard.
    work_dir = Path(tempfile.mkdtemp(prefix="eval_job_"))
    try:
        script_path = work_dir / f"{spec['name']}_evaluation.py"
        script_path.write_text(render_bundle(spec, args.spec or "<catalog>"))

        gold_path = args.gold_summaries or _synthesise_gold(args.llm_summaries, work_dir)

        argv_child = [
            sys.executable, str(script_path),
            "--llm_summaries", args.llm_summaries,
            "--gold_summaries", gold_path,
            "--output_file", str(output_path),
        ]
        if "posts" in references:
            argv_child += ["--posts", args.posts]

        # The generated script does sys.path.append(<its own dir>) then imports
        # `metrics.*`. The script lives in a temp dir, so that append is useless
        # and the import path has to come from the environment instead.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(BUNDLES_DIR)

        print(f"Running {spec['name']} with metrics: "
              f"{', '.join(e['id'] for e in spec['metrics'])}", flush=True)

        try:
            completed = run(argv_child, env=env, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"error: evaluation timed out after {args.timeout}s", file=sys.stderr)
            return EXIT_BUNDLE_FAILED

        if completed.returncode != 0:
            stderr = getattr(completed, "stderr", None) or ""
            print(stderr if isinstance(stderr, str) else stderr.decode(errors="replace"),
                  file=sys.stderr)
            print(f"error: bundle exited with code {completed.returncode}", file=sys.stderr)
            return EXIT_BUNDLE_FAILED

        if not output_path.exists():
            print("error: bundle exited cleanly but wrote no results", file=sys.stderr)
            return EXIT_BUNDLE_FAILED

        print(f"Wrote results to {output_path}", flush=True)
        return EXIT_OK
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _synthesise_gold(llm_summaries_path: str, work_dir: Path) -> str:
    """Write empty gold summaries when no metric reads them.

    render_bundle emits `gold_summary = gold_summaries[document_id]`
    unconditionally and the generated CLI requires the flag. Validation above
    guarantees no gold-referencing metric is present, so these are provably
    unread -- this just saves the caller hand-making a dummy file.
    """
    document_ids = json.loads(Path(llm_summaries_path).read_text())
    gold_path = work_dir / "gold_summaries.json"
    gold_path.write_text(json.dumps({doc_id: "" for doc_id in document_ids}))
    return str(gold_path)


if __name__ == "__main__":
    raise SystemExit(main())
