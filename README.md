<!-- @format -->

# AdSoLve Utilities

This repository gathers code that has been developed to support the AdSoLve project, like evaluation tools and generation models.

## Evaluation Bundles

Resources to evaluate LLMs for different use cases are gathered in the directory <a href="evaluation_bundles">evaluation_bundles</a>.

### Generating a new evaluation bundle

Instead of hand-writing a new bundle script, describe your metric selection in a YAML spec under
`evaluation_bundles/bundle_specs/` and generate it:

```bash
python evaluation_bundles/generate_bundle.py --spec evaluation_bundles/bundle_specs/my_use_case.yaml
```

This writes `evaluation_bundles/my_use_case_evaluation.py`, structured like the existing bundles.
See `evaluation_bundles/metric_registry.py` for the list of available metrics and
`evaluation_bundles/bundle_specs/social_media_example.yaml` for an example spec.

## Evaluation API

Runs evaluations on request instead of by hand. The platform submits model outputs and a
list of metrics; the API generates a bundle script, runs it as a subprocess, and returns
results by polling and (optionally) by callback.

```bash
pip install -e ".[api]"
export METRIC_API_TOKEN=...            # required; unset means 503, never open
export METRIC_JOB_ROOT=./job_data
export METRIC_DB_PATH=./jobs.db
uvicorn evaluation_api.app:app --factory --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, no auth |
| `GET /metrics` | Available metric ids, labels, and which reference data each needs |
| `POST /evaluations` | Submit a run, returns `202 {job_id}` |
| `GET /evaluations/{job_id}` | Status, then results |

All but `/health` require the `X-Api-Token` header.

Metric ids come from `METRIC_CATALOG` in `evaluation_bundles/metric_registry.py` — this
repo is the source of truth for them. Adding a metric means one `METRIC_REGISTRY` entry
(how to run it) plus one or more `METRIC_CATALOG` entries (what callers may ask for);
`test_metric_catalog.py` fails if the two drift apart.

Raw data is written to the job directory, never to SQLite, and deleted when the job
finishes. For `sensitive: true` runs, per-sentence `detail` is stripped from results and
no document text is transmitted or logged.

See `docs/superpowers/specs/2026-08-28-metric-calculation-api-design.md` for the design.

## Models

Deep learning models, e.g. for summary generation, are gathered in the directory <a href="models">models</a>.
