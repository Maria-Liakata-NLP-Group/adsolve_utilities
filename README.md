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

### Setup

Set up a virtual environment, e.g. with `venv`, and install the package with the API
extras from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[api]"
```

For local testing you can set up a `.env` file in the repo root that looks like this (see [`.env.example`](.env.example) for more details):

```bash
METRIC_API_TOKEN=test_token
METRIC_JOB_ROOT=./job_data
METRIC_DB_PATH=./jobs.db
METRIC_JOB_TIMEOUT_SECONDS=21600
METRIC_RETENTION_HOURS=168
```

Settings are read from `.env` in the repo root, so nothing needs exporting. A real
environment variable overrides the file, which is handy for a one-off change:
`METRIC_JOB_TIMEOUT_SECONDS=60 uvicorn ...`. `.env` holds the API token and is
gitignored; `.env.example` documents every setting. `METRIC_API_TOKEN` is required —
if it is unset every authenticated route returns 503 rather than serving openly.

### Run API

```bash
uvicorn evaluation_api.app:app --factory --port 8000
```

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, no auth |
| `GET /metrics` | Available metric ids, labels, and which reference data each needs |
| `POST /evaluations` | Submit a run, returns `202 {job_id}` |
| `GET /evaluations/{job_id}` | Status, then results |

All but `/health` require the `X-Api-Token` header. A browser can't send that header, so
call the endpoints from the terminal. With the local `.env` above, the token is `test_token`.

`GET /health`:

```bash
curl -s http://localhost:8000/health
```

`GET /metrics`:

```bash
curl -s http://localhost:8000/metrics -H "X-Api-Token: test_token" | jq
```

`POST /evaluations` (see the example below for the request body):

```bash
curl -s -X POST http://localhost:8000/evaluations \
  -H "X-Api-Token: test_token" -H "Content-Type: application/json" \
  -d @models/example_data/social_media_evaluation_request.json
```

`GET /evaluations/{job_id}`, with the `job_id` returned by `POST /evaluations`:

```bash
curl -s http://localhost:8000/evaluations/<job_id> -H "X-Api-Token: test_token" | jq
```

FastAPI also serves interactive docs at http://localhost:8000/docs, where you can try
every endpoint in the browser: expand one, click **Try it out**, and enter the token in the
`x-api-token` field.

### Example: evaluating a social media timeline summary

[`models/example_data/social_media_evaluation_request.json`](models/example_data/social_media_evaluation_request.json)
is a ready-made request built from the example timeline in
[`clpsych_example.json`](models/example_data/clpsych_example.json). It asks for the metrics
of the [`social_media_example`](evaluation_bundles/bundle_specs/social_media_example.yaml)
bundle. It holds the timeline's posts (`posts`), the expert timeline summary
(`gold_summaries`) and the summary to evaluate (`llm_summaries`). That last one is a
hand-written placeholder; in practice it comes from a model, e.g.
`models/summary_generation_with_autoclass.py`.

With the server running (see above), submit it:

```bash
TOKEN=test_token

curl -s -X POST http://localhost:8000/evaluations \
  -H "X-Api-Token: $TOKEN" -H "Content-Type: application/json" \
  -d @models/example_data/social_media_evaluation_request.json
# {"job_id":"3f2c...","status":"queued"}
```

A malformed request, such as an unknown metric id or a missing reference, is rejected
straight away with a `422` that lists every problem. Otherwise, poll the job until its
`status` moves from `queued` through `running` to `succeeded` or `failed`:

```bash
curl -s http://localhost:8000/evaluations/3f2c... -H "X-Api-Token: $TOKEN" | jq '.status, .error'
```

Once it has succeeded, `results` holds `document_ids` plus one entry per metric id with a
`mean` and per-document `document_level` scores:

```bash
curl -s http://localhost:8000/evaluations/3f2c... -H "X-Api-Token: $TOKEN" \
  | jq '.results | del(.document_ids) | map_values(.mean)'
```

The first run downloads several Hugging Face models and is slow without a GPU. For a
quick check that the server and the subprocess work, submit the same request with only
`flesch_kincaid_grade_level`, which needs no model download:

```bash
jq '.metrics = ["flesch_kincaid_grade_level"]' models/example_data/social_media_evaluation_request.json \
  | curl -s -X POST http://localhost:8000/evaluations \
      -H "X-Api-Token: $TOKEN" -H "Content-Type: application/json" \
      -d @-
```

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
