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
python -m spacy download en_core_web_sm
```

The last line installs the spaCy English model that the `style_similarity` metric needs;
pip does not install it with the package.

For local testing, create a `.env` file in the repo root by running this from the repo
root (see [`.env.example`](.env.example) for more details). It overwrites any existing
`.env`:

```bash
cat > .env <<'EOF'
METRIC_API_TOKEN=test_token
METRIC_JOB_ROOT=./job_data
METRIC_DB_PATH=./jobs.db
METRIC_JOB_TIMEOUT_SECONDS=21600
METRIC_RETENTION_HOURS=168
EOF
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
  -d @evaluation_request.json
```

`GET /evaluations/{job_id}`, with the `job_id` returned by `POST /evaluations`:

```bash
curl -s http://localhost:8000/evaluations/<job_id> -H "X-Api-Token: test_token" | jq
```

FastAPI also serves interactive docs at http://localhost:8000/docs, where you can try
every endpoint in the browser: expand one, click **Try it out**, and enter the token in the
`x-api-token` field.

### Example: evaluating the example summaries

[`evaluation_bundles/example_data`](evaluation_bundles/example_data) holds three synthetic
documents in the shape the API expects (see its
[README](evaluation_bundles/example_data/README.md)): the source text (`posts.json`), a
reference summary (`gold_summaries.json`) and the summary to evaluate
(`llm_summaries.json`). In practice the last one comes from a model, e.g.
`models/summary_generation_with_autoclass.py`.

Build a request from them that asks for the metrics of the
[`social_media_example`](evaluation_bundles/bundle_specs/social_media_example.yaml) bundle:

```bash
jq -n \
  --slurpfile llm evaluation_bundles/example_data/llm_summaries.json \
  --slurpfile gold evaluation_bundles/example_data/gold_summaries.json \
  --slurpfile posts evaluation_bundles/example_data/posts.json \
  '{name: "social_media_example",
    metrics: ["mhic", "intra_nli", "fc_expert", "fc_document", "style_similarity", "bertscore"],
    llm_summaries: $llm[0], gold_summaries: $gold[0], posts: $posts[0]}' \
  > evaluation_request.json
```

With the server running (see above), submit it:

```bash
curl -s -X POST http://localhost:8000/evaluations \
  -H "X-Api-Token: test_token" -H "Content-Type: application/json" \
  -d @evaluation_request.json
```

The response has the form `{"job_id": "<job_id>", "status": "queued"}`, where
`<job_id>` is the id of the new job, a UUID such as `1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed`.
A malformed request, such as an unknown metric id or a missing reference, is rejected
straight away with a `422` that lists every problem instead.

Store the `job_id` from your response in a variable, so the following commands can use
it. Replace `<job_id>` with your id, without quotes or angle brackets:

```bash
JOB_ID=<job_id>
```

Then run this every so often to follow the job's progress. Its `status` moves from
`queued` through `running` to `succeeded` or `failed`, and `error` explains a failure:

```bash
curl -s http://localhost:8000/evaluations/$JOB_ID -H "X-Api-Token: test_token" | jq '.status, .error'
```

If this prints `null` twice, the API returned an error instead of the job. Run the
command without `| jq ...` to see it, e.g. `{"detail":"Unknown job id"}` when `JOB_ID`
is wrong or not set in the current terminal.

Once the job has succeeded, its results are in `job_data/<job_id>/results.json`.

## Models

Deep learning models, e.g. for summary generation, are gathered in the directory <a href="models">models</a>.
