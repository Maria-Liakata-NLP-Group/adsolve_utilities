#!/bin/sh
# Container entrypoint: warm the model cache from S3, run the evaluation, push
# any newly downloaded models back.
#
# All arguments are passed straight through to run_job, so the container is
# invoked exactly like the CLI:
#
#   docker run adsolve-eval:cpu --metrics fc_expert,mhic --name my_run \
#     --llm-summaries /data/in/llm.json --output /data/out/results.json
#
# Set MODEL_CACHE_S3_URI to enable the sync; leave it unset and the job just uses
# whatever is already in HF_HOME.
set -u

CACHE_DIR="${HF_HOME:-/models}"
S3_URI="${MODEL_CACHE_S3_URI:-}"

if [ -n "$S3_URI" ]; then
    echo "Warming model cache from $S3_URI"
    mkdir -p "$CACHE_DIR"
    # Non-fatal: a cold or missing bucket just means models download from
    # HuggingFace instead, which is slower but still correct.
    aws s3 sync "$S3_URI" "$CACHE_DIR" --only-show-errors || \
        echo "warning: could not warm cache from S3; models will download on demand" >&2
fi

python -m evaluation_bundles.run_job "$@"
STATUS=$?

# Capture the status BEFORE syncing: otherwise the sync's exit code becomes the
# pod's, and a failed evaluation would report success to Kubernetes.
if [ -n "$S3_URI" ]; then
    echo "Writing model cache back to $S3_URI"
    # Runs even when the evaluation failed -- models downloaded before the
    # failure are still worth keeping. `sync` uploads only what changed, so a
    # warm cache transfers nothing.
    aws s3 sync "$CACHE_DIR" "$S3_URI" --only-show-errors || \
        echo "warning: could not write model cache back to S3" >&2
fi

exit $STATUS
