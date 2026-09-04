"""
Kubeflow driver — AdSoLve evaluation bundle run.

Submits a PyTorchJob that renders an evaluation bundle for the requested metrics
and runs it against input JSON on the shared filesystem. Results land in
OUT_DIR/results.json.

Prereqs on the shared FS, before submitting:
  1. This repo copied to REPO_DIR (the image supplies the environment; the code
     is read from the shared FS, so edits do not need an image rebuild).
  2. Input JSON in IN_DIR: llm_summaries.json, plus gold_summaries.json and/or
     posts.json if the chosen metrics need them (see METRICS below).
  3. s3_credentials.json in /home/jovyan/eey828-shared-fs/ (only needed when
     STAGE_MODELS_FROM_S3 is True).

Image: built from evaluation_bundles/Dockerfile and pushed to Harbor.
  docker build -f evaluation_bundles/Dockerfile \
    --build-arg BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
    -t harbor.a.comp-teach.qmul.ac.uk/nlp/adsolve-eval:latest .
  docker push harbor.a.comp-teach.qmul.ac.uk/nlp/adsolve-eval:latest

nltk corpora (punkt_tab) are baked into that image, so no staging is needed for
them -- unlike the model weights below.
"""
import json
import os
from pathlib import Path

from kubernetes import client, config


# ── Cluster settings from .env ──────────────────────────────────────────────
def _load_env() -> None:
    """Read .env into os.environ, without depending on python-dotenv.

    Looked for next to this driver first, then at the repo root. A real
    environment variable always wins, so a single value can be overridden for
    one submission without editing the file.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here / ".env", here.parents[1] / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        break


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. Copy kubernetes_drivers/.env.example to .env "
            f"and fill it in (see that file for what each setting means)."
        )
    return value


_load_env()

# ── Credentials ─────────────────────────────────────────────────────────────
MOUNTPATH = _require("KF_MOUNTPATH")

SECRET_FILE = f"{MOUNTPATH}/s3_credentials.json"
if not os.path.exists(SECRET_FILE):
    if os.path.exists("/home/jovyan/s3_credentials.json"):
        SECRET_FILE = "/home/jovyan/s3_credentials.json"
    else:
        raise FileNotFoundError(f"Put s3_credentials.json in {MOUNTPATH}/")
with open(SECRET_FILE, "r") as f:
    s3_creds = json.load(f)
access_key = s3_creds.get("AWS_ACCESS_KEY_ID") or s3_creds.get("FSSPEC_S3_KEY")
secret_key = s3_creds.get("AWS_SECRET_ACCESS_KEY") or s3_creds.get("FSSPEC_S3_SECRET")
if not access_key or not secret_key:
    raise ValueError("No S3 keys found in s3_credentials.json")

# ── Cluster configuration (from .env) ───────────────────────────────────────
NAMESPACE = _require("KF_NAMESPACE")
CLAIMNAME = _require("KF_CLAIMNAME")
IMAGE_NAME = _require("KF_IMAGE")
QUEUE_NAME = os.environ.get("KF_QUEUE", "nlp")
GPU_RESOURCE = os.environ.get("KF_GPU_RESOURCE", "nvidia.com/a40")
S3_ENDPOINT = os.environ.get(
    "KF_S3_ENDPOINT",
    "http://rook-ceph-rgw-ceph-objectstore-private.rook-ceph.svc.cluster.local")

REPO_DIR = os.environ.get("KF_REPO_DIR", f"{MOUNTPATH}/adsolve_utilities")

# ── Model cache ─────────────────────────────────────────────────────────────
# HF_HOME is what metrics/cache.py resolve_cache_dir() reads. Staging from S3
# avoids every job re-downloading several GB from HuggingFace, and is required
# if the GPU nodes have no outbound internet.
S3_MODELS = os.environ.get("KF_S3_MODELS", "")
STAGE_MODELS_FROM_S3 = bool(S3_MODELS)
TMP_MODELS = "/tmp/models"
OFFLINE = os.environ.get("KF_HF_OFFLINE", "1") == "1"

# ── Job configuration (edit per run) ────────────────────────────────────────
JOB_NAME = "adsolve-eval-affiniti"

IN_DIR = f"{MOUNTPATH}/adsolve_eval_data/affiniti/in"
OUT_DIR = f"{MOUNTPATH}/adsolve_eval_data/affiniti/out"

# ── Eval run configuration ──────────────────────────────────────────────────
# The affiniti spec: intra_nli, mhic, fc_document.
SPEC_PATH = f"{REPO_DIR}/evaluation_bundles/bundle_specs/affiniti_therapy_session_summary.yaml"
METRICS = ""      # ignored when SPEC_PATH is set
RUN_NAME = ""     # empty: the run is named by the spec

LLM_SUMMARIES = f"{IN_DIR}/llm_summaries.json"
GOLD_SUMMARIES = ""                # affiniti uses no gold reference
POSTS = f"{IN_DIR}/posts.json"     # required by mhic and fc_document
OUTPUT = f"{OUT_DIR}/results.json"

cli = (
    "python -m evaluation_bundles.run_job"
    + (f" --spec {SPEC_PATH}" if SPEC_PATH else f" --metrics {METRICS}")
    + (f" --name {RUN_NAME}" if RUN_NAME else "")
    + f" --llm-summaries {LLM_SUMMARIES}"
    + (f" --gold-summaries {GOLD_SUMMARIES}" if GOLD_SUMMARIES else "")
    + (f" --posts {POSTS}" if POSTS else "")
    + f" --output {OUTPUT}"
)

# ── Container script ────────────────────────────────────────────────────────
stage_block = f"""
                                    printf "[1/3] Install s3cmd\\n"
                                    PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install \
                                        --quiet --no-warn-script-location s3cmd

                                    printf "[2/3] Pull staged models from S3 to {TMP_MODELS}\\n"
                                    mkdir -p {TMP_MODELS} /tmp/torch
                                    s3cmd -c {MOUNTPATH}/.s3cfg --recursive get {S3_MODELS}/ {TMP_MODELS}/
                                    df -h /tmp
""" if STAGE_MODELS_FROM_S3 else ""

offline_block = """
                                    export HF_HUB_OFFLINE=1
                                    export TRANSFORMERS_OFFLINE=1
""" if OFFLINE else ""

script = f"""
                                    set -euo pipefail
{stage_block}
                                    printf "[3/3] Run evaluation\\n"
                                    export HF_HOME="{TMP_MODELS}"
                                    export TORCH_HOME=/tmp/torch
                                    export TOKENIZERS_PARALLELISM=false
{offline_block}
                                    mkdir -p "{OUT_DIR}"
                                    cd "{REPO_DIR}"
                                    echo "{cli}"
                                    {cli}

                                    printf "Done. Results at {OUTPUT}\\n"
                                    ls -la "{OUT_DIR}"
"""

# ── Manifest ────────────────────────────────────────────────────────────────
manifest = {
    "apiVersion": "kubeflow.org/v1",
    "kind": "PyTorchJob",
    "metadata": {
        "name": JOB_NAME,
        "namespace": NAMESPACE,
        "labels": {"kueue.x-k8s.io/queue-name": QUEUE_NAME},
    },
    "spec": {
        "pytorchReplicaSpecs": {
            "Master": {
                "replicas": 1,
                # Never, not OnFailure: a failed run has usually burned GPU time
                # already, and run_job exits 2 for a bad request, which retrying
                # cannot fix.
                "restartPolicy": "Never",
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "pytorch",
                                "image": IMAGE_NAME,
                                # Overrides the image ENTRYPOINT, which is only
                                # used for local `docker run`.
                                "command": ["bash", "-lc"],
                                "args": [script],
                                "env": [
                                    {"name": "FSSPEC_S3_ENDPOINT_URL",
                                     "value": S3_ENDPOINT},
                                    {"name": "FSSPEC_S3_KEY", "value": access_key},
                                    {"name": "FSSPEC_S3_SECRET", "value": secret_key},
                                    {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
                                ],
                                "resources": {
                                    "limits": {
                                        GPU_RESOURCE: "1",
                                        "memory": "24Gi",
                                        "cpu": "8",
                                    }
                                },
                                "volumeMounts": [
                                    {"name": "workspace", "mountPath": MOUNTPATH}
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "workspace",
                             "persistentVolumeClaim": {"claimName": CLAIMNAME}}
                        ],
                    }
                },
            }
        }
    },
}

# ── Submit ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()

    print(f"Submitting job: {JOB_NAME}")
    print(f"CLI: {cli}")
    try:
        client.CustomObjectsApi().create_namespaced_custom_object(
            group="kubeflow.org", version="v1", namespace=NAMESPACE,
            plural="pytorchjobs", body=manifest,
        )
        print(f"Submitted. Logs: kubectl -n {NAMESPACE} logs -f {JOB_NAME}-master-0")
        print(f"Output: {OUTPUT}")
    except client.exceptions.ApiException as e:
        if e.status == 409:
            print(f"Job '{JOB_NAME}' exists — delete or bump JOB_NAME:")
            print(f"  kubectl -n {NAMESPACE} delete pytorchjob {JOB_NAME}")
        else:
            print(f"API Error: {e}")
