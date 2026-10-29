"""
Kubeflow driver — FULL evaluation run (eey828), AlignScore-LARGE.
Phase A (own-battery, all datasets) as configured below.
Phase B (cross-battery): set JOB_NAME="slm-eval-all-b", SKIP_CROSS=False,
RECOMPUTE=False and resubmit — Phase A caches are reused.

Prereqs on shared FS: this v16 repo folder + eval_data/<dataset>/ raw files
(or canonical.csv) copied in. Staging (incl. roberta-large + large ckpt,
~4GB one-time) runs automatically on submit; already-staged models skip.
"""
import json
import os
import subprocess
import sys

from kubernetes import client, config

# ── Credentials ─────────────────────────────────────────────────────────────
SECRET_FILE = "/home/jovyan/eey828-shared-fs/s3_credentials.json"
if not os.path.exists(SECRET_FILE):
    if os.path.exists("/home/jovyan/s3_credentials.json"):
        SECRET_FILE = "/home/jovyan/s3_credentials.json"
    else:
        raise FileNotFoundError("Put s3_credentials.json in /home/jovyan/eey828-shared-fs/")
with open(SECRET_FILE, "r") as f:
    s3_creds = json.load(f)
access_key = s3_creds.get("AWS_ACCESS_KEY_ID") or s3_creds.get("FSSPEC_S3_KEY")
secret_key = s3_creds.get("AWS_SECRET_ACCESS_KEY") or s3_creds.get("FSSPEC_S3_SECRET")
if not access_key or not secret_key:
    raise ValueError("No S3 keys found in s3_credentials.json")

# ── Job configuration ───────────────────────────────────────────────────────
JOB_NAME = "slm-eval-worstpool"
NAMESPACE = "kf-eey828"
IMAGE_NAME = "harbor.a.comp-teach.qmul.ac.uk/nlp/mingwei-slmcaseone:latest"
MOUNTPATH = "/home/jovyan/eey828-shared-fs"
CLAIMNAME = "eey828-shared-fs"

REPO_DIR = f"{MOUNTPATH}/2026July24-Multifactor-SLM-V6-eval-v19"
EVAL_DATA_DIR = "/home/jovyan/eey828-shared-fs/2026July22-Multifactor-SLM-V6-eval-v17/eval_data"
#OUT_ROOT = f"{MOUNTPATH}/results/full_v17" 
OUT_ROOT = f"{MOUNTPATH}/results/worstpool"

S3_MODELS = "s3://nlp-01/eey828/hf_models"
TMP_MODELS = "/tmp/models"

# ── Model staging (idempotent; only new items download) ─────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "s3cmd", "s3fs"], check=False)
r = subprocess.run([sys.executable, f"{REPO_DIR}/scripts/stage_models_to_s3.py"])
if r.returncode != 0:
    raise SystemExit("Model staging failed — fix before submitting.")

# ── Eval run configuration ──────────────────────────────────────────────────
EVAL_DATASET = "ragtruth"
SAMPLE_LIMIT = 1000
SKIP_CROSS = False         # single run = Phase A + B together
RECOMPUTE = True

cli = (
    f"python -m evaluation.run_eval"
    f" --eval-dataset {EVAL_DATASET}"
    f" --sample-limit {SAMPLE_LIMIT}"
    f" --out-root {OUT_ROOT}"
    + (" --skip-cross" if SKIP_CROSS else "")
    + (" --recompute-indicators" if RECOMPUTE else "")
)

# ── Manifest ────────────────────────────────────────────────────────────────
manifest = {
    "apiVersion": "kubeflow.org/v1",
    "kind": "PyTorchJob",
    "metadata": {
        "name": JOB_NAME,
        "namespace": NAMESPACE,
        "labels": {"kueue.x-k8s.io/queue-name": "nlp"},
    },
    "spec": {
        "pytorchReplicaSpecs": {
            "Master": {
                "replicas": 1,
                "restartPolicy": "Never",
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "pytorch",
                                "image": IMAGE_NAME,
                                "command": ["bash", "-lc"],
                                "args": [
                                    f"""
                                    set -euo pipefail

                                    printf "[1/4] Install s3cmd\\n"
                                    PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet --no-warn-script-location --index-url https://pypi.org/simple s3cmd sacrebleu

                                    printf "[2/4] Pull staged models from S3 to {TMP_MODELS}\\n"
                                    mkdir -p {TMP_MODELS} /tmp/torch
                                    s3cmd -c {MOUNTPATH}/.s3cfg --recursive get {S3_MODELS}/ {TMP_MODELS}/
                                    ls {TMP_MODELS}
                                    df -h /tmp

                                    printf "[3/4] Sanity-check staged files\\n"
                                    test -f "{TMP_MODELS}/roberta-base/config.json"
                                    test -f "{TMP_MODELS}/roberta-large/config.json"
                                    test -f "{TMP_MODELS}/cross-encoder--nli-deberta-v3-small/config.json"
                                    test -f "{TMP_MODELS}/facebook--bart-large-cnn/config.json"
                                    test -f "{TMP_MODELS}/tals--albert-xlarge-vitaminc-mnli/config.json"
                                    test -f "{TMP_MODELS}/yzha--AlignScore/AlignScore-large.ckpt"
                                    test -d "{TMP_MODELS}/nltk_data/corpora" || printf "WARN: nltk_data missing (meteor will disable)\n"

                                    printf "[4/4] Run eval (models by LOCAL PATH, AlignScore-LARGE)\\n"
                                    export EVAL_MODELS_DIR="{TMP_MODELS}"
                                    export NLTK_DATA="{TMP_MODELS}/nltk_data"
                                    export CC_NLI_MAX_PAIRS=2000
                                    export ALIGNSCORE_CKPT="{TMP_MODELS}/yzha--AlignScore/AlignScore-large.ckpt"
                                    export HF_HUB_OFFLINE=1
                                    export TRANSFORMERS_OFFLINE=1
                                    export HF_EVALUATE_OFFLINE=1
                                    export TORCH_HOME=/tmp/torch
                                    export IC_WORST_POOL=1
                                    export EVAL_DATA_DIR="{EVAL_DATA_DIR}"
                                    cd "{REPO_DIR}"
                                    echo "{cli}"
                                    {cli}

                                    printf "Done. Reports under {OUT_ROOT}\\n"
                                    ls "{OUT_ROOT}" || true
                                    """
                                ],
                                "env": [
                                    {"name": "FSSPEC_S3_ENDPOINT_URL", "value": "http://rook-ceph-rgw-ceph-objectstore-private.rook-ceph.svc.cluster.local"},
                                    {"name": "FSSPEC_S3_KEY", "value": access_key},
                                    {"name": "FSSPEC_S3_SECRET", "value": secret_key},
                                    {"name": "CUDA_LAUNCH_BLOCKING", "value": "1"},
                                    {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
                                ],
                                "resources": {
                                    "limits": {
                                        "nvidia.com/a40": "1",
                                        "memory": "24Gi",
                                        "cpu": "8",
                                        #"ephemeral-storage": "40Gi",
                                    }
                                },
                                "volumeMounts": [{"name": "workspace", "mountPath": MOUNTPATH}],
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
        print(f"Outputs: {OUT_ROOT}")
    except client.exceptions.ApiException as e:
        if e.status == 409:
            print(f"Job '{JOB_NAME}' exists — delete or bump JOB_NAME:")
            print(f"  kubectl -n {NAMESPACE} delete pytorchjob {JOB_NAME}")
        else:
            print(f"API Error: {e}")
