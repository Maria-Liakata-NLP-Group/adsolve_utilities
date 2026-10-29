import json
import os
from kubernetes import client, config

SECRET_FILE = "/home/jovyan/eey828-shared-fs/s3_credentials.json"
if not os.path.exists(SECRET_FILE):
    SECRET_FILE_HOME = "/home/jovyan/s3_credentials.json"
    if os.path.exists(SECRET_FILE_HOME):
        SECRET_FILE = SECRET_FILE_HOME
    else:
        raise FileNotFoundError("Please put s3_credentials.json in /home/jovyan/eey828-shared-fs/")

with open(SECRET_FILE, "r") as f:
    s3_creds = json.load(f)

JOB_NAME = "icl-attn-exp2-ift"
NAMESPACE = "kf-eey828"
IMAGE_NAME = "harbor.a.comp-teach.qmul.ac.uk/nlp/qwen-unsloth:latest"

# Put these updated scripts on shared fs before submitting.
SCRIPT_ANALYSIS_PATH = "/home/jovyan/eey828-shared-fs/ICL/Attention_viz/attention_analysis_exp2_fullforward.py"
SCRIPT_VIZ_PATH = "/home/jovyan/eey828-shared-fs/ICL/Attention_viz/attention_viz_exp2_v3.py"

# ---- INPUTS ----
S3_EVALPACK_PREFIX = "s3://nlp-01/ia180/EVAL_PACKS/olmo7b/olmo7b_ift_full_valid/"
EVALPACK_BASENAME = "test_eval_mood_k1.jsonl.gz"
#S3_TOKENIZER_PREFIX = "s3://nlp-01/ia180/IFT_OUTPUTCuricullum/olmo7bfull/tokenizer/"
#TMP_TOKENIZER_DIR = "/tmp/tokenizer"
#HF_BASE_MODEL = "allenai/Olmo-3-7B-Instruct"

S3_MODEL_PREFIX = "s3://nlp-01/ia180/IFT_OUTPUTCuricullum/olmo7bfull/Final_Model/full_model/"

TMP_EVALPACK_DIR = "/tmp/olmo7b_evalpacks"
TMP_MODEL_DIR = "/tmp/full_model"
EVALPACK_PATH = f"{TMP_EVALPACK_DIR}/{EVALPACK_BASENAME}"

# ---- OPTIONAL correctness split ----
# Turn this on later when you patch the nested label-token loader in the analysis script.
CONDITION_ON_CORRECT = False
LABEL_TOKEN_IDS_JSON = "/home/jovyan/eey828-shared-fs/ICL/Attention_viz/label_token_ids.json"
LABEL_TOKEN_IDS_KEY = "talklife_spec"

# ---- ATTENTION SETTINGS ----
MODEL_LABEL = "sft_model"
MAX_EXAMPLES = 394
BATCH_SIZE = 1
DTYPE = "bf16"
ATTN_IMPL = "eager"
PRED_INDEX_SOURCE = "last_curr"
REGIONS = "instruction,fewshot,hist,curr"
HISTBIN_MODE = "rel_t"
MAX_HIST_BINS = 10
ADD_OTHER_REGION = True
CHECK_NORMALIZATION = True
RUN_VIZ = True
DELTA_COMPARE = False  # keep false for single-model runs

LOCAL_OUT_DIR = "/home/jovyan/eey828-shared-fs/ICL/Attention_viz/Results/IFT"
LOCAL_FIG_DIR = "/home/jovyan/eey828-shared-fs/ICL/Attention_viz/Results/IFT"
OUT_JSON = f"{LOCAL_OUT_DIR}/{MODEL_LABEL}.json"

MOUNTPATH = "/home/jovyan/eey828-shared-fs"
CLAIMNAME = "eey828-shared-fs"

access_key = s3_creds.get("AWS_ACCESS_KEY_ID") or s3_creds.get("FSSPEC_S3_KEY")
secret_key = s3_creds.get("AWS_SECRET_ACCESS_KEY") or s3_creds.get("FSSPEC_S3_SECRET")
if not access_key or not secret_key:
    raise ValueError("Could not find keys in s3_credentials.json! Checked AWS_ and FSSPEC_ prefixes.")

analysis_cmd = [
    "--evalpack", EVALPACK_PATH,
    "--base_model", TMP_MODEL_DIR,
    #"--tokenizer_path", TMP_TOKENIZER_DIR,
    "--attn_implementation", ATTN_IMPL,
    "--pred_index_source", PRED_INDEX_SOURCE,
    "--regions", REGIONS,
    "--max_examples", str(MAX_EXAMPLES),
    "--batch_size", str(BATCH_SIZE),
    "--dtype", DTYPE,
    "--histbin_mode", HISTBIN_MODE,
    "--max_hist_bins", str(MAX_HIST_BINS),
    "--out_json", OUT_JSON,
]
if ADD_OTHER_REGION:
    analysis_cmd.append("--add_other_region")
if CHECK_NORMALIZATION:
    analysis_cmd.append("--check_normalization")
if CONDITION_ON_CORRECT:
    analysis_cmd.extend([
        "--condition_on_correct",
        "--label_token_ids_json", LABEL_TOKEN_IDS_JSON,
        "--label_token_ids_key", LABEL_TOKEN_IDS_KEY,
    ])
analysis_cmd_str = " ".join(analysis_cmd)

viz_cmd = [
    "--inputs", OUT_JSON,
    "--labels", MODEL_LABEL,
    "--bucket", "all",
    "--out_dir", LOCAL_FIG_DIR,
]
viz_cmd_str = " ".join(viz_cmd)
#s3cmd -c /home/jovyan/eey828-shared-fs/.s3cfg --recursive get {S3_TOKENIZER_PREFIX} {TMP_TOKENIZER_DIR}/
manifest = {
    "apiVersion": "kubeflow.org/v1",
    "kind": "PyTorchJob",
    "metadata": {
        "name": JOB_NAME,
        "namespace": NAMESPACE,
        "labels": {
            "kueue.x-k8s.io/queue-name": "nlp",
        },
    },
    "spec": {
        "pytorchReplicaSpecs": {
            "Master": {
                "replicas": 1,
                "restartPolicy": "OnFailure",
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "pytorch",
                                "image": IMAGE_NAME,
                                "command": ["bash", "-lc"],
                                "args": [f'''
                                    set -euo pipefail

                                    export HF_HOME=/tmp/hf
                                    export TRANSFORMERS_CACHE=/tmp/hf/hub
                                    export HF_DATASETS_CACHE=/tmp/hf/datasets
                                    export TORCH_HOME=/tmp/torch

                                    printf "[1/5] Install s3cmd\\n"
                                    PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet --no-warn-script-location --index-url https://pypi.org/simple s3cmd

                                    printf "[2/5] Create local dirs\\n"
                                    mkdir -p {TMP_EVALPACK_DIR} {TMP_MODEL_DIR} {LOCAL_OUT_DIR} {LOCAL_FIG_DIR} 

                                    printf "[3/5] Download evalpack + single model\\n"
                                    s3cmd -c /home/jovyan/eey828-shared-fs/.s3cfg --recursive get {S3_EVALPACK_PREFIX} {TMP_EVALPACK_DIR}/ 
                                    s3cmd -c /home/jovyan/eey828-shared-fs/.s3cfg --recursive get {S3_MODEL_PREFIX} {TMP_MODEL_DIR}/                                  

                                    printf "[5/5] Run attention analysis\\n"
                                    python {SCRIPT_ANALYSIS_PATH} {analysis_cmd_str}

                                    if [ "{str(RUN_VIZ).lower()}" = "true" ]; then
                                      if [ -f {OUT_JSON} ]; then
                                        printf "Running attention visualization...\\n"
                                        python {SCRIPT_VIZ_PATH} {viz_cmd_str}
                                      else
                                        printf "Skipping viz because output JSON is missing.\\n"
                                      fi
                                    fi

                                    printf "Done.\\n"
                                '''],
                                "env": [
                                    {"name": "FSSPEC_S3_ENDPOINT_URL", "value": "http://rook-ceph-rgw-ceph-objectstore-private.rook-ceph.svc.cluster.local"},
                                    {"name": "FSSPEC_S3_KEY", "value": access_key},
                                    {"name": "FSSPEC_S3_SECRET", "value": secret_key},
                                    {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
                                    {"name": "CUDA_LAUNCH_BLOCKING", "value": "1"},
                                ],
                                "resources": {
                                    "limits": {
                                        "nvidia.com/h100-96g": "1",
                                        "memory": "48Gi",
                                        "cpu": "8",
                                    }
                                },
                                "volumeMounts": [
                                    {
                                        "name": "workspace",
                                        "mountPath": MOUNTPATH,
                                    }
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "workspace",
                                "persistentVolumeClaim": {
                                    "claimName": CLAIMNAME,
                                },
                            }
                        ],
                    }
                },
            }
        }
    }
}

if __name__ == "__main__":
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()

    api_client = client.CustomObjectsApi()
    print(f"🚀 Submitting {JOB_NAME} (Manifest Mode)...")
    try:
        api_client.create_namespaced_custom_object(
            group="kubeflow.org",
            version="v1",
            namespace=NAMESPACE,
            plural="pytorchjobs",
            body=manifest,
        )
        print("✅ Job Submitted Successfully!")
    except client.exceptions.ApiException as e:
        if e.status == 409:
            print(f"❌ Error: Job '{JOB_NAME}' already exists. Please change the JOB_NAME.")
        else:
            print(f"❌ API Error: {e}")