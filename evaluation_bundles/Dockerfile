# Environment image for AdSoLve evaluation jobs.
#
# This image contains NO application code. Argo builds it without access to the
# repository, so there is nothing to COPY -- and the cluster pattern is to keep
# the code on the shared filesystem anyway, so a metric change needs no rebuild.
# The driver does `cd $KF_REPO_DIR` and runs `python -m evaluation_bundles.run_job`
# from there; cwd puts the shared-FS copy on sys.path.
#
# Dependencies are pinned here rather than read from pyproject.toml for the same
# reason. Keep this list in sync with [project.dependencies] in pyproject.toml.
#
# Defaults to a CUDA base, because a CPU base silently produces an image that
# cannot use the GPU at all: torch.version.cuda is None, nvidia-smi shows the
# card at 0% with no processes, and nothing errors. The tag matches the group's
# other Kubeflow images and is known to work on this cluster (CUDA 12.1 against a
# 12.8 driver is fine -- drivers are backward compatible).
#
# NOTE: this base ships torch 2.4.0, and the build below deliberately does NOT
# upgrade it -- doing so would replace the CUDA build with a CPU one. pyproject's
# torch>=2.7 is therefore not enforced inside this image.
#
#   CPU (local testing only, no GPU support):
#     --build-arg BASE_IMAGE=python:3.12-slim --build-arg REQUIRE_CUDA=0
ARG BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
FROM ${BASE_IMAGE}

# Set to 0 for a deliberate CPU build; 1 makes the smoke test below refuse to
# produce a CUDA-less image by accident.
ARG REQUIRE_CUDA=1

# Where torch comes from when the base image has none. The default PyPI wheel
# bundles CUDA and is ~2.5GB even for CPU-only use; this index serves a slim CPU
# build instead.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_SYMLINKS=1 \
    NLTK_DATA=/usr/share/nltk_data

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install torch ONLY if the base image has none. A pytorch/* base ships a torch
# built against its own CUDA runtime; reinstalling from TORCH_INDEX_URL would
# replace it with a CPU build and the job would silently lose the GPU.
RUN python -c "import torch" 2>/dev/null \
    && echo "torch $(python -c 'import torch; print(torch.__version__)') from base image; leaving it alone" \
    || pip install --no-cache-dir "torch>=2.7" --index-url ${TORCH_INDEX_URL}

# Runtime dependencies of evaluation_bundles/metrics/*.
# py-readability-metrics, NOT readability-lxml: they claim the same `readability`
# module name but only the former provides the Readability grade-level scorer.
RUN pip install --no-cache-dir \
        "bert-score>=0.3.13" \
        "datasets>=4.0.0" \
        "nltk>=3.9.1" \
        "numpy>=2.3" \
        "pandas>=2.3" \
        "peft>=0.15.2" \
        "PyYAML>=6.0" \
        "py-readability-metrics>=1.4.5" \
        "rouge>=1.0.1" \
        "spacy>=3.8" \
        "tqdm>=4.67" \
        "transformers>=4.52"

# Cluster tooling: awscli and s3cmd for staging the model cache from S3.
RUN pip install --no-cache-dir awscli s3cmd

# nltk corpora, baked in rather than downloaded at runtime: a GPU pod should not
# need to reach the internet to tokenise a sentence, and on a cluster it may not
# be able to. punkt_tab is what nltk >= 3.9 looks for (py-readability-metrics
# calls sent_tokenize); punkt is kept for anything still on the old name.
RUN python -m nltk.downloader -d ${NLTK_DATA} punkt_tab punkt

# Smoke test: fail the BUILD rather than a GPU job if the environment is broken.
# torch.cuda.is_available() is expected to be False here -- builders have no GPU.
# What matters is torch.version.cuda: None means a CPU-only wheel, which would
# run on the cluster at 0% GPU utilisation without ever raising an error.
RUN python -c "\
import torch, transformers, nltk, readability, yaml; \
print('torch       :', torch.__version__); \
print('cuda build  :', torch.version.cuda); \
print('transformers:', transformers.__version__); \
nltk.data.find('tokenizers/punkt_tab'); \
assert torch.version.cuda or '${REQUIRE_CUDA}' == '0', 'CPU-only torch: this image cannot use the GPU (set REQUIRE_CUDA=0 if deliberate)'; \
print('image OK')"

WORKDIR /workspace
