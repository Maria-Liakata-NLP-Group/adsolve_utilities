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
#   GPU: pick a base whose bundled torch already satisfies torch>=2.7 --
#        this build will not upgrade it.
#     BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

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

WORKDIR /workspace
