# Batch evaluation job. Build from the repo root:
#
#   CPU (default, builds anywhere, testable locally):
#     docker build -f evaluation_bundles/Dockerfile -t adsolve-eval:cpu .
#
#   GPU (for the cluster) -- pick a tag whose bundled torch already satisfies
#   pyproject's torch>=2.7, because the build below will NOT replace it:
#     docker build -f evaluation_bundles/Dockerfile \
#       --build-arg BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime \
#       -t adsolve-eval:gpu .
#
# Base image is the only thing that changes between the two, which is why it is
# an ARG rather than two Dockerfiles that would drift apart.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# Where torch comes from when the base image has none. The default PyPI wheel
# bundles CUDA and is ~2.5GB even for CPU-only use; this index serves a slim CPU
# build instead.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_SYMLINKS=1

WORKDIR /app

# git: some pip dependencies resolve from git refs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# awscli powers the model-cache sync in entrypoint.sh. Installed via pip so this
# works identically on a debian- or conda-based base image.
RUN pip install --no-cache-dir awscli

COPY pyproject.toml README.md ./
# Install torch ONLY if the base image has none. A pytorch/* base ships a torch
# built against its own CUDA runtime; reinstalling from TORCH_INDEX_URL would
# replace it with a CPU build and the job would silently lose the GPU. The
# consequence is that a GPU base tag must already satisfy pyproject's
# torch>=2.7 -- this build will not upgrade it for you.
RUN python -c "import torch" 2>/dev/null \
    && echo "torch $(python -c 'import torch; print(torch.__version__)') from base image; leaving it alone" \
    || pip install --no-cache-dir "torch>=2.7" --index-url ${TORCH_INDEX_URL}

# evaluation_bundles must keep this path: run_job.py derives BUNDLES_DIR from its
# own location and injects it as PYTHONPATH, which is how generated scripts find
# `metrics.*`. Flatten this layout and every bundle fails on import.
COPY evaluation_bundles/ evaluation_bundles/
RUN pip install --no-cache-dir -e .

# nltk corpora, baked in rather than downloaded at runtime: a GPU pod should not
# need to reach the internet to tokenise a sentence, and on a cluster it may not
# be able to. punkt_tab is what nltk >= 3.9 looks for (py-readability-metrics
# calls sent_tokenize); punkt is kept for anything still on the old name.
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -m nltk.downloader -d ${NLTK_DATA} punkt_tab punkt

COPY evaluation_bundles/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
