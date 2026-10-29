"""Where metrics cache their HuggingFace downloads.

Shared so no metric hardcodes a machine-specific path again: several once
defaulted to the group cluster's `/import/nlp-datasets/LLMs`, which made them
unrunnable anywhere else.
"""

from __future__ import annotations

import os
from typing import Optional


def resolve_cache_dir(explicit: Optional[str] = None) -> Optional[str]:
    """The cache directory to hand transformers.

    Precedence: an explicit argument, then HF_HOME, then None — which lets
    transformers fall back to its own ~/.cache/huggingface. Returning None
    rather than a path of our own is deliberate: the default has to be
    writable on every machine, and only transformers knows where that is.
    """
    return explicit or os.environ.get("HF_HOME") or None
