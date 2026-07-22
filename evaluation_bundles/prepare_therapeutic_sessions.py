"""Convert the therapeutic_sessions dataset into llm_summaries.json/posts.json,
the input shape evaluation_bundles-generated bundles expect for intra_nli/mhic/fc_document.

See docs/superpowers/specs/2026-07-22-therapeutic-sessions-bundle-data-prep-design.md.
"""
from __future__ import annotations

LEAK_PREFIXES = (
    "system prompt",
    "a patient has selected",
    "a therapist asked a patient",
    "from this point, imagine that the ai is",
)


def is_instruction_leak(text: str) -> bool:
    stripped = text.strip().lower()
    return any(stripped.startswith(prefix) for prefix in LEAK_PREFIXES)


def extract_posts(content: list) -> list:
    return [turn["content"] for turn in content if not is_instruction_leak(turn["content"])]


def build_llm_summary(summary: dict) -> str:
    return f"{summary['problem']} {summary['activity']} {summary['outcome']}"
