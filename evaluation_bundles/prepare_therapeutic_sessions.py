"""Convert the therapeutic_sessions dataset into llm_summaries.json/posts.json,
the input shape evaluation_bundles-generated bundles expect for intra_nli/mhic/fc_document.

See docs/superpowers/specs/2026-07-22-therapeutic-sessions-bundle-data-prep-design.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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


SUMMARY_SUFFIX = "_session_summary.json"


def find_session_pairs(input_dir: Path) -> list:
    pairs = []
    for user_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        for summary_path in sorted(user_dir.glob(f"*{SUMMARY_SUFFIX}")):
            document_id = summary_path.name[: -len(SUMMARY_SUFFIX)]
            conversation_path = user_dir / f"{document_id}.json"
            if not conversation_path.exists():
                print(
                    f"warning: skipping {document_id} - no matching conversation file "
                    f"({conversation_path})",
                    file=sys.stderr,
                )
                continue
            pairs.append((document_id, conversation_path, summary_path))
    return pairs


def load_session_data(content: list, summary: dict):
    if not all(summary.get(field, "").strip() for field in ("problem", "activity", "outcome")):
        return None
    return build_llm_summary(summary), extract_posts(content)


def load_session(conversation_path: Path, summary_path: Path):
    conversation = json.loads(conversation_path.read_text())
    summary = json.loads(summary_path.read_text())
    return load_session_data(conversation["content"], summary)
