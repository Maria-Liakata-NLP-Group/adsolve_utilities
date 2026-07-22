import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prepare_therapeutic_sessions as pts


def test_is_instruction_leak_matches_known_prefixes():
    assert pts.is_instruction_leak("System prompt\n\nRole: ...") is True
    assert pts.is_instruction_leak("A patient has selected to complete a Box Breathing exercise.") is True
    assert pts.is_instruction_leak("A therapist asked a patient to complete a Assertiveness Training exercise.") is True
    assert pts.is_instruction_leak("From this point, imagine that the AI is used in conjunction with human-led therapy.") is True


def test_is_instruction_leak_false_for_genuine_message():
    assert pts.is_instruction_leak("Let us do box breathing together.") is False
    assert pts.is_instruction_leak("That sounds like a lovely routine. I'll check in with you at 6:30 AM.") is False


def test_is_instruction_leak_strips_whitespace_before_matching():
    assert pts.is_instruction_leak("  \n System Prompt\nRole: ...") is True


def test_extract_posts_filters_leak_turn_and_keeps_order():
    content = [
        {"role": "assistant", "content": "A patient has selected to complete a Box Breathing exercise."},
        {"role": "assistant", "content": "I'm here to support you with Box Breathing."},
        {"role": "user", "content": "Yes"},
        {"role": "assistant", "content": "Great, let's begin."},
    ]
    assert pts.extract_posts(content) == [
        "I'm here to support you with Box Breathing.",
        "Yes",
        "Great, let's begin.",
    ]


def test_extract_posts_keeps_all_turns_when_no_leak_present():
    content = [
        {"role": "assistant", "content": "Let us do box breathing together."},
        {"role": "user", "content": "Ok"},
    ]
    assert pts.extract_posts(content) == ["Let us do box breathing together.", "Ok"]


def test_build_llm_summary_joins_fields_in_order():
    summary = {"problem": "Anxiety.", "activity": "Did breathing exercise.", "outcome": "Felt calmer."}
    assert pts.build_llm_summary(summary) == "Anxiety. Did breathing exercise. Felt calmer."
