import json
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


def _write_session(user_dir, user_id, session_id, content, summary):
    conv_path = user_dir / f"{user_id}_{session_id}.json"
    conv_path.write_text(json.dumps({
        "user_id": user_id, "session_id": session_id,
        "session_type": "therapist_initiated", "content": content,
    }))
    summary_path = user_dir / f"{user_id}_{session_id}_session_summary.json"
    summary_path.write_text(json.dumps(summary))
    return conv_path, summary_path


def test_find_session_pairs_pairs_by_prefix(tmp_path):
    user_dir = tmp_path / "user1"
    user_dir.mkdir()
    conv_path, summary_path = _write_session(
        user_dir, "user1", "sess1",
        [{"role": "user", "content": "hi"}],
        {"problem": "p", "activity": "a", "outcome": "o"},
    )
    pairs = pts.find_session_pairs(tmp_path)
    assert pairs == [("user1_sess1", conv_path, summary_path)]


def test_find_session_pairs_skips_unpaired_summary_with_warning(tmp_path, capsys):
    user_dir = tmp_path / "user1"
    user_dir.mkdir()
    (user_dir / "user1_sess1_session_summary.json").write_text(
        json.dumps({"problem": "p", "activity": "a", "outcome": "o"})
    )
    pairs = pts.find_session_pairs(tmp_path)
    assert pairs == []
    assert "user1_sess1" in capsys.readouterr().err


def test_find_session_pairs_ignores_patient_summary_files(tmp_path):
    user_dir = tmp_path / "user1"
    user_dir.mkdir()
    (user_dir / "user1_patient_summary.json").write_text(json.dumps({"Key Insights": "x"}))
    pairs = pts.find_session_pairs(tmp_path)
    assert pairs == []


def test_load_session_returns_summary_and_posts():
    summary = {"problem": "Anxiety.", "activity": "Breathing.", "outcome": "Calmer."}
    content = [{"role": "user", "content": "hi"}]
    result = pts.load_session_data(content, summary)
    assert result == ("Anxiety. Breathing. Calmer.", ["hi"])


def test_load_session_returns_none_for_empty_field():
    summary = {"problem": "", "activity": "Breathing.", "outcome": "Calmer."}
    content = [{"role": "user", "content": "hi"}]
    assert pts.load_session_data(content, summary) is None


def test_load_session_reads_from_disk(tmp_path):
    user_dir = tmp_path / "user1"
    user_dir.mkdir()
    conv_path, summary_path = _write_session(
        user_dir, "user1", "sess1",
        [{"role": "user", "content": "hi"}],
        {"problem": "p", "activity": "a", "outcome": "o"},
    )
    assert pts.load_session(conv_path, summary_path) == ("p a o", ["hi"])


def _write_dataset(tmp_path, n=3):
    for i in range(n):
        user_dir = tmp_path / f"user{i}"
        user_dir.mkdir(parents=True, exist_ok=True)
        _write_session(
            user_dir, f"user{i}", f"sess{i}",
            [{"role": "user", "content": f"message {i}"}],
            {"problem": f"p{i}", "activity": f"a{i}", "outcome": f"o{i}"},
        )
    return tmp_path


def test_convert_dataset_writes_expected_json_files(tmp_path):
    input_dir = _write_dataset(tmp_path / "input", n=2)
    output_dir = tmp_path / "output"
    llm_summaries, posts = pts.convert_dataset(input_dir, output_dir)
    assert llm_summaries == {"user0_sess0": "p0 a0 o0", "user1_sess1": "p1 a1 o1"}
    assert posts == {"user0_sess0": ["message 0"], "user1_sess1": ["message 1"]}
    assert json.loads((output_dir / "llm_summaries.json").read_text()) == llm_summaries
    assert json.loads((output_dir / "posts.json").read_text()) == posts


def test_convert_dataset_skips_session_with_empty_summary_field(tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    user_dir = input_dir / "user0"
    user_dir.mkdir()
    _write_session(
        user_dir, "user0", "sess0",
        [{"role": "user", "content": "hi"}],
        {"problem": "", "activity": "a", "outcome": "o"},
    )
    llm_summaries, posts = pts.convert_dataset(input_dir, tmp_path / "output")
    assert llm_summaries == {}
    assert posts == {}
    assert "user0_sess0" in capsys.readouterr().err


def test_convert_dataset_skips_session_with_invalid_json_in_conversation(tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    user_dir = input_dir / "user0"
    user_dir.mkdir()
    # Write an invalid JSON conversation file
    conv_path = user_dir / "user0_sess0.json"
    conv_path.write_text("not valid json")
    # Write a valid summary file
    summary_path = user_dir / "user0_sess0_session_summary.json"
    summary_path.write_text(json.dumps({"problem": "p", "activity": "a", "outcome": "o"}))

    llm_summaries, posts = pts.convert_dataset(input_dir, tmp_path / "output")
    assert llm_summaries == {}
    assert posts == {}
    err = capsys.readouterr().err
    assert "user0_sess0" in err
    assert "JSONDecodeError" in err


def test_convert_dataset_skips_session_with_missing_content_key(tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    user_dir = input_dir / "user0"
    user_dir.mkdir()
    # Write a valid JSON conversation file but without the "content" key
    conv_path = user_dir / "user0_sess0.json"
    conv_path.write_text(json.dumps({"user_id": "user0", "session_id": "sess0"}))
    # Write a valid summary file
    summary_path = user_dir / "user0_sess0_session_summary.json"
    summary_path.write_text(json.dumps({"problem": "p", "activity": "a", "outcome": "o"}))

    llm_summaries, posts = pts.convert_dataset(input_dir, tmp_path / "output")
    assert llm_summaries == {}
    assert posts == {}
    err = capsys.readouterr().err
    assert "user0_sess0" in err
    assert "KeyError" in err


def test_convert_dataset_limit_and_seed_are_deterministic(tmp_path):
    input_dir = _write_dataset(tmp_path / "input", n=5)
    llm_summaries_1, _ = pts.convert_dataset(input_dir, tmp_path / "out1", limit=2, seed=42)
    llm_summaries_2, _ = pts.convert_dataset(input_dir, tmp_path / "out2", limit=2, seed=42)
    assert len(llm_summaries_1) == 2
    assert llm_summaries_1 == llm_summaries_2


def test_convert_dataset_limit_larger_than_dataset_returns_all(tmp_path):
    input_dir = _write_dataset(tmp_path / "input", n=2)
    llm_summaries, _ = pts.convert_dataset(input_dir, tmp_path / "output", limit=10)
    assert len(llm_summaries) == 2


def test_main_writes_files_with_explicit_args(tmp_path):
    input_dir = _write_dataset(tmp_path / "input", n=1)
    output_dir = tmp_path / "output"
    exit_code = pts.main(["--input_dir", str(input_dir), "--output_dir", str(output_dir)])
    assert exit_code == 0
    assert (output_dir / "llm_summaries.json").exists()
    assert (output_dir / "posts.json").exists()


def test_main_respects_limit_and_seed(tmp_path):
    input_dir = _write_dataset(tmp_path / "input", n=5)
    output_dir = tmp_path / "output"
    exit_code = pts.main([
        "--input_dir", str(input_dir), "--output_dir", str(output_dir),
        "--limit", "2", "--seed", "7",
    ])
    assert exit_code == 0
    llm_summaries = json.loads((output_dir / "llm_summaries.json").read_text())
    assert len(llm_summaries) == 2
