import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation_api.sensitivity import safe_error, strip_results

RESULTS = {
    "intra_nli": {"document_level": [0.9], "mean": 0.9},
    "fc_document": {
        "document_level": [0.6], "mean": 0.6,
        "detail": [{"scores": [0.6], "sentences": ["a private sentence"]}],
    },
    "document_ids": ["doc1"],
}


def test_non_sensitive_results_are_unchanged():
    assert strip_results(RESULTS, sensitive=False) == RESULTS


def test_detail_is_removed_for_sensitive_results():
    stripped = strip_results(RESULTS, sensitive=True)
    assert "detail" not in stripped["fc_document"]


def test_numeric_scores_survive_stripping():
    stripped = strip_results(RESULTS, sensitive=True)
    assert stripped["fc_document"]["document_level"] == [0.6]
    assert stripped["fc_document"]["mean"] == 0.6
    assert stripped["intra_nli"] == RESULTS["intra_nli"]


def test_document_ids_survive_stripping():
    assert strip_results(RESULTS, sensitive=True)["document_ids"] == ["doc1"]


def test_stripping_does_not_mutate_the_input():
    strip_results(RESULTS, sensitive=True)
    assert "detail" in RESULTS["fc_document"]


def test_no_document_text_remains_anywhere_in_stripped_results():
    assert "a private sentence" not in str(strip_results(RESULTS, sensitive=True))


def test_errors_are_passed_through_for_non_sensitive_jobs():
    assert safe_error("Traceback ...\nValueError: bad doc text", False).endswith("bad doc text")


def test_only_the_last_line_of_an_error_is_kept_for_sensitive_jobs():
    """A traceback can quote document text, so sensitive jobs keep only the summary."""
    assert safe_error("Traceback ...\n  print(doc)\nValueError: boom", True) == "ValueError: boom"
