"""Removes document text from results and errors for sensitive datasets."""
from __future__ import annotations

DOCUMENT_IDS_KEY = "document_ids"
TEXT_BEARING_KEYS = ("detail",)


def strip_results(results: dict, sensitive: bool) -> dict:
    """Drop text-bearing fields, keeping numeric scores.

    `detail` is {scores: [float], sentences: [str]} -- the sentences are verbatim
    document text, so the whole key goes. document_level and mean are floats and
    stay.
    """
    if not sensitive:
        return results

    stripped = {}
    for key, value in results.items():
        if key == DOCUMENT_IDS_KEY or not isinstance(value, dict):
            stripped[key] = value
            continue
        stripped[key] = {k: v for k, v in value.items() if k not in TEXT_BEARING_KEYS}
    return stripped


def safe_error(error: str, sensitive: bool) -> str:
    """A traceback can quote document text; for sensitive jobs keep only its last line."""
    if not sensitive:
        return error
    lines = [line for line in error.strip().splitlines() if line.strip()]
    return lines[-1].strip() if lines else ""
