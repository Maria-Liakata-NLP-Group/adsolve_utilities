"""Where metrics cache their HuggingFace downloads.

Metrics used to hardcode the group cluster's NFS path as the default cache
directory, which made them fail on any other machine — on macOS with
`OSError: [Errno 30] Read-only file system: '/import'`.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.cache import resolve_cache_dir


def test_nothing_configured_defers_to_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None makes transformers use its own ~/.cache/huggingface default."""
    monkeypatch.delenv("HF_HOME", raising=False)
    assert resolve_cache_dir() is None


def test_uses_hf_home_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", "/tmp/hf-cache")
    assert resolve_cache_dir() == "/tmp/hf-cache"


def test_an_explicit_directory_wins_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HOME", "/tmp/hf-cache")
    assert resolve_cache_dir("/tmp/explicit") == "/tmp/explicit"


def test_an_empty_hf_home_is_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported-but-empty variable must not become a cache path of ''."""
    monkeypatch.setenv("HF_HOME", "")
    assert resolve_cache_dir() is None


def test_no_metric_defaults_to_a_machine_specific_path() -> None:
    """The regression itself: no constructor may hardcode an absolute path."""
    import inspect

    from metrics.evidence_appropriateness import EA
    from metrics.intra_nli import IntraNLI

    for metric_class in (IntraNLI, EA):
        default = inspect.signature(metric_class.__init__).parameters[
            "hf_cache_dir"
        ].default
        assert default is None, (
            f"{metric_class.__name__} hardcodes a cache path: {default!r}"
        )
