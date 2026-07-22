"""Generate an evaluation_bundles/<name>_evaluation.py script from a YAML metric spec.

See evaluation_bundles/bundle_specs/social_media_example.yaml for an example spec, and
evaluation_bundles/metric_registry.py for the list of available metrics.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metric_registry import METRIC_REGISTRY, InputKind, MetricInfo


class SpecError(Exception):
    """Raised when a bundle spec fails validation. Message lists every problem found."""


NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
REFERENCE_KINDS = (InputKind.SINGLE, InputKind.LIST, InputKind.DUAL, InputKind.BATCH)
VALID_REFERENCES = {"gold", "posts"}
VALID_FACT_REFERENCES = {"gold", "posts", "llm"}
VALID_FACT_CLAIM_SOURCES = {"llm", "gold"}
VALID_FACT_MODES = {"recall", "precision"}


def load_spec(spec_path: str) -> dict:
    with open(spec_path, "r") as f:
        return yaml.safe_load(f)


def validate_spec(spec: dict) -> None:
    errors: list = []

    name = spec.get("name")
    if not name or not NAME_PATTERN.match(name):
        errors.append(f"'name' must be a lowercase snake_case identifier (got {name!r})")

    metrics = spec.get("metrics") or []
    if not metrics:
        errors.append("'metrics' must be a non-empty list")

    seen_ids = set()
    for i, entry in enumerate(metrics):
        prefix = f"metrics[{i}]"
        entry_id = entry.get("id")
        if not entry_id:
            errors.append(f"{prefix}: missing 'id'")
            continue
        if entry_id in seen_ids:
            errors.append(f"{prefix}: duplicate id '{entry_id}'")
        seen_ids.add(entry_id)

        metric_key = entry.get("metric")
        info = METRIC_REGISTRY.get(metric_key)
        if info is None:
            errors.append(
                f"{prefix} ('{entry_id}'): unknown metric '{metric_key}', "
                f"expected one of {sorted(METRIC_REGISTRY)}"
            )
            continue

        errors.extend(_validate_entry_fields(prefix, entry_id, entry, info))
        errors.extend(_validate_entry_params(prefix, entry_id, entry, info))

    if errors:
        raise SpecError("\n".join(errors))


def _validate_entry_fields(prefix: str, entry_id: str, entry: dict, info: MetricInfo) -> list:
    errors = []
    if info.kind in REFERENCE_KINDS:
        reference = entry.get("reference")
        if reference not in VALID_REFERENCES:
            errors.append(
                f"{prefix} ('{entry_id}'): 'reference' must be one of "
                f"{sorted(VALID_REFERENCES)} for metric kind '{info.kind.value}' "
                f"(got {reference!r})"
            )
    elif info.kind == InputKind.PRECOMPUTE_CLAIMS:
        mode = entry.get("mode")
        if mode not in VALID_FACT_MODES:
            errors.append(
                f"{prefix} ('{entry_id}'): 'mode' must be one of "
                f"{sorted(VALID_FACT_MODES)} (got {mode!r})"
            )
        claim_source = entry.get("claim_source")
        if claim_source not in VALID_FACT_CLAIM_SOURCES:
            errors.append(
                f"{prefix} ('{entry_id}'): 'claim_source' must be one of "
                f"{sorted(VALID_FACT_CLAIM_SOURCES)} (got {claim_source!r})"
            )
        reference = entry.get("reference")
        if reference not in VALID_FACT_REFERENCES:
            errors.append(
                f"{prefix} ('{entry_id}'): 'reference' must be one of "
                f"{sorted(VALID_FACT_REFERENCES)} (got {reference!r})"
            )
    return errors


def _validate_entry_params(prefix: str, entry_id: str, entry: dict, info: MetricInfo) -> list:
    params = entry.get("params") or {}
    bad_keys = set(params) - info.allowed_params
    if bad_keys:
        return [
            f"{prefix} ('{entry_id}'): unknown params {sorted(bad_keys)} for "
            f"metric '{info.key}', allowed: {sorted(info.allowed_params)}"
        ]
    return []
