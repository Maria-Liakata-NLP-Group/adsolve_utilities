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


def to_pascal_case(snake: str) -> str:
    return "".join(part.capitalize() for part in snake.split("_"))


def _uses_posts(spec: dict) -> bool:
    return any(entry.get("reference") == "posts" for entry in spec["metrics"])


def _instance_groups(spec: dict):
    """Group metric entries sharing the same (metric, params) so identical, possibly
    GPU-heavy, metric instances are only constructed once and reused across ids -
    e.g. `fc_expert`/`fc_document` in the social-media bundle both reuse one
    FactualConsistency instance today."""
    group_attr: dict = {}
    entry_attr: dict = {}
    instances: list = []
    for entry in spec["metrics"]:
        metric_key = entry["metric"]
        params = entry.get("params", {})
        gkey = (metric_key, tuple(sorted(params.items())))
        if gkey not in group_attr:
            attr_name = entry["id"]
            group_attr[gkey] = attr_name
            instances.append((attr_name, metric_key, params))
        entry_attr[entry["id"]] = group_attr[gkey]
    return entry_attr, instances


def _reference_expr(entry: dict, info: MetricInfo) -> str:
    reference = entry["reference"]
    source_expr = {
        "gold": "gold_summary",
        "posts": "document_posts",
        "llm": "llm_summary",
    }[reference]
    if info.kind == InputKind.SINGLE and reference == "posts":
        return '" ".join(document_posts)'
    if info.kind == InputKind.LIST and reference == "gold":
        return "[gold_summary]"
    if info.kind == InputKind.LIST and reference == "llm":
        return "[llm_summary]"
    if info.kind == InputKind.PRECOMPUTE_CLAIMS and reference == "posts":
        return '" ".join(document_posts)'
    return source_expr


def _format_ctor_call(info: MetricInfo, entry_params: dict) -> str:
    merged = {**info.fixed_params, **info.default_params, **entry_params}
    args = ", ".join(f"{k}={v!r}" for k, v in merged.items())
    return f"{info.class_name}({args})"
