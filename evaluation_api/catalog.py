"""Translates public metric ids into bundle-spec entries.

METRIC_CATALOG names the combinations of (implementation, reference, params)
that this service offers. This module turns a list of those names into exactly
the spec a person would otherwise hand-write under bundle_specs/, so everything
downstream -- validate_spec, render_bundle -- is unchanged.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Set

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_bundles"))

from metric_registry import METRIC_CATALOG, METRIC_REGISTRY


class UnknownMetricError(Exception):
    """Raised when a request names metric ids that are not in the catalog."""

    def __init__(self, unknown: List[str]) -> None:
        self.unknown = unknown
        super().__init__(f"Unknown metric ids: {', '.join(unknown)}")


def _entry(metric_id: str):
    return METRIC_CATALOG[metric_id]


def expand(metric_ids: List[str]) -> List[dict]:
    """Turn catalog ids into bundle-spec metric entries, preserving order."""
    unknown = [m for m in metric_ids if m not in METRIC_CATALOG]
    if unknown:
        raise UnknownMetricError(unknown)

    entries: List[dict] = []
    for metric_id in metric_ids:
        entry_spec = _entry(metric_id)
        entry: dict = {"id": metric_id, "metric": entry_spec.metric}
        if entry_spec.reference is not None:
            entry["reference"] = entry_spec.reference
        if entry_spec.params:
            entry["params"] = dict(entry_spec.params)
        entry.update(entry_spec.extra)
        entries.append(entry)
    return entries


def build_spec(name: str, metric_ids: List[str]) -> dict:
    """Build a full bundle spec, the same shape load_spec() returns for YAML."""
    return {"name": name, "metrics": expand(metric_ids)}


def all_ids() -> List[str]:
    """Every selectable metric id, in catalog order.

    Callers go through this rather than importing METRIC_CATALOG directly: this
    module is what puts evaluation_bundles/ on sys.path, so importing the
    registry elsewhere depends on import order.
    """
    return list(METRIC_CATALOG)


def required_references(metric_ids: List[str]) -> Set[str]:
    """Which reference data the run needs: subset of {'gold', 'posts'}."""
    required = set()
    for metric_id in metric_ids:
        reference = _entry(metric_id).reference
        if reference in {"gold", "posts"}:
            required.add(reference)
    return required


def environments_used(metric_ids: List[str]) -> Set[str]:
    """Which container environments the run needs."""
    return {METRIC_REGISTRY[_entry(m).metric].environment for m in metric_ids}


def requires_of(metric_id: str) -> Optional[str]:
    """The reference a single metric needs, for the API listing."""
    reference = _entry(metric_id).reference
    return reference if reference in {"gold", "posts"} else None


def describe(metric_id: str, available: bool) -> dict:
    """One row of GET /metrics."""
    entry_spec = _entry(metric_id)
    return {
        "id": metric_id,
        "label": entry_spec.label,
        "description": entry_spec.description,
        "requires": requires_of(metric_id),
        "environment": METRIC_REGISTRY[entry_spec.metric].environment,
        "available": available,
    }
