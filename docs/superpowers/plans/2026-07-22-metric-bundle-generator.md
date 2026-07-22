# Metric Bundle Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `evaluation_bundles/generate_bundle.py`, a CLI that reads a YAML spec selecting
metrics from `evaluation_bundles/metrics/` and writes a new `evaluation_bundles/<name>_evaluation.py`
bundle script, structured like the existing hand-written bundles.

**Architecture:** A static registry (`evaluation_bundles/metric_registry.py`) describes each metric
wrapper's import path, per-document input shape ("kind"), constructor params, and return shape. The
generator loads a YAML spec, validates every metric selection against the registry, then assembles
the output file as plain Python source via small line-generating functions (no templating engine —
PyYAML is the only new dependency, already present in `pyproject.toml`).

**Tech Stack:** Python 3.9+ (repo floor), PyYAML (already a dependency), pytest (dev-only, new).

## Global Constraints

- No changes to any existing bundle script (`court_case_summarisation_evaluation.py`,
  `social_media_summarisation_evaluation.py`, `chest_xray_report_generation.py`) or any class under
  `evaluation_bundles/metrics/`.
- `generate_bundle.py` and `metric_registry.py` must be importable and testable **without** torch,
  transformers, bert_score, spacy, or any other heavy ML dependency installed — they only write
  Python source referencing the metric classes, they never import the classes themselves. This
  matters because the current dev machine has none of those installed (verified: `pip list` shows
  no torch/transformers/pyyaml/pytest — only base Python).
- Generated bundle files must follow the existing convention exactly: `results[id] = {"document_level": [...], "mean": ...}` per metric, `sys.path.append(os.path.dirname(os.path.abspath(__file__)))` then bare `from metrics.x import Y` imports, `argparse` CLI at the bottom with `--llm_summaries`/`--gold_summaries`/`--combined_summaries`/`--output_file`.
- Python version floor: `requires-python = ">=3.9"` (from `pyproject.toml`) — do not use syntax newer than 3.9 in `metric_registry.py` / `generate_bundle.py` (e.g. no `X | Y` union syntax in runtime type hints outside `from __future__ import annotations`; use `Optional`/`typing.FrozenSet` etc. or add `from __future__ import annotations` at the top of both files to allow modern hint syntax safely).
- `.venv` is already in `.gitignore` — a local dev venv must not be committed.

## Deviations from the approved design doc (`docs/superpowers/specs/2026-07-22-metric-bundle-generator-design.md`)

Two refinements discovered while working out exact code generation, both strictly narrower/simpler
than the original design, no capability lost:

1. **`uses_posts` is auto-derived, not a spec field.** The design doc's YAML schema had an explicit
   `uses_posts: true/false` field. Requiring the user to keep it in sync with the metric list is a
   pure footgun (forget to set it → wrong CLI generated). The generator instead scans all metric
   entries for `reference: posts` and adds `--posts` automatically. Simpler spec, same behavior.
2. **`FactScorer` gets empty-dict placeholders for `llm_text`/`reference`, not deferred construction.**
   `FactScorer.__init__(self, llm_text, reference, min_claim=1, max_claim=10)` requires `llm_text`/
   `reference` positionally, but — verified by reading `evaluation_bundles/metrics/fact.py` in full —
   `self.llm_text`/`self.reference` are never read anywhere else in the class; `get_claims()` and
   `calculate_metric()` take their own `text`/`claims`/`reference` parameters instead. The existing
   hand-written `court_case_summarisation_evaluation.py` actually has a latent bug here: it calls
   `FactScorer(llm_text=llm_summaries, reference=gold_summaries, ...)` inside `__init__`, referencing
   `llm_summaries`/`gold_summaries` as bare names that only happen to exist as module-level globals
   set later in the `__main__` block — this only works by accident of that one script's exact
   execution order and would break if the class were ever instantiated another way (e.g. imported
   into a notebook). The generator avoids reproducing that bug: it passes `llm_text={}, reference={}`
   (dead placeholders) to `FactScorer`'s constructor in `__init__`, and gets the real data into
   `get_claims(...)` where it's actually used, inside `evaluate()`.

## File Structure

- `evaluation_bundles/metric_registry.py` — new. `InputKind` enum, `MetricInfo` dataclass, the
  `METRIC_REGISTRY` dict (11 entries, one per metric wrapper class).
- `evaluation_bundles/generate_bundle.py` — new. `load_spec`, `validate_spec`/`SpecError`, the line-
  generating helpers, `render_bundle`, `main`.
- `evaluation_bundles/tests/test_metric_registry.py` — new.
- `evaluation_bundles/tests/test_generate_bundle.py` — new.
- `evaluation_bundles/bundle_specs/social_media_example.yaml` — new, example spec reproducing
  `social_media_summarisation_evaluation.py`'s metric selection, for manual fidelity verification.
- `pyproject.toml` — modified: add a `dev` optional-dependency group with `pytest`.
- `README.md` — modified: short "Generating a new evaluation bundle" subsection.

---

### Task 1: Dev environment + `metric_registry.py`

**Files:**
- Create: `evaluation_bundles/metric_registry.py`
- Create: `evaluation_bundles/tests/test_metric_registry.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `InputKind` (str Enum: `NONE`, `SINGLE`, `LIST`, `DUAL`, `PRECOMPUTE_CLAIMS`, `BATCH`),
  `MetricInfo` (frozen dataclass: `key: str`, `module: str`, `class_name: str`, `kind: InputKind`,
  `returns_detail: bool`, `default_params: dict`, `allowed_params: FrozenSet[str]`,
  `fixed_params: dict`), `METRIC_REGISTRY: Dict[str, MetricInfo]` — used by every later task.

- [ ] **Step 1: Create a local dev venv with pytest and PyYAML**

```bash
cd /Users/sebastian/Documents/adsolve_utilities
python3 -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install pytest pyyaml
./.venv/bin/python -c "import pytest, yaml; print('ok', pytest.__version__)"
```

Expected: prints `ok 9.x.x` (or similar). `.venv` is already listed in `.gitignore`, so it will not
be committed.

- [ ] **Step 2: Add `pytest` as a dev dependency in `pyproject.toml`**

In `pyproject.toml`, under `[project.optional-dependencies]`, add a `dev` group next to the existing
`torch` group:

```toml
[project.optional-dependencies]
# Let users choose their Torch build (CPU/CUDA) explicitly:
torch = ["torch>=2.7"]
dev = ["pytest>=8.0"]
```

- [ ] **Step 3: Write the failing test file**

Create `evaluation_bundles/tests/test_metric_registry.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metric_registry import METRIC_REGISTRY, InputKind


EXPECTED_KEYS = {
    "rouge", "bertscore", "style_roberta", "evidence_appropriateness",
    "cross_nli", "intra_nli", "readability", "mhic", "fc", "fact", "greenscore",
}


def test_registry_has_all_expected_metrics():
    assert set(METRIC_REGISTRY) == EXPECTED_KEYS


def test_registry_keys_match_entry_key_field():
    for key, info in METRIC_REGISTRY.items():
        assert info.key == key


def test_registry_kinds_are_valid_input_kind_members():
    for info in METRIC_REGISTRY.values():
        assert isinstance(info.kind, InputKind)


def test_default_params_are_subset_of_allowed_params():
    for key, info in METRIC_REGISTRY.items():
        assert set(info.default_params) <= info.allowed_params, (
            f"{key}: default_params {set(info.default_params)} not a subset of "
            f"allowed_params {info.allowed_params}"
        )


def test_fact_has_fixed_placeholder_params():
    fact = METRIC_REGISTRY["fact"]
    assert fact.fixed_params == {"llm_text": {}, "reference": {}}


def test_greenscore_is_batch_kind_without_detail():
    greenscore = METRIC_REGISTRY["greenscore"]
    assert greenscore.kind == InputKind.BATCH
    assert greenscore.returns_detail is False


def test_fc_is_dual_kind_with_detail():
    fc = METRIC_REGISTRY["fc"]
    assert fc.kind == InputKind.DUAL
    assert fc.returns_detail is True


def test_intra_nli_and_readability_are_none_kind():
    assert METRIC_REGISTRY["intra_nli"].kind == InputKind.NONE
    assert METRIC_REGISTRY["readability"].kind == InputKind.NONE
```

- [ ] **Step 4: Run the test to verify it fails**

```bash
./.venv/bin/python -m pytest evaluation_bundles/tests/test_metric_registry.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'metric_registry'`.

- [ ] **Step 5: Write `evaluation_bundles/metric_registry.py`**

```python
"""Static metadata describing the metric wrapper classes in evaluation_bundles/metrics/.

Used by generate_bundle.py to validate bundle specs and render bundle scripts without
needing to import the (heavy, GPU-dependent) metric classes themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet


class InputKind(str, Enum):
    NONE = "none"
    SINGLE = "single"
    LIST = "list"
    DUAL = "dual"
    PRECOMPUTE_CLAIMS = "precompute_claims"
    BATCH = "batch"


@dataclass(frozen=True)
class MetricInfo:
    key: str
    module: str
    class_name: str
    kind: InputKind
    returns_detail: bool
    default_params: dict = field(default_factory=dict)
    allowed_params: FrozenSet[str] = field(default_factory=frozenset)
    fixed_params: dict = field(default_factory=dict)


METRIC_REGISTRY: Dict[str, MetricInfo] = {
    "rouge": MetricInfo(
        key="rouge",
        module="metrics.rouge",
        class_name="ROUGE",
        kind=InputKind.SINGLE,
        returns_detail=False,
        default_params={"configuration": "1", "metric": "p"},
        allowed_params=frozenset({"configuration", "metric"}),
    ),
    "bertscore": MetricInfo(
        key="bertscore",
        module="metrics.bertscore",
        class_name="BERTScore",
        kind=InputKind.SINGLE,
        returns_detail=False,
        default_params={"model_type": "microsoft/deberta-xlarge-mnli", "lang": "en"},
        allowed_params=frozenset({"model_type", "lang"}),
    ),
    "style_roberta": MetricInfo(
        key="style_roberta",
        module="metrics.style_roberta",
        class_name="StyleSimilarity",
        kind=InputKind.SINGLE,
        returns_detail=False,
        default_params={},
        allowed_params=frozenset(),
    ),
    "evidence_appropriateness": MetricInfo(
        key="evidence_appropriateness",
        module="metrics.evidence_appropriateness",
        class_name="EA",
        kind=InputKind.SINGLE,
        returns_detail=False,
        default_params={},
        allowed_params=frozenset({"hf_cache_dir", "hg_model_hub_name"}),
    ),
    "cross_nli": MetricInfo(
        key="cross_nli",
        module="metrics.cross_nli",
        class_name="Cross_NLI",
        kind=InputKind.SINGLE,
        returns_detail=True,
        default_params={},
        allowed_params=frozenset(),
    ),
    "intra_nli": MetricInfo(
        key="intra_nli",
        module="metrics.intra_nli",
        class_name="IntraNLI",
        kind=InputKind.NONE,
        returns_detail=False,
        default_params={},
        allowed_params=frozenset({"hf_cache_dir", "hg_model_hub_name"}),
    ),
    "readability": MetricInfo(
        key="readability",
        module="metrics.readability_metric",
        class_name="ReadabilityMetric",
        kind=InputKind.NONE,
        returns_detail=False,
        default_params={"readability_type": "flesch_kincaid"},
        allowed_params=frozenset({"readability_type"}),
    ),
    "mhic": MetricInfo(
        key="mhic",
        module="metrics.mhic",
        class_name="MHIC",
        kind=InputKind.LIST,
        returns_detail=False,
        default_params={},
        allowed_params=frozenset(),
    ),
    "fc": MetricInfo(
        key="fc",
        module="metrics.fc",
        class_name="FactualConsistency",
        kind=InputKind.DUAL,
        returns_detail=True,
        default_params={},
        allowed_params=frozenset(),
    ),
    "fact": MetricInfo(
        key="fact",
        module="metrics.fact",
        class_name="FactScorer",
        kind=InputKind.PRECOMPUTE_CLAIMS,
        returns_detail=True,
        default_params={"min_claim": 1, "max_claim": 30},
        allowed_params=frozenset({"min_claim", "max_claim"}),
        # FactScorer's constructor requires llm_text/reference positionally, but neither
        # attribute is read anywhere else in the class (get_claims/calculate_metric take
        # their own parameters) -- these are dead placeholders, not tunable inputs.
        fixed_params={"llm_text": {}, "reference": {}},
    ),
    "greenscore": MetricInfo(
        key="greenscore",
        module="metrics.greenscore",
        class_name="GreenScorer",
        kind=InputKind.BATCH,
        returns_detail=False,
        default_params={"model_name": "StanfordAIMI/GREEN-radllama2-7b"},
        allowed_params=frozenset({"model_name"}),
    ),
}
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
./.venv/bin/python -m pytest evaluation_bundles/tests/test_metric_registry.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add evaluation_bundles/metric_registry.py evaluation_bundles/tests/test_metric_registry.py pyproject.toml .venv 2>/dev/null; git reset .venv 2>/dev/null
git add evaluation_bundles/metric_registry.py evaluation_bundles/tests/test_metric_registry.py pyproject.toml
git commit -m "feat: add metric registry for bundle generation

Static metadata (import path, per-document input kind, constructor
params, return shape) for each metric wrapper in evaluation_bundles/metrics/,
consumed by the upcoming bundle generator."
```

---

### Task 2: Spec loading and validation

**Files:**
- Create: `evaluation_bundles/generate_bundle.py`
- Modify: `evaluation_bundles/tests/test_generate_bundle.py` (create)

**Interfaces:**
- Consumes: `METRIC_REGISTRY`, `InputKind` from `metric_registry` (Task 1).
- Produces: `SpecError(Exception)`, `load_spec(spec_path: str) -> dict`,
  `validate_spec(spec: dict) -> None` (raises `SpecError` listing every problem found, does nothing
  on success) — used by every later task.

- [ ] **Step 1: Write the failing tests**

Create `evaluation_bundles/tests/test_generate_bundle.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import generate_bundle as gb


MINIMAL_VALID_SPEC = """
name: my_use_case
metrics:
  - id: rouge_1
    metric: rouge
    reference: gold
"""


def _write_spec(tmp_path, content: str) -> str:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(content)
    return str(spec_path)


def test_load_spec_reads_yaml(tmp_path):
    path = _write_spec(tmp_path, MINIMAL_VALID_SPEC)
    spec = gb.load_spec(path)
    assert spec["name"] == "my_use_case"
    assert spec["metrics"][0]["metric"] == "rouge"


def test_validate_spec_accepts_minimal_valid_spec(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, MINIMAL_VALID_SPEC))
    gb.validate_spec(spec)  # should not raise


def test_validate_spec_rejects_unknown_metric_key(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, """
name: my_use_case
metrics:
  - id: made_up
    metric: not_a_real_metric
    reference: gold
"""))
    with pytest.raises(gb.SpecError, match="unknown metric 'not_a_real_metric'"):
        gb.validate_spec(spec)


def test_validate_spec_rejects_duplicate_ids(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, """
name: my_use_case
metrics:
  - id: rouge_1
    metric: rouge
    reference: gold
  - id: rouge_1
    metric: bertscore
    reference: gold
"""))
    with pytest.raises(gb.SpecError, match="duplicate id 'rouge_1'"):
        gb.validate_spec(spec)


def test_validate_spec_rejects_missing_reference_for_single_kind(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, """
name: my_use_case
metrics:
  - id: rouge_1
    metric: rouge
"""))
    with pytest.raises(gb.SpecError, match="'reference' must be one of"):
        gb.validate_spec(spec)


def test_validate_spec_rejects_bad_param_key(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, """
name: my_use_case
metrics:
  - id: rouge_1
    metric: rouge
    reference: gold
    params:
      not_a_real_param: 1
"""))
    with pytest.raises(gb.SpecError, match="unknown params"):
        gb.validate_spec(spec)


def test_validate_spec_rejects_invalid_name_format(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, """
name: "My Use Case"
metrics:
  - id: rouge_1
    metric: rouge
    reference: gold
"""))
    with pytest.raises(gb.SpecError, match="lowercase snake_case"):
        gb.validate_spec(spec)


def test_validate_spec_rejects_fact_missing_mode_and_claim_source(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, """
name: my_use_case
metrics:
  - id: conciseness
    metric: fact
    reference: gold
"""))
    with pytest.raises(gb.SpecError, match="'mode' must be one of"):
        gb.validate_spec(spec)


def test_validate_spec_collects_multiple_errors(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, """
name: my_use_case
metrics:
  - id: a
    metric: rouge
  - id: b
    metric: does_not_exist
"""))
    with pytest.raises(gb.SpecError) as exc_info:
        gb.validate_spec(spec)
    message = str(exc_info.value)
    assert "'reference' must be one of" in message
    assert "unknown metric 'does_not_exist'" in message
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./.venv/bin/python -m pytest evaluation_bundles/tests/test_generate_bundle.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'generate_bundle'`.

- [ ] **Step 3: Write `evaluation_bundles/generate_bundle.py` (spec loading/validation only)**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./.venv/bin/python -m pytest evaluation_bundles/tests/test_generate_bundle.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation_bundles/generate_bundle.py evaluation_bundles/tests/test_generate_bundle.py
git commit -m "feat: add bundle spec loading and validation

load_spec/validate_spec read a YAML metric selection and check every
entry against the metric registry, collecting all errors before
raising so a user sees every problem in one pass."
```

---

### Task 3: Naming, instance-dedup, and reference-binding helpers

**Files:**
- Modify: `evaluation_bundles/generate_bundle.py`
- Modify: `evaluation_bundles/tests/test_generate_bundle.py`

**Interfaces:**
- Consumes: `MINIMAL_VALID_SPEC`, `_write_spec` from Task 2's test file (same module).
- Produces: `to_pascal_case(snake: str) -> str`, `_uses_posts(spec: dict) -> bool`,
  `_instance_groups(spec: dict) -> tuple[dict, list]` (returns `(entry_id -> attr_name,
  [(attr_name, metric_key, params), ...])`), `_reference_expr(entry: dict, info: MetricInfo) -> str`,
  `_format_ctor_call(info: MetricInfo, entry_params: dict) -> str` — all used by Task 4/5's
  line generators.

- [ ] **Step 1: Write the failing tests**

Append to `evaluation_bundles/tests/test_generate_bundle.py`:

```python
def test_to_pascal_case():
    assert gb.to_pascal_case("my_use_case") == "MyUseCase"
    assert gb.to_pascal_case("court_case") == "CourtCase"
    assert gb.to_pascal_case("x") == "X"


def test_uses_posts_true_when_any_reference_is_posts():
    spec = {"metrics": [
        {"id": "a", "metric": "rouge", "reference": "gold"},
        {"id": "b", "metric": "mhic", "reference": "posts"},
    ]}
    assert gb._uses_posts(spec) is True


def test_uses_posts_false_when_no_reference_is_posts():
    spec = {"metrics": [{"id": "a", "metric": "rouge", "reference": "gold"}]}
    assert gb._uses_posts(spec) is False


def test_instance_groups_dedups_identical_metric_and_params():
    spec = {"metrics": [
        {"id": "fc_expert", "metric": "fc", "reference": "gold"},
        {"id": "fc_document", "metric": "fc", "reference": "posts"},
    ]}
    entry_attr, instances = gb._instance_groups(spec)
    assert entry_attr == {"fc_expert": "fc_expert", "fc_document": "fc_expert"}
    assert instances == [("fc_expert", "fc", {})]


def test_instance_groups_separate_instances_for_different_params():
    spec = {"metrics": [
        {"id": "rouge_1", "metric": "rouge", "reference": "gold", "params": {"configuration": "1"}},
        {"id": "rouge_l", "metric": "rouge", "reference": "gold", "params": {"configuration": "l"}},
    ]}
    entry_attr, instances = gb._instance_groups(spec)
    assert entry_attr == {"rouge_1": "rouge_1", "rouge_l": "rouge_l"}
    assert len(instances) == 2


def test_reference_expr_single_kind_gold():
    entry = {"reference": "gold"}
    info = gb.METRIC_REGISTRY["rouge"]
    assert gb._reference_expr(entry, info) == "gold_summary"


def test_reference_expr_single_kind_posts_joins():
    entry = {"reference": "posts"}
    info = gb.METRIC_REGISTRY["rouge"]
    assert gb._reference_expr(entry, info) == '" ".join(document_posts)'


def test_reference_expr_list_kind_gold_wraps():
    entry = {"reference": "gold"}
    info = gb.METRIC_REGISTRY["mhic"]
    assert gb._reference_expr(entry, info) == "[gold_summary]"


def test_reference_expr_list_kind_posts_passthrough():
    entry = {"reference": "posts"}
    info = gb.METRIC_REGISTRY["mhic"]
    assert gb._reference_expr(entry, info) == "document_posts"


def test_reference_expr_dual_kind_passthrough():
    entry = {"reference": "posts"}
    info = gb.METRIC_REGISTRY["fc"]
    assert gb._reference_expr(entry, info) == "document_posts"


def test_reference_expr_precompute_claims_posts_joins():
    entry = {"reference": "posts"}
    info = gb.METRIC_REGISTRY["fact"]
    assert gb._reference_expr(entry, info) == '" ".join(document_posts)'


def test_format_ctor_call_merges_default_and_entry_params():
    info = gb.METRIC_REGISTRY["rouge"]
    call = gb._format_ctor_call(info, {"metric": "f"})
    assert call == "ROUGE(configuration='1', metric='f')"


def test_format_ctor_call_includes_fact_fixed_params():
    info = gb.METRIC_REGISTRY["fact"]
    call = gb._format_ctor_call(info, {})
    assert call == "FactScorer(llm_text={}, reference={}, min_claim=1, max_claim=30)"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./.venv/bin/python -m pytest evaluation_bundles/tests/test_generate_bundle.py -v -k "pascal_case or uses_posts or instance_groups or reference_expr or format_ctor_call"
```

Expected: FAIL — `AttributeError: module 'generate_bundle' has no attribute 'to_pascal_case'` (and
similarly for the other new names).

- [ ] **Step 3: Add the helper functions to `evaluation_bundles/generate_bundle.py`**

Append these functions after the validation functions from Task 2:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./.venv/bin/python -m pytest evaluation_bundles/tests/test_generate_bundle.py -v
```

Expected: all tests PASS (21 total so far).

- [ ] **Step 5: Commit**

```bash
git add evaluation_bundles/generate_bundle.py evaluation_bundles/tests/test_generate_bundle.py
git commit -m "feat: add naming, dedup, and reference-binding helpers to bundle generator"
```

---

### Task 4: Per-document, claims-precompute, and batch line generators

**Files:**
- Modify: `evaluation_bundles/generate_bundle.py`
- Modify: `evaluation_bundles/tests/test_generate_bundle.py`

**Interfaces:**
- Consumes: `_reference_expr`, `METRIC_REGISTRY` (Task 3).
- Produces: `_loop_body_lines(entry: dict, info: MetricInfo, attr: str) -> list[str]`,
  `_claims_block_lines(entry: dict, attr: str) -> list[str]`,
  `_batch_block_lines(entry: dict, attr: str) -> list[str]` — all return flat, unindented lines;
  used by Task 5's `render_bundle`.

- [ ] **Step 1: Write the failing tests**

Append to `evaluation_bundles/tests/test_generate_bundle.py`:

```python
def test_loop_body_lines_none_kind_no_detail():
    entry = {"id": "intra", "metric": "intra_nli"}
    info = gb.METRIC_REGISTRY["intra_nli"]
    lines = gb._loop_body_lines(entry, info, "intra")
    assert lines == [
        "intra_score = self.intra.calculate_metric(llm_summary)",
        "results['intra']['document_level'].append(intra_score)",
    ]


def test_loop_body_lines_single_kind_no_detail():
    entry = {"id": "rouge_1", "metric": "rouge", "reference": "gold"}
    info = gb.METRIC_REGISTRY["rouge"]
    lines = gb._loop_body_lines(entry, info, "rouge_1")
    assert lines == [
        "rouge_1_score = self.rouge_1.calculate_metric(llm_summary, gold_summary)",
        "results['rouge_1']['document_level'].append(rouge_1_score)",
    ]


def test_loop_body_lines_dual_kind_with_detail_uses_shared_attr():
    entry = {"id": "fc_document", "metric": "fc", "reference": "posts"}
    info = gb.METRIC_REGISTRY["fc"]
    lines = gb._loop_body_lines(entry, info, "fc_expert")
    assert lines == [
        "fc_document_score, fc_document_detail = self.fc_expert.calculate_metric(llm_summary, document_posts)",
        "results['fc_document']['document_level'].append(fc_document_score)",
        "results['fc_document']['detail'].append(fc_document_detail)",
    ]


def test_loop_body_lines_precompute_claims_kind():
    entry = {
        "id": "conciseness", "metric": "fact", "mode": "recall",
        "claim_source": "llm", "reference": "gold",
    }
    info = gb.METRIC_REGISTRY["fact"]
    lines = gb._loop_body_lines(entry, info, "conciseness")
    assert lines == [
        'conciseness_score, conciseness_detail = self.conciseness.calculate_metric('
        'type="recall", claims=conciseness_claims[document_id], reference=gold_summary)',
        "results['conciseness']['document_level'].append(conciseness_score)",
        "results['conciseness']['detail'].append(conciseness_detail)",
    ]


def test_claims_block_lines_llm_source():
    entry = {"id": "conciseness", "claim_source": "llm"}
    lines = gb._claims_block_lines(entry, "conciseness")
    assert lines == [
        "print(\"Generating claims for 'conciseness'...\")",
        "conciseness_claims = self.conciseness.get_claims(llm_summaries)",
    ]


def test_claims_block_lines_gold_source():
    entry = {"id": "conciseness", "claim_source": "gold"}
    lines = gb._claims_block_lines(entry, "conciseness")
    assert lines[1] == "conciseness_claims = self.conciseness.get_claims(gold_summaries)"


def test_batch_block_lines_gold_reference():
    entry = {"id": "green_score", "reference": "gold"}
    lines = gb._batch_block_lines(entry, "green_score")
    assert lines == [
        "green_score_references = [gold_summaries[doc_id] for doc_id in results['document_ids']]",
        "green_score_hypotheses = [llm_summaries[doc_id] for doc_id in results['document_ids']]",
        "green_score_mean, green_score_std, green_score_score_list, green_score_summary, _ = "
        "self.green_score.calculate_metric(green_score_references, green_score_hypotheses)",
        "results['green_score']['document_level'] = green_score_score_list",
        "results['green_score']['mean'] = green_score_mean",
        "results['green_score']['std'] = green_score_std",
        "results['green_score']['summary'] = green_score_summary",
    ]


def test_batch_block_lines_posts_reference_joins():
    entry = {"id": "green_score", "reference": "posts"}
    lines = gb._batch_block_lines(entry, "green_score")
    assert lines[0] == (
        'green_score_references = [" ".join(posts[doc_id]) for doc_id in '
        "results['document_ids']]"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./.venv/bin/python -m pytest evaluation_bundles/tests/test_generate_bundle.py -v -k "loop_body_lines or claims_block_lines or batch_block_lines"
```

Expected: FAIL — `AttributeError: module 'generate_bundle' has no attribute '_loop_body_lines'`.

- [ ] **Step 3: Add the line generators to `evaluation_bundles/generate_bundle.py`**

Append after `_format_ctor_call`:

```python
def _loop_body_lines(entry: dict, info: MetricInfo, attr: str) -> list:
    entry_id = entry["id"]
    if info.kind == InputKind.NONE:
        call = f"self.{attr}.calculate_metric(llm_summary)"
    elif info.kind == InputKind.PRECOMPUTE_CLAIMS:
        ref_expr = _reference_expr(entry, info)
        call = (
            f'self.{attr}.calculate_metric(type="{entry["mode"]}", '
            f"claims={entry_id}_claims[document_id], reference={ref_expr})"
        )
    else:
        ref_expr = _reference_expr(entry, info)
        call = f"self.{attr}.calculate_metric(llm_summary, {ref_expr})"

    if info.returns_detail:
        return [
            f"{entry_id}_score, {entry_id}_detail = {call}",
            f"results['{entry_id}']['document_level'].append({entry_id}_score)",
            f"results['{entry_id}']['detail'].append({entry_id}_detail)",
        ]
    return [
        f"{entry_id}_score = {call}",
        f"results['{entry_id}']['document_level'].append({entry_id}_score)",
    ]


def _claims_block_lines(entry: dict, attr: str) -> list:
    entry_id = entry["id"]
    source = "llm_summaries" if entry["claim_source"] == "llm" else "gold_summaries"
    return [
        f"print(\"Generating claims for '{entry_id}'...\")",
        f"{entry_id}_claims = self.{attr}.get_claims({source})",
    ]


def _batch_block_lines(entry: dict, attr: str) -> list:
    entry_id = entry["id"]
    if entry["reference"] == "gold":
        refs_expr = "gold_summaries[doc_id]"
    else:
        refs_expr = '" ".join(posts[doc_id])'
    return [
        f"{entry_id}_references = [{refs_expr} for doc_id in results['document_ids']]",
        f"{entry_id}_hypotheses = [llm_summaries[doc_id] for doc_id in results['document_ids']]",
        f"{entry_id}_mean, {entry_id}_std, {entry_id}_score_list, {entry_id}_summary, _ = "
        f"self.{attr}.calculate_metric({entry_id}_references, {entry_id}_hypotheses)",
        f"results['{entry_id}']['document_level'] = {entry_id}_score_list",
        f"results['{entry_id}']['mean'] = {entry_id}_mean",
        f"results['{entry_id}']['std'] = {entry_id}_std",
        f"results['{entry_id}']['summary'] = {entry_id}_summary",
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./.venv/bin/python -m pytest evaluation_bundles/tests/test_generate_bundle.py -v
```

Expected: all tests PASS (29 total so far).

- [ ] **Step 5: Commit**

```bash
git add evaluation_bundles/generate_bundle.py evaluation_bundles/tests/test_generate_bundle.py
git commit -m "feat: add per-document, claims-precompute, and batch line generators"
```

---

### Task 5: `render_bundle` — full source assembly

**Files:**
- Modify: `evaluation_bundles/generate_bundle.py`
- Modify: `evaluation_bundles/tests/test_generate_bundle.py`

**Interfaces:**
- Consumes: everything from Tasks 2-4 (`validate_spec`, `to_pascal_case`, `_uses_posts`,
  `_instance_groups`, `_loop_body_lines`, `_claims_block_lines`, `_batch_block_lines`,
  `METRIC_REGISTRY`, `InputKind`).
- Produces: `render_bundle(spec: dict, spec_path: str) -> str` — the full generated bundle source,
  used by Task 6's `main`.

- [ ] **Step 1: Write the failing tests**

Append to `evaluation_bundles/tests/test_generate_bundle.py`:

```python
FULL_FEATURED_SPEC = """
name: full_featured
metrics:
  - id: intra
    metric: intra_nli
  - id: rouge_1
    metric: rouge
    reference: gold
  - id: mhic
    metric: mhic
    reference: posts
  - id: fc_expert
    metric: fc
    reference: gold
  - id: fc_document
    metric: fc
    reference: posts
  - id: conciseness
    metric: fact
    mode: recall
    claim_source: llm
    reference: gold
  - id: green_score
    metric: greenscore
    reference: gold
"""


def test_render_bundle_is_valid_python_for_full_featured_spec(tmp_path):
    spec = gb.load_spec(_write_spec(tmp_path, FULL_FEATURED_SPEC))
    gb.validate_spec(spec)
    source = gb.render_bundle(spec, "bundle_specs/full_featured.yaml")
    compile(source, "<generated:full_featured>", "exec")  # raises SyntaxError if invalid


def test_render_bundle_class_name_and_imports():
    spec = {"name": "my_use_case", "metrics": [
        {"id": "rouge_1", "metric": "rouge", "reference": "gold"},
    ]}
    source = gb.render_bundle(spec, "spec.yaml")
    assert "class MyUseCaseEvaluationBundle:" in source
    assert "from metrics.rouge import ROUGE" in source
    assert "self.rouge_1 = ROUGE(configuration='1', metric='p')" in source


def test_render_bundle_dedups_shared_instance_in_init():
    spec = {"name": "social", "metrics": [
        {"id": "fc_expert", "metric": "fc", "reference": "gold"},
        {"id": "fc_document", "metric": "fc", "reference": "posts"},
    ]}
    source = gb.render_bundle(spec, "spec.yaml")
    assert source.count("FactualConsistency()") == 1
    assert "self.fc_expert.calculate_metric(llm_summary, gold_summary)" in source
    assert "self.fc_expert.calculate_metric(llm_summary, document_posts)" in source


def test_render_bundle_omits_posts_cli_arg_when_unused():
    spec = {"name": "simple", "metrics": [
        {"id": "rouge_1", "metric": "rouge", "reference": "gold"},
    ]}
    source = gb.render_bundle(spec, "spec.yaml")
    assert "--posts" not in source
    assert "def evaluate(self, llm_summaries: dict, gold_summaries: dict) -> dict:" in source


def test_render_bundle_includes_posts_cli_arg_when_used():
    spec = {"name": "social", "metrics": [
        {"id": "mhic", "metric": "mhic", "reference": "posts"},
    ]}
    source = gb.render_bundle(spec, "spec.yaml")
    assert "--posts" in source
    assert "posts: dict = None) -> dict:" in source
    assert "document_posts = posts[document_id]" in source


def test_render_bundle_header_references_spec_path():
    spec = {"name": "simple", "metrics": [
        {"id": "rouge_1", "metric": "rouge", "reference": "gold"},
    ]}
    source = gb.render_bundle(spec, "bundle_specs/simple.yaml")
    assert source.startswith("# Generated by generate_bundle.py from bundle_specs/simple.yaml")


def test_render_bundle_final_mean_loop_excludes_batch_metrics():
    spec = {"name": "mixed", "metrics": [
        {"id": "rouge_1", "metric": "rouge", "reference": "gold"},
        {"id": "green_score", "metric": "greenscore", "reference": "gold"},
    ]}
    source = gb.render_bundle(spec, "spec.yaml")
    assert "for metric_id in ['rouge_1']:" in source
    assert "results['green_score']['mean'] = green_score_mean" in source
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./.venv/bin/python -m pytest evaluation_bundles/tests/test_generate_bundle.py -v -k render_bundle
```

Expected: FAIL — `AttributeError: module 'generate_bundle' has no attribute 'render_bundle'`.

- [ ] **Step 3: Add `render_bundle` and its remaining helpers to `evaluation_bundles/generate_bundle.py`**

Append after `_batch_block_lines`:

```python
def _import_lines(spec: dict) -> list:
    seen = set()
    lines = []
    for entry in spec["metrics"]:
        info = METRIC_REGISTRY[entry["metric"]]
        key = (info.module, info.class_name)
        if key not in seen:
            seen.add(key)
            lines.append(f"from {info.module} import {info.class_name}")
    return sorted(lines)


def _init_lines(instances) -> list:
    lines = []
    for attr, metric_key, params in instances:
        info = METRIC_REGISTRY[metric_key]
        lines.append(f"self.{attr} = {_format_ctor_call(info, params)}")
    return lines


def _results_seed_lines(spec: dict) -> list:
    lines = []
    for entry in spec["metrics"]:
        info = METRIC_REGISTRY[entry["metric"]]
        entry_id = entry["id"]
        if info.kind == InputKind.BATCH:
            lines.append(
                f"'{entry_id}': {{\"document_level\": [], \"mean\": None, "
                f'"std": None, "summary": None}},'
            )
        elif info.returns_detail:
            lines.append(f"'{entry_id}': {{\"document_level\": [], \"mean\": None, \"detail\": []}},")
        else:
            lines.append(f"'{entry_id}': {{\"document_level\": [], \"mean\": None}},")
    return lines


def _cli_lines(spec: dict, class_name: str, uses_posts: bool) -> list:
    name = spec["name"]
    lines = [
        'if __name__ == "__main__":',
        f'    parser = argparse.ArgumentParser(description="Evaluate {name}.")',
        "    parser.add_argument('--llm_summaries', type=str, help='Path to the LLM summaries JSON file.')",
        "    parser.add_argument('--gold_summaries', type=str, help='Path to the gold summaries JSON file.')",
        "    parser.add_argument('--combined_summaries', type=str, default=None, help='Path to the combined summaries JSON file (optional). If provided, it will be used instead of LLM summaries and gold summaries.')",
    ]
    if uses_posts:
        lines.append(
            "    parser.add_argument('--posts', type=str, required=True, help='Path to the posts JSON file (document_id -> list of source texts).')"
        )
    lines.append(
        f"    parser.add_argument('--output_file', type=str, default='{name}_evaluation_results.json', help='Path to save the evaluation results JSON file.')"
    )
    lines += [
        "    args = parser.parse_args()",
        "",
        "    if args.combined_summaries:",
        "        print(f\"Loading combined summaries from {args.combined_summaries}\")",
        "        with open(args.combined_summaries, 'r') as f:",
        "            combined_summaries = json.load(f)",
        '        llm_summaries = {key: value["summary"] for key, value in combined_summaries.items()}',
        '        gold_summaries = {key: value["reference"] for key, value in combined_summaries.items()}',
        "    elif args.llm_summaries and args.gold_summaries:",
        "        print(f\"Loading LLM summaries from {args.llm_summaries}\")",
        "        with open(args.llm_summaries, 'r') as f:",
        "            llm_summaries = json.load(f)",
        "        print(f\"Loading gold summaries from {args.gold_summaries}\")",
        "        with open(args.gold_summaries, 'r') as f:",
        "            gold_summaries = json.load(f)",
        "    else:",
        '        raise ValueError("Either --combined_summaries or both --llm_summaries and --gold_summaries must be provided.")',
        "",
    ]
    if uses_posts:
        lines += [
            "    print(f\"Loading posts from {args.posts}\")",
            "    with open(args.posts, 'r') as f:",
            "        posts = json.load(f)",
            "",
        ]
    call_args = "llm_summaries, gold_summaries, posts" if uses_posts else "llm_summaries, gold_summaries"
    lines += [
        f'    print("Creating evaluation bundle for {name}.")',
        f"    evaluation_bundle = {class_name}()",
        '    print("Evaluating LLM.")',
        f"    results = evaluation_bundle.evaluate({call_args})",
        "",
        "    print(f\"Saving evaluation results to {args.output_file}\")",
        "    output_file = args.output_file",
        "    with open(output_file, 'w') as f:",
        "        json.dump(results, f, indent=4)",
    ]
    return lines


def render_bundle(spec: dict, spec_path: str) -> str:
    class_name = f"{to_pascal_case(spec['name'])}EvaluationBundle"
    uses_posts = _uses_posts(spec)
    entry_attr, instances = _instance_groups(spec)

    loop_entries = [e for e in spec["metrics"] if METRIC_REGISTRY[e["metric"]].kind != InputKind.BATCH]
    precompute_entries = [e for e in loop_entries if METRIC_REGISTRY[e["metric"]].kind == InputKind.PRECOMPUTE_CLAIMS]
    batch_entries = [e for e in spec["metrics"] if METRIC_REGISTRY[e["metric"]].kind == InputKind.BATCH]
    non_batch_ids = [e["id"] for e in loop_entries]

    lines = [
        f"# Generated by generate_bundle.py from {spec_path}",
        "# Regenerating from that spec will overwrite manual edits to this file.",
        "import argparse",
        "import sys",
        "import os",
        "import json",
        "from tqdm import tqdm",
        "import numpy as np",
        "sys.path.append(os.path.dirname(os.path.abspath(__file__)))",
        "",
    ]
    lines.extend(_import_lines(spec))
    lines += ["", "", f"class {class_name}:", "    def __init__(self):"]
    for line in _init_lines(instances):
        lines.append(f"        {line}")
    lines.append("")

    evaluate_sig = "    def evaluate(self, llm_summaries: dict, gold_summaries: dict"
    evaluate_sig += ", posts: dict = None) -> dict:" if uses_posts else ") -> dict:"
    lines.append(evaluate_sig)
    lines.append("        results = {")
    for line in _results_seed_lines(spec):
        lines.append(f"            {line}")
    lines.append("            'document_ids': list(llm_summaries.keys())")
    lines.append("        }")
    lines.append("")

    for entry in precompute_entries:
        attr = entry_attr[entry["id"]]
        for line in _claims_block_lines(entry, attr):
            lines.append(f"        {line}")
        lines.append("")

    lines.append("        for document_id in tqdm(results['document_ids']):")
    lines.append("            llm_summary = llm_summaries[document_id]")
    lines.append("            gold_summary = gold_summaries[document_id]")
    if uses_posts:
        lines.append("            document_posts = posts[document_id]")
    lines.append("")
    for entry in loop_entries:
        info = METRIC_REGISTRY[entry["metric"]]
        attr = entry_attr[entry["id"]]
        for line in _loop_body_lines(entry, info, attr):
            lines.append(f"            {line}")
        lines.append("")

    for entry in batch_entries:
        attr = entry_attr[entry["id"]]
        for line in _batch_block_lines(entry, attr):
            lines.append(f"        {line}")
        lines.append("")

    lines.append(f"        for metric_id in {non_batch_ids!r}:")
    lines.append("            results[metric_id]['mean'] = float(np.mean(results[metric_id]['document_level']))")
    lines.append("")
    lines.append("        return results")
    lines.append("")
    lines.append("")
    lines.extend(_cli_lines(spec, class_name, uses_posts))
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./.venv/bin/python -m pytest evaluation_bundles/tests/test_generate_bundle.py -v
```

Expected: all tests PASS (36 total so far).

- [ ] **Step 5: Commit**

```bash
git add evaluation_bundles/generate_bundle.py evaluation_bundles/tests/test_generate_bundle.py
git commit -m "feat: assemble full bundle source in render_bundle"
```

---

### Task 6: CLI entrypoint, example spec, README, final wiring

**Files:**
- Modify: `evaluation_bundles/generate_bundle.py`
- Modify: `evaluation_bundles/tests/test_generate_bundle.py`
- Create: `evaluation_bundles/bundle_specs/social_media_example.yaml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `load_spec`, `validate_spec`, `SpecError`, `render_bundle` (Tasks 2-5).
- Produces: `main(argv: list = None) -> int` — the script's CLI entrypoint.

- [ ] **Step 1: Write the failing tests**

Append to `evaluation_bundles/tests/test_generate_bundle.py` (add `import subprocess` and
`import sys` near the top of the file alongside the existing imports):

```python
import subprocess


def test_main_writes_generated_file(tmp_path):
    spec_path = _write_spec(tmp_path, MINIMAL_VALID_SPEC)
    exit_code = gb.main(["--spec", spec_path, "--output-dir", str(tmp_path)])
    assert exit_code == 0
    output_path = tmp_path / "my_use_case_evaluation.py"
    assert output_path.exists()
    content = output_path.read_text()
    assert "class MyUseCaseEvaluationBundle:" in content
    compile(content, "<generated>", "exec")


def test_main_returns_nonzero_and_writes_nothing_on_invalid_spec(tmp_path, capsys):
    spec_path = _write_spec(tmp_path, """
name: my_use_case
metrics:
  - id: bad
    metric: not_a_real_metric
""")
    exit_code = gb.main(["--spec", spec_path, "--output-dir", str(tmp_path)])
    assert exit_code == 1
    assert not (tmp_path / "my_use_case_evaluation.py").exists()
    captured = capsys.readouterr()
    assert "unknown metric" in captured.err


def test_cli_subprocess_generates_file(tmp_path):
    spec_path = _write_spec(tmp_path, MINIMAL_VALID_SPEC)
    generator = Path(__file__).resolve().parents[1] / "generate_bundle.py"
    result = subprocess.run(
        [sys.executable, str(generator), "--spec", spec_path, "--output-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "my_use_case_evaluation.py").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./.venv/bin/python -m pytest evaluation_bundles/tests/test_generate_bundle.py -v -k "main or subprocess"
```

Expected: FAIL — `AttributeError: module 'generate_bundle' has no attribute 'main'`.

- [ ] **Step 3: Add `main` to `evaluation_bundles/generate_bundle.py`**

Append at the end of the file:

```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate an evaluation_bundles/<name>_evaluation.py script from a YAML spec."
    )
    parser.add_argument("--spec", type=str, required=True, help="Path to the YAML bundle spec.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to write the generated bundle script into.",
    )
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    try:
        validate_spec(spec)
    except SpecError as exc:
        print(f"Invalid spec '{args.spec}':\n{exc}", file=sys.stderr)
        return 1

    source = render_bundle(spec, args.spec)
    output_path = Path(args.output_dir) / f"{spec['name']}_evaluation.py"
    output_path.write_text(source)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the full test suite to verify everything passes**

```bash
./.venv/bin/python -m pytest evaluation_bundles/tests/ -v
```

Expected: all tests PASS (39 total).

- [ ] **Step 5: Create the example spec**

Create `evaluation_bundles/bundle_specs/social_media_example.yaml`, reproducing
`social_media_summarisation_evaluation.py`'s metric selection to prove real-world fidelity:

```yaml
name: social_media_example
metrics:
  - id: mhic
    metric: mhic
    reference: posts
  - id: intra_nli
    metric: intra_nli
  - id: fc_expert
    metric: fc
    reference: gold
  - id: fc_document
    metric: fc
    reference: posts
  - id: style_similarity
    metric: style_roberta
    reference: gold
  - id: bert_score
    metric: bertscore
    reference: gold
```

- [ ] **Step 6: Manually verify the example spec generates a sensible bundle**

```bash
./.venv/bin/python evaluation_bundles/generate_bundle.py \
  --spec evaluation_bundles/bundle_specs/social_media_example.yaml \
  --output-dir /tmp
cat /tmp/social_media_example_evaluation.py
```

Read the output and confirm by eye:
- `class SocialMediaExampleEvaluationBundle:` with one `__init__` line per metric (5 lines: mhic,
  intra_nli, fc_expert reused for fc_document, style_similarity, bert_score — `FactualConsistency()`
  should appear exactly once).
- `evaluate(self, llm_summaries: dict, gold_summaries: dict, posts: dict = None) -> dict:` (posts
  present, since `mhic`/`fc_document` reference it).
- The per-document loop calls `self.fc_expert.calculate_metric(llm_summary, gold_summary)` for
  `fc_expert` and `self.fc_expert.calculate_metric(llm_summary, document_posts)` for `fc_document`.
- The bottom CLI block has `--posts` as a required argument.

This mirrors the hand-written `social_media_summarisation_evaluation.py` metric-for-metric. Discard
the generated file afterwards — it's a manual check, not something to commit:

```bash
rm /tmp/social_media_example_evaluation.py
```

- [ ] **Step 7: Update `README.md`**

Add this subsection right after the existing "## Evaluation Bundles" paragraph in `README.md`:

```markdown
### Generating a new evaluation bundle

Instead of hand-writing a new bundle script, describe your metric selection in a YAML spec under
`evaluation_bundles/bundle_specs/` and generate it:

```bash
python evaluation_bundles/generate_bundle.py --spec evaluation_bundles/bundle_specs/my_use_case.yaml
```

This writes `evaluation_bundles/my_use_case_evaluation.py`, structured like the existing bundles.
See `evaluation_bundles/metric_registry.py` for the list of available metrics and
`evaluation_bundles/bundle_specs/social_media_example.yaml` for an example spec.
```

- [ ] **Step 8: Run the full test suite one more time**

```bash
./.venv/bin/python -m pytest evaluation_bundles/tests/ -v
```

Expected: all 39 tests PASS.

- [ ] **Step 9: Commit**

```bash
git add evaluation_bundles/generate_bundle.py evaluation_bundles/tests/test_generate_bundle.py \
        evaluation_bundles/bundle_specs/social_media_example.yaml README.md
git commit -m "feat: add generate_bundle.py CLI entrypoint, example spec, and docs

Completes the bundle generator: python evaluation_bundles/generate_bundle.py
--spec <spec.yaml> now writes a working evaluation_bundles/<name>_evaluation.py,
matching the structure of the existing hand-written bundles."
```

---

## Post-implementation notes for the user

- Running a generated bundle still requires the real ML dependencies (torch, transformers,
  bert_score, spacy, rouge, readability_lxml, green_score) installed separately — `generate_bundle.py`
  itself never imports them, only the dev deps (`pytest`, `pyyaml`) are needed to develop/test the
  generator.
- `evaluation_bundles/tests/` has no `__init__.py` by design, so `pyproject.toml`'s
  `exclude = ["tests*", ...]` packaging rule and pytest's default rootdir-based test collection both
  work without extra configuration.
