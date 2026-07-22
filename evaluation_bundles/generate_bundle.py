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

    # Guard against overwriting non-generated files
    if output_path.exists():
        existing_content = output_path.read_text()
        if not existing_content.startswith("# Generated by generate_bundle.py"):
            message = (
                f"Refusing to overwrite '{output_path}': it exists and was not generated by this tool "
                f"(no '# Generated by generate_bundle.py' header). Rename or remove it first if you really want to replace it."
            )
            print(message, file=sys.stderr)
            return 1

    output_path.write_text(source)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
