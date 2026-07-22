# Metric Bundle Generator — Design

## Problem

`evaluation_bundles/` contains hand-written scripts (`court_case_summarisation_evaluation.py`,
`social_media_summarisation_evaluation.py`, `chest_xray_report_generation.py`) that each wire up
a fixed set of classes from `evaluation_bundles/metrics/` into a bundle: a class with an
`evaluate()` method that loops over documents, calls each metric's `calculate_metric(...)`, and
assembles a `results` dict; plus an `argparse` CLI at the bottom for loading inputs and saving
results. Every new use case today means copy-pasting one of these files and hand-editing the
metric selection and wiring.

Goal: a way to pick existing metrics from `evaluation_bundles/metrics/` and generate a new bundle
script in the same style, without hand-wiring imports/instantiation/loop code each time.

## Metric interface survey

The metric wrapper classes are heterogeneous in what `calculate_metric` needs per document:

- **none** — only the LLM text: `IntraNLI`, `ReadabilityMetric`
- **single reference** — LLM text + one reference string: `ROUGE`, `BERTScore`,
  `StyleSimilarity`, `EA` (evidence appropriateness), `Cross_NLI`
- **list of references** — LLM text + a list of reference strings: `MHIC`
- **dual** — accepts a reference that is either a string or a list, passthrough:
  `FactualConsistency` (`fc.py`) — already reused in the social-media bundle as both `fc_expert`
  (vs. gold, string) and `fc_document` (vs. posts, list)
- **precompute_claims** — needs a claims-generation pass over one input set *before* the
  per-document loop, then verifies claims per document: `FactScorer` (`fact.py`)
- **batch** — operates once over the whole corpus (ordered lists), not per document:
  `GreenScorer` (`greenscore.py`)

Return shape also varies: most return a bare float; `Cross_NLI`, `FactualConsistency`, and
`FactScorer` return `(score, detail_dict)`.

## Architecture

Three new pieces, no change to existing bundles or metric classes:

1. **`evaluation_bundles/metric_registry.py`** — static metadata per metric: import path/class,
   `input_kind` (one of the six kinds above), default constructor params, `returns_detail`.
2. **`evaluation_bundles/generate_bundle.py`** — CLI:
   `python generate_bundle.py --spec <spec.yaml> [--output-dir evaluation_bundles/]`.
   Reads a YAML spec, validates every metric selection against the registry (unknown metric key,
   missing required field for that kind, duplicate `id`, invalid constructor param — checked via
   `inspect.signature` on the real class), then renders the bundle file via plain string
   templates (no new dependency; PyYAML is already a project dependency).
3. **`evaluation_bundles/bundle_specs/`** — where YAML specs live, checked into git so bundle
   generation is reproducible and diffable.

Output is a normal `evaluation_bundles/<name>_evaluation.py` file, structurally identical to the
existing hand-written bundles (same class-with-`evaluate()` shape, same
`results[id] = {"document_level": [...], "mean": ...}` convention, same CLI flags at the bottom).
Running a generated bundle works exactly like running an existing one today.

## Data contract

Each document has up to two input sources, loaded by the generated bundle's CLI:

- `gold_summary` — single reference string (via `--gold_summaries` / `--combined_summaries`,
  matching today's pattern)
- `posts` — list of reference strings, loaded via `--posts` (JSON dict `document_id -> [str]`),
  only added to the generated CLI if some selected metric binds to it

For `single`-kind metrics bound to `posts`, the generator auto-joins the list with `" ".join(...)`.
For `list`-kind metrics bound to `gold`, the generator auto-wraps it as `[gold_summary]`. `dual`
kind (`fc`) passes through whichever source is bound, unmodified — matching how
`FactualConsistency.calculate_metric` already branches on `isinstance(reference_text, str | list)`.

## Metric registry

| key | class | kind | returns_detail | default params |
|---|---|---|---|---|
| `rouge` | `ROUGE` | single | no | `configuration="1", metric="p"` |
| `bertscore` | `BERTScore` | single | no | `model_type="microsoft/deberta-xlarge-mnli", lang="en"` |
| `style_roberta` | `StyleSimilarity` | single | no | — |
| `evidence_appropriateness` | `EA` | single | no | class defaults |
| `cross_nli` | `Cross_NLI` | single | yes | — |
| `intra_nli` | `IntraNLI` | none | no | class defaults |
| `readability` | `ReadabilityMetric` | none | no | `readability_type="flesch_kincaid"` |
| `mhic` | `MHIC` | list | no | — |
| `fc` | `FactualConsistency` | dual | yes | — |
| `fact` | `FactScorer` | precompute_claims | yes | `min_claim=1, max_claim=30` |
| `greenscore` | `GreenScorer` | batch | no (has mean/std/summary instead) | `model_name="StanfordAIMI/GREEN-radllama2-7b"` |

Note: `EA` and `IntraNLI` default to a hardcoded `hf_cache_dir="/import/nlp-datasets/LLMs"`
inside their own class (pre-existing, not introduced by this design). Spec `params` can override
it like any other constructor arg.

## YAML spec schema

```yaml
name: my_use_case            # -> writes evaluation_bundles/my_use_case_evaluation.py
                              # -> class MyUseCaseEvaluationBundle
uses_posts: false             # true if any metric below binds to posts; adds --posts CLI arg

metrics:
  - id: rouge_1                # unique result key in the bundle's output dict
    metric: rouge               # registry key
    reference: gold              # gold | posts   (only for single/list/dual kinds)
    params:                      # optional: overrides registry defaults, passed to __init__
      configuration: "1"
      metric: p

  - id: bert_score
    metric: bertscore
    reference: gold

  - id: mhic
    metric: mhic
    reference: posts

  - id: fc_expert                # same underlying metric, reused twice under different ids —
    metric: fc                   # exactly like today's court_case/social_media bundles
    reference: gold
  - id: fc_document
    metric: fc
    reference: posts

  - id: conciseness
    metric: fact
    mode: recall                 # recall | precision (fact-specific)
    claim_source: llm            # llm | gold — which side's claims get extracted
    reference: gold               # the other side, verified against the claims
    params:
      min_claim: 1
      max_claim: 30

  - id: green_score
    metric: greenscore
    reference: gold
```

## Generated bundle structure

**Instance deduplication:** several metrics load heavy transformer models onto GPU (`fc`,
`evidence_appropriateness`, `intra_nli`, `mhic`, `greenscore`, ...). If a spec reuses the same
`metric` key with identical `params` under two different `id`s (e.g. `fc_expert`/`fc_document`),
the generator instantiates it once in `__init__` and both `id`s call that shared instance —
avoiding duplicate model loads. Different `params` for the same metric key get separate instances.

**`__init__`:** one attribute per unique `(metric, params)` group.

**`evaluate(llm_summaries, gold_summaries, posts=None)`:**
1. Seed `results` per `id`: `{"document_level": [], "mean": None}`, plus `"detail": []` if
   `returns_detail`.
2. If any metric is `precompute_claims`: run its claims-generation block
   (`self.<attr>.get_claims(llm_summaries or gold_summaries)`, per `claim_source`) before the loop.
3. Main `for document_id in tqdm(...)` loop: resolve each non-batch metric's reference from
   `gold_summary`/`posts` per its binding, call `calculate_metric(...)`, append to
   `results[id]["document_level"]` (and `["detail"]` if applicable).
4. After the loop: any `batch`-kind metric (`greenscore`) runs once over the full ordered lists,
   populating `document_level` from its per-item score list, plus `mean`/`std`/`summary`.
5. Final pass computes `mean = float(np.mean(...))` for every non-batch metric.

**CLI block:** identical to existing bundles — `--llm_summaries`/`--gold_summaries`/
`--combined_summaries`, plus `--posts` only if `uses_posts: true`, `--output_file` defaulting to
`<name>_evaluation_results.json`.

**Header comment:** generated files start with a comment noting the source spec path — since
regenerating from that spec overwrites the file, hand edits won't survive a re-run.

## Out of scope

- No change to any existing bundle script or any class in `metrics/`.
- No interactive wizard — spec files only.
- No support for metric kinds beyond the six identified above; adding a genuinely new kind (e.g.
  a metric needing three inputs) requires extending the registry's kind enum and the generator's
  templates, not just adding a registry row.
