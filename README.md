<!-- @format -->

# AdSoLve Utilities

This repository gathers code that has been developed to support the AdSoLve project, like evaluation tools and generation models.

## Evaluation Bundles

Resources to evaluate LLMs for different use cases are gathered in the directory <a href="evaluation_bundles">evaluation_bundles</a>.

### Generating a new evaluation bundle

Instead of hand-writing a new bundle script, describe your metric selection in a YAML spec under
`evaluation_bundles/bundle_specs/` and generate it:

```bash
python evaluation_bundles/generate_bundle.py --spec evaluation_bundles/bundle_specs/my_use_case.yaml
```

This writes `evaluation_bundles/my_use_case_evaluation.py`, structured like the existing bundles.
See `evaluation_bundles/metric_registry.py` for the list of available metrics and
`evaluation_bundles/bundle_specs/social_media_example.yaml` for an example spec.

## Models

Deep learning models, e.g. for summary generation, are gathered in the directory <a href="models">models</a>.
