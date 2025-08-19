<!-- @format -->

# AdSolVe Models

This directory contains models for various tasks, e.g. summarisation

The directory <a href="example_data">example_data</a> contains small examples for testing and demonstration purposes.

## summary_generation_with_autoclass.py

This script implements summary generation with <a href="https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM">AutoModelForCausalLM</a> from the AutoClass module in the Hugging Face Transformers library. The script allows to load Huggingface models and perform summary generation tasks. It also includes an implementation of the summary generation from the paper [Temporal reasoning for timeline summarisation in social media](https://aclanthology.org/2025.acl-long.1362/) (Song et al., ACL 2025). An example of how to use the script is provided in <a href="generation_examples.ipynb">generation_examples.ipynb</a>.
