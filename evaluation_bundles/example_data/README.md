# Example data

Synthetic example (3 documents) of the input shape the AdSoLve evaluation API expects. Every file is a JSON object keyed by the same `document_id`:

| File | Shape | Description | Needed when |
|---|---|---|---|
| `posts.json` | `{document_id: [str, ...]}` | The input / source text (posts, utterances, document chunks) | A metric uses `reference: posts` |
| `llm_summaries.json` | `{document_id: str}` | The LLM-generated output to evaluate | Always |
| `gold_summaries.json` | `{document_id: str}` | Reference output, e.g. a human-written gold summary | A metric uses `reference: gold` |

The ids in `llm_summaries.json` decide which documents get evaluated, so each
of them must also appear in the other files you pass.

**`posts` values must always be a list of strings, never a single string** (`[str, str, ...]`). If your input is one long text, either wrap in a list with one item (`["<text>"]`) or,
preferably, split it into chunks such as paragraphs (`["<chunk1>", "<chunk2>", ...]`) because some metrics compare the output against each item separately and truncate items longer than ~512 tokens.
