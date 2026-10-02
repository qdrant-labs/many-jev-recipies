# jevqu

Query understanding for Qdrant, powered by [Jev](https://openrouter.ai/typesafe/jev-1.13) (TypeSafe's System One decision model, reached through OpenRouter).

Jev answers many yes/no and pick-one questions about a piece of text in one cheap request. `jevqu` uses that to:

1. **Induce a taxonomy** from your collection alone (no queries or relevance labels needed): classes are named from the collection's own text and kept only if Jev can apply them reliably.
2. **Label** every point with those classes, stored in the payload (`classes`, `class_probs`).
3. **Understand each query**: which classes (and facets) does it belong to?
4. **Search** with that understanding as a filter or a score boost over hybrid (dense + sparse, RRF) search, with a fallback to plain search.

The taxonomy is stored in the collection (`_query_understanding`), so a collection is self-describing.

## Install

```bash
uv sync --extra dev                 # core + tests
uv sync --extra eval --extra notebook   # STaRK evaluation, notebooks
uv sync --extra chunk               # recipes/chunker baselines
export OPENROUTER_API_KEY=...       # the code reads the environment, not .env
```

Jev costs about $0.00002 per request; responses are cached on disk in `.jev_cache/`.

## Usage

```python
from qdrant_client import QdrantClient
from jevqu.jev import Jev
from jevqu.api import (create_query_understanding, set_query_understanding,
                       upload_points, understand_query, query_points)

client, jev = QdrantClient(...), Jev()

tax, report = create_query_understanding(client, "products", jev, sample=2000, embed=embed)  # 1. induce
set_query_understanding(client, "products", tax)                                             # 2. store + index
upload_points(client, "products", new_points, jev)                                           # 3. label on ingest
hits = query_points(client, "products", "waterproof hiking boots", jev,
                    dense=embed(["waterproof hiking boots"])[0], sparse=sparse_vec,
                    understanding="auto", limit=10)                                          # 4. search
```

`understanding` is one of `auto`, `filter`, `boost`, `off`. `understand_query` returns the matched classes and facets without searching.

`create_query_understanding(..., llm=LLM())` (from `jevqu.llm`) swaps in a chat model that writes class names and definitions, with Jev validating them. It exists as a baseline; the default path needs no chat model.

## Layout

| Path | What |
|---|---|
| `jevqu/api.py` | the five public calls |
| `jevqu/induce.py` | taxonomy induction and its gates |
| `jevqu/classify.py`, `cascade.py` | labeling points; local classifier cascade |
| `jevqu/understand.py`, `qquery.py` | query -> classes/facets; filter/boost search |
| `jevqu/jev.py`, `llm.py` | OpenRouter clients with on-disk cache and cost tracking |
| `recipes/` | standalone Jev recipes: `prune` (rerank/dedupe top results), `chunker` (semantic chunking), `wands` (shared WANDS loader) |
| `notebooks/` | C4 and WANDS demos, induction experiments |
| `scripts/` | STaRK collection builder and recall-loss evaluation |
| `docs/` | design notes and implementation plan |

## Recipe results

Measured with `typesafe/jev-1.13`; the numbers live in each recipe's notebook.

- **Rerank (WANDS, 240 held-out queries):** sorting the hybrid top 20 by Jev's relevance answer gives nDCG@10 +0.086 over hybrid and cuts irrelevant results in the top 10 from 102 to 38. It beats MiniLM-L-6 by +0.022 and bge-reranker-base by +0.039. It is not faster than local cross-encoders.
- **Chunker (five corpora):** at matched mean chunk size, recall is +0.05 to +0.08 over fixed, recursive and chonkie baselines.

Both are one dataset family each, so treat them as evidence, not a guarantee.

## Tests

```bash
uv run pytest        # tests/ and recipes/; no network, Jev is faked
```
