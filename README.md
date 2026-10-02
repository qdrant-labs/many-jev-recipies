# Jev recipes for Qdrant

Small, tested recipes that use [Jev](https://openrouter.ai/typesafe/jev-1.13) (TypeSafe's System One decision model, reached through OpenRouter) next to a Qdrant collection. Jev answers many yes/no and pick-one questions about a piece of text in one cheap request (about $0.00002 each, cached on disk), so each recipe is "ask Jev a batch of questions, then apply a plain function to the probabilities".

| Recipe | Use it to | Code |
|---|---|---|
| [Query understanding](#query-understanding-jevqu) | Turn a query into classes and facets, then filter or boost the search | `jevqu/` |
| [Rerank / prune](#rerank--prune) | Re-order and de-duplicate the top results of any search | `recipes/prune/` |
| [Semantic chunker](#semantic-chunker) | Split documents at topic changes before embedding | `recipes/chunker/` |

The recipes share the Jev client (`jevqu/jev.py`) and nothing else, so you can take one without the others.

## Install

```bash
uv sync --extra dev                       # core + tests
uv sync --extra chunk                     # chunker baselines (chonkie, chunking-evaluation)
uv sync --extra eval --extra notebook     # STaRK evaluation, notebooks
export OPENROUTER_API_KEY=...             # the code reads the environment, not .env
```

## Query understanding (`jevqu`)

Induce a taxonomy from your collection alone (no queries or relevance labels), label every point with it, and understand each query against it.

```python
from qdrant_client import QdrantClient
from jevqu.jev import Jev
from jevqu.api import (create_query_understanding, set_query_understanding,
                       upload_points, understand_query, query_points)

client, jev = QdrantClient(...), Jev()

tax, report = create_query_understanding(client, "products", jev, sample=2000, embed=embed)  # induce
set_query_understanding(client, "products", tax)                                             # store + index
upload_points(client, "products", new_points, jev)                                           # label on ingest
hits = query_points(client, "products", "waterproof hiking boots", jev,
                    dense=embed(["waterproof hiking boots"])[0], sparse=sparse_vec,
                    understanding="auto", limit=10)                                          # search
```

`understanding` is `auto`, `filter`, `boost` or `off`. `understand_query` returns the matched classes and facets without searching. The taxonomy is stored in the collection (`_query_understanding`).

`create_query_understanding(..., llm=LLM())` (from `jevqu.llm`) lets a chat model write the class names and Jev validate them. It is a baseline; the default path needs no chat model.

Evaluation: `scripts/build_stark_collection.py` and `scripts/run_eval.py` (STaRK-Amazon recall-loss curve). Demos: `notebooks/`.

## Rerank / prune

One Jev request per query reads the top 20 results and answers, per result, "is it relevant?" and "is it a near-duplicate of a higher-ranked result?". Thresholds apply afterwards, so one set of requests serves any sweep.

```python
from recipes.prune.prune import relevance, judge, prune

rel = relevance(jev, query, texts)                      # rerank: sort the top 20 by this
rel, dup = judge(jev, query, texts)                     # or also get duplicate probabilities
ids, backfilled = prune(ids, rel, dup, rel_min=0.5, dup_max=0.5, keep=10)
```

Measured on WANDS (240 held-out queries): sorting the hybrid top 20 by Jev's relevance gives nDCG@10 +0.086 over hybrid and cuts irrelevant results in the top 10 from 102 to 38. It beats MiniLM-L-6 by +0.022 and bge-reranker-base by +0.039, but it is not faster than local cross-encoders. The notebook (`recipes/prune/prune.ipynb`) also covers variety (MMR with Jev relevance as the relevance term) and sweeps the thresholds.

## Semantic chunker

Jev reads windows of numbered sentences and judges, per gap, whether the next sentence starts a new topic. Cutting is a pure function of those probabilities, so one set of requests serves every threshold and size bound.

```python
from recipes.chunker.chunker import sentences, gap_probs, cut, chunks

spans = sentences(text)
probs = gap_probs([text[a:b] for a, b in spans], jev)
starts = cut(probs, sizes=[b - a for a, b in spans], threshold=0.5, min_size=200, max_size=800)
pieces = chunks(text, spans, starts)                    # exact slices of `text`
```

Measured on five corpora from Chroma's chunking-evaluation, at matched mean chunk size: recall +0.05 to +0.08 over fixed, recursive and chonkie baselines, with precision never lower. See `recipes/chunker/chunker.ipynb`.

Both results are one dataset family each, so treat them as evidence, not a guarantee.

## Layout

| Path | What |
|---|---|
| `jevqu/jev.py`, `llm.py` | OpenRouter clients with on-disk cache and cost tracking |
| `jevqu/api.py` | the five public query-understanding calls |
| `jevqu/induce.py`, `classify.py`, `cascade.py` | taxonomy induction, point labeling, local classifier cascade |
| `jevqu/understand.py`, `qquery.py` | query to classes/facets; filter/boost search |
| `recipes/wands.py` | shared WANDS loader and hybrid index used by the recipe notebooks |
| `docs/` | design notes and implementation plan |

## Tests

```bash
uv run pytest        # tests/ and recipes/; Jev is faked
```
