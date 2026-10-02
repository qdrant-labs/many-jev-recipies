"""Post-retrieval prune with Jev. One request per query: Jev reads the top results in rank order and
answers, for each one, whether it is relevant to the query and whether it near-duplicates a result
ranked above it. Thresholds are applied afterwards in `prune`, so one request serves any sweep."""
from __future__ import annotations
from jevqu.jev import noul

TOP, KEEP, SNIP = 20, 10, 300  # results Jev reads, results kept, characters of each result shown

def questions(n_items: int) -> dict[str, dict]:
    qs = {}
    for n in range(1, n_items + 1):
        qs[f"rel_{n}"] = noul(f"Is result {n} relevant to the query?",
                              f"Someone searching for the query would accept result {n} as an answer",
                              f"Result {n} is not what the query asks for")
        if n > 1:
            qs[f"dup_{n}"] = noul(f"Is result {n} a near-duplicate of a result ranked above it (results 1 to {n - 1})?",
                                  f"Result {n} is the same item as a result ranked above it, or a trivial variant of that item",
                                  f"Result {n} is a different item from every result ranked above it")
    return qs

def judge(jev, query: str, texts: list[str]) -> tuple[list[float], list[float]]:
    """(relevance, duplicate) probability per result, in rank order; result 1 has duplicate probability 0."""
    state = {"query": query, "results": [{"result": n, "text": t[:SNIP]} for n, t in enumerate(texts, 1)]}
    ans = jev.ask(state, questions(len(texts)))
    rel = [ans[f"rel_{n}"]["noul"] for n in range(1, len(texts) + 1)]
    dup = [0.0] + [ans[f"dup_{n}"]["noul"] for n in range(2, len(texts) + 1)]
    return rel, dup

def relevance(jev, query: str, texts: list[str], batch: int = TOP) -> list[float]:
    """Jev's relevance answer for every result, in rank order: relevance questions only, `batch` results
    per request, numbered 1..batch within each request (the format the pilot checked)."""
    out = []
    for s in range(0, len(texts), batch):
        part = texts[s:s + batch]
        state = {"query": query, "results": [{"result": n, "text": t[:SNIP]} for n, t in enumerate(part, 1)]}
        ans = jev.ask(state, {k: q for k, q in questions(len(part)).items() if k.startswith("rel_")})
        out += [ans[f"rel_{n}"]["noul"] for n in range(1, len(part) + 1)]
    return out

def prune(ids: list, rel: list[float], dup: list[float], rel_min: float = 0.5, dup_max: float = 0.5,
          keep: int = KEEP) -> tuple[list, bool]:
    """First `keep` results with rel >= rel_min and dup < dup_max. If too few survive, the page is
    backfilled with the dropped results, most relevant first (so a relevant duplicate comes back before
    an irrelevant result); the flag says whether that happened."""
    ok = [r >= rel_min and d < dup_max for r, d in zip(rel, dup)]
    survivors = [i for i, k in zip(ids, ok) if k]
    dropped = [i for i, _ in sorted(((i, r) for i, r, k in zip(ids, rel, ok) if not k), key=lambda x: -x[1])]
    return (survivors + dropped)[:keep], len(survivors) < keep
