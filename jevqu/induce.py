from __future__ import annotations
import math, re
from collections import Counter
import numpy as np
from jevqu.jev import MAX_QUESTIONS, noul
from jevqu.metrics import ndcg_at_k, paired_bootstrap
from jevqu.schema import ClassDef, Facet
from jevqu.text import state_from_payload

STOP = set("the a an and or of for with to in on by is are this that it its from as at be".split())

def _tokens(t: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP and len(w) > 2]

def _kmeans(x: np.ndarray, k: int, seed: int, iters: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    cent = x[rng.choice(len(x), size=min(k, len(x)), replace=False)]
    for _ in range(iters):
        lab = np.argmax(x @ cent.T, axis=1)
        for j in range(len(cent)):
            m = x[lab == j]
            if len(m):
                cent[j] = m.mean(axis=0)
    return np.argmax(x @ cent.T, axis=1)

def propose(texts: list[str], vectors: np.ndarray, k: int = 40, top_terms: int = 4, seed: int = 0) -> list[ClassDef]:
    lab = _kmeans(np.asarray(vectors, dtype=float), k, seed)
    docs = [set(_tokens(t)) for t in texts]
    df = Counter(w for d in docs for w in d)
    out, seen = [], set()
    for j in sorted(set(lab)):
        members = [docs[i] for i in range(len(docs)) if lab[i] == j]
        cf = Counter(w for d in members for w in d)
        score = {w: (c / len(members)) * math.log(len(docs) / df[w]) for w, c in cf.items() if c >= 2 or len(members) < 3}
        terms = [w for w, _ in sorted(score.items(), key=lambda kv: (-kv[1], -cf[kv[0]], kv[0]))[:top_terms]]
        cid = "-".join(terms)
        if not terms or cid in seen:
            continue
        seen.add(cid)
        out.append(ClassDef(cid, " ".join(terms).title(), f"Items primarily about {', '.join(terms)}",
                            f"Items that mention {', '.join(terms)} only incidentally, or are about something else"))
    return out

def _q(c: ClassDef, side: str) -> dict:
    lead = "Does this item belong to" if side == "point" else "Does the query ask for items in"
    return noul(f"{lead} the class `{c.name}`?", c.definition, c.exclusions)

def coverage(cands: list[ClassDef], states: list[dict], jev) -> dict[str, list[float]]:
    out = {c.id: [] for c in cands}
    for s in states:
        ans = jev.ask(s, {c.id: _q(c, "point") for c in cands})
        for c in cands:
            out[c.id].append(ans[c.id]["noul"])
    return out

def demand(cands: list[ClassDef], queries: list[str], jev, above: float = 0.6) -> dict[str, list[int]]:
    out = {c.id: [] for c in cands}
    for i, q in enumerate(queries):
        ans = jev.ask(state_from_payload({"query": q}), {c.id: _q(c, "query") for c in cands})
        for c in cands:
            if ans[c.id]["noul"] >= above:
                out[c.id].append(i)
    return out

def retrieval_value(cid: str, qidx: list[int], queries: list[str], relevant: list[set], run_fn) -> tuple[float, float, float]:
    qidx = [i for i in qidx if relevant[i]]
    if not qidx:
        return (0.0, 0.0, 0.0)
    base = [ndcg_at_k(run_fn(queries[i], None), relevant[i], 10) for i in qidx]
    filt = [ndcg_at_k(run_fn(queries[i], cid), relevant[i], 10) for i in qidx]
    return paired_bootstrap(base, filt)

def select(cands, cov, dem, values, facets: list[Facet], *, n_queries: int, band=(0.005, 0.40), min_demand: float = 0.02) -> list[ClassDef]:
    facet_values = {v.lower() for f in facets for v in f.values}
    kept = []
    for c in cands:
        frac = sum(p >= 0.5 for p in cov[c.id]) / max(1, len(cov[c.id]))
        dem_frac = len(dem[c.id]) / max(1, n_queries)
        dup = c.name.lower() in facet_values or any(t in facet_values for t in c.id.split("-"))
        mean, lo, _ = values[c.id]
        if band[0] <= frac <= band[1] and dem_frac >= min_demand and not dup and lo > 0:
            kept.append(c)
    return kept

def induce(texts, vectors, states, queries, relevant, run_fn, jev, facets, max_classes: int = MAX_QUESTIONS - 1,
           rounds: int = 5, k: int = 40, band=(0.005, 0.40), min_demand: float = 0.02, label_fn=None) -> tuple[list[ClassDef], list[dict]]:
    max_classes = min(max_classes, MAX_QUESTIONS - max(1, len(facets)))  # level-1 classes + injection/facet question must fit one request
    kept, report, residual = [], [], list(range(len(texts)))
    for _ in range(rounds):
        if len(kept) >= max_classes or not residual:
            break
        cands = [c for c in propose([texts[i] for i in residual], np.asarray(vectors)[residual], k=k)
                 if c.id not in {x.id for x in kept}]
        if not cands:
            break
        # ponytail: coverage/demand stay sequential (~15k Jev calls); parallelize if induction latency is measured as a problem
        cov, dem = coverage(cands, states, jev), demand(cands, queries, jev)
        if label_fn is not None:
            for c in cands:
                label_fn(c.id, [i for i, p in enumerate(cov[c.id]) if p >= 0.5])
        values = {c.id: retrieval_value(c.id, dem[c.id], queries, relevant, run_fn) for c in cands}
        chosen = select(cands, cov, dem, values, facets, n_queries=len(queries), band=band, min_demand=min_demand)[: max_classes - len(kept)]
        for c in cands:
            report.append({"id": c.id, "coverage": sum(p >= 0.5 for p in cov[c.id]) / len(states),
                           "demand": len(dem[c.id]), "value": values[c.id], "kept": c in chosen})
        kept += chosen
        explained = {i for c in chosen for i, p in enumerate(cov[c.id]) if p >= 0.5}
        residual = [i for i in residual if i not in explained]
        if not chosen:
            break
    return kept, report
