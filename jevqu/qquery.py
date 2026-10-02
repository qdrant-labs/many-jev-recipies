from __future__ import annotations
from qdrant_client import models
from jevqu.schema import Taxonomy, Understanding

MODES = {"auto", "filter", "boost", "off"}

def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; must be one of {sorted(MODES)}")

def _sure(u: Understanding, tax: Taxonomy) -> list[str]:
    return [cid for cid, p in u.classes if p >= tax.thresholds.filter_above]

def _boosts(u: Understanding, tax: Taxonomy, mode: str) -> list[tuple[str, str, float]]:
    t = tax.thresholds
    ceiling = t.filter_above if mode == "auto" else float("inf")
    cands = [("classes", cid, p) for cid, p in u.classes] + [(f, v, p) for f, (v, p) in u.facets.items()]
    return [(key, v, p) for key, v, p in cands if t.boost_above <= p < ceiling]

def build_filter(u: Understanding | None, tax: Taxonomy, mode: str) -> models.Filter | None:
    _check_mode(mode)
    if u is None or mode in ("off", "boost"):
        return None
    must = [models.FieldCondition(key="classes", match=models.MatchValue(value=cid)) for cid in _sure(u, tax)]
    must += [models.FieldCondition(key=f, match=models.MatchValue(value=v)) for f, (v, p) in u.facets.items()
             if p >= tax.thresholds.filter_above]
    return models.Filter(must=must) if must else None

def build_formula(u: Understanding | None, tax: Taxonomy, mode: str) -> models.FormulaQuery | None:
    _check_mode(mode)
    if u is None or mode in ("off", "filter"):
        return None
    boosts = _boosts(u, tax, mode)
    if not boosts:
        return None
    terms = [models.MultExpression(mult=[tax.thresholds.boost_scale * p,
             models.FieldCondition(key=key, match=models.MatchValue(value=v))]) for key, v, p in boosts]
    return models.FormulaQuery(formula=models.SumExpression(sum=["$score", *terms]))

def _search(client, collection, dense, sparse, flt, formula, limit, prefetch_limit):
    legs = [models.Prefetch(query=dense, using=None if sparse is None else "dense", filter=flt, limit=prefetch_limit)]
    if sparse is not None:
        legs.append(models.Prefetch(query=sparse, using="sparse", filter=flt, limit=prefetch_limit))
    fused = models.Prefetch(prefetch=legs, query=models.FusionQuery(fusion=models.Fusion.RRF), limit=prefetch_limit) \
        if len(legs) > 1 else legs[0]
    # boosted or not, one path: RRF has many exact ties, and two paths break them differently (measured 0.25-0.9 nDCG
    # points), which biases every comparison of a boost against plain search
    return client.query_points(collection, prefetch=[fused], query=formula or models.FormulaQuery(formula="$score"),
                               limit=limit).points

def run(client, collection: str, dense, sparse, u: Understanding | None, tax: Taxonomy,
        limit: int = 10, mode: str = "auto", prefetch_limit: int = 100) -> list[models.ScoredPoint]:
    flt, formula = build_filter(u, tax, mode), build_formula(u, tax, mode)
    hits = _search(client, collection, dense, sparse, flt, formula, limit, prefetch_limit)
    if flt is not None and len(hits) < limit:  # fallback: fill the page from the unfiltered ranking
        seen = {h.id for h in hits}
        extra = _search(client, collection, dense, sparse, None, formula, limit, prefetch_limit)
        hits += [h for h in extra if h.id not in seen][: limit - len(hits)]
    return hits
