from __future__ import annotations
import re
from jevqu.jev import choice, noul
from jevqu.schema import Taxonomy, Understanding
from jevqu.text import state_from_payload

def local_facets(query: str, tax: Taxonomy) -> dict[str, tuple[str, float]]:
    q = f" {re.sub(r'[^a-z0-9 ]', ' ', query.lower())} "
    out = {}
    for f in tax.facets:
        for v in f.values:
            if f" {v.lower()} " in q:
                out[f.field] = (v, 1.0); break
    return out

def understand(query: str, tax: Taxonomy, jev, stored_model: str | None = None, labeled_ids: set[str] | None = None) -> Understanding:
    if stored_model is not None and stored_model != tax.model:
        raise ValueError(f"labels were written by {stored_model}, taxonomy expects {tax.model}; relabel first")
    if labeled_ids is not None:
        level1_ids = {c.id for c in tax.level1()}
        if not level1_ids <= labeled_ids:
            raise ValueError(f"stored labels predate taxonomy {tax.version}; relabel first")
    facets = local_facets(query, tax)
    qs = {c.id: noul(f"Does the query ask for items in the class `{c.name}`?", c.definition, c.exclusions)
          for c in tax.level1()}
    for f in tax.facets:
        if f.field not in facets:
            opts = {v: None for v in f.values[:250]}
            opts["none"] = "The query does not constrain on this field"
            qs[f"facet_{f.field}"] = choice(f"Which `{f.field}` does the query constrain on, if any?", opts)
    state = state_from_payload({"query": query})
    ans = jev.ask(state, qs)
    classes = [(c.id, ans[c.id]["noul"]) for c in tax.level1()]
    for f in tax.facets:
        a = ans.get(f"facet_{f.field}")
        if a and a["choice"] != "none" and a["confidence"] >= tax.thresholds.boost_above:
            facets[f.field] = (a["choice"], a["confidence"])
    return Understanding(classes=classes, facets=facets, taxonomy_version=tax.version)
