from __future__ import annotations
import datetime as dt
import numpy as np
from qdrant_client import models
from jevqu.classify import label_collection
from jevqu.induce import induce
from jevqu.qquery import run
from jevqu.schema import Facet, Taxonomy, Thresholds, Understanding
from jevqu.store import load_taxonomy, save_taxonomy
from jevqu.text import state_from_payload
from jevqu.understand import understand

def _sample(client, collection: str, n: int):
    pts, _ = client.scroll(collection, limit=n, with_payload=True, with_vectors=True)
    return pts

def create_query_understanding(client, collection: str, jev, sample: int = 2000, queries=None, relevant=None,
                               facets: list[Facet] | None = None, embed=None, run_fn=None, **gates) -> tuple[Taxonomy, list[dict]]:
    pts = _sample(client, collection, sample)
    texts = [" ".join(str(v) for v in p.payload.values() if isinstance(v, str)) for p in pts]
    vectors = embed(texts) if embed else np.array([p.vector if isinstance(p.vector, list) else p.vector["dense"] for p in pts])
    states = [state_from_payload(p.payload) for p in pts]
    queries, relevant, facets = queries or [], relevant or [], facets or []
    label_fn = None
    if run_fn is None:
        ids, members = [p.id for p in pts], {}
        using = None if isinstance(pts[0].vector, list) else "dense"
        def label_fn(cid: str, idx: list[int]) -> None:
            members[cid] = [ids[i] for i in idx]
        def run_fn(q: str, cid: str | None):  # value gate inside the sample: all sampled points vs the candidate's members
            if embed is None:
                return []
            flt = models.Filter(must=[models.HasIdCondition(has_id=members[cid] if cid else ids)])
            return [h.id for h in client.query_points(collection, query=embed([q])[0].tolist(), using=using,
                                                      query_filter=flt, limit=10).points]
    classes, report = induce(texts, vectors, states, queries, relevant, run_fn, jev, facets, label_fn=label_fn, **gates)
    tax = Taxonomy(version=dt.datetime.now(dt.UTC).strftime("v%Y%m%d%H%M%S"), model=jev.model, classes=classes,
                   facets=facets, thresholds=Thresholds())
    return tax, report

def set_query_understanding(client, collection: str, tax: Taxonomy) -> None:
    save_taxonomy(client, collection, tax)
    if "classes" not in (client.get_collection(collection).payload_schema or {}):
        client.create_payload_index(collection, "classes", models.PayloadSchemaType.KEYWORD)

def upload_points(client, collection: str, points, jev) -> list:
    client.upsert(collection, points)
    tax = load_taxonomy(client, collection)
    return label_collection(client, collection, tax, jev) if tax else []

def _stored_labels(client, collection: str) -> tuple[str | None, set[str] | None]:
    # ponytail: checks one labeled point; scan all if collections ever mix label sets
    pts, _ = client.scroll(collection, limit=1, with_payload=["class_model", "class_probs"], scroll_filter=models.Filter(
        must=[models.FieldCondition(key="needs_review", match=models.MatchValue(value=False))]))
    return (pts[0].payload["class_model"], set(pts[0].payload["class_probs"])) if pts else (None, None)

def understand_query(client, collection: str, query: str, jev) -> Understanding:
    tax = load_taxonomy(client, collection)
    if tax is None:
        raise RuntimeError(f"no query understanding set for {collection}")
    stored_model, labeled_ids = _stored_labels(client, collection)
    return understand(query, tax, jev, stored_model=stored_model, labeled_ids=labeled_ids)

def query_points(client, collection: str, query: str, jev, dense, sparse=None, understanding: str = "auto", limit: int = 10):
    tax = load_taxonomy(client, collection)
    u = None
    if understanding != "off" and tax is not None:
        stored_model, labeled_ids = _stored_labels(client, collection)
        u = understand(query, tax, jev, stored_model=stored_model, labeled_ids=labeled_ids)
    return run(client, collection, dense, sparse, u, tax or Taxonomy("none", jev.model, [], []), limit=limit, mode=understanding)
