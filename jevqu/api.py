from __future__ import annotations
import datetime as dt
import random
import numpy as np
from qdrant_client import models
from jevqu.classify import label_collection
from jevqu.induce import induce, induce_llm
from jevqu.qquery import run
from jevqu.schema import Facet, Taxonomy, Thresholds, Understanding
from jevqu.store import load_taxonomy, save_taxonomy
from jevqu.text import state_from_payload
from jevqu.understand import understand

def _sample(client, collection: str, n: int, seed: int = 0):
    ids, offset = [], None
    while True:  # ids only, no payload: cheap even at 55k points
        pts, offset = client.scroll(collection, limit=10_000, offset=offset, with_payload=False, with_vectors=False)
        ids += [p.id for p in pts]
        if offset is None:
            break
    pick = random.Random(seed).sample(ids, min(n, len(ids)))
    return client.retrieve(collection, ids=pick, with_payload=True, with_vectors=True)

def create_query_understanding(client, collection: str, jev, sample: int = 2000, facets: list[Facet] | None = None,
                               embed=None, judge_groups: bool = False, encoder=None, llm=None, **gates) -> tuple[Taxonomy, list[dict]]:
    """Induce a taxonomy from the collection alone (no queries): see jevqu.induce.induce.
    encoder ranks candidate class names; defaults to embed, then to bge-small. With llm (jevqu.llm.LLM), the LLM writes
    the class names, definitions and parents instead, and Jev validates them (jevqu.induce.induce_llm)."""
    pts = _sample(client, collection, sample)
    if not pts:
        raise ValueError(f"sample of {collection} is empty")
    states = [state_from_payload(p.payload) for p in pts]
    texts = [" ".join(str(v) for v in st.values() if isinstance(v, str)) for st in states]
    vectors = embed(texts) if embed else np.array([p.vector if isinstance(p.vector, list) else p.vector["dense"] for p in pts])
    if llm is not None:  # the LLM writes the classes, Jev validates them (Jev chooses, it cannot write)
        classes, report = induce_llm(texts, vectors, states, jev, llm, facets or [], judge_groups=judge_groups, **gates)
    else:
        classes, report = induce(texts, vectors, states, jev, facets or [], judge_groups=judge_groups,
                                 encoder=encoder or embed, **gates)
    tax = Taxonomy(version=dt.datetime.now(dt.UTC).strftime("v%Y%m%d%H%M%S"), model=jev.model, classes=classes,
                   facets=facets or [], thresholds=Thresholds())
    return tax, report

def set_query_understanding(client, collection: str, tax: Taxonomy) -> None:
    save_taxonomy(client, collection, tax)
    schema = client.get_collection(collection).payload_schema or {}
    if "classes" not in schema:
        client.create_payload_index(collection, "classes", models.PayloadSchemaType.KEYWORD)
    for f in tax.facets:  # facet matches become hard filters
        if f.field not in schema:
            client.create_payload_index(collection, f.field, models.PayloadSchemaType.KEYWORD)

def upload_points(client, collection: str, points, jev, workers: int = 8) -> list:
    client.upsert(collection, points)
    tax = load_taxonomy(client, collection)
    return label_collection(client, collection, tax, jev, ids=[p.id for p in points], workers=workers) if tax else []

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
