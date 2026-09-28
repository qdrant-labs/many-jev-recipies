import numpy as np
import pytest
from qdrant_client import QdrantClient, models
from jevqu.api import _sample, create_query_understanding, query_points, set_query_understanding, understand_query, upload_points
from jevqu.jev import FakeJev
from jevqu.schema import ClassDef, Facet, Taxonomy, Thresholds

class TermJev(FakeJev):
    """Yes (0.9) when a word of the class name, between backticks in the question, appears in the state text."""
    def ask(self, state, questions):
        text = " ".join(str(v) for v in state.values()).lower()
        out = {}
        for qid, q in questions.items():
            name = q["instructions"].split("`")[1].lower().split() if "`" in q["instructions"] else []
            out[qid] = {"type": "noul", "noul": 0.9 if any(w in text for w in name) else 0.1}
        return out

def _client():
    c = QdrantClient(":memory:")
    c.create_collection("p", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT))
    c.create_payload_index("p", "classes", models.PayloadSchemaType.KEYWORD)
    return c

def test_end_to_end_with_fake_jev():
    c = _client()
    pts = [models.PointStruct(id=i, vector=v, payload={"text": t}) for i, (v, t) in enumerate(
        [([1.0, 0.0], "hiking boots trail"), ([0.9, 0.1], "trail boots waterproof"),
         ([0.0, 1.0], "frying pan steel"), ([0.1, 0.9], "steel pan nonstick")])]
    c.upsert("p", pts)
    jev = TermJev()
    embed = lambda texts: np.array([[1.0, 0.0] if "boot" in t or "trail" in t else [0.0, 1.0] for t in texts])
    tax, report = create_query_understanding(c, "p", jev, sample=4, queries=["waterproof hiking footwear"],
                                             relevant=[{0, 1}], embed=embed, k=2, band=(0.0, 1.0), min_demand=0.0)
    assert len(tax.classes) == 1 and "boots" in tax.classes[0].id and report
    set_query_understanding(c, "p", tax)
    assert upload_points(c, "p", pts, jev) == []
    labeled = {p.id: p.payload for p in c.scroll("p", limit=10, with_payload=True)[0]}
    assert sorted(i for i, p in labeled.items() if p["classes"]) == [0, 1]
    u = understand_query(c, "p", "trail boots", jev)
    assert u.taxonomy_version == tax.version
    hits = query_points(c, "p", "trail boots", jev, dense=[1.0, 0.0], understanding="auto", limit=2)
    assert sorted(h.id for h in hits) == [0, 1]

def test_sample_is_a_reproducible_random_sample_not_the_lowest_ids():
    c = QdrantClient(":memory:")
    c.create_collection("p", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT))
    c.upsert("p", [models.PointStruct(id=i, vector=[1.0, 0.0], payload={}) for i in range(100)])
    ids1 = {p.id for p in _sample(c, "p", 10)}
    ids2 = {p.id for p in _sample(c, "p", 10)}
    assert ids1 == ids2
    assert ids1 != set(range(10))

def test_set_query_understanding_indexes_facet_fields():
    c = _client()
    tax = Taxonomy("v1", "jev-1.13.0", [ClassDef("boots", "Boots", "d", "e")], [Facet("brand", ["Nike"])], Thresholds())
    calls = []
    real = c.create_payload_index
    c.create_payload_index = lambda collection, field_name, *a, **k: (calls.append(field_name), real(collection, field_name, *a, **k))[1]
    set_query_understanding(c, "p", tax)
    assert "brand" in calls

def test_upload_points_only_labels_the_newly_uploaded_points():
    c = _client()
    tax = Taxonomy("v1", "jev-1.13.0", [ClassDef("boots", "Boots", "d", "e")], [], Thresholds())
    set_query_understanding(c, "p", tax)
    stale = models.PointStruct(id=0, vector=[1.0, 0.0], payload={"text": "old unlabeled point"})
    c.upsert("p", [stale])
    calls = []
    class Spy(FakeJev):
        def ask(self, state, questions):
            calls.append(state); return super().ask(state, questions)
    new = models.PointStruct(id=1, vector=[1.0, 0.0], payload={"text": "trail boots"})
    upload_points(c, "p", [new], Spy(nouls={"boots": 0.9}))
    assert calls == [{"text": "trail boots"}]

def test_create_query_understanding_validates_arguments():
    c = _client()
    c.upsert("p", [models.PointStruct(id=0, vector=[1.0, 0.0], payload={"text": "boots"})])
    c.create_collection("empty", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT))
    embed = lambda texts: np.array([[1.0, 0.0] for _ in texts])
    with pytest.raises(ValueError):
        create_query_understanding(c, "p", FakeJev(), queries=[], relevant=[], embed=embed)
    with pytest.raises(ValueError):
        create_query_understanding(c, "p", FakeJev(), queries=["q"], relevant=[], embed=embed)
    with pytest.raises(ValueError):
        create_query_understanding(c, "p", FakeJev(), queries=["q"], relevant=[{0}], embed=None, run_fn=None)
    with pytest.raises(ValueError):
        create_query_understanding(c, "empty", FakeJev(), queries=["q"], relevant=[{0}], embed=embed)

def test_labels_from_another_model_or_taxonomy_are_refused():
    c = _client()
    tax = Taxonomy("v2", "jev-1.13.0", [ClassDef("boots", "Boots", "d", "e")], [], Thresholds())
    set_query_understanding(c, "p", tax)
    for model, probs in [("jev-1.12", {"boots": 0.1}), ("jev-1.13.0", {"old-class": 0.9})]:
        c.upsert("p", [models.PointStruct(id=0, vector=[1.0, 0.0], payload={
            "text": "boots", "classes": [], "class_probs": probs, "class_model": model, "needs_review": False})])
        with pytest.raises(ValueError):
            understand_query(c, "p", "boots", FakeJev())
        with pytest.raises(ValueError):
            query_points(c, "p", "boots", FakeJev(), dense=[1.0, 0.0])
