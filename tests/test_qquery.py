import pytest
from qdrant_client import QdrantClient, models
from jevqu.qquery import build_filter, build_formula, run
from jevqu.schema import ClassDef, Facet, Taxonomy, Thresholds, Understanding

TAX = Taxonomy("v1", "jev-1.13.0",
    [ClassDef("outdoor", "Outdoor", "d", "e"), ClassDef("kitchen", "Kitchen", "d", "e")],
    [Facet("brand", ["Salomon"])], Thresholds(boost_scale=0.5))

def test_auto_filters_when_sure_boosts_when_unsure():
    u = Understanding([("outdoor", 0.95), ("kitchen", 0.7)], {"brand": ("Salomon", 1.0)}, "v1")
    f = build_filter(u, TAX, "auto")
    keys = sorted((c.key, c.match.value) for c in f.must)
    assert keys == [("brand", "Salomon"), ("classes", "outdoor")]
    fq = build_formula(u, TAX, "auto")
    assert fq is not None and "kitchen" in str(fq)

def test_no_signal_means_no_filter_no_formula():
    u = Understanding([("outdoor", 0.2), ("kitchen", 0.1)], {}, "v1")
    assert build_filter(u, TAX, "auto") is None and build_formula(u, TAX, "auto") is None

def test_run_rejects_unknown_mode():
    c = QdrantClient(":memory:")
    c.create_collection("p", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT))
    with pytest.raises(ValueError):
        run(c, "p", [1.0, 0.0], None, None, TAX, mode="Off")

def _mem():
    c = QdrantClient(":memory:")
    c.create_collection("p", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT))
    c.create_payload_index("p", "classes", models.PayloadSchemaType.KEYWORD)
    c.upsert("p", [
        models.PointStruct(id=1, vector=[1.0, 0.0], payload={"classes": ["kitchen"]}),
        models.PointStruct(id=2, vector=[0.9, 0.1], payload={"classes": ["outdoor"]}),
        models.PointStruct(id=3, vector=[0.8, 0.2], payload={"classes": ["kitchen"]}),
    ])
    return c

def test_run_off_equals_plain_search_and_boost_reorders():
    c = _mem()
    plain = [p.id for p in run(c, "p", [1.0, 0.0], None, None, TAX, limit=3, mode="off")]
    assert plain == [1, 2, 3]
    u = Understanding([("outdoor", 0.7)], {}, "v1")
    boosted = [p.id for p in run(c, "p", [1.0, 0.0], None, u, TAX, limit=3, mode="auto")]
    assert boosted[0] == 2

def test_filtered_query_with_too_few_hits_falls_back_and_merges():
    c = _mem()
    u = Understanding([("outdoor", 0.95)], {}, "v1")
    ids = [p.id for p in run(c, "p", [1.0, 0.0], None, u, TAX, limit=3, mode="auto")]
    assert ids[0] == 2 and sorted(ids) == [1, 2, 3]

def test_mid_confidence_facet_boosts_instead_of_vanishing():
    u = Understanding([], {"brand": ("Salomon", 0.7)}, "v1")
    assert build_filter(u, TAX, "auto") is None
    fq = build_formula(u, TAX, "auto")
    assert fq is not None and "Salomon" in str(fq)

def _hybrid():
    c = QdrantClient(":memory:")
    c.create_collection("h", vectors_config={"dense": models.VectorParams(size=2, distance=models.Distance.DOT)},
                        sparse_vectors_config={"sparse": models.SparseVectorParams()})
    c.upsert("h", [models.PointStruct(id=i, vector={"dense": d, "sparse": models.SparseVector(indices=[0], values=[s])},
                                      payload={"classes": cl}) for i, d, s, cl in [
        (1, [1.0, 0.0], 1.0, ["kitchen"]), (2, [0.9, 0.1], 0.9, ["outdoor"]), (3, [0.8, 0.2], 0.8, ["kitchen"])]])
    return c

def test_hybrid_no_signal_matches_off_and_boost_and_fallback_work():
    c, sp = _hybrid(), models.SparseVector(indices=[0], values=[1.0])
    off = run(c, "h", [1.0, 0.0], sp, None, TAX, limit=3, mode="off")
    quiet = run(c, "h", [1.0, 0.0], sp, Understanding([("outdoor", 0.2)], {}, "v1"), TAX, limit=3, mode="auto")
    assert [p.id for p in off] == [1, 2, 3]
    assert [(p.id, p.score) for p in quiet] == [(p.id, p.score) for p in off]
    boosted = run(c, "h", [1.0, 0.0], sp, Understanding([("outdoor", 0.7)], {}, "v1"), TAX, limit=3, mode="auto")
    assert boosted[0].id == 2
    filled = run(c, "h", [1.0, 0.0], sp, Understanding([("outdoor", 0.95)], {}, "v1"), TAX, limit=3, mode="auto")
    assert filled[0].id == 2 and sorted(p.id for p in filled) == [1, 2, 3]
