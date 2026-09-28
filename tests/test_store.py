from qdrant_client import QdrantClient
from jevqu.schema import ClassDef, Taxonomy, Thresholds
from jevqu.store import load_taxonomy, save_taxonomy

def test_roundtrip_and_overwrite():
    c = QdrantClient(":memory:")
    t = Taxonomy("v1", "jev-1.13.0", [ClassDef("a", "A", "d", "e")], [], Thresholds())
    assert load_taxonomy(c, "p") is None
    save_taxonomy(c, "p", t)
    assert load_taxonomy(c, "p") == t
    t2 = Taxonomy("v2", "jev-1.13.0", [], [], Thresholds(filter_above=0.8))
    save_taxonomy(c, "p", t2)
    assert load_taxonomy(c, "p") == t2 and load_taxonomy(c, "other") is None
