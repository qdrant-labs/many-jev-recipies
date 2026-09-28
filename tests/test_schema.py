from jevqu.schema import ClassDef, Facet, Taxonomy, Thresholds
from jevqu.text import state_from_payload

def make_tax():
    return Taxonomy(version="v1", model="jev-1.13.0",
        classes=[ClassDef("outdoor", "Outdoor", "Gear used outside", "Indoor items"),
                 ClassDef("boots", "Hiking boots", "Footwear for hiking", "Other shoes", parent="outdoor")],
        facets=[Facet("brand", ["Nike", "Salomon"])], thresholds=Thresholds())

def test_hierarchy_and_roundtrip():
    t = make_tax()
    assert [c.id for c in t.level1()] == ["outdoor"]
    assert [c.id for c in t.children("outdoor")] == ["boots"]
    assert t.by_id("boots").parent == "outdoor"
    t2 = Taxonomy.from_json(t.to_json())
    assert t2 == t

def test_state_truncates_and_selects_fields():
    s = state_from_payload({"title": "x" * 5000, "price": 3, "junk": "y"}, fields=["title", "price"], max_chars=10)
    assert s == {"title": "xxxxxxxxxx", "price": 3}
