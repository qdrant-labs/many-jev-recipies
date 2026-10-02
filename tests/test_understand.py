import pytest
from jevqu.jev import FakeJev
from jevqu.schema import ClassDef, Facet, Taxonomy, Thresholds
from jevqu.understand import local_facets, understand

TAX = Taxonomy("v1", "jev-1.13.0",
    [ClassDef("outdoor", "Outdoor", "Gear used outside", "Indoor items"),
     ClassDef("kitchen", "Kitchen", "Cooking items", "Non-cooking items")],
    [Facet("brand", ["Salomon", "Nike"]), Facet("color", ["red", "blue"])], Thresholds())

def test_local_facets_exact_match_case_insensitive():
    assert local_facets("salomon red boots", TAX) == {"brand": ("Salomon", 1.0), "color": ("red", 1.0)}

def test_understand_combines_local_facets_and_jev_classes():
    jev = FakeJev(nouls={"outdoor": 0.93, "kitchen": 0.02}, choices={"facet_color": "none"})
    u = understand("salomon hiking boots", TAX, jev)
    assert u.classes == [("outdoor", 0.93), ("kitchen", 0.02)]
    assert u.facets == {"brand": ("Salomon", 1.0)}
    assert u.taxonomy_version == "v1"

def test_facet_choice_only_asked_when_not_matched_locally():
    asked = []
    class Spy(FakeJev):
        def ask(self, state, questions):
            asked.extend(questions); return super().ask(state, questions)
    understand("nike", TAX, Spy())
    assert "facet_brand" not in asked and "facet_color" in asked

def test_facet_choice_none_is_dropped_and_confident_pick_kept():
    jev = FakeJev(choices={"facet_brand": "Nike", "facet_color": "none"})
    u = understand("running shoes", TAX, jev)
    assert u.facets == {"brand": ("Nike", 0.9)}

def test_model_mismatch_refused():
    with pytest.raises(ValueError):
        understand("x", TAX, FakeJev(), stored_model="jev-1.12")

def test_labels_from_an_older_taxonomy_refused():
    with pytest.raises(ValueError):
        understand("x", TAX, FakeJev(), stored_model="jev-1.13.0", labeled_ids={"outdoor"})
    u = understand("x", TAX, FakeJev(), stored_model="jev-1.13.0", labeled_ids={"outdoor", "kitchen", "extra"})
    assert u.taxonomy_version == "v1"

TAX_HIER = Taxonomy("v1", "jev-1.13.0",
    [ClassDef("outdoor", "Outdoor", "Gear used outside", "Indoor items"),
     ClassDef("boots", "Hiking boots", "Footwear for hiking", "Other shoes", parent="outdoor")],
    [], Thresholds())

def test_children_are_asked_with_their_parents_in_one_request():
    asked = []
    class Spy(FakeJev):
        def ask(self, state, questions):
            asked.append(sorted(questions)); return super().ask(state, questions)
    u = understand("hiking boots", TAX_HIER, Spy(nouls={"outdoor": 0.7, "boots": 0.92}))
    assert asked == [["boots", "outdoor"]]  # one request for both levels
    assert u.classes == [("outdoor", 0.7), ("boots", 0.92)]

def test_a_child_is_dropped_when_the_query_is_not_in_its_parent():
    # items carry a child label only inside the parent, so a child filter here would drop the target
    u = understand("hiking boots", TAX_HIER, FakeJev(nouls={"outdoor": 0.4, "boots": 0.95}))
    assert u.classes == [("outdoor", 0.4)]

def test_long_query_is_truncated_before_it_enters_the_state():
    seen = []
    class Spy(FakeJev):
        def ask(self, state, questions):
            seen.append(state); return super().ask(state, questions)
    understand("x" * 5000, TAX, Spy())
    assert len(seen[0]["query"]) == 1500
