from collections import Counter
import numpy as np
from jevqu.induce import _children, _promote, induce_llm, _key, _phrases, _rejected_by, _dups, classes_for, coverage, induce, judge, name_clusters, name_options, propose, select
from jevqu.jev import FakeJev
from jevqu.schema import ClassDef, Facet

enc = lambda ts: np.array([[("boot" in t) + ("trail" in t), ("pan" in t) + ("steel" in t), 0.01] for t in ts])

def _jev(yes=lambda text, qid: 0.1):
    """Picks the first (encoder's top) name option; answers yes/no questions with yes(state text, question id)."""
    class J(FakeJev):
        def ask(self, state, questions):
            text = " ".join(map(str, state.values())).lower()
            return {q: {"type": "choice", "choice": next(iter(v["criteria"])), "confidence": 0.9} if v["type"] == "choice"
                    else {"type": "noul", "noul": yes(text, q)} for q, v in questions.items()}
    return J()

def test_propose_lets_jev_pick_each_name_from_encoder_ranked_words_of_the_cluster():
    texts = ["hiking boots, trail", "waterproof boots, mud", "frying pan, steel", "nonstick pan, steel"]
    vecs = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=float)
    lab = np.array([0, 0, 1, 1])
    opts = name_options(texts, vecs, lab, enc)
    o = opts[0][0]
    assert o[0] == "boots" and o.index("trail") < o.index("hiking")  # the encoder ranks words it relates to the cluster first
    assert opts[1][0][0] == "frying pan" and all(len(o) <= 8 for o, _ in opts)
    states = [{"t": t} for t in texts]
    names_fit = _jev(lambda text, q: 0.9 if any(w in text for w in q.split("-")) else 0.1)
    cands = propose(texts, vecs, states, names_fit, enc, k=2)
    assert sorted(c.id for c in cands) == ["boots", "frying-pan"] and cands[0].definition.startswith("The item is a kind of ")
    assert propose(texts, vecs, states, FakeJev(), enc, k=2) == []  # no member fits any name: no class

def test_a_name_that_fits_few_members_is_replaced_and_one_that_fits_everything_is_dropped():
    texts = ["boots trail"] * 4 + ["pan steel"] * 4
    vecs = np.array([[1, 0]] * 4 + [[0, 1]] * 4, dtype=float)
    lab = np.array([0] * 4 + [1] * 4)
    # Jev picks the encoder's top name ("boots trail"), but only "boots" fits members; "steel" fits every item
    def yes(text, q):
        return 0.9 if q == "steel" or (q in ("boots", "pan") and q in text) else 0.1
    rows = name_clusters(texts, vecs, [{"t": t} for t in texts], lab, _jev(yes), enc)
    assert [r["name"] for r in rows] == ["boots", "pan"] and [r["jev_pick"] for r in rows] == ["boots trail", "pan steel"]
    assert rows[1]["options"]["steel"] == (1.0, 1.0)  # fits all members, and every item of the other cluster

def test_name_phrases_skip_ids_numbers_stop_words_and_punctuation():
    assert _phrases("B01C5ZG0BK Sterling Silver Necklace for Women, 18 inch") == {
        "sterling", "silver", "necklace", "sterling silver", "silver necklace", "sterling silver necklace", "women", "inch"}
    assert _phrases("It was founded about his world and has been") == {"founded", "world"}

def test_singular_keys():
    assert [_key(w) for w in ("dog collars", "glasses", "dress", "batteries")] == ["dog collar", "glass", "dress", "battery"]

def test_clusters_jev_gives_the_same_name_merge_and_unnamed_ones_drop():
    assert [c.id for c in classes_for(["dog collars", None, "Dog Collars", "mug"])] == ["dog-collars", "mug"]

def test_coverage_uses_one_request_per_item_and_keeps_item_order():
    cands = [ClassDef("a", "A", "d", "e"), ClassDef("b", "B", "d", "e")]
    calls = []
    class ByItem(FakeJev):
        def ask(self, state, questions):
            calls.append(len(questions))
            return {q: {"type": "noul", "noul": state["p"] if q == "a" else 0.05} for q in questions}
    cov = coverage(cands, [{"p": 0.1}, {"p": 0.2}, {"p": 0.3}], ByItem(), workers=3)
    assert cov == {"a": [0.1, 0.2, 0.3], "b": [0.05, 0.05, 0.05]} and calls == [2, 2, 2]

def test_select_rejects_out_of_band_and_duplicate_facet():
    cands = [ClassDef("a", "A", "d", "e"), ClassDef("rare", "Rare", "d", "e"), ClassDef("nike", "Nike", "d", "e")]
    cov = {"a": [0.9, 0.9, 0.1, 0.1], "rare": [0.9, 0.0, 0.0, 0.0], "nike": [0.9, 0.9, 0.1, 0.1]}
    kept = select(cands, cov, facets=[Facet("brand", ["Nike"])], band=(0.3, 0.9))
    assert [c.id for c in kept] == ["a"]

def test_judge_gate_drops_groups_jev_calls_niche_or_incoherent():
    cands = [ClassDef(x, x.upper(), "d", "e") for x in ("a", "b", "c")]
    cov = {"a": [0.9, 0.1, 0.1], "b": [0.1, 0.9, 0.1], "c": [0.1, 0.1, 0.9]}  # distinct, so none merge
    judged = {"a": (0.9, 0.8), "b": (0.2, 0.9), "c": (0.9, 0.3)}
    assert [c.id for c in select(cands, cov, facets=[], band=(0.0, 1.0))] == ["a", "b", "c"]
    assert [c.id for c in select(cands, cov, facets=[], band=(0.0, 1.0), judged=judged)] == ["a"]

def test_judge_asks_once_per_group_with_member_examples():
    seen = []
    class Spy(FakeJev):
        def ask(self, state, questions):
            seen.append((state["group"]["examples"], sorted(questions)))
            return {q: {"type": "noul", "noul": 0.8} for q in questions}
    cands = [ClassDef("a", "A", "d", "e"), ClassDef("b", "B", "d", "e")]
    out = judge(cands, {"a": [0.9, 0.1], "b": [0.1, 0.9]}, ["boots", "pan"], Spy(), workers=1)
    assert out == {"a": (0.8, 0.8), "b": (0.8, 0.8)}
    assert sorted(seen) == [(["boots"], ["coherent", "popular"]), (["pan"], ["coherent", "popular"])]

def test_induce_uses_items_only_and_reports_why_each_candidate_was_rejected():
    texts = ["boots trail"] * 3 + ["pan steel"] * 3
    vecs = np.array([[1, 0]] * 3 + [[0, 1]] * 3, dtype=float)
    by_text = _jev(lambda text, q: 0.9 if q.split("-")[0] in text else 0.1)
    kept, report = induce(texts, vecs, [{"t": t} for t in texts], by_text, facets=[], k=2, band=(0.0, 1.0), encoder=enc)
    assert sorted(c.id for c in kept) == ["boots-trail", "pan-steel"]
    assert all(r["kept"] and r["rejected_by"] is None and r["coverage"] == 0.5 for r in report)
    kept, report = induce(texts, vecs, [{"t": t} for t in texts], by_text, facets=[], k=2, band=(0.0, 0.4), encoder=enc)
    assert kept == [] and {r["rejected_by"] for r in report} == {"size"}

def test_induce_stops_at_max_classes():
    texts = ["boots trail"] * 3 + ["pan steel"] * 3
    vecs = np.array([[1, 0]] * 3 + [[0, 1]] * 3, dtype=float)
    kept, report = induce(texts, vecs, [{"t": t} for t in texts], _jev(lambda text, q: 0.6 if q.split("-")[0] in text else 0.1), facets=[],
                          max_classes=1, k=2, band=(0.0, 1.0), encoder=enc)
    assert len(kept) == 1 and sorted(r["rejected_by"] or "" for r in report) == ["", "max_classes"]

def test_spelling_ties_break_alphabetically_and_spellings_share_one_class():
    opts = name_options(["red boot", "blue boots"], np.array([[1.0, 0], [1.0, 0]]), np.array([0, 0]), enc)[0][0]
    assert "boot" in opts and "boots" not in opts  # equal counts: not decided by Python's per-process set order
    assert [c.id for c in classes_for(["decorations", "decoration"])] == ["decorations"]

def test_classes_jev_accepts_for_the_same_items_are_merged_into_the_bigger_one():
    cands = [ClassDef(x, x.upper(), "d", "e") for x in ("shoes", "footwear", "pan")]
    cov = {"shoes": [.9, .9, .9, .1, .1], "footwear": [.9, .9, .1, .1, .1], "pan": [.1, .1, .1, .9, .9]}
    assert [c.id for c in select(cands, cov, facets=[], band=(0.0, 1.0))] == ["shoes", "pan"]
    assert _rejected_by(cands[1], cov, [], (0.0, 1.0), None, 0.5, _dups(cands, cov, [], (0.0, 1.0))) == "duplicates shoes"

def test_children_go_under_the_parent_that_accepts_their_items_and_are_scored_only_there():
    texts = ["boots red", "boots blue", "pan steel", "pan iron", "mug coffee"]
    words = ("boots", "red", "blue", "pan", "steel", "iron", "mug", "coffee")
    enc2 = lambda ts: np.array([[float(w in t) for w in words] + [0.01] for t in ts])
    parents = [ClassDef("boots", "Boots", "d", "e"), ClassDef("pan", "Pan", "d", "e")]
    pcov = {"boots": [.9, .9, .1, .1, .1], "pan": [.1, .1, .9, .9, .1]}
    asked = []
    class J(FakeJev):
        def ask(self, state, questions):
            asked.append((state.get("t", ""), sorted(questions)))
            text = " ".join(map(str, state.values()))
            return {q: {"type": "choice", "choice": next(iter(v["criteria"])), "confidence": .9} if v["type"] == "choice"
                    else {"type": "noul", "noul": .9 if all(w in text for w in q.split("-")) else .1} for q, v in questions.items()}
    kids, orphans, report = _children(parents, pcov, texts, np.eye(5), [{"t": t} for t in texts], J(), enc2, child_k=5,
                                      band=(0.0, 1.0), judge_groups=False, min_judge=.5, workers=1)
    assert sorted((k.id, k.parent) for k in kids) == [("boots-blue", "boots"), ("boots-red", "boots"),
                                                      ("pan-iron", "pan"), ("pan-steel", "pan")]
    assert [c.id for c in orphans] == ["mug-coffee"]
    kid_ids = {k.id for k in kids} | {"mug-coffee"}
    scored = [(t, set(qs)) for t, qs in asked if qs and set(qs) <= kid_ids]  # the children's coverage requests
    assert sorted(t for t, _ in scored) == ["boots blue", "boots red", "pan iron", "pan steel"]
    assert all(qs == ({"boots-red", "boots-blue"} if t.startswith("boots") else {"pan-steel", "pan-iron"}) for t, qs in scored)

def test_a_name_fitting_exactly_min_fit_of_members_is_accepted():
    texts = ["boots"] * 6 + ["shoes"] * 9 + ["pan"] * 15
    vecs = np.array([[1, 0]] * 15 + [[0, 1]] * 15, dtype=float)
    rows = name_clusters(texts, vecs, [{"t": t} for t in texts], np.array([0] * 15 + [1] * 15),
                         _jev(lambda text, q: 0.9 if q in text else 0.1), enc, fit_n=15)
    assert rows[0]["options"]["boots"][0] == 0.4 and rows[0]["name"] in ("boots", "shoes")

def test_orphans_become_level1_classes_unless_they_duplicate_a_parent():
    parents = [ClassDef("shoes", "Shoes", "d", "e")]
    pcov = {"shoes": [.9, .9, .1, .1]}
    orphans = [ClassDef("mugs", "Mugs", "d", "e"), ClassDef("footwear", "Footwear", "d", "e")]
    jev = _jev(lambda text, q: {"mugs": .9 if "mug" in text else .1, "footwear": .9 if "shoe" in text else .1}[q])
    kept, report = _promote(parents, pcov, orphans, [{"t": t} for t in ("shoe", "shoe", "mug", "mug")], [], jev, [],
                            (0.0, 1.0), False, .5, room=10, workers=1)
    assert [c.id for c in kept] == ["mugs"] and all(c.parent is None for c in kept)
    assert {r["id"]: r["rejected_by"] for r in report} == {"mugs": None, "footwear": "duplicates shoes"}

class FakeLLM:
    """Names a group by its most common word; puts every group under one parent, "gear"."""
    def ask_json(self, system, user):
        if user.startswith("Example items"):
            words = Counter(w for line in user.splitlines()[1:] for w in line.split(". ", 1)[1].split())
            w = words.most_common(1)[0][0]
            return {"name": w, "definition": f"The item is {w}", "exclusions": "Other items", "coherent": True}
        n = len(user.splitlines()) - 1
        return {"parents": [{"name": "gear", "definition": "Any item", "exclusions": "Nothing", "children": list(range(n))}]}

def test_llm_proposes_two_levels_and_jev_keeps_children_inside_their_parent():
    texts = ["boots boots trail"] * 6 + ["pan pan steel"] * 6
    vecs = np.array([[1, 0]] * 6 + [[0, 1]] * 6, dtype=float)
    jev = _jev(lambda text, q: 0.9 if q == "gear" or q in text else 0.1)
    classes, report = induce_llm(texts, vecs, [{"t": t} for t in texts], jev, FakeLLM(), facets=[], k=2, band=(0.0, 1.0),
                                 fit_n=6, pool_n=2, workers=1)
    assert sorted((c.id, c.parent) for c in classes) == [("boots", "gear"), ("gear", None), ("pan", "gear")]
    assert {r["id"]: r["kept"] for r in report} == {"gear": True, "boots": True, "pan": True}

def test_llm_names_that_fit_few_members_are_dropped_before_parents():
    texts = ["boots boots trail"] * 6 + ["pan pan steel"] * 6
    vecs = np.array([[1, 0]] * 6 + [[0, 1]] * 6, dtype=float)
    jev = _jev(lambda text, q: 0.9 if q == "gear" or (q == "boots" and q in text) else 0.1)  # nothing fits "pan"
    classes, report = induce_llm(texts, vecs, [{"t": t} for t in texts], jev, FakeLLM(), facets=[], k=2, band=(0.0, 1.0),
                                 fit_n=6, pool_n=2, workers=1)
    assert sorted(c.id for c in classes) == ["boots", "gear"]
    assert {r["id"]: r["rejected_by"] for r in report}["pan"] == "fits few members"
