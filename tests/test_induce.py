import numpy as np
from jevqu.induce import coverage, demand, induce, propose, retrieval_value, select
from jevqu.jev import FakeJev
from jevqu.schema import ClassDef, Facet

def test_propose_names_clusters_by_distinctive_terms():
    texts = ["hiking boots trail", "trail boots waterproof", "frying pan steel", "steel pan nonstick"]
    vecs = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=float)
    cands = propose(texts, vecs, k=2, top_terms=2)
    names = {c.id for c in cands}
    assert len(cands) == 2 and any("boots" in n or "trail" in n for n in names) and any("pan" in n or "steel" in n for n in names)

def test_coverage_and_demand_use_one_request_per_item():
    cands = [ClassDef("a", "A", "d", "e"), ClassDef("b", "B", "d", "e")]
    calls = []
    class Spy(FakeJev):
        def ask(self, state, questions):
            calls.append(len(questions)); return super().ask(state, questions)
    cov = coverage(cands, [{"t": 1}, {"t": 2}], Spy(nouls={"a": 0.9}))
    assert cov == {"a": [0.9, 0.9], "b": [0.05, 0.05]} and calls == [2, 2]
    dem = demand(cands, ["q1", "q2"], FakeJev(nouls={"b": 0.7}))
    assert dem == {"a": [], "b": [0, 1]}

def test_retrieval_value_positive_when_filter_helps():
    rel = [{1}, {1}]
    def run_fn(q, cid):
        return [1, 2] if cid == "a" else [2, 1]
    d, lo, hi = retrieval_value("a", [0, 1], ["q", "q"], rel, run_fn)
    assert d > 0 and lo > 0

def test_select_rejects_out_of_band_duplicate_facet_and_negative_value():
    cands = [ClassDef("a", "A", "d", "e"), ClassDef("rare", "Rare", "d", "e"),
             ClassDef("nike", "Nike", "d", "e"), ClassDef("bad", "Bad", "d", "e")]
    cov = {"a": [0.9, 0.9, 0.1, 0.1], "rare": [0.9, 0.0, 0.0, 0.0], "nike": [0.9, 0.9, 0.1, 0.1], "bad": [0.9, 0.9, 0.1, 0.1]}
    dem = {"a": [0, 1], "rare": [0, 1], "nike": [0, 1], "bad": [0, 1]}
    values = {"a": (0.05, 0.01, 0.09), "rare": (0.2, 0.1, 0.3), "nike": (0.2, 0.1, 0.3), "bad": (-0.02, -0.05, 0.01)}
    kept = select(cands, cov, dem, values, facets=[Facet("brand", ["Nike"])], band=(0.3, 0.9))
    assert [c.id for c in kept] == ["a"]

def test_induce_stops_at_max_classes_and_reports():
    texts = ["boots trail"] * 3 + ["pan steel"] * 3
    vecs = np.array([[1, 0]] * 3 + [[0, 1]] * 3, dtype=float)
    states = [{"t": t} for t in texts]
    jev = FakeJev(default_noul=0.6)
    kept, report = induce(texts, vecs, states, ["q"], [{0}], lambda q, cid: [0] if cid else [1], jev, facets=[],
                          max_classes=1, rounds=2, k=2, band=(0.0, 1.0), min_demand=0.0)
    assert len(kept) == 1 and report and {"id", "coverage", "demand", "value"} <= set(report[0])

def test_induce_hands_each_candidates_members_to_label_fn():
    texts = ["boots trail"] * 2 + ["pan steel"] * 2
    vecs = np.array([[1, 0]] * 2 + [[0, 1]] * 2, dtype=float)
    class ByText(FakeJev):
        def ask(self, state, questions):
            text = " ".join(map(str, state.values()))
            return {q: {"type": "noul", "noul": 0.9 if q.split("-")[0] in text else 0.1} for q in questions}
    seen = {}
    induce(texts, vecs, [{"t": t} for t in texts], [], [], lambda q, cid: [], ByText(), facets=[],
           rounds=1, k=2, band=(0.0, 1.0), min_demand=0.0, label_fn=lambda cid, idx: seen.update({cid: idx}))
    assert seen == {"boots-trail": [0, 1], "pan-steel": [2, 3]}
