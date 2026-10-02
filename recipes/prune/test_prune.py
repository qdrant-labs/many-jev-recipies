from jevqu.jev import FakeJev
from recipes.prune.prune import judge, prune, relevance

def test_judge_then_prune_with_backfill():
    ids = list(range(100, 112))  # 12 results
    jev = FakeJev(nouls={f"rel_{n}": 0.9 for n in range(1, 13)} | {"rel_2": 0.1, "dup_4": 0.8})
    rel, dup = judge(jev, "q", [f"text {i}" for i in ids])
    assert rel[1] == 0.1 and dup[0] == 0.0 and dup[3] == 0.8 and dup[2] == 0.05
    page, backfilled = prune(ids, rel, dup)
    assert page == [100, 102] + list(range(104, 112)) and not backfilled  # result 2 irrelevant, result 4 a duplicate
    page, backfilled = prune(ids, rel, dup, keep=11)
    assert page[-1] == 103 and backfilled  # 10 survivors; the relevant duplicate (rel 0.9) beats the irrelevant one (0.1)
    assert prune(ids, rel, dup, rel_min=0.0, dup_max=1.01)[0] == ids[:10]  # thresholds off: plain top 10

def test_relevance_batches():
    class Counting(FakeJev):
        calls: int = 0
        def ask(self, state, questions):
            self.calls += 1
            assert len(state["results"]) <= 20 and all(k.startswith("rel_") for k in questions)
            return super().ask(state, questions)
    jev = Counting(nouls={"rel_3": 0.9})
    rel = relevance(jev, "q", [f"text {i}" for i in range(45)])
    assert jev.calls == 3 and len(rel) == 45
    assert [i for i, r in enumerate(rel) if r == 0.9] == [2, 22, 42]  # local numbering maps back to global ranks
