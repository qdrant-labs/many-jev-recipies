from jevqu.jev import FakeJev
from recipes.chunker.chunker import chunks, cut, gap_probs, sentences, windows

class TopicJev:  # a gap is a topic change when the leading letter of the two sentences differs
    def ask(self, state, questions):
        t = {s["sentence"]: s["text"][0] for s in state["sentences"]}
        return {q: {"noul": 0.9 if t[int(q[4:])] != t[int(q[4:]) + 1] else 0.1} for q in questions}

def test_chunker():
    text = " ".join(f"{c}{i} is a sentence." for c, n in (("A", 7), ("B", 9), ("C", 6)) for i in range(n)) + "\n"
    spans = sentences(text)
    texts = [text[a:b] for a, b in spans]
    assert len(texts) == 22 and texts[0] == "A0 is a sentence." and texts[-1] == "C5 is a sentence."
    plan = windows(len(texts), 6)
    assert sorted(g for _, gaps in plan for g in gaps) == list(range(21)) and len(plan) > 3
    assert gap_probs(texts, FakeJev(default_noul=0.05), window=6) == [0.05] * 21
    probs = gap_probs(texts, TopicJev(), window=6)
    assert [g for g, p in enumerate(probs) if p > 0.5] == [6, 15]  # local sentence numbers map back to global gaps
    starts = cut(probs, [1] * 22, threshold=0.5)
    assert starts == [0, 7, 16]
    assert [c[:2] for c in chunks(text, spans, starts)] == ["A0", "B0", "C0"]
    assert all(c in text for c in chunks(text, spans, starts))
    assert cut(probs, [1] * 22, 0.5, min_size=8) == [0, 16]  # first topic too small to stand alone
    capped = cut(probs, [1] * 22, 0.5, max_size=5)
    assert {7, 16} <= set(capped) and max(b - a for a, b in zip(capped, capped[1:] + [22])) <= 5
    assert cut([0.1, 0.4, 0.1, 0.1], [1] * 5, 0.5, max_size=3) == [0, 2]  # forced cut moves back to the likeliest gap
    assert windows(1, 40) == [] and gap_probs(["one"], FakeJev()) == [] and cut([], [], 0.5) == []
