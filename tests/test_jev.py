import json
from pathlib import Path
from jevqu.jev import FakeJev, Jev, choice, noul

def test_question_builders():
    q = noul("Is it red?", "The item is red", "Any other colour")
    assert q == {"type": "noul", "instructions": "Is it red?", "criteria": {"true": "The item is red", "false": "Any other colour"}}
    c = choice("Which brand?", {"nike": None, "none": "No brand named"})
    assert c["type"] == "choice" and set(c["criteria"]) == {"nike", "none"}

def test_fake_jev_answers_by_question_id():
    j = FakeJev(nouls={"cls_a": 0.9}, choices={"brand": "nike"})
    a = j.ask({"q": "x"}, {"cls_a": noul("a", "t", "f"), "cls_b": noul("b", "t", "f"),
                          "brand": choice("b", {"nike": None, "none": None})})
    assert a["cls_a"]["noul"] == 0.9 and a["cls_b"]["noul"] == 0.05
    assert a["brand"]["choice"] == "nike" and a["brand"]["probabilities"]["nike"] > 0.5

def test_real_client_uses_cache_without_network(tmp_path, monkeypatch):
    calls = []
    def fake_post(self, body):
        calls.append(body)
        return {"model": "jev-1.13.0", "answers": {"x": {"type": "noul", "noul": 0.42}}}
    monkeypatch.setattr(Jev, "_post", fake_post)
    j = Jev(api_key="k", cache_dir=tmp_path)
    a1 = j.ask("s", {"x": noul("q", "t", "f")})
    a2 = j.ask("s", {"x": noul("q", "t", "f")})
    assert a1 == a2 == {"x": {"type": "noul", "noul": 0.42}}
    assert len(calls) == 1 and calls[0]["model"] == "jev-1.13.0"
    assert len(list(tmp_path.glob("*.json"))) == 1

def test_too_many_questions_rejected():
    j = FakeJev()
    import pytest
    with pytest.raises(ValueError):
        j.ask("s", {f"q{i}": noul("q", "t", "f") for i in range(201)})
