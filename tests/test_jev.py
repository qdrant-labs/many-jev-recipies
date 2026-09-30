import json
import urllib.error
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
        return {"model": "typesafe/jev-1.13-20260917", "answers": {"x": {"type": "noul", "noul": 0.42}},
                "usage": {"input_tokens": 12, "output_tokens": 3, "cost": 1.9e-05}}
    monkeypatch.setattr(Jev, "_post", fake_post)
    j = Jev(api_key="k", cache_dir=tmp_path)
    a1 = j.ask("s", {"x": noul("q", "t", "f")})
    a2 = j.ask("s", {"x": noul("q", "t", "f")})
    assert a1 == a2 == {"x": {"type": "noul", "noul": 0.42}}
    assert len(calls) == 1 and calls[0]["model"] == "typesafe/jev-1.13" and j.usage == [1.9e-05]
    assert len(list(tmp_path.glob("*.json"))) == 1

def test_too_many_questions_rejected():
    j = FakeJev()
    import pytest
    with pytest.raises(ValueError):
        j.ask("s", {f"q{i}": noul("q", "t", "f") for i in range(201)})

def test_jev_fails_fast_on_401(tmp_path, monkeypatch):
    import pytest
    calls = []
    def fake_post(self, body):
        calls.append(body)
        raise urllib.error.HTTPError("https://api.typesafe.ai/v1/systemone", 401, "unauthorized", {}, None)
    monkeypatch.setattr(Jev, "_post", fake_post)
    monkeypatch.setattr("jevqu.jev.time.sleep", lambda x: None)
    j = Jev(api_key="k", cache_dir=tmp_path)
    with pytest.raises(urllib.error.HTTPError):
        j.ask("s", {"x": noul("q", "t", "f")})
    assert len(calls) == 1

def test_jev_retries_on_500(tmp_path, monkeypatch):
    calls = []
    def fake_post(self, body):
        calls.append(body)
        if len(calls) < 4:
            raise urllib.error.HTTPError("https://api.typesafe.ai/v1/systemone", 500, "server error", {}, None)
        return {"model": "jev-1.13.0", "answers": {"x": {"type": "noul", "noul": 0.42}}}
    monkeypatch.setattr(Jev, "_post", fake_post)
    monkeypatch.setattr("jevqu.jev.time.sleep", lambda x: None)
    j = Jev(api_key="k", cache_dir=tmp_path)
    a = j.ask("s", {"x": noul("q", "t", "f")})
    assert a == {"x": {"type": "noul", "noul": 0.42}}
    assert len(calls) == 4
