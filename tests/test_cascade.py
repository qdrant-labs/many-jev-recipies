import numpy as np
import pytest
from jevqu.cascade import Cascade, LocalClassifier, error_independence
from jevqu.jev import FakeJev
from jevqu.schema import ClassDef, Taxonomy, Thresholds

TAX = Taxonomy("v1", "jev-1.13.0", [ClassDef("a", "A", "d", "e"), ClassDef("b", "B", "d", "e")], [], Thresholds())

def _data():
    X = np.array([[1, 0]] * 20 + [[0, 1]] * 20, dtype=float)
    Y = {"a": [1] * 20 + [0] * 20, "b": [0] * 20 + [1] * 20}
    return X, Y

def test_local_classifier_learns_separable_classes():
    X, Y = _data()
    m = LocalClassifier(["a", "b"]).fit(X, Y)
    p = m.predict_proba(np.array([1.0, 0.0]))
    assert p["a"] > 0.8 and p["b"] < 0.2

def test_cascade_escalates_only_uncertain():
    X, Y = _data()
    m = LocalClassifier(["a", "b"]).fit(X, Y)
    calls = []
    class Spy(FakeJev):
        def ask(self, state, questions):
            calls.append(1); return super().ask(state, questions)
    c = Cascade(m, Spy(nouls={"a": 0.9}), TAX)
    labels, who = c.classify(np.array([1.0, 0.0]), {"text": "x"})
    assert who == "local" and labels["classes"] == ["a"] and not calls
    assert labels["class_model"] == "jev-1.13.0"
    labels, who = c.classify(np.array([0.5, 0.5]), {"text": "x"})
    assert who == "jev" and calls

def test_error_independence():
    gold = [{"a"}, {"a"}, {"b"}, {"b"}]
    local = [{"a"}, {"b"}, {"b"}, {"a"}]   # wrong on 1, 3
    jevl = [{"a"}, {"a"}, {"a"}, {"a"}]    # wrong on 2, 3
    r = error_independence(gold, local, jevl)
    assert r == {"p_jev_wrong": 0.5, "p_jev_wrong_given_local_wrong": 0.5, "n": 4}

def test_error_independence_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        error_independence([{"a"}, {"a"}], [{"a"}], [{"a"}, {"a"}])
