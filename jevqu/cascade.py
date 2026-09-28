from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from jevqu.classify import classify_point
from jevqu.schema import Taxonomy

class LocalClassifier:
    def __init__(self, class_ids: list[str]):
        self.class_ids, self.models = class_ids, {}

    def fit(self, X: np.ndarray, Y: dict[str, list[int]]) -> "LocalClassifier":
        for cid in self.class_ids:
            y = np.asarray(Y[cid])
            if y.min() == y.max():  # ponytail: constant class, predict its constant
                self.models[cid] = float(y[0]); continue
            self.models[cid] = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, y)
        return self

    def predict_proba(self, x: np.ndarray) -> dict[str, float]:
        out = {}
        for cid, m in self.models.items():
            out[cid] = m if isinstance(m, float) else float(m.predict_proba(x.reshape(1, -1))[0, 1])
        return out

class Cascade:
    def __init__(self, local: LocalClassifier, jev, tax: Taxonomy, band=(0.35, 0.65)):
        self.local, self.jev, self.tax, self.band = local, jev, tax, band

    def classify(self, vector: np.ndarray, payload: dict) -> tuple[dict, str]:
        probs = self.local.predict_proba(vector)
        if any(self.band[0] < p < self.band[1] for p in probs.values()):
            return classify_point(payload, self.tax, self.jev), "jev"
        t = self.tax.thresholds.label_above
        return {"classes": [c for c, p in probs.items() if p >= t], "class_probs": probs,
                "class_model": self.tax.model, "needs_review": False}, "local"

def error_independence(gold: list[set], local: list[set], jevl: list[set]) -> dict:
    n = len(gold)
    jev_wrong = [j != g for g, j in zip(gold, jevl, strict=True)]
    local_wrong = [l != g for g, l in zip(gold, local, strict=True)]
    both = sum(1 for a, b in zip(jev_wrong, local_wrong, strict=True) if a and b)
    lw = sum(local_wrong)
    return {"p_jev_wrong": sum(jev_wrong) / n, "p_jev_wrong_given_local_wrong": (both / lw) if lw else 0.0, "n": n}
