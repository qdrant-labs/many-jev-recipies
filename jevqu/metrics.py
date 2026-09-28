from __future__ import annotations
import math, random

def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)

def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / idcg

def paired_bootstrap(a: list[float], b: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    assert len(a) == len(b) and a
    rng = random.Random(seed)
    deltas = [y - x for x, y in zip(a, b)]
    mean = sum(deltas) / len(deltas)
    samples = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(n))
    return mean, samples[int(0.025 * n)], samples[int(0.975 * n) - 1]
