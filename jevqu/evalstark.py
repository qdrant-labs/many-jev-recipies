from __future__ import annotations
from jevqu.metrics import ndcg_at_k, recall_at_k

def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)

def recall_loss_curve(runs: dict[str, list[list]], relevant: list[set], thresholds: list[float], run_at) -> list[dict]:
    base = runs["off"]
    base_recall = _mean(recall_at_k(r, rel, 20) for r, rel in zip(base, relevant))
    base_ndcg10 = _mean(ndcg_at_k(r, rel, 10) for r, rel in zip(base, relevant))
    rows = []
    for t in thresholds:
        ranked = run_at(t)
        rec = _mean(recall_at_k(r, rel, 20) for r, rel in zip(ranked, relevant))
        ndcg10 = _mean(ndcg_at_k(r, rel, 10) for r, rel in zip(ranked, relevant))
        rows.append({"threshold": t,
                     "ndcg10": ndcg10, "ndcg_gain": ndcg10 - base_ndcg10,
                     "recall20": rec, "recall_loss": max(0.0, base_recall - rec),
                     "filtered_share": _mean(1.0 if r != b else 0.0 for r, b in zip(ranked, base))})
    return rows

def report_markdown(rows: list[dict], base_ndcg: float) -> str:
    out = [f"Unfiltered hybrid nDCG@10: {base_ndcg:.4f}", "", "| threshold | nDCG@10 | nDCG gained | recall@20 | recall lost | queries changed |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['threshold']} | {r['ndcg10']:.4f} | {r['ndcg_gain']:+.4f} | {r['recall20']:.4f} | {r['recall_loss']:.4f} | {r['filtered_share']:.1%} |")
    return "\n".join(out)
