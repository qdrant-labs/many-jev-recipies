"""Baseline hybrid vs understanding modes on STaRK-Amazon test queries; prints the recall-loss curve."""
from __future__ import annotations
import argparse, json
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from stark_qa import load_qa
from jevqu.evalstark import recall_loss_curve, report_markdown
from jevqu.jev import Jev
from jevqu.metrics import ndcg_at_k
from jevqu.qquery import run
from jevqu.schema import Thresholds
from jevqu.store import load_taxonomy
from jevqu.understand import understand

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333"); ap.add_argument("--collection", default="stark_amazon")
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--out", default="results/stark_curve.md")
    a = ap.parse_args()
    client, jev = QdrantClient(url=a.url), Jev()
    tax = load_taxonomy(client, a.collection); assert tax, "run create_query_understanding first"
    dense, sparse = TextEmbedding("BAAI/bge-small-en-v1.5"), SparseTextEmbedding("Qdrant/bm25")
    qa = [(q, {int(i) for i in ans}) for q, _, ans, _ in load_qa("amazon").get_subset("test")][: a.n]
    queries, relevant = [q for q, _ in qa], [r for _, r in qa]
    dv, sv = list(dense.embed(queries)), list(sparse.embed(queries))
    us = [understand(q, tax, jev) for q in queries]
    def ranked(mode, thr):
        tax.thresholds = thr
        return [[p.id for p in run(client, a.collection, d.tolist(),
                 models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()), u, tax, limit=20, mode=mode)]
                for d, s, u in zip(dv, sv, us)]
    off = ranked("off", Thresholds())
    rows = recall_loss_curve({"off": off}, relevant, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
                             lambda t: ranked("auto", Thresholds(filter_above=t)))
    base = sum(ndcg_at_k(r, rel, 10) for r, rel in zip(off, relevant)) / len(off)
    md = report_markdown(rows, base)
    from pathlib import Path; Path(a.out).parent.mkdir(exist_ok=True, parents=True); Path(a.out).write_text(md)
    print(md); print(json.dumps({"queries": len(queries), "classes": len(tax.classes)}))

if __name__ == "__main__":
    main()
