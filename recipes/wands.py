"""WANDS (Wayfair's product-search benchmark), shared by the recipes: loader, graded labels,
and the full 43k-product hybrid index (dense bge-small + BM25), built the same way as
notebooks/jevqu_wands_demo.ipynb but without the demo's 10k-product subsample."""
from __future__ import annotations
import csv, math, sys, urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np
from qdrant_client import models

DATA = Path(__file__).resolve().parent.parent / "data" / "wands"
BASE_URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/"
GRADE = {"Exact": 2, "Partial": 1, "Irrelevant": 0}
DENSE_MODEL, SPARSE_MODEL = "BAAI/bge-small-en-v1.5", "Qdrant/bm25"
COLLECTION = "wands"

def load(name: str) -> list[dict]:
    path = DATA / f"{name}.csv"
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BASE_URL + path.name, path)
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def text(p: dict) -> str:
    return f"{p['product_name']}. {p['product_description']}"[:600]

def grades(labels: list[dict]) -> dict[str, dict[int, int]]:
    """query_id -> {product_id: 2 Exact / 1 Partial / 0 Irrelevant}; unjudged products are absent."""
    out = defaultdict(dict)
    for r in labels:
        out[r["query_id"]][int(r["product_id"])] = GRADE[r["label"]]
    return dict(out)

def ndcg(ranked: list[int], graded: dict[int, int], k: int = 10) -> float:
    """Graded nDCG@k (gain 2^g - 1); unjudged products count as gain 0."""
    gain = lambda g: 2 ** g - 1
    dcg = sum(gain(graded.get(d, 0)) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    idcg = sum(gain(g) / math.log2(i + 2) for i, g in enumerate(sorted(graded.values(), reverse=True)[:k]))
    return dcg / idcg if idcg else 0.0

def sparse_vec(s) -> models.SparseVector:
    return models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())

def build_index(client, products: list[dict], collection: str = COLLECTION):
    """Index every product with dense + BM25 vectors. Dense vectors are cached in data/wands/.
    Returns (dense, bm25) FastEmbed models for embedding queries."""
    from fastembed import SparseTextEmbedding, TextEmbedding
    dense, bm25 = TextEmbedding(DENSE_MODEL), SparseTextEmbedding(SPARSE_MODEL)
    texts = [text(p) for p in products]
    cache = DATA / f"dense_full_{len(products)}.npy"
    if cache.exists():
        dvecs = np.load(cache)
    else:
        dvecs = np.array(list(dense.embed(texts, batch_size=64)), dtype=np.float32)
        np.save(cache, dvecs)
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(collection,
        vectors_config={"dense": models.VectorParams(size=dvecs.shape[1], distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    for i in range(0, len(products), 1024):  # ponytail: batches bound memory; sparse vectors are recomputed each run
        batch = products[i:i + 1024]
        svecs = bm25.embed(texts[i:i + 1024])
        client.upsert(collection, [models.PointStruct(
            id=int(p["product_id"]), vector={"dense": d.tolist(), "sparse": sparse_vec(s)},
            payload={"product_id": int(p["product_id"]), "name": p["product_name"], "text": t})
            for p, d, s, t in zip(batch, dvecs[i:i + 1024], svecs, texts[i:i + 1024])], wait=True)
    return dense, bm25

def hybrid(client, qdense: list[float], qsparse: models.SparseVector, limit: int = 10,
           collection: str = COLLECTION, prefetch_limit: int = 100, mmr: models.Mmr | None = None):
    """Dense + BM25 fused with RRF. With `mmr`, Qdrant's MMR re-ranks the fused candidates;
    note MMR's relevance term is dense similarity to the query, not the RRF score."""
    legs = [models.Prefetch(query=qdense, using="dense", limit=prefetch_limit),
            models.Prefetch(query=qsparse, using="sparse", limit=prefetch_limit)]
    if mmr is None:
        return client.query_points(collection, prefetch=legs, query=models.FusionQuery(fusion=models.Fusion.RRF),
                                   limit=limit, with_payload=True).points
    fused = models.Prefetch(prefetch=legs, query=models.FusionQuery(fusion=models.Fusion.RRF),
                            limit=mmr.candidates_limit or prefetch_limit)
    return client.query_points(collection, prefetch=[fused], query=models.NearestQuery(nearest=qdense, mmr=mmr),
                               using="dense", limit=limit, with_payload=True).points
