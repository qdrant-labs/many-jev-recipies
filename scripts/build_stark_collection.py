"""Build a Qdrant collection from STaRK-Amazon: answer nodes for all QA splits plus RANDOM_EXTRA random nodes."""
from __future__ import annotations
import argparse, json, random
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from stark_qa import load_qa, load_skb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333"); ap.add_argument("--collection", default="stark_amazon")
    ap.add_argument("--random-extra", type=int, default=50000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    skb = load_skb("amazon", download_processed=True)
    ids = set()
    qa = load_qa("amazon")
    for split in ("train", "val", "test"):
        for _, _, answer_ids, _ in qa.get_subset(split):
            ids.update(int(i) for i in answer_ids)
    rng = random.Random(a.seed)
    ids.update(rng.sample(range(skb.num_nodes()), a.random_extra))
    dense, sparse = TextEmbedding("BAAI/bge-small-en-v1.5"), SparseTextEmbedding("Qdrant/bm25")
    client = QdrantClient(url=a.url)
    if client.collection_exists(a.collection):
        client.delete_collection(a.collection)
    client.create_collection(a.collection,
        vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    for key in ("classes", "brand", "category"):
        client.create_payload_index(a.collection, key, models.PayloadSchemaType.KEYWORD)
    ids = sorted(ids); batch = 256
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        docs = [skb.get_doc_info(n, add_rel=False, compact=True) for n in chunk]
        texts = [d[:1500] for d in docs]
        dv, sv = list(dense.embed(texts)), list(sparse.embed(texts))
        pts = [models.PointStruct(id=n, vector={"dense": d.tolist(), "sparse": models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())},
                                  payload={"text": t, **_attrs(skb, n)}) for n, d, s, t in zip(chunk, dv, sv, texts)]
        client.upsert(a.collection, pts)
    print(json.dumps({"points": len(ids)}))

def _attrs(skb, n: int) -> dict:
    info = skb.node_info.get(n, {}) if hasattr(skb, "node_info") else {}
    return {k: info[k] for k in ("brand", "category") if isinstance(info.get(k), str)}

if __name__ == "__main__":
    main()
