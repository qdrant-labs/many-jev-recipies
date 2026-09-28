from __future__ import annotations
import hashlib, uuid
from qdrant_client import models
from jevqu.schema import Taxonomy

STORE = "_query_understanding"

def _id(collection: str) -> str:
    return str(uuid.UUID(hashlib.sha256(collection.encode()).hexdigest()[:32]))

def _ensure(client) -> None:
    if not client.collection_exists(STORE):
        client.create_collection(STORE, vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT))

def save_taxonomy(client, collection: str, tax: Taxonomy) -> None:
    _ensure(client)
    client.upsert(STORE, [models.PointStruct(id=_id(collection), vector=[0.0],
                                             payload={"collection": collection, "taxonomy": tax.to_json()})])

def load_taxonomy(client, collection: str) -> Taxonomy | None:
    if not client.collection_exists(STORE):
        return None
    pts = client.retrieve(STORE, ids=[_id(collection)], with_payload=True)
    return Taxonomy.from_json(pts[0].payload["taxonomy"]) if pts else None
