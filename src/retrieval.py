"""Dense, BM25, and hybrid retrieval helpers."""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi


# ── Dense retrieval ─────────────────────────────────────────────────────

def retrieve_ids(query: str, embedder, collection, k: int) -> tuple[list, list]:
    """ChromaDB dense retrieval. Returns (chunk_ids, cosine_distances)."""
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=['metadatas', 'distances'],
    )
    ids   = [m['chunk_id'] for m in res['metadatas'][0]]
    dists = res['distances'][0]
    return ids, dists


def retrieve_ids_numpy(
    query: str, embedder, embeddings: np.ndarray, chunk_ids: list, k: int
) -> tuple[list, list]:
    """
    Numpy cosine retrieval — used in Exp A where every strategy has its own
    embedding matrix but only the reference strategy is in ChromaDB.
    Returns (chunk_ids, similarity_scores).
    """
    q_emb = embedder.encode(query, normalize_embeddings=True)
    sims  = embeddings @ q_emb
    top   = np.argsort(-sims)[:k]
    return [chunk_ids[i] for i in top], sims[top].tolist()


# ── OOD gate ────────────────────────────────────────────────────────────

def min_distance(query: str, embedder, collection, k: int = 5) -> float:
    _, dists = retrieve_ids(query, embedder, collection, k)
    return min(dists) if dists else 1.0


def is_ood(query: str, embedder, collection, threshold: float, k: int = 5) -> bool:
    return min_distance(query, embedder, collection, k) > threshold


# ── BM25 ────────────────────────────────────────────────────────────────

def build_bm25(chunks: list, chunk_ids: list) -> tuple[BM25Okapi, dict]:
    """Returns (BM25Okapi index, id_to_idx mapping)."""
    tokenized = [c.lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenized)
    id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
    return bm25, id_to_idx


# ── Hybrid ──────────────────────────────────────────────────────────────

def retrieve_hybrid_ids(
    query: str,
    embedder,
    collection,
    bm25: BM25Okapi,
    chunk_ids: list,
    k: int = 5,
    alpha: float = 0.7,
) -> list:
    """Linear combination: alpha * dense + (1-alpha) * BM25."""
    pool = k * 2

    # Dense scores (cosine similarity = 1 - distance)
    dense_ids, dists = retrieve_ids(query, embedder, collection, k=pool)
    dense_scores = {cid: 1.0 - d for cid, d in zip(dense_ids, dists)}

    # BM25 scores (normalized to [0,1])
    raw     = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(raw)[::-1][:pool]
    max_s   = raw[top_idx[0]] if raw[top_idx[0]] > 0 else 1.0
    bm25_scores = {chunk_ids[i]: raw[i] / max_s for i in top_idx}

    all_ids = set(dense_scores) | set(bm25_scores)
    scored  = {
        cid: alpha * dense_scores.get(cid, 0.0) + (1 - alpha) * bm25_scores.get(cid, 0.0)
        for cid in all_ids
    }
    return sorted(scored, key=scored.get, reverse=True)[:k]


# ── CrossEncoder rerank ─────────────────────────────────────────────────

def retrieve_with_rerank(
    query: str,
    embedder,
    collection,
    cross_encoder,
    k_retrieve: int = 10,
    k_final: int = 5,
) -> tuple[list, list]:
    """Two-stage: dense retrieval then CrossEncoder rerank.
    Returns (final_chunk_ids, final_docs)."""
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    res   = collection.query(
        query_embeddings=q_emb,
        n_results=k_retrieve,
        include=['documents', 'metadatas'],
    )
    docs  = res['documents'][0]
    metas = res['metadatas'][0]

    pairs  = [[query, d] for d in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)[:k_final]

    final_ids  = [m['chunk_id'] for _, _, m in ranked]
    final_docs = [d for _, d, _ in ranked]
    return final_ids, final_docs
