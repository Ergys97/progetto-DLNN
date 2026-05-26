"""ChromaDB setup, embedding loading, chunk ID assignment."""
from __future__ import annotations
import hashlib
import os

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import torch


def make_chunk_id(source: str, chunk_idx: int, text: str) -> str:
    h = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return f'{source.replace(" ", "_").replace("/", "_")}::{chunk_idx:04d}::{h}'


def assign_chunk_ids(
    strategies: dict, strategies_meta: dict
) -> tuple[dict, dict]:
    """
    Generates deterministic chunk IDs for every strategy.
    Returns (chunk_ids_per_strategy, updated_strategies_meta).
    """
    chunk_ids_per_strategy: dict[str, list] = {}
    new_metas: dict[str, list] = {}
    for name, chunks in strategies.items():
        counters: dict[str, int] = {}
        ids, metas_new = [], []
        for chunk, meta in zip(chunks, strategies_meta[name]):
            src = meta['source']
            counters.setdefault(src, 0)
            cid = make_chunk_id(src, counters[src], chunk)
            counters[src] += 1
            ids.append(cid)
            metas_new.append({**meta, 'chunk_id': cid})
        chunk_ids_per_strategy[name] = ids
        new_metas[name] = metas_new
    return chunk_ids_per_strategy, new_metas


def load_embedder(model_name: str, device: str | None = None) -> SentenceTransformer:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedder = SentenceTransformer(model_name, device=device)
    print(f'Embedder: {model_name} su {device}')
    return embedder


def load_all_embeddings(strategies: dict, emb_ckpt_fn) -> dict[str, np.ndarray]:
    """
    Returns name -> L2-normalized float32 ndarray.
    Skips strategies whose .npz checkpoint is missing.
    """
    result: dict[str, np.ndarray] = {}
    for name in strategies:
        path = emb_ckpt_fn(name)
        if os.path.exists(path):
            e = np.load(path)['embeddings'].astype('float32')
            norms = np.linalg.norm(e, axis=1, keepdims=True)
            result[name] = e / np.maximum(norms, 1e-9)
            print(f'  {name}: {e.shape[0]} embeddings caricate')
        else:
            print(f'  [SKIP] {name}: .npz non trovato')
    return result


def build_chromadb(
    strategies: dict,
    chunk_ids_per_strategy: dict,
    strategies_meta: dict,
    emb_ckpt_fn,
    strategies_to_load: list | None = None,
    batch: int = 5000,
) -> tuple[chromadb.Client, dict]:
    """
    Builds in-memory ChromaDB collections.
    strategies_to_load: subset of strategy names to load (None = all).
    Returns (chroma_client, collections_dict).
    """
    client = chromadb.Client()
    collections: dict = {}
    names = strategies_to_load or list(strategies.keys())

    for name in names:
        if name not in strategies:
            print(f'  [SKIP] {name}: non in strategies')
            continue
        path = emb_ckpt_fn(name)
        if not os.path.exists(path):
            print(f'  [SKIP] {name}: .npz non trovato')
            continue

        embs = np.load(path)['embeddings']
        try:
            client.delete_collection(name)
        except Exception:
            pass
        col = client.create_collection(name, metadata={'hnsw:space': 'cosine'})

        ids_s    = chunk_ids_per_strategy[name]
        metas_s  = strategies_meta[name]
        chunks_s = strategies[name]
        for start in range(0, len(chunks_s), batch):
            end = min(start + batch, len(chunks_s))
            col.add(
                ids=ids_s[start:end],
                embeddings=embs[start:end].tolist(),
                documents=list(chunks_s[start:end]),
                metadatas=metas_s[start:end],
            )
        collections[name] = col
        print(f'  {name}: {col.count()} chunk in ChromaDB')

    return client, collections
