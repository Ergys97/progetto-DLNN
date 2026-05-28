"""
Annotation helper per il gold set RAG.

Uso:
    python scripts/annotate_gold_set.py

Per ogni query in-domain non ancora annotata lo script mostra i top-20 chunk
recuperati da ChromaDB. Inserisci i numeri (es. "1 3 5") dei chunk rilevanti,
poi INVIO per passare alla query successiva. Per OOD e prompt_injection
lascia vuoto (INVIO) — expected_chunk_ids=[] e' il comportamento corretto.

Prerequisiti:
    pip install chromadb sentence-transformers numpy
    I file checkpoint/chunks.pkl e checkpoint/embeddings_*.npz devono esistere.
"""

import os
import sys
import json
import pickle
import hashlib
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
CHECKPOINT_DIR = _HERE.parent / "checkpoint"
STRATEGY        = "recursive_512"   # strategia di riferimento (cambia se preferisci un'altra)
EMBEDDING_MODEL = "BAAI/bge-m3"
TOP_K           = 20


def make_chunk_id(source: str, chunk_idx: int, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f'{source.replace(" ","_").replace("/","_")}::{chunk_idx:04d}::{h}'


def build_collection(strategy: str):
    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"Caricamento chunks.pkl ...", flush=True)
    with open(CHECKPOINT_DIR / "chunks.pkl", "rb") as f:
        saved = pickle.load(f)
    strategies      = saved["strategies"]
    strategies_meta = saved["strategies_meta"]

    chunks = strategies[strategy]
    metas  = strategies_meta[strategy]

    # Ricostruisci ID deterministici
    counters: dict[str, int] = {}
    ids: list[str] = []
    for chunk, meta in zip(chunks, metas):
        src = meta["source"]
        counters.setdefault(src, 0)
        ids.append(make_chunk_id(src, counters[src], chunk))
        counters[src] += 1

    # Carica embeddings pre-calcolati
    ckpt = CHECKPOINT_DIR / f"embeddings_{strategy}.npz"
    if not ckpt.exists():
        sys.exit(f"[ERRORE] {ckpt} non trovato. Esegui prima il notebook.")
    embs = np.load(ckpt)["embeddings"]

    # Crea collection ChromaDB in-memory
    print("Creazione ChromaDB in-memory ...", flush=True)
    client = chromadb.Client()
    col = client.create_collection(strategy, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=ids,
        embeddings=embs.tolist(),
        documents=chunks,
        metadatas=[{**m, "chunk_id": cid} for m, cid in zip(metas, ids)],
    )
    print(f"Collection '{strategy}': {col.count()} chunk caricati.", flush=True)

    # Carica embedder per le query
    print(f"Caricamento embedder {EMBEDDING_MODEL} ...", flush=True)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print(f"Embedder pronto su {device}.\n", flush=True)

    return col, embedder, ids


def main():
    col, embedder, _ = build_collection(STRATEGY)

    gs_path = CHECKPOINT_DIR / "gold_set.json"
    with open(gs_path, "r", encoding="utf-8") as f:
        gold_set = json.load(f)

    total = len(gold_set)
    annotated_before = sum(1 for q in gold_set if q.get("expected_chunk_ids"))
    print(f"{'='*60}")
    print(f"ANNOTATION HELPER  |  Strategia: {STRATEGY}  |  Top-k: {TOP_K}")
    print(f"Gold set: {total} query, {annotated_before} gia annotate")
    print(f"{'='*60}\n")

    changed = False
    for q in gold_set:
        if q.get("expected_chunk_ids"):
            print(f"[SKIP] {q['id']} — gia annotato ({len(q['expected_chunk_ids'])} chunk)")
            continue

        print(f"\n{'─'*60}")
        print(f"[{q['id']}]  categoria: {q['category']}")
        print(f"QUERY: {q['query']}")
        if q.get("gt_answer"):
            print(f"GT:    {q['gt_answer'][:120]}...")
        print(f"{'─'*60}")

        qemb = embedder.encode(q["query"], normalize_embeddings=True)
        results = col.query(
            query_embeddings=[qemb.tolist()],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        docs   = results["documents"][0]
        metas  = results["metadatas"][0]
        dists  = results["distances"][0]

        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            cid = meta.get("chunk_id", "?")
            src = meta.get("source", "?")
            preview = doc[:200].replace("\n", " ")
            print(f"\n  [{rank+1:02d}] dist={dist:.4f}  src={src}")
            print(f"       id={cid}")
            print(f"       {preview}...")

        raw = input(
            "\n  Indici rilevanti (es: 1 3 5) — INVIO per skip (OOD/injection): "
        ).strip()

        if raw:
            try:
                idxs = [int(x) - 1 for x in raw.split()]
                q["expected_chunk_ids"] = [
                    metas[i].get("chunk_id", "?") for i in idxs if 0 <= i < len(metas)
                ]
                print(f"  -> Salvati {len(q['expected_chunk_ids'])} chunk ID")
                changed = True
            except ValueError:
                print("  [WARN] Input non valido, saltato.")
        else:
            print("  -> Saltato (expected_chunk_ids rimane [])")

    if changed:
        with open(gs_path, "w", encoding="utf-8") as f:
            json.dump(gold_set, f, ensure_ascii=False, indent=2)
        annotated_after = sum(1 for q in gold_set if q.get("expected_chunk_ids"))
        print(f"\nGold set salvato: {gs_path}")
        print(f"Annotate: {annotated_after}/{total} query")
    else:
        print("\nNessuna modifica.")


if __name__ == "__main__":
    main()
