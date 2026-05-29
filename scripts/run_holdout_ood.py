"""
Holdout test del gate OOD su query MAI usate per il tuning di theta.

Calcola min_distance in modo ESATTO (numpy, deterministico) usando gli
embeddings del corpus gia' pre-calcolati in checkpoint/embeddings_recursive_512.npz,
quindi NON richiede ChromaDB ne' il ricalcolo del corpus, e non e' soggetto
alla variabilita' numerica dell'indice HNSW in-memory (vedi report.md §2.2).

La distanza coseno di ChromaDB per vettori normalizzati e' 1 - cos_sim;
min_distance = min_d (1 - cos_sim(q, d)) = 1 - max_d cos_sim(q, d).

Output: checkpoint/holdout_ood_results.json + tabella TPR/FPR a theta=0.40.
Solo embedder locale (BGE-M3), nessuna chiamata LLM/API.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as cfg
from src.embeddings import load_embedder

THETA = 0.40
REFERENCE_STRATEGY = "recursive_512"


def load_corpus_matrix(strategy: str) -> np.ndarray:
    """Carica e L2-normalizza la matrice di embedding del corpus."""
    path = cfg.emb_ckpt(strategy)
    e = np.load(path)["embeddings"].astype("float32")
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    return e / np.maximum(norms, 1e-9)


def main() -> None:
    holdout_path = os.path.join(cfg.CHECKPOINT_DIR, "holdout_set.json")
    with open(holdout_path, encoding="utf-8") as f:
        holdout = json.load(f)
    queries = holdout["queries"]

    print(f"Holdout: {len(queries)} query | theta = {THETA} | strategia = {REFERENCE_STRATEGY}")
    print("Carico embeddings del corpus dal checkpoint...")
    corpus = load_corpus_matrix(REFERENCE_STRATEGY)
    print(f"  {corpus.shape[0]} chunk x {corpus.shape[1]} dim")

    print("Carico embedder BGE-M3 (solo per le query di holdout)...")
    embedder = load_embedder(cfg.EMBEDDING_MODEL)

    q_texts = [q["query"] for q in queries]
    q_emb = embedder.encode(q_texts, normalize_embeddings=True, show_progress_bar=False)
    q_emb = np.asarray(q_emb, dtype="float32")

    # min_distance esatta = 1 - max cosine similarity
    sims = q_emb @ corpus.T                 # (n_query, n_chunk)
    min_dist = 1.0 - sims.max(axis=1)       # (n_query,)

    results = []
    for q, d in zip(queries, min_dist):
        status = "refused_ood" if d > THETA else "answered"
        results.append({
            "id": q["id"],
            "category": q["category"],
            "materia": q.get("materia"),
            "query": q["query"],
            "min_distance": round(float(d), 4),
            "status": status,
        })

    # ── Metriche: classe positiva = "deve essere bloccata" ──────────────
    should_block = [r for r in results if r["category"] in ("out_of_domain", "prompt_injection")]
    should_pass  = [r for r in results if r["category"].startswith("in_domain")]

    tp = sum(r["status"] == "refused_ood" for r in should_block)   # OOD correttamente bloccate
    fp = sum(r["status"] == "refused_ood" for r in should_pass)    # in-domain bloccate per errore

    tpr = tp / len(should_block) if should_block else float("nan")
    fpr = fp / len(should_pass) if should_pass else float("nan")

    summary = {
        "theta": THETA,
        "n_total": len(results),
        "n_should_block": len(should_block),
        "n_should_pass": len(should_pass),
        "TPR_ood_blocked": round(tpr, 4),
        "FPR_indomain_blocked": round(fpr, 4),
        "tp": tp, "fp": fp,
    }

    out = {"summary": summary, "results": results}
    out_path = os.path.join(cfg.CHECKPOINT_DIR, "holdout_ood_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ── Stampa ──────────────────────────────────────────────────────────
    print("\n=== Risultati holdout (theta = 0.40) ===")
    print("ID    Categoria           min_dist  Stato         Esito")
    for r in sorted(results, key=lambda x: x["min_distance"]):
        should = "block" if r["category"] in ("out_of_domain", "prompt_injection") else "pass"
        ok = (should == "block" and r["status"] == "refused_ood") or \
             (should == "pass" and r["status"] == "answered")
        mark = "OK" if ok else "MISS"
        print(f"{r['id']:<5} {r['category']:<18} {r['min_distance']:>7.4f}  {r['status']:<12} {mark}")

    print("\n=== Sommario ===")
    print(f"  Should-block (OOD+inj): {summary['n_should_block']}  -> TPR = {tpr:.1%} ({tp} bloccate)")
    print(f"  Should-pass (in-domain): {summary['n_should_pass']}  -> FPR = {fpr:.1%} ({fp} bloccate per errore)")
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
