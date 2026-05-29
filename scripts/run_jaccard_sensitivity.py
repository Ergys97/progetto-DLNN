"""
Analisi di sensibilità della metrica Hit@5 testuale (Esperimento A) alla soglia
di Jaccard, che nel report è fissata arbitrariamente a 0.5.

Replica ESATTAMENTE la pipeline di Exp A (retrieval denso numpy sugli embeddings
.npz pre-calcolati + hit_at_k_textual), ma ripete il calcolo per più soglie di
Jaccard. Serve a mostrare che il ranking delle strategie di chunking è robusto e
non un artefatto della soglia scelta.

Solo embedder locale (BGE-M3) + embeddings pre-calcolati. Nessuna chiamata API.
Output: checkpoint/exp_a_jaccard_sensitivity.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import src.config as cfg
from src.embeddings import assign_chunk_ids, load_embedder, load_all_embeddings
from src.metrics import hit_at_k_textual

REF_STRATEGY = "recursive_512"
K = 5
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def main() -> None:
    with open(cfg.CHUNKS_CHECKPOINT, "rb") as f:
        saved = pickle.load(f)
    strategies = saved["strategies"]
    strategies_meta = saved["strategies_meta"]
    chunk_ids_per_strategy, strategies_meta = assign_chunk_ids(strategies, strategies_meta)

    with open(cfg.GOLD_SET_PATH, encoding="utf-8") as f:
        gold = json.load(f)
    indomain = [q for q in gold if q["category"].startswith("in_domain") and q["expected_chunk_ids"]]
    print(f"Query in-domain valutabili: {len(indomain)}")

    embs_per_strategy = load_all_embeddings(strategies, cfg.emb_ckpt)

    # Lookup id -> testo sulla strategia di riferimento (come Exp A)
    ref_ids = chunk_ids_per_strategy[REF_STRATEGY]
    id_to_text_ref = dict(zip(ref_ids, list(strategies[REF_STRATEGY])))

    def expected_texts(q):
        return [id_to_text_ref[c] for c in q["expected_chunk_ids"] if c in id_to_text_ref]

    print("Carico embedder BGE-M3...")
    embedder = load_embedder(cfg.EMBEDDING_MODEL)
    q_emb = np.asarray(
        embedder.encode([q["query"] for q in indomain], normalize_embeddings=True,
                        show_progress_bar=False), dtype="float32")

    # Pre-calcola i top-5 doc per ogni strategia e query (indipendenti dalla soglia)
    retrieved_docs = {}  # sname -> list (per query) di list di doc top-5
    for sname, embs_s in embs_per_strategy.items():
        docs_s = list(strategies[sname])
        sims = q_emb @ embs_s.T                     # (n_query, n_chunk)
        topk = np.argsort(-sims, axis=1)[:, :K]     # (n_query, K)
        retrieved_docs[sname] = [[docs_s[i] for i in row] for row in topk]

    # Hit@5 per strategia per soglia
    table = {}  # sname -> {threshold: hit5}
    for sname in embs_per_strategy:
        table[sname] = {}
        for th in THRESHOLDS:
            hits = [
                hit_at_k_textual(retrieved_docs[sname][qi], expected_texts(q), K, threshold=th)
                for qi, q in enumerate(indomain)
            ]
            table[sname][th] = round(float(np.mean(hits)), 3)

    # Ordina strategie per Hit@5 alla soglia di riferimento 0.5
    order = sorted(table, key=lambda s: table[s][0.5], reverse=True)

    out = {"k": K, "thresholds": THRESHOLDS, "metric": "Hit@5 textual",
           "hit5_by_strategy_threshold": table, "ranking_at_0.5": order}
    out_path = os.path.join(cfg.CHECKPOINT_DIR, "exp_a_jaccard_sensitivity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Stampa tabella
    print("\n=== Hit@5 testuale per soglia di Jaccard ===")
    header = "Strategia".ljust(16) + "".join(f"θ={th}".ljust(9) for th in THRESHOLDS)
    print(header)
    for sname in order:
        row = sname.ljust(16) + "".join(f"{table[sname][th]:.3f}".ljust(9) for th in THRESHOLDS)
        print(row)

    # Verifica robustezza del ranking
    rankings = {th: sorted(table, key=lambda s: table[s][th], reverse=True) for th in THRESHOLDS}
    stable = all(rankings[th][0] == order[0] for th in THRESHOLDS)
    print(f"\nMiglior strategia a ogni soglia: {[rankings[th][0] for th in THRESHOLDS]}")
    print(f"Ranking del vincitore stabile su tutte le soglie: {stable}")
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
