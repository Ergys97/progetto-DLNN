"""
Controprova dell'Esperimento A con metrica di coverage asimmetrica.

La Jaccard simmetrica usata in Exp A penalizza strutturalmente le strategie
con chunk di dimensione diversa dal riferimento recursive_512 (su cui e'
annotato il gold set): un chunk fixed_1024 che CONTIENE interamente il testo
atteso ha Jaccard ~0.5, due chunk sentence_5 che insieme lo coprono hanno
Jaccard ~0.25 ciascuno. La coverage misura invece la frazione dei token del
testo atteso coperti dall'unione dei top-k recuperati: a retrieval perfetto
vale 1.0 per qualunque strategia.

I due bias sono di segno opposto (Jaccard anti-chunk-grandi, coverage
leggermente pro-chunk-grandi): se il ranking coincide sotto entrambe,
il vincitore non e' un artefatto della metrica.

Replica la pipeline di retrieval di Exp A (denso numpy su embeddings .npz
pre-calcolati). Solo embedder locale BGE-M3, nessuna chiamata API.
Output: checkpoint/exp_a_coverage_comparison.json
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as cfg
from src.embeddings import assign_chunk_ids, load_embedder, load_all_embeddings
from src.metrics import coverage_at_k, hit_at_k_coverage

REF_STRATEGY = "recursive_512"
K = 5
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


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

    # Top-5 doc per strategia e query (indipendenti dalla soglia)
    retrieved_docs = {}
    for sname, embs_s in embs_per_strategy.items():
        docs_s = list(strategies[sname])
        sims = q_emb @ embs_s.T
        topk = np.argsort(-sims, axis=1)[:, :K]
        retrieved_docs[sname] = [[docs_s[i] for i in row] for row in topk]

    # Coverage media + Hit@5 a coverage >= soglia, per strategia
    mean_cov = {}
    hit_table = {}
    for sname in embs_per_strategy:
        covs = [coverage_at_k(retrieved_docs[sname][qi], expected_texts(q), K)
                for qi, q in enumerate(indomain)]
        mean_cov[sname] = round(float(np.mean(covs)), 3)
        hit_table[sname] = {}
        for th in THRESHOLDS:
            hits = [hit_at_k_coverage(retrieved_docs[sname][qi], expected_texts(q), K, threshold=th)
                    for qi, q in enumerate(indomain)]
            hit_table[sname][th] = round(float(np.mean(hits)), 3)

    order = sorted(mean_cov, key=mean_cov.get, reverse=True)

    out = {"k": K, "thresholds": THRESHOLDS,
           "mean_coverage_at_5": mean_cov,
           "hit5_by_strategy_coverage_threshold": hit_table,
           "ranking_by_mean_coverage": order}
    out_path = os.path.join(cfg.CHECKPOINT_DIR, "exp_a_coverage_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n=== Coverage media @5 e Hit@5 per soglia di coverage ===")
    header = "Strategia".ljust(16) + "cov_media".ljust(11) + "".join(f"th={th}".ljust(9) for th in THRESHOLDS)
    print(header)
    for sname in order:
        row = (sname.ljust(16) + f"{mean_cov[sname]:.3f}".ljust(11)
               + "".join(f"{hit_table[sname][th]:.3f}".ljust(9) for th in THRESHOLDS))
        print(row)

    rankings = {th: sorted(hit_table, key=lambda s: hit_table[s][th], reverse=True) for th in THRESHOLDS}
    print(f"\nMiglior strategia per coverage media: {order[0]}")
    print(f"Miglior strategia a ogni soglia Hit@5: {[rankings[th][0] for th in THRESHOLDS]}")
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
