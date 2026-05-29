"""
Holdout di RETRIEVAL: genera i candidati per l'annotazione e/o valuta le
metriche di retrieval (Hit@5/MRR/Recall@5 + bootstrap CI) sulle query in-domain
di holdout una volta che hanno `expected_chunk_ids`.

Tutto OFFLINE: retrieval = prodotto matriciale numpy sugli embeddings .npz
pre-calcolati della strategia di riferimento (recursive_512). Nessuna API.

Modalità:
    python scripts/run_holdout_retrieval.py dump   # crea holdout_candidates.json + holdout_audit.md
    python scripts/run_holdout_retrieval.py eval    # calcola metriche (richiede expected_chunk_ids)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import src.config as cfg
from src.embeddings import assign_chunk_ids, load_embedder
from src.metrics import hit_at_k, mrr, recall_at_k

REF_STRATEGY = "recursive_512"
TOP_K_DUMP = 20
K = 5


def _load_ref():
    with open(cfg.CHUNKS_CHECKPOINT, "rb") as f:
        saved = pickle.load(f)
    strategies, strategies_meta = saved["strategies"], saved["strategies_meta"]
    chunk_ids_per_strategy, _ = assign_chunk_ids(strategies, strategies_meta)
    ids = chunk_ids_per_strategy[REF_STRATEGY]
    docs = list(strategies[REF_STRATEGY])
    metas = strategies_meta[REF_STRATEGY]
    e = np.load(cfg.emb_ckpt(REF_STRATEGY))["embeddings"].astype("float32")
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    return ids, docs, metas, e


def _holdout_indomain():
    with open(os.path.join(cfg.CHECKPOINT_DIR, "holdout_set.json"), encoding="utf-8") as f:
        h = json.load(f)
    return h, [q for q in h["queries"] if q["category"].startswith("in_domain")]


def dump():
    ids, docs, metas, E = _load_ref()
    _, indomain = _holdout_indomain()
    embedder = load_embedder(cfg.EMBEDDING_MODEL)
    Q = np.asarray(embedder.encode([q["query"] for q in indomain], normalize_embeddings=True),
                   dtype="float32")
    sims = Q @ E.T
    cand = {}
    md = ["# Holdout — candidati per annotazione (top-20, recursive_512)\n"]
    for qi, q in enumerate(indomain):
        order = np.argsort(-sims[qi])[:TOP_K_DUMP]
        rows = []
        md.append(f"\n## {q['id']} [{q.get('materia','?')}] — {q['query']}\n")
        for rank, i in enumerate(order, 1):
            dist = float(1.0 - sims[qi, i])
            src = metas[i].get("source", "?")
            preview = " ".join(docs[i].split())[:500]
            rows.append({"rank": rank, "chunk_id": ids[i], "source": src,
                         "distance": round(dist, 4), "preview": preview})
            md.append(f"- **[{rank:02d}]** d={dist:.4f} `{ids[i]}` _(src: {src})_\n  {preview}\n")
        cand[q["id"]] = rows
    with open(os.path.join(cfg.CHECKPOINT_DIR, "holdout_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(cand, f, indent=2, ensure_ascii=False)
    with open(os.path.join(cfg.CHECKPOINT_DIR, "holdout_audit.md"), "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"[SAVED] holdout_candidates.json + holdout_audit.md ({len(indomain)} query, top-{TOP_K_DUMP})")


def eval_():
    ids, docs, metas, E = _load_ref()
    h, indomain = _holdout_indomain()
    annotated = [q for q in indomain if q.get("expected_chunk_ids")]
    if not annotated:
        sys.exit("[ERRORE] Nessuna query con expected_chunk_ids. Annota prima holdout_set.json.")
    embedder = load_embedder(cfg.EMBEDDING_MODEL)
    Q = np.asarray(embedder.encode([q["query"] for q in annotated], normalize_embeddings=True),
                   dtype="float32")
    sims = Q @ E.T
    per_q = []
    for qi, q in enumerate(annotated):
        order = np.argsort(-sims[qi])[:max(K, 10)]
        ret_ids = [ids[i] for i in order]
        exp = set(q["expected_chunk_ids"])
        per_q.append({"id": q["id"],
                      "Hit@5": hit_at_k(ret_ids, exp, K),
                      "MRR": mrr(ret_ids, exp),
                      "Recall@5": recall_at_k(ret_ids, exp, K)})

    def boot(vals, B=10000, seed=42):
        rng = np.random.default_rng(seed)
        v = np.asarray(vals, float)
        n = len(v)
        means = v[rng.integers(0, n, size=(B, n))].mean(1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return round(float(v.mean()), 3), [round(float(lo), 3), round(float(hi), 3)]

    summary = {"n": len(per_q)}
    for m in ["Hit@5", "MRR", "Recall@5"]:
        mean, ci = boot([r[m] for r in per_q])
        summary[m] = {"mean": mean, "ci95": ci}

    out = {"strategy": REF_STRATEGY, "k": K, "summary": summary, "per_query": per_q}
    with open(os.path.join(cfg.CHECKPOINT_DIR, "holdout_retrieval_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n=== Holdout retrieval (n={len(per_q)} query in-domain annotate, {REF_STRATEGY}) ===")
    for m in ["Hit@5", "MRR", "Recall@5"]:
        s = summary[m]
        print(f"  {m:<9} = {s['mean']:.3f}  [95% CI {s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]")
    print("\nPer-query:")
    for r in per_q:
        print(f"  {r['id']}: Hit@5={r['Hit@5']:.0f}  MRR={r['MRR']:.3f}  Recall@5={r['Recall@5']:.3f}")
    print("\n[SAVED] holdout_retrieval_results.json")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dump"
    (dump if mode == "dump" else eval_)()
