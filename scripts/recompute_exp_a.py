"""Ricalcola l'Esperimento A con vero MRR@k + statistiche chunk + breakdown per categoria.

Aggiorna:
  checkpoint/exp_a_retrieval.json          (MRR ora dipende da k: MRR@k)
  checkpoint/exp_a_chunk_stats.json        (statistiche chunk per strategia)
  checkpoint/exp_a_category_breakdown.json (metriche@5 per categoria di query)
  images/exp_a_jaccard_sensitivity.png     (line plot sensibilita' soglia)
  images/exp_a_chunk_stats.png             (n. chunk + distribuzione lunghezze)
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import src.config as cfg
from src.embeddings import assign_chunk_ids, load_embedder, load_all_embeddings
from src.metrics import hit_at_k_textual, recall_at_k_textual, mrr_textual

TOP_K_VALUES = [3, 5, 10]
REF_STRATEGY = 'recursive_512'

# ── Caricamento dati ─────────────────────────────────────────────────────
with open(cfg.CHUNKS_CHECKPOINT, 'rb') as f:
    _ck = pickle.load(f)
strategies, strategies_meta = _ck['strategies'], _ck['strategies_meta']
chunk_ids_per_strategy, _ = assign_chunk_ids(strategies, strategies_meta)

with open(cfg.GOLD_SET_PATH, 'r', encoding='utf-8') as f:
    GOLD_SET = json.load(f)
indomain_queries = [
    q for q in GOLD_SET
    if q['category'].startswith('in_domain') and q['expected_chunk_ids']
]
print(f'Query in-domain valutabili: {len(indomain_queries)}/{len(GOLD_SET)}')

embs_per_strategy = load_all_embeddings(strategies, cfg.emb_ckpt)
embedder = load_embedder(cfg.EMBEDDING_MODEL)

_ref_ids = chunk_ids_per_strategy[REF_STRATEGY]
id_to_text_ref = dict(zip(_ref_ids, list(strategies[REF_STRATEGY])))


def expected_texts(q):
    return [id_to_text_ref[c] for c in q['expected_chunk_ids'] if c in id_to_text_ref]


# ── Retrieval top-10 per strategia (una sola encodifica delle query) ─────
Q = np.asarray(embedder.encode([q['query'] for q in indomain_queries],
                               normalize_embeddings=True), dtype='float32')
rdocs_per_strategy = {}
for sname, E in embs_per_strategy.items():
    docs = list(strategies[sname])
    topk = np.argsort(-(Q @ E.T), axis=1)[:, :max(TOP_K_VALUES)]
    rdocs_per_strategy[sname] = [[docs[i] for i in row] for row in topk]

# ── Exp A con vero MRR@k (lista troncata a k anche per MRR) ──────────────
rows = []
for sname in embs_per_strategy:
    for k in TOP_K_VALUES:
        hits, mrrs, recs = [], [], []
        for qi, q in enumerate(indomain_queries):
            exp = expected_texts(q)
            rdocs = rdocs_per_strategy[sname][qi]
            hits.append(hit_at_k_textual(rdocs, exp, k))
            mrrs.append(mrr_textual(rdocs[:k], exp))
            recs.append(recall_at_k_textual(rdocs, exp, k))
        rows.append({'Strategia': sname, 'Top-k': k,
                     'Hit@k': round(sum(hits) / len(hits), 3),
                     'MRR@k': round(sum(mrrs) / len(mrrs), 3),
                     'Recall@k': round(sum(recs) / len(recs), 3)})

best = max((r for r in rows if r['Top-k'] == 5), key=lambda r: r['Hit@k'])
with open(cfg.EXP_A_PATH, 'w', encoding='utf-8') as f:
    json.dump({'rows': rows, 'best_strategy': best['Strategia']}, f, indent=2)
print(f"[SAVED] {cfg.EXP_A_PATH}  (best: {best['Strategia']})")
for r in sorted(rows, key=lambda r: (-r['Hit@k'], r['Top-k'])):
    print(r)

# ── Breakdown per categoria (metriche @5) ────────────────────────────────
K_BD = 5
breakdown = {}
for cat in ('in_domain_direct', 'in_domain_complex'):
    idxs = [i for i, q in enumerate(indomain_queries) if q['category'] == cat]
    breakdown[cat] = {'n': len(idxs)}
    for sname in embs_per_strategy:
        h, m, r = [], [], []
        for qi in idxs:
            exp = expected_texts(indomain_queries[qi])
            rdocs = rdocs_per_strategy[sname][qi]
            h.append(hit_at_k_textual(rdocs, exp, K_BD))
            m.append(mrr_textual(rdocs[:K_BD], exp))
            r.append(recall_at_k_textual(rdocs, exp, K_BD))
        breakdown[cat][sname] = {
            'Hit@5': round(sum(h) / len(h), 3),
            'MRR@5': round(sum(m) / len(m), 3),
            'Recall@5': round(sum(r) / len(r), 3),
        }
_BD_PATH = os.path.join(cfg.CHECKPOINT_DIR, 'exp_a_category_breakdown.json')
with open(_BD_PATH, 'w', encoding='utf-8') as f:
    json.dump({'k': K_BD, 'breakdown': breakdown}, f, indent=2, ensure_ascii=False)
print(f'[SAVED] {_BD_PATH}')
print(json.dumps(breakdown, indent=1, ensure_ascii=False))

# ── Statistiche chunk per strategia (lunghezze in parole) ────────────────
chunk_stats = {}
for sname, chunks in strategies.items():
    lengths = np.array([len(c.split()) for c in chunks])
    chunk_stats[sname] = {
        'n_chunks': int(len(chunks)),
        'mean_words': round(float(lengths.mean()), 1),
        'median_words': float(np.median(lengths)),
        'p10_words': float(np.percentile(lengths, 10)),
        'p90_words': float(np.percentile(lengths, 90)),
    }
_ST_PATH = os.path.join(cfg.CHECKPOINT_DIR, 'exp_a_chunk_stats.json')
with open(_ST_PATH, 'w', encoding='utf-8') as f:
    json.dump(chunk_stats, f, indent=2)
print(f'[SAVED] {_ST_PATH}')
print(json.dumps(chunk_stats, indent=1))

# ── Grafici ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('images', exist_ok=True)

# 1) Sensibilita' Jaccard (dal checkpoint esistente)
_JAC_PATH = os.path.join(cfg.CHECKPOINT_DIR, 'exp_a_jaccard_sensitivity.json')
with open(_JAC_PATH, 'r', encoding='utf-8') as f:
    _jac = json.load(f)
tab, order = _jac['hit5_by_strategy_threshold'], _jac['ranking_at_0.5']
ths = [float(t) for t in _jac['thresholds']] if 'thresholds' in _jac else [0.3, 0.4, 0.5, 0.6, 0.7]

fig, ax = plt.subplots(figsize=(9, 5))
for s in order:
    ax.plot(ths, [tab[s][str(t)] for t in ths], marker='o', label=s)
ax.axvline(0.5, color='gray', ls='--', lw=1, alpha=0.7)
ax.text(0.505, 0.05, 'soglia usata (0.5)', color='gray', fontsize=9)
ax.set_xlabel('Soglia Jaccard θ_jac')
ax.set_ylabel('Hit@5 testuale')
ax.set_ylim(0, 1.05)
ax.set_title('Exp A — Sensibilità di Hit@5 alla soglia di Jaccard')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/exp_a_jaccard_sensitivity.png', dpi=150)
print('[SAVED] images/exp_a_jaccard_sensitivity.png')

# 2) Statistiche chunk: numero + distribuzione lunghezze
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
names = list(strategies.keys())
axes[0].bar(names, [chunk_stats[s]['n_chunks'] for s in names], color='seagreen')
axes[0].set_title('Numero di chunk per strategia')
axes[0].set_ylabel('n. chunk')
axes[0].tick_params(axis='x', rotation=20)
axes[1].boxplot([[len(c.split()) for c in strategies[s]] for s in names],
                tick_labels=names, showfliers=False)
axes[1].set_title('Distribuzione lunghezza chunk (parole)')
axes[1].set_ylabel('parole per chunk')
axes[1].tick_params(axis='x', rotation=20)
plt.suptitle('Exp A — Statistiche dei chunk per strategia', fontsize=13)
plt.tight_layout()
plt.savefig('images/exp_a_chunk_stats.png', dpi=150)
print('[SAVED] images/exp_a_chunk_stats.png')
