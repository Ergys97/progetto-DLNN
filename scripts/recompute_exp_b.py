"""Rigenera l'Esperimento B (gate OOD) con distanze numpy ESATTE.

Le distanze min coseno sono calcolate direttamente sugli embeddings .npz
(deterministico), eliminando la variabilita' HNSW di ChromaDB in-memory che
aveva contaminato il checkpoint precedente (es. q01 a 0.4554 invece del
valore pulito). Stesso metodo gia' usato per l'holdout OOD.

Sweep esteso sotto 0.40 per verificare che l'ottimo non sia un artefatto
del bordo della griglia.

Aggiorna: checkpoint/exp_b_ood.json, images/ood_gate_analysis.png
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import src.config as cfg
from src.embeddings import load_embedder, load_all_embeddings

REF_STRATEGY = 'recursive_512'
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

with open(cfg.CHUNKS_CHECKPOINT, 'rb') as f:
    strategies = pickle.load(f)['strategies']
with open(cfg.GOLD_SET_PATH, 'r', encoding='utf-8') as f:
    GOLD_SET = json.load(f)

E = load_all_embeddings({REF_STRATEGY: strategies[REF_STRATEGY]}, cfg.emb_ckpt)[REF_STRATEGY]
embedder = load_embedder(cfg.EMBEDDING_MODEL)

Q = np.asarray(embedder.encode([q['query'] for q in GOLD_SET],
                               normalize_embeddings=True), dtype='float32')
min_dists = 1.0 - (Q @ E.T).max(axis=1)

dist_records = [
    {'id': q['id'], 'category': q['category'], 'min_dist': round(float(d), 4)}
    for q, d in zip(GOLD_SET, min_dists)
]
df_cat = {}
sweep_rows = []
pos = np.array([q['category'] in ('out_of_domain', 'prompt_injection') for q in GOLD_SET])
neg = np.array([q['category'].startswith('in_domain') for q in GOLD_SET])
for theta in THRESHOLDS:
    blocked = min_dists > theta
    tpr = float((blocked & pos).sum() / pos.sum())
    fpr = float((blocked & neg).sum() / neg.sum())
    sweep_rows.append({'threshold': theta, 'TPR': round(tpr, 3),
                       'FPR': round(fpr, 3), 'Youden J': round(tpr - fpr, 3)})

best = max(sweep_rows, key=lambda r: r['Youden J'])
with open(cfg.EXP_B_PATH, 'w', encoding='utf-8') as f:
    json.dump({'dist_records': dist_records, 'sweep_rows': sweep_rows,
               'ood_threshold': best['threshold'],
               'distance_method': 'numpy_exact'}, f, indent=2)
print(f'[SAVED] {cfg.EXP_B_PATH}')

import pandas as pd
df = pd.DataFrame(dist_records)
print(pd.DataFrame(sweep_rows).to_string(index=False))
print(f"\nMiglior J sulla griglia: theta={best['threshold']} (J={best['Youden J']})")
print()
print(df.groupby('category')['min_dist'].agg(['count', 'min', 'mean', 'max']).round(4))
print()
ind = df[df['category'].str.startswith('in_domain')].sort_values('min_dist', ascending=False)
print('FP a theta=0.40:', ind[ind['min_dist'] > 0.40][['id', 'min_dist']].to_dict('records'))
pi = df[df['category'] == 'prompt_injection'].sort_values('min_dist')
print('Prompt injection:', pi[['id', 'min_dist']].to_dict('records'))
ood = df[df['category'] == 'out_of_domain']
print('OOD min:', ood['min_dist'].min())

# ── Grafico ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df_sweep = pd.DataFrame(sweep_rows)
THETA = 0.40  # soglia finale della pipeline (vedi report §4.2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(df_sweep['FPR'], df_sweep['TPR'], 'o-', color='steelblue', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
for _, r in df_sweep.iterrows():
    ax.annotate(f'{r["threshold"]:.2f}', (r['FPR'], r['TPR']), fontsize=7,
                textcoords='offset points', xytext=(5, -2))
ax.set(xlabel='FPR', ylabel='TPR', title='Curva ROC Gate OOD')
ax.grid(True, alpha=0.3)

ax = axes[1]
cmap = {'in_domain_direct': '#2ecc71', 'in_domain_complex': '#27ae60',
        'out_of_domain': '#e74c3c', 'prompt_injection': '#c0392b'}
for cat in df['category'].unique():
    sub = df[df['category'] == cat]
    ax.scatter([cat] * len(sub), sub['min_dist'], color=cmap.get(cat, 'gray'), s=60, alpha=0.7)
ax.axhline(THETA, color='black', linestyle='--', alpha=0.6, label=f'θ = {THETA}')
ax.set(ylabel='min cosine distance', title='Distanze per categoria (numpy esatto)')
ax.tick_params(axis='x', rotation=20)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('images/ood_gate_analysis.png', dpi=150)
print('[SAVED] images/ood_gate_analysis.png')
