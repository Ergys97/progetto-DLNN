"""
Exp F — Confronto Multi-LLM (standalone, run in background).
Esegue la pipeline RAG su tutte le 34 query in-domain per ogni modello,
giudica ogni risposta, salva il checkpoint dopo ogni query.
"""
import sys, os, json, time, importlib, pickle

sys.stdout.reconfigure(encoding='utf-8')

# ── Setup path ──────────────────────────────────────────────────────────
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

import src.config as cfg
import src.embeddings as emb_mod
import src.retrieval as ret_mod
import src.pipeline as pipe_mod
import src.judge as judge_mod

for m in [cfg, emb_mod, ret_mod, pipe_mod, judge_mod]:
    importlib.reload(m)

# ── Load gold set ────────────────────────────────────────────────────────
with open(cfg.GOLD_SET_PATH, 'r', encoding='utf-8') as f:
    GOLD_SET = json.load(f)
queries_multi = [q for q in GOLD_SET if q['category'].startswith('in_domain')]
print(f'Query in-domain: {len(queries_multi)}')

# ── Load components ──────────────────────────────────────────────────────
with open(cfg.CHUNKS_CHECKPOINT, 'rb') as f:
    saved = pickle.load(f)
strategies      = saved['strategies']
strategies_meta = saved['strategies_meta']

REF = 'recursive_512'
chunk_ids_per_strategy, strategies_meta = emb_mod.assign_chunk_ids(strategies, strategies_meta)
embedder = emb_mod.load_embedder(cfg.EMBEDDING_MODEL)
emb_mod.load_all_embeddings(strategies, cfg.emb_ckpt)  # pre-warm cache
_, collections = emb_mod.build_chromadb(
    strategies, chunk_ids_per_strategy, strategies_meta,
    cfg.emb_ckpt, strategies_to_load=[REF],
)

best_chunks = list(strategies[REF])
best_ids    = chunk_ids_per_strategy[REF]

bm25, _ = ret_mod.build_bm25(best_chunks, best_ids)

from sentence_transformers import CrossEncoder
print('Carico CrossEncoder...')
cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3')

# Load OOD threshold and hybrid alpha from checkpoints
with open(cfg.EXP_B_PATH, 'r', encoding='utf-8') as f:
    _b = json.load(f)
cfg.OOD_THRESHOLD = _b['ood_threshold']
with open(cfg.EXP_D_PATH, 'r', encoding='utf-8') as f:
    _d = json.load(f)
cfg.HYBRID_ALPHA = _d['hybrid_alpha']
print(f'OOD_THRESHOLD={cfg.OOD_THRESHOLD}  HYBRID_ALPHA={cfg.HYBRID_ALPHA}')

components = pipe_mod.RAGComponents(
    embedder      = embedder,
    collection    = collections[REF],
    bm25          = bm25,
    chunk_ids     = best_ids,
    chunks        = best_chunks,
    cross_encoder = cross_encoder,
)

# ── Build clients ────────────────────────────────────────────────────────
clients = pipe_mod.build_clients(cfg)

judge_client = clients[cfg.JUDGE_BACKEND]

# ── Model list ───────────────────────────────────────────────────────────
ALL_MODELS = [
    {'id': 'gemma4:e2b',              'label': 'Gemma 4 2B (locale)',   'backend': 'ollama'},
    {'id': 'gemma4:e4b',              'label': 'Gemma 4 4B (locale)',   'backend': 'ollama'},
    {'id': 'llama-3.3-70b-versatile', 'label': 'Llama 3.3 70B (Groq)', 'backend': 'groq'},
    {'id': 'deepseek-v4-flash',       'label': 'DeepSeek V4 Flash',     'backend': 'deepseek'},
]

# ── Resume ───────────────────────────────────────────────────────────────
if os.path.exists(cfg.MULTI_LLM_RESULTS_PATH):
    with open(cfg.MULTI_LLM_RESULTS_PATH, 'r', encoding='utf-8') as f:
        multi_records = json.load(f)
    done_keys = {(r['model'], r['id']) for r in multi_records}
    print(f'[RESUME] {len(multi_records)} record gia presenti.')
else:
    multi_records, done_keys = [], set()

def save():
    with open(cfg.MULTI_LLM_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(multi_records, f, ensure_ascii=False, indent=2)

# ── Main loop ────────────────────────────────────────────────────────────
judge_delay = judge_mod.judge_delay(cfg.JUDGE_BACKEND, cfg.JUDGE_MODEL)

for model in ALL_MODELS:
    gen_client = clients.get(model['backend'])
    if gen_client is None:
        print(f'[SKIP] {model["label"]}: client non disponibile')
        continue

    todo = [q for q in queries_multi if (model['label'], q['id']) not in done_keys]
    if not todo:
        print(f'[SKIP] {model["label"]}: gia completato')
        continue

    print(f'\n=== {model["label"]} ===  ({len(todo)} query da eseguire)')

    for q in todo:
        try:
            t0 = time.time()
            out = pipe_mod.rag_pipeline(
                q['query'], components, gen_client, model['id'],
                cfg.OOD_THRESHOLD, cfg.HYBRID_ALPHA,
                generator_backend=model['backend'],
            )
            gen_lat = round(time.time() - t0, 2)

            if out['status'] != 'answered':
                multi_records.append({
                    'model': model['label'], 'id': q['id'],
                    'category': q['category'], 'answer': out['answer'],
                    'gt_answer': q['gt_answer'],
                    'faithfulness': None, 'answer_relevance': None,
                    'judge_reasoning': f'skipped: {out["status"]}',
                    'latency_s': out['latency_s'],
                })
                done_keys.add((model['label'], q['id']))
                save()
                print(f'  [{q["id"]}] {out["status"]}  gen_lat={gen_lat}s')
                continue

            j = judge_mod.call_judge_with_retry(
                {'id': q['id'], 'query': q['query'], 'context': out['context'],
                 'answer': out['answer'], 'gt_answer': q['gt_answer']},
                judge_client, cfg.JUDGE_MODEL, cfg.JUDGE_BACKEND,
            )
            multi_records.append({
                'model': model['label'], 'id': q['id'],
                'category': q['category'],
                'faithfulness': j['faithfulness'],
                'answer_relevance': j['answer_relevance'],
                'judge_reasoning': j['reasoning'],
                'latency_s': out['latency_s'],
                'answer': out['answer'],
                'gt_answer': q['gt_answer'],
            })
            done_keys.add((model['label'], q['id']))
            save()
            print(f'  [{q["id"]}] F={j["faithfulness"]} AR={j["answer_relevance"]}  gen={out["latency_s"]}s')

            time.sleep(judge_delay)

        except Exception as e:
            print(f'  [{q["id"]}] ERR: {e}')

        if model['backend'] == 'groq':
            time.sleep(2.5)

    # Riepilogo per modello
    model_recs = [r for r in multi_records if r['model'] == model['label']]
    judged = [r for r in model_recs if r.get('faithfulness') is not None]
    if judged:
        f_avg  = sum(r['faithfulness']     for r in judged) / len(judged)
        ar_avg = sum(r['answer_relevance'] for r in judged) / len(judged)
        l_avg  = sum(r['latency_s']        for r in model_recs) / len(model_recs)
        print(f'  --> F_avg={f_avg:.2f}  AR_avg={ar_avg:.2f}  lat_avg={l_avg:.2f}s  ({len(judged)}/{len(model_recs)} giudicati)')

# ── Riepilogo finale ──────────────────────────────────────────────────────
print('\n=== RIEPILOGO FINALE ===')
from collections import defaultdict
by_model = defaultdict(list)
for r in multi_records:
    by_model[r['model']].append(r)

for mlabel, recs in by_model.items():
    judged = [r for r in recs if r.get('faithfulness') is not None]
    if judged:
        f_avg  = sum(r['faithfulness']     for r in judged) / len(judged)
        ar_avg = sum(r['answer_relevance'] for r in judged) / len(judged)
        l_avg  = sum(r['latency_s']        for r in recs)   / len(recs)
        print(f'  {mlabel:<28} F={f_avg:.2f}  AR={ar_avg:.2f}  lat={l_avg:.2f}s  n={len(judged)}')
    else:
        print(f'  {mlabel:<28} nessuna risposta giudicata')

print('\nDone.')
