import os
from dotenv import load_dotenv

load_dotenv(override=False)

# ── API keys ────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.environ.get('GROQ_API_KEY')
GEMINI_API_KEY    = os.environ.get('GEMINI_API_KEY')
DEEPSEEK_API_KEY  = os.environ.get('DEEPSEEK_API_KEY')

# ── Backend URLs ────────────────────────────────────────────────────────
OLLAMA_BASE_URL   = 'http://localhost:11434/v1'
DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
GEMINI_BASE_URL   = 'https://generativelanguage.googleapis.com/v1beta/openai/'

# ── Model selection (override in the notebook if needed) ────────────────
GENERATOR_BACKEND = 'ollama'          # 'ollama' | 'groq' | 'deepseek'
GENERATOR_MODEL   = 'gemma4:e4b'
JUDGE_BACKEND     = 'deepseek'        # 'ollama' | 'groq' | 'gemini' | 'deepseek'
JUDGE_MODEL       = 'deepseek-v4-pro'
EMBEDDING_MODEL   = 'BAAI/bge-m3'

# ── Pipeline hyperparameters ────────────────────────────────────────────
OOD_THRESHOLD  = 0.6
TOP_K_RETRIEVE = 10
TOP_K_FINAL    = 5
HYBRID_ALPHA   = 0.7

# ── Checkpoints ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.normpath(os.path.join(_HERE, '..', 'checkpoint'))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CORPUS_CHECKPOINT      = os.path.join(CHECKPOINT_DIR, 'corpus.pkl')
CHUNKS_CHECKPOINT      = os.path.join(CHECKPOINT_DIR, 'chunks.pkl')
GOLD_SET_PATH          = os.path.join(CHECKPOINT_DIR, 'gold_set.json')
PIPELINE_RESULTS_PATH  = os.path.join(CHECKPOINT_DIR, 'pipeline_results.json')
JUDGE_RESULTS_PATH     = os.path.join(CHECKPOINT_DIR, 'judge_results.json')
MULTI_LLM_RESULTS_PATH = os.path.join(CHECKPOINT_DIR, 'multi_llm_results.json')
MANUAL_EVAL_PATH       = os.path.join(CHECKPOINT_DIR, 'manual_eval.json')
EXP_A_PATH             = os.path.join(CHECKPOINT_DIR, 'exp_a_retrieval.json')
EXP_B_PATH             = os.path.join(CHECKPOINT_DIR, 'exp_b_ood.json')
EXP_C_PATH             = os.path.join(CHECKPOINT_DIR, 'exp_c_rerank.json')
EXP_D_PATH             = os.path.join(CHECKPOINT_DIR, 'exp_d_hybrid.json')


def emb_ckpt(name: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f'embeddings_{name}.npz')


def print_status():
    for key in ('GROQ_API_KEY', 'GEMINI_API_KEY', 'DEEPSEEK_API_KEY'):
        val = globals()[key]
        print(f'  {key:<22}: {"OK" if val else "MANCANTE"}')
    print(f'  Generator: {GENERATOR_MODEL} [{GENERATOR_BACKEND}]')
    print(f'  Judge:     {JUDGE_MODEL} [{JUDGE_BACKEND}]')
    print()
    for label, path in [
        ('corpus.pkl',   CORPUS_CHECKPOINT),
        ('chunks.pkl',   CHUNKS_CHECKPOINT),
        ('gold_set.json', GOLD_SET_PATH),
    ]:
        ok   = os.path.exists(path)
        size = f'{os.path.getsize(path)/1e6:.1f} MB' if ok else 'non trovato'
        print(f'  {label:<18}: {"OK" if ok else "MANCANTE"}  {size}')
