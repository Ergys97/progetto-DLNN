"""
LLM-as-a-Judge utilities.

Key features vs. the original notebook:
  - run_judge_batch()   → resumes from checkpoint, saves every N records
  - call_judge_with_retry() → exponential backoff on 429, parses retry-after
  - Both the main judge loop (Exp E) and multi-LLM (Exp F) use the same helpers
"""
from __future__ import annotations

import json
import os
import re
import time


# ── Prompt templates ─────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """\
Sei un valutatore esperto di sistemi RAG (Retrieval-Augmented Generation).
Valuta la risposta di un assistente AI sulla base del CONTESTO fornito e della RISPOSTA DI RIFERIMENTO.

DOMANDA:
{query}

CONTESTO FORNITO ALL'ASSISTENTE:
{context}

RISPOSTA DELL'ASSISTENTE:
{answer}

RISPOSTA DI RIFERIMENTO (ground truth):
{gt_answer}

Valuta su due dimensioni INDIPENDENTEMENTE, scala intera 1-5:

1. FAITHFULNESS: la risposta è supportata dal CONTESTO?
   5 = ogni affermazione è verificabile nel contesto
   4 = quasi tutto supportato, qualche dettaglio minore non verificabile
   3 = parzialmente supportata, qualche estrapolazione
   2 = poche affermazioni verificabili
   1 = allucinazioni gravi o contraddizioni col contesto

2. ANSWER_RELEVANCE: la risposta risponde alla DOMANDA?
   5 = completa, centrata, accurata rispetto al riferimento
   4 = corretta ma incompleta
   3 = parziale o leggermente off-topic
   2 = poco pertinente
   1 = non risponde / off-topic completo

Rispondi ESCLUSIVAMENTE in JSON valido, senza testo prima o dopo, senza backtick:
{{"faithfulness": <int 1-5>, "answer_relevance": <int 1-5>, "reasoning": "<una frase>"}}
"""

JUDGE_SYSTEM_PROMPT = (
    "Sei un valutatore RAG. "
    "Rispondi ESCLUSIVAMENTE con un oggetto JSON valido, nessun testo aggiuntivo."
)

# Models that don't support response_format=json_object on their backend
_NO_JSON_MODE: set[str] = {'openai/gpt-oss-120b'}


# ── Response parsing ─────────────────────────────────────────────────────

def parse_judge_response(raw: str) -> dict:
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw.strip()).strip()

    # Direct parse
    try:
        d = json.loads(raw)
        return {
            'faithfulness':     int(d.get('faithfulness', 0)),
            'answer_relevance': int(d.get('answer_relevance', 0)),
            'reasoning':        d.get('reasoning', ''),
        }
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: find first {...} block (handles stray text around JSON)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            return {
                'faithfulness':     int(d.get('faithfulness', 0)),
                'answer_relevance': int(d.get('answer_relevance', 0)),
                'reasoning':        d.get('reasoning', ''),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        'faithfulness':     None,
        'answer_relevance': None,
        'reasoning':        f'PARSE_ERROR: {raw[:200]}',
    }


# ── Single judge call ────────────────────────────────────────────────────

def judge_response(
    query: str,
    context: str,
    answer: str,
    gt_answer: str,
    client,
    judge_model: str,
    judge_backend: str,
) -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        query=query, context=context, answer=answer, gt_answer=gt_answer,
    )
    kwargs: dict = dict(
        model=judge_model,
        messages=[
            {'role': 'system', 'content': JUDGE_SYSTEM_PROMPT},
            {'role': 'user',   'content': prompt},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    if judge_backend == 'ollama':
        # Suppress chain-of-thought on thinking models (Ollama ≥ 0.5)
        kwargs['extra_body'] = {'think': False}
    elif judge_backend == 'deepseek':
        # Suppress chain-of-thought on DeepSeek models (thinking mode)
        kwargs['extra_body'] = {
            'thinking': {
                'type': 'disabled'
            }
        }

    if judge_backend != 'ollama' and judge_model not in _NO_JSON_MODE:
        kwargs['response_format'] = {'type': 'json_object'}

    resp    = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ''

    # Some thinking models put the answer in reasoning_content
    if not content.strip():
        msg     = resp.choices[0].message
        content = (
            getattr(msg, 'reasoning_content', '') or
            getattr(msg, 'thinking', '') or ''
        )
    return parse_judge_response(content)


# ── Retry wrapper ────────────────────────────────────────────────────────

def _parse_retry_after(error_msg: str, default: int = 65) -> int:
    """Extract seconds from a 429 error message like 'retry in 54s'."""
    m = re.search(r'retry.{0,20}?(\d+)\s*s', error_msg, re.IGNORECASE)
    return int(m.group(1)) + 3 if m else default


def call_judge_with_retry(
    record: dict,
    client,
    judge_model: str,
    judge_backend: str,
    max_retries: int = 5,
) -> dict:
    """
    Calls judge_response with exponential backoff on 429 rate-limit errors.
    Other exceptions propagate immediately.
    """
    base_delay = 10
    for attempt in range(max_retries):
        try:
            return judge_response(
                record['query'], record['context'],
                record['answer'], record['gt_answer'],
                client, judge_model, judge_backend,
            )
        except Exception as e:
            msg = str(e)
            if '429' in msg:
                wait = _parse_retry_after(msg, default=base_delay * 2)
                print(
                    f'  [{record["id"]}] 429 rate-limit '
                    f'(attempt {attempt + 1}/{max_retries}) – attendo {wait}s'
                )
                time.sleep(wait)
                base_delay = min(base_delay * 2, 180)
            else:
                raise
    raise RuntimeError(
        f'Max retries ({max_retries}) superato per record {record["id"]}'
    )


# ── Batch runner with checkpoint resume ─────────────────────────────────

def run_judge_batch(
    pipeline_results: list,
    client,
    judge_model: str,
    judge_backend: str,
    checkpoint_path: str,
    delay_s: float = 0.0,
    max_retries: int = 5,
    save_every: int = 5,
) -> list:
    """
    Evaluates answered pipeline results with the judge.

    Resume semantics: if `checkpoint_path` already contains records with
    non-null faithfulness scores, those records are skipped and the function
    only processes the remainder.  Saves to checkpoint every `save_every`
    new evaluations so progress is never lost to a crash or rate-limit abort.

    Returns the full list (pre-existing + newly evaluated).
    """
    # Load existing checkpoint, index by query id
    done: dict[str, dict] = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        done = {r['id']: r for r in existing if r.get('faithfulness') is not None}
        print(f'[RESUME] {len(done)} record già valutati nel checkpoint.')

    results: list = list(done.values())
    todo    = [r for r in pipeline_results if r['id'] not in done]
    todo_answered = [r for r in todo if r.get('status') == 'answered']
    todo_skipped = len(todo) - len(todo_answered)
    print(
        f'Da giudicare: {len(todo_answered)} record answered '
        f'(da registrare come skipped/non valutabili: {todo_skipped}; '
        f'già presenti: {len(done)}).'
    )

    new_count = 0
    for r in todo:
        if r['status'] != 'answered':
            results.append({
                **r,
                'faithfulness':     None,
                'answer_relevance': None,
                'judge_reasoning':  f'skipped: status={r["status"]}',
            })
            continue

        try:
            j = call_judge_with_retry(
                r, client, judge_model, judge_backend, max_retries
            )
            results.append({
                **r,
                'faithfulness':     j['faithfulness'],
                'answer_relevance': j['answer_relevance'],
                'judge_reasoning':  j['reasoning'],
            })
            print(f'  [{r["id"]}] F={j["faithfulness"]} AR={j["answer_relevance"]}')
        except Exception as e:
            print(f'  [{r["id"]}] ERR: {e}')
            results.append({
                **r,
                'faithfulness':     None,
                'answer_relevance': None,
                'judge_reasoning':  str(e),
            })

        new_count += 1
        if delay_s > 0:
            time.sleep(delay_s)

        # Incremental checkpoint save
        if new_count % save_every == 0:
            _save_checkpoint(results, checkpoint_path)
            print(f'  [CKPT] Salvati {len(results)} record.')

    _save_checkpoint(results, checkpoint_path)
    evaluated = sum(1 for r in results if r.get('faithfulness') is not None)
    print(f'\nSalvati {len(results)} record. Valutati: {evaluated}/{len(results)}')
    return results


def _save_checkpoint(records: list, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ── Delay helper ─────────────────────────────────────────────────────────

def judge_delay(judge_backend: str, judge_model: str) -> float:
    """
    Conservative inter-call delay in seconds.
    Groq free tier: ~8K TPM; each judge call ~2500 tokens → ~3 calls/min → 20s.
    Gemini free tier: 15 RPM → 4s, but we use 6s for safety.
    DeepSeek paid API: rate limits >> Groq free tier; backoff in call_judge_with_retry
                       gestisce i 429, quindi basta un buffer minimo di 1s.
    Ollama: no limit.
    """
    if judge_backend == 'ollama':
        return 0.0
    if judge_backend == 'gemini':
        return 6.0
    if judge_backend == 'deepseek':
        return 1.0
    # groq / openai
    return 20.0
