"""Quick smoke tests for src/ modules."""
from src.metrics import hit_at_k, mrr, recall_at_k, jaccard, hit_at_k_textual

assert hit_at_k(['a','b','c'], {'b'}, 2) == 1.0
assert hit_at_k(['a','b','c'], {'b'}, 1) == 0.0
assert abs(mrr(['a','b','c'], {'b'}) - 0.5) < 1e-9
assert abs(recall_at_k(['a','b','c'], {'b','d'}, 3) - 0.5) < 1e-9
assert abs(jaccard('hello world', 'hello there') - 1/3) < 1e-9
assert hit_at_k_textual(['ciao mondo bello', 'foo'], ['ciao mondo'], 1) == 1.0
print('src.metrics OK')

from src.judge import parse_judge_response
r = parse_judge_response('{"faithfulness": 4, "answer_relevance": 5, "reasoning": "ok"}')
assert r['faithfulness'] == 4 and r['answer_relevance'] == 5, r
r2 = parse_judge_response('```json\n{"faithfulness": 3, "answer_relevance": 3, "reasoning": "x"}\n```')
assert r2['faithfulness'] == 3, r2
r3 = parse_judge_response('some text before {"faithfulness": 2, "answer_relevance": 1, "reasoning": "bad"} after')
assert r3['faithfulness'] == 2, r3
print('src.judge parse_judge_response OK')

from src.judge import judge_delay
assert judge_delay('ollama', 'any') == 0.0
assert judge_delay('groq', 'any') == 20.0
assert judge_delay('gemini', 'any') == 6.0
print('src.judge judge_delay OK')

from src.embeddings import make_chunk_id
cid = make_chunk_id('Fisica/file.pdf', 0, 'testo di prova')
assert '::' in cid and cid.startswith('Fisica_file.pdf')
print('src.embeddings make_chunk_id OK')

from src.config import emb_ckpt
path = emb_ckpt('recursive_512')
assert path.endswith('embeddings_recursive_512.npz')
print('src.config emb_ckpt OK')

print('\nTutti i test passati.')
