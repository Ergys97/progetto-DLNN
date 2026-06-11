"""Pure IR metrics — no external state, no imports beyond builtins."""
from __future__ import annotations


# ── ID-based metrics ────────────────────────────────────────────────────

def hit_at_k(retrieved_ids: list, expected_ids: set, k: int) -> float:
    if not expected_ids:
        return float('nan')
    return 1.0 if any(rid in expected_ids for rid in retrieved_ids[:k]) else 0.0


def mrr(retrieved_ids: list, expected_ids: set) -> float:
    if not expected_ids:
        return float('nan')
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in expected_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: list, expected_ids: set, k: int) -> float:
    if not expected_ids:
        return float('nan')
    return sum(1 for rid in retrieved_ids[:k] if rid in expected_ids) / len(expected_ids)


def precision_at_k(retrieved_ids: list, expected_ids: set, k: int) -> float:
    if not expected_ids or k == 0:
        return float('nan')
    return sum(1 for rid in retrieved_ids[:k] if rid in expected_ids) / k


# ── Textual (Jaccard) metrics — used in Exp A across chunking strategies ─

def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def hit_at_k_textual(
    retrieved_docs: list, expected_texts: list, k: int, threshold: float = 0.5
) -> float:
    if not expected_texts:
        return float('nan')
    for doc in retrieved_docs[:k]:
        if any(jaccard(doc, e) >= threshold for e in expected_texts):
            return 1.0
    return 0.0


def recall_at_k_textual(
    retrieved_docs: list, expected_texts: list, k: int, threshold: float = 0.5
) -> float:
    if not expected_texts:
        return float('nan')
    found = sum(
        1 for e in expected_texts
        if any(jaccard(doc, e) >= threshold for doc in retrieved_docs[:k])
    )
    return found / len(expected_texts)


# ── Coverage (containment) metrics — counter-check for Exp A ────────────
# La Jaccard simmetrica penalizza i chunk di dimensione diversa dal
# riferimento anche a retrieval perfetto; la coverage asimmetrica no,
# ma favorisce leggermente i chunk grandi. Riportate entrambe, i due bias
# sono di segno opposto e delimitano il risultato vero.

def coverage_of_expected(retrieved_docs: list, expected_text: str, k: int) -> float:
    """Frazione dei token del testo atteso coperti dall'unione dei top-k."""
    exp_tokens = set(expected_text.lower().split())
    if not exp_tokens:
        return 0.0
    union: set = set()
    for doc in retrieved_docs[:k]:
        union |= set(doc.lower().split())
    return len(union & exp_tokens) / len(exp_tokens)


def coverage_at_k(retrieved_docs: list, expected_texts: list, k: int) -> float:
    """Coverage media sui testi attesi (metrica continua, senza soglia)."""
    if not expected_texts:
        return float('nan')
    covs = [coverage_of_expected(retrieved_docs, e, k) for e in expected_texts]
    return sum(covs) / len(covs)


def hit_at_k_coverage(
    retrieved_docs: list, expected_texts: list, k: int, threshold: float = 0.8
) -> float:
    """Hit se almeno un testo atteso è coperto >= threshold dai top-k."""
    if not expected_texts:
        return float('nan')
    return 1.0 if any(
        coverage_of_expected(retrieved_docs, e, k) >= threshold
        for e in expected_texts
    ) else 0.0


def mrr_textual(
    retrieved_docs: list, expected_texts: list, threshold: float = 0.5
) -> float:
    if not expected_texts:
        return float('nan')
    for rank, doc in enumerate(retrieved_docs, start=1):
        if any(jaccard(doc, e) >= threshold for e in expected_texts):
            return 1.0 / rank
    return 0.0
