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


def mrr_textual(
    retrieved_docs: list, expected_texts: list, threshold: float = 0.5
) -> float:
    if not expected_texts:
        return float('nan')
    for rank, doc in enumerate(retrieved_docs, start=1):
        if any(jaccard(doc, e) >= threshold for e in expected_texts):
            return 1.0 / rank
    return 0.0
