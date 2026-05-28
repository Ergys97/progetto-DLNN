"""
Script di audit per verificare la coerenza del Gold Set.
Genera un report in Markdown ('gold_set_audit.md') contenente:
- Ogni domanda con la relativa Ground Truth (GT).
- Il testo completo di ciascun chunk annotato come 'expected_chunk_id'.
"""

import os
import json
import pickle
import hashlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
CHECKPOINT_DIR = _HERE.parent / "checkpoint"
STRATEGY = "recursive_512"

def make_chunk_id(source: str, chunk_idx: int, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f'{source.replace(" ","_").replace("/","_")}::{chunk_idx:04d}::{h}'

def main():
    gs_path = CHECKPOINT_DIR / "gold_set.json"
    chunks_path = CHECKPOINT_DIR / "chunks.pkl"

    if not gs_path.exists():
        print(f"[ERRORE] {gs_path} non trovato. Assicurati che esista.")
        return
    if not chunks_path.exists():
        print(f"[ERRORE] {chunks_path} non trovato. Esegui prima il caricamento dei chunks.")
        return

    # 1. Carica il Gold Set
    with open(gs_path, "r", encoding="utf-8") as f:
        gold_set = json.load(f)

    # 2. Carica i chunk per ricostruire il dizionario ID -> Testo
    print("Caricamento chunks.pkl per la ricostruzione del testo...")
    with open(chunks_path, "rb") as f:
        saved = pickle.load(f)
    
    strategies = saved["strategies"]
    strategies_meta = saved["strategies_meta"]

    if STRATEGY not in strategies:
        print(f"[ERRORE] La strategia {STRATEGY} non è presente in chunks.pkl.")
        return

    chunks_text = strategies[STRATEGY]
    chunks_meta = strategies_meta[STRATEGY]

    # Ricostruisci il mapping ID -> Testo
    id_to_chunk = {}
    counters = {}
    for chunk, meta in zip(chunks_text, chunks_meta):
        src = meta["source"]
        counters.setdefault(src, 0)
        cid = make_chunk_id(src, counters[src], chunk)
        id_to_chunk[cid] = (chunk, src)
        counters[src] += 1

    # 3. Genera il file di audit in Markdown
    audit_path = CHECKPOINT_DIR / "gold_set_audit.md"
    print(f"Generazione del report di audit in: {audit_path}")

    with open(audit_path, "w", encoding="utf-8") as out:
        out.write("# Audit del Gold Set RAG\n\n")
        out.write(f"Questo report mostra per ogni domanda del Gold Set i testi effettivi dei chunk annotati come rilevanti (strategia di riferimento: `{STRATEGY}`).\n\n")
        
        # Statistiche
        total = len(gold_set)
        in_domain = sum(1 for q in gold_set if q['category'].startswith('in_domain'))
        ood = sum(1 for q in gold_set if q['category'] == 'out_of_domain')
        injection = sum(1 for q in gold_set if q['category'] == 'prompt_injection')
        annotated = sum(1 for q in gold_set if q.get('expected_chunk_ids'))
        
        out.write("## Statistiche Generali\n")
        out.write(f"- **Totale query**: {total}\n")
        out.write(f"- **In-Domain**: {in_domain} (con expected chunk)\n")
        out.write(f"- **Out-of-Domain**: {ood} (atteso block/refusal)\n")
        out.write(f"- **Prompt Injection**: {injection} (atteso block/refusal)\n")
        out.write(f"- **Query in-domain annotate**: {annotated}/{in_domain}\n\n")
        
        out.write("---\n\n")

        for q in gold_set:
            qid = q["id"]
            cat = q["category"]
            query_txt = q["query"]
            gt = q.get("gt_answer", "N/A")
            expected_ids = q.get("expected_chunk_ids", [])

            out.write(f"### [{qid}] {query_txt}\n")
            out.write(f"- **Categoria**: `{cat}`\n")
            if gt:
                out.write(f"- **Ground Truth (GT)**: *{gt}*\n")
            else:
                out.write("- **Ground Truth (GT)**: (Nessuna - OOD/Injection)\n")
            
            out.write(f"- **Expected Chunks** ({len(expected_ids)}):\n")
            
            if not expected_ids:
                out.write("  *Nessun chunk atteso (corretto per OOD/Injection o query non ancora annotata).*\n")
            else:
                for idx, cid in enumerate(expected_ids):
                    out.write(f"  #### Chunk {idx + 1}: `{cid}`\n")
                    if cid in id_to_chunk:
                        text, src = id_to_chunk[cid]
                        out.write(f"  * **Fonte**: `{src}`\n")
                        # indentiamo il testo per renderlo leggibile in un blocco citazione
                        indented_text = "\n  > ".join(text.split("\n"))
                        out.write(f"  > {indented_text}\n\n")
                    else:
                        out.write(f"  * **[ERRORE]** ID chunk non trovato in chunks.pkl per la strategia `{STRATEGY}`.\n\n")
            
            out.write("\n---\n\n")

    print("Completato! Puoi aprire il file 'checkpoint/gold_set_audit.md' per iniziare l'analisi.")

if __name__ == "__main__":
    main()
