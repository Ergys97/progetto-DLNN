# RAG Evaluation Framework — Pipeline RAG su Note Universitarie

Questo progetto implementa e valuta quantitativamente un sistema **RAG (Retrieval-Augmented Generation)** locale basato sui materiali del corso di laurea triennale dell'autore (vault Obsidian + PDF dei corsi). 

L'obiettivo principale è analizzare l'efficacia delle diverse strategie di chunking, implementare un gate per query fuori dominio (**OOD**), ottimizzare la ricerca ibrida e valutare le risposte generate tramite un approccio **LLM-as-a-Judge** allineato a valutazioni umane.

Per tutti i dettagli metodologici, le analisi teoriche e la discussione dei risultati sperimentali, si rimanda al [Report Finale (report.md)](file:///C:/Users/ergys/Desktop/Git_Repositories/progetto-DLNN/report.md).

---

## 🛠️ Architettura del Sistema

```
PDF universitari ──> Chunking (5 strategie) ──> Embeddings BGE-M3 ──> ChromaDB in-memory
                                                                          │
                                                                          ▼
Query ──> Gate OOD (cosine θ) ──> Hybrid Search (BM25 + Dense, α) ──> CrossEncoder Rerank ──> LLM Generatore
```

---

## 📁 Struttura della Repository

*   [src/](file:///C:/Users/ergys/Desktop/Git_Repositories/progetto-DLNN/src/) — Contiene la logica modulare Python del sistema RAG:
    *   [config.py](file:///C:/Users/ergys/Desktop/Git_Repositories/progetto-DLNN/src/config.py) — Gestione delle chiavi API, iperparametri e percorsi dei checkpoint.
    *   [embeddings.py](file:///C:/Users/ergys/Desktop/Git_Repositories/progetto-DLNN/src/embeddings.py) — Pipeline di estrazione, hashing e caricamento vettoriale in ChromaDB.
    *   [retrieval.py](file:///C:/Users/ergys/Desktop/Git_Repositories/progetto-DLNN/src/retrieval.py) — Algoritmi di ricerca densa, BM25 (ibrido) e CrossEncoder Reranking.
    *   [pipeline.py](file:///C:/Users/ergys/Desktop/Git_Repositories/progetto-DLNN/src/pipeline.py) — Orchestratore del flusso RAG end-to-end e batch runner.
    *   [judge.py](file:///C:/Users/ergys/Desktop/Git_Repositories/progetto-DLNN/src/judge.py) — Valutatore LLM-as-a-Judge con gestione del rate limiting e retry.
    *   [metrics.py](file:///C:/Users/ergys/Desktop/Git_Repositories/progetto-DLNN/src/metrics.py) — Metriche di retrieval (Hit@K, MRR, Recall) e Jaccard testuale.
*   [checkpoint/](file:///C:/Users/ergys/Desktop/Git_Repositories/progetto-DLNN/checkpoint/) — Checkpoint serialized degli esperimenti e dataset:
    *   `gold_set.json` — Il dataset di test con 50 query annotate.
    *   `gold_set_audit.md` — Report dettagliato sui chunk attesi per ciascuna query in-domain.
    *   `*_results.json` — Risultati intermedi dei singoli esperimenti (A, B, C, D, E, F).
*   [rag_experiment.ipynb](file:///C:/Users/ergys/Desktop/Git_Repositories/progetto-DLNN/rag_experiment.ipynb) — Notebook Jupyter principale, vetrina di visualizzazione ed esecuzione degli esperimenti.
*   [report.md](file:///C:/Users/ergys/Desktop/Git_Repositories/progetto-DLNN/report.md) — Relazione scientifica finale con risultati dettagliati e grafici.
*   [requirements.txt](file:///C:/Users/ergys/Desktop/Git_Repositories/progetto-DLNN/requirements.txt) — Dipendenze Python necessarie.

---

## 🚀 Setup e Installazione

### 1. Prerequisiti
Assicurati di avere installato Python (consigliato >= 3.9, < 3.12) e `pip`.

### 2. Installazione delle dipendenze
Installa le librerie richieste eseguendo:
```bash
pip install -r requirements.txt
```

### 3. Configurazione delle API Keys
Il sistema supporta generatori locali (Ollama) e provider cloud. Rinomina il file [.env.example](file:///C:/Users/ergys/Desktop/Git_Repositories/progetto-DLNN/.env.example) in `.env` e inserisci le tue chiavi API:
```bash
cp .env.example .env
```
Quindi modifica il file inserendo le tue credenziali per Groq, Gemini o DeepSeek:
```env
GROQ_API_KEY=tua_chiave_qui
GEMINI_API_KEY=tua_chiave_qui
DEEPSEEK_API_KEY=tua_chiave_qui
```

---

## 💻 Utilizzo degli Script Helper

La repository include diversi script standalone (ora organizzati nella directory `scripts/`) per gestire e testare il flusso di lavoro:

### 🧪 Eseguire i test automatici sui moduli
Verifica che tutti i moduli della cartella `src/` funzionino correttamente eseguendo lo smoke test:
```bash
python scripts/_test_modules.py
```

### 📝 Annotazione interattiva del Gold Set
Se desideri annotare nuove query o modificare l'assegnazione dei chunk attesi (`expected_chunk_ids`) nel gold set:
```bash
python scripts/annotate_gold_set.py
```

### 🔍 Generare l'audit del Gold Set
Genera il file markdown di ispezione visuale per confrontare le domande con i testi effettivi dei chunk annotati:
```bash
python scripts/analyze_gold_set.py
```

### ⚡ Eseguire l'esperimento Multi-LLM in Background
Per lanciare la valutazione comparativa dei sei generatori (DeepSeek V4 Flash, Llama 3.3 70B, Qwen 2.5 14B, Granite 4.1 8B, Gemma 4 2B/4B) gestendo in background la latenza e i checkpoint di salvataggio:
```bash
python scripts/run_exp_f.py
```

### 🎯 Validare il gate OOD su holdout (generalizzazione di θ)
Esegue il test del gate su 18 query mai usate per il tuning di θ, calcolando TPR/FPR a θ=0.40 (solo embedder locale, nessuna API):
```bash
python scripts/run_holdout_ood.py
```

### 📐 Analisi di sensibilità della soglia Jaccard (Exp A)
Verifica che il ranking delle strategie di chunking sia robusto alla soglia di Jaccard (sweep 0.3–0.7):
```bash
python scripts/run_jaccard_sensitivity.py
```

### 📊 Holdout di retrieval (Hit@5/MRR/Recall su query nuove)
Genera i candidati da annotare, poi calcola le metriche di retrieval sull'holdout annotato:
```bash
python scripts/run_holdout_retrieval.py dump   # crea holdout_candidates.json + holdout_audit.md
python scripts/run_holdout_retrieval.py eval   # metriche + bootstrap CI (richiede expected_chunk_ids)
```

### 🔨 Ricostruire il Notebook da zero
Il notebook `rag_experiment.ipynb` è generato in modo programmatico da uno script sorgente Python per garantire la consistenza formale delle sezioni. Se apporti modifiche al template, puoi rigenerare il notebook tramite:
```bash
python scripts/_build_notebook.py
```
