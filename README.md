# ASSISTENTE AI PER MANUALI TECNICI

Questo progetto è un'applicazione locale di RAG (Retrieval-Augmented Generation) che risponde a domande sui manuali tecnici dei dispositivi WENDY ed EGO.

---

## Cosa fa il progetto

L’assistente permette agli utenti di fare domande in linguaggio naturale sulla documentazione tecnica.

Il sistema recupera le sezioni più rilevanti dai manuali e genera una risposta utilizzando un LLM locale.

L’obiettivo è fornire risposte accurate basate esclusivamente sui manuali caricati.

---

## Funzionalità principali

* LLM locale con Ollama
* Pipeline RAG con ChromaDB
* Retrieval ibrido:
  * ricerca per parole chiave
  * ricerca vettoriale
* Risposte basate sulle fonti
* Interfaccia chat da terminale
* Tool MCP per integrazione con agenti

---

## Struttura del progetto

project/
├── build_index.py        # Costruisce l'indice vettoriale
├── chat_terminal.py      # Interfaccia chat da terminale
├── rag_core.py           # Logica principale del RAG
├── rag_mcp_server.py     # Tool MCP per agenti
├── requirements.txt
├── *.md                  # Manuali tecnici utilizzati come dataset


---

## Modelli e impostazioni

* LLM: `qwen2.5:1.5b`
* Modello di embedding: `intfloat/multilingual-e5-small`
* Database vettoriale: ChromaDB
* Dimensione chunk: 1800
* Overlap chunk: 300
* Top-K: 4

---

## Installazione

```PowerShell
pip install -r requirements.txt
```

Assicurati che Ollama sia installato e in esecuzione.

---

## Utilizzo

Avvia il modello locale:

```PowerShell
ollama run qwen2.5:1.5b
```

Costruisci l’indice vettoriale:

```PowerShell
python build_index.py
```

Avvia l’assistente da terminale:

```PowerShell
python chat_terminal.py
```

---

## Esempi di domande

* Come eseguo un backup in WENDY?
* Come inserisco il rotolo di carta?

---

## Dataset

La base di conoscenza è composta da manuali tecnici in formato Markdown:

* Manuale Operativo WENDY
* Manuale Assistenza EGO
* Guida Rapida EGO

---

## Autore

Cecilia Morici
Laboratorio di Data Science
2026

---


