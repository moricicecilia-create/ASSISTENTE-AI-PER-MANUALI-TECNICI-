"""
rag_core.py
-----------
Logica RAG condivisa tra interfaccia terminale e web.

Versione migliorata:
- Embedding locale con sentence-transformers
- LLM locale con Ollama
- ChromaDB persistente
- Retrieval ibrido: keyword search + vector search
"""

import re
import pathlib
from typing import List, Tuple

# ─── Costanti ───────────────────────────────────────────────────────────────

DOCS_DIR = pathlib.Path(__file__).parent.parent
CHROMA_DIR = pathlib.Path(__file__).parent / "chroma_db"

# Modello leggero per PC lenti
OLLAMA_LLM_MODEL = "qwen2.5:1.5b"

# Embedding migliore rispetto a MiniLM, ma ancora abbastanza leggero
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"

# Parametri alleggeriti
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 300
TOP_K = 4

# ─── Pulizia documenti ──────────────────────────────────────────────────────

_NOISE_PATTERNS = [
    re.compile(r'!\[.*?\]\(.*?\)'),
    re.compile(r'^WENDY\s+\d+\s*$'),
    re.compile(r'^Manuale per l.assistenza B\d+V\d+\s*$', re.IGNORECASE),
    re.compile(r'^PERFECT_BLOCK\s*$'),
]


def _clean_line(line: str) -> str:
    line = re.sub(r'<[^>]+>', '', line)
    line = re.sub(r'!\[.*?\]\(.*?\)', '', line)
    line = re.sub(r'&[a-z#0-9]+;', ' ', line)
    return line.strip()


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()

    if not stripped:
        return False

    for pat in _NOISE_PATTERNS:
        if pat.fullmatch(stripped):
            return True

    if re.fullmatch(r'\d{1,3}', stripped):
        return True

    return False


def _remove_duplicate_sections(text: str) -> str:
    """
    Rimuove blocchi duplicati, per esempio sommari ripetuti nei manuali.
    """
    lines = text.split('\n')
    seen_blocks = set()
    result = []

    i = 0
    window = 8

    while i < len(lines):
        block_key = '\n'.join(lines[i:i + window]).strip()

        if len(block_key) > 100 and block_key in seen_blocks:
            i += window
            continue

        seen_blocks.add(block_key)
        result.append(lines[i])
        i += 1

    return '\n'.join(result)


def clean_document(text: str) -> str:
    """
    Pipeline di pulizia completa per un documento Markdown.
    """
    text = _remove_duplicate_sections(text)

    cleaned_lines = []
    prev_empty = False

    for line in text.split('\n'):
        if _is_noise_line(line):
            continue

        cleaned = _clean_line(line)

        if cleaned == '':
            if prev_empty:
                continue
            prev_empty = True
        else:
            prev_empty = False

        cleaned_lines.append(cleaned)

    return '\n'.join(cleaned_lines)


# ─── Caricamento documenti ───────────────────────────────────────────────────

def load_and_clean_docs() -> List[dict]:
    docs = []
    md_files = sorted(DOCS_DIR.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(f"Nessun file .md trovato in {DOCS_DIR}")

    for filepath in md_files:
        raw = filepath.read_text(encoding='utf-8', errors='ignore')
        cleaned = clean_document(raw)

        name = filepath.stem

        if 'EGO' in name.upper() and 'ASSISTENZA' in name.upper():
            title = "Manuale Assistenza EGO"
        elif 'WENDY' in name.upper():
            title = "Manuale Operativo WENDY"
        elif 'OUTPUT' in name.upper():
            title = "Guida Rapida EGO"
        else:
            title = name

        docs.append({
            "content": cleaned,
            "source": filepath.name,
            "title": title
        })

        print(f"  ✓ {filepath.name}  ({len(cleaned):,} caratteri dopo pulizia)")

    return docs


# ─── Chunking ────────────────────────────────────────────────────────────────

def split_into_chunks(docs: List[dict]) -> List[dict]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3")
    ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []

    for doc in docs:
        header_docs = md_splitter.split_text(doc["content"])

        for hd in header_docs:
            split_texts = char_splitter.split_text(hd.page_content)

            for chunk_text in split_texts:
                if len(chunk_text.strip()) < 50:
                    continue

                all_chunks.append({
                    "content": chunk_text,
                    "source": doc["source"],
                    "title": doc["title"],
                    "metadata": {
                        **hd.metadata,
                        "source": doc["source"],
                        "title": doc["title"]
                    },
                })

    print(f"  → {len(all_chunks)} chunk generati")
    return all_chunks


# ─── Embedding ───────────────────────────────────────────────────────────────

_embedder = None


def get_embedder():
    global _embedder

    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Caricamento modello embedding: {EMBED_MODEL_NAME}")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)

    return _embedder


def _prepare_for_embedding(texts: List[str], is_query: bool = False) -> List[str]:
    """
    I modelli E5 funzionano meglio se usano prefissi:
    - "query: ..." per le domande
    - "passage: ..." per i documenti
    """
    if "e5" not in EMBED_MODEL_NAME.lower():
        return texts

    prefix = "query: " if is_query else "passage: "
    return [prefix + text for text in texts]


def embed_texts(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """
    Calcola gli embedding di una lista di testi.
    """
    model = get_embedder()
    prepared_texts = _prepare_for_embedding(texts, is_query=is_query)

    vectors = model.encode(
        prepared_texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    return vectors.tolist()


# ─── ChromaDB ────────────────────────────────────────────────────────────────

def get_chroma_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    return client.get_or_create_collection(
        name="manuali_buffetti",
        metadata={"hnsw:space": "cosine"},
    )


def index_exists() -> bool:
    try:
        return get_chroma_collection().count() > 0
    except Exception:
        return False


# ─── Retrieval ibrido ────────────────────────────────────────────────────────

STOPWORDS = {
    "come", "cosa", "quale", "quali", "dove", "quando", "perche", "perché",
    "devo", "posso", "fare", "faccio", "fa", "si", "mi", "spieghi",
    "su", "sul", "sulla", "nel", "nella", "nei", "nelle", "con", "per",
    "il", "lo", "la", "i", "gli", "le", "un", "una", "uno", "di", "del",
    "della", "dei", "degli", "delle", "a", "ad", "da", "e", "o", "in",
    "registratore", "telematico", "manuale", "manuali", "buffetti"
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("à", "a").replace("è", "e").replace("é", "e")
    text = text.replace("ì", "i").replace("ò", "o").replace("ù", "u")
    return text


def query_terms(query: str) -> List[str]:
    q = normalize_text(query)
    words = re.findall(r"[a-z0-9]+", q)
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def title_for_file(filepath: pathlib.Path) -> str:
    name = filepath.stem.upper()

    if "EGO" in name and "ASSISTENZA" in name:
        return "Manuale Assistenza EGO"

    if "WENDY" in name:
        return "Manuale Operativo WENDY"

    if "OUTPUT" in name:
        return "Guida Rapida EGO"

    return filepath.stem


def split_markdown_sections(text: str) -> List[str]:
    """
    Divide il Markdown in sezioni basate sui titoli #, ##, ###.
    """
    sections = re.split(r"(?=^#{1,3}\s+)", text, flags=re.MULTILINE)
    return [s.strip() for s in sections if len(s.strip()) > 80]


def infer_target_titles(query: str) -> List[str]:
    """
    Capisce se la domanda riguarda EGO, WENDY o entrambi.
    """
    q = normalize_text(query)
    titles = []

    if "wendy" in q:
        titles.append("Manuale Operativo WENDY")

    if "ego" in q:
        titles.append("Guida Rapida EGO")
        titles.append("Manuale Assistenza EGO")

    return titles


def important_phrases(query: str) -> List[str]:
    """
    Frasi tecniche da cercare esattamente nei manuali.
    """
    q = normalize_text(query)
    phrases = []

    if "rotolo" in q and "carta" in q:
        phrases.append("rotolo carta")
        phrases.append("inserimento del rotolo carta")
        phrases.append("inserimento rotolo carta")

    if "backup" in q:
        phrases.append("backup")
        phrases.append("backup/ripristina")
        phrases.append("backup dati")
        phrases.append("backup immagini")
        phrases.append("ripristina dati")
        phrases.append("ripristina immagini")

    if "agenzia" in q and "entrate" in q:
        phrases.append("agenzia delle entrate")
        phrases.append("agenzia entrate")

    if "stampanti" in q and "reparto" in q:
        phrases.append("stampanti di reparto")
        phrases.append("stampante di reparto")

    if "dgfe" in q:
        phrases.append("dgfe")

    if "wifi" in q or "wi-fi" in q:
        phrases.append("wi-fi")
        phrases.append("wifi")

    if "fattura" in q:
        phrases.append("fattura")
        phrases.append("fattura elettronica")
        phrases.append("fattura riepilogativa")

    if "lotteria" in q:
        phrases.append("lotteria")
        phrases.append("lotteria degli scontrini")

    if "errore" in q or "errori" in q:
        phrases.append("errore")
        phrases.append("errori")

    return phrases


def keyword_section_search(query: str, max_results: int = 6) -> List[dict]:
    """
    Cerca nei file .md sezioni che contengono parole esatte della domanda.
    Questa ricerca aiuta molto con manuali tecnici.
    """
    terms = query_terms(query)
    phrases = important_phrases(query)
    target_titles = infer_target_titles(query)

    if not terms and not phrases:
        return []

    results = []

    for md_path in sorted(DOCS_DIR.glob("*.md")):
        title = title_for_file(md_path)

        try:
            raw = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        cleaned = clean_document(raw)
        sections = split_markdown_sections(cleaned)

        for section in sections:
            ns = normalize_text(section)
            heading = normalize_text(section.split("\n", 1)[0]) if section else ""

            score = 0.0

            # Match parole singole
            for term in terms:
                if term in ns:
                    score += 1.0

                if term in heading:
                    score += 4.0

            # Match frasi tecniche
            for phrase in phrases:
                if phrase in ns:
                    score += 8.0

                if phrase in heading:
                    score += 12.0

            # Bonus manuale corretto
            if target_titles and title in target_titles:
                score += 5.0

            # Penalità manuale sbagliato
            if target_titles and title not in target_titles:
                score -= 4.0

            if score <= 0:
                continue

            excerpt = " ".join(section.split())

            if len(excerpt) > 1800:
                excerpt = excerpt[:1800] + "..."

            results.append({
                "content": excerpt,
                "source": md_path.name,
                "title": title,
                "score": round(score, 3),
                "kind": "keyword",
            })

    results.sort(key=lambda c: c["score"], reverse=True)
    return results[:max_results]


def vector_search(query: str, max_results: int = 8) -> List[dict]:
    """
    Retrieval vettoriale classico con ChromaDB.
    """
    collection = get_chroma_collection()
    query_vector = embed_texts([query], is_query=True)[0]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=max_results,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []

    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        chunks.append({
            "content": doc,
            "source": meta.get("source", ""),
            "title": meta.get("title", ""),
            "score": round(1 - distance, 3),
            "kind": "vector",
        })

    return chunks


def deduplicate_chunks(chunks: List[dict]) -> List[dict]:
    """
    Rimuove duplicati tra keyword search e vector search.
    """
    seen = set()
    output = []

    for chunk in chunks:
        content = chunk.get("content", "")
        key = normalize_text(content[:300])

        if key in seen:
            continue

        seen.add(key)
        output.append(chunk)

    return output


def rerank_chunks(query: str, chunks: List[dict]) -> List[dict]:
    """
    Riordina i risultati dando più peso a:
    - keyword match
    - manuale giusto
    - frasi tecniche
    - score vettoriale
    """
    terms = query_terms(query)
    phrases = important_phrases(query)
    target_titles = infer_target_titles(query)

    reranked = []

    for chunk in chunks:
        content = normalize_text(chunk.get("content", ""))
        heading = normalize_text(content[:250])
        title = chunk.get("title", "")

        final_score = 0.0
        original_score = chunk.get("score", 0)

        if chunk.get("kind") == "vector":
            final_score += original_score * 5

        if chunk.get("kind") == "keyword":
            final_score += original_score

        for term in terms:
            if term in content:
                final_score += 1.5

            if term in heading:
                final_score += 3.0

        for phrase in phrases:
            if phrase in content:
                final_score += 6.0

            if phrase in heading:
                final_score += 8.0

        if target_titles and title in target_titles:
            final_score += 5.0

        if target_titles and title not in target_titles:
            final_score -= 4.0

        new_chunk = dict(chunk)
        new_chunk["score"] = round(final_score, 3)
        reranked.append(new_chunk)

    reranked.sort(key=lambda c: c["score"], reverse=True)
    return reranked


def retrieve(query: str, top_k: int = TOP_K) -> List[dict]:
    """
    Retrieval ibrido:
    1. keyword search nei Markdown
    2. vector search in ChromaDB
    3. unione risultati
    4. reranking
    5. restituzione dei migliori top_k
    """
    keyword_chunks = keyword_section_search(query, max_results=6)
    vector_chunks = vector_search(query, max_results=8)

    combined = deduplicate_chunks(keyword_chunks + vector_chunks)
    reranked = rerank_chunks(query, combined)

    return reranked[:top_k]


# ─── Prompt ──────────────────────────────────────────────────────────────────

def build_prompt(query: str, chunks: List[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Fonte: {c['title']} | Tipo ricerca: {c.get('kind', 'n/d')} | Score: {c.get('score', '')}]\n{c['content']}"
        for c in chunks
    )

    return f"""Rispondi alla domanda dell'utente usando ESCLUSIVAMENTE i testi forniti.

REGOLE OBBLIGATORIE:
1. Non inventare informazioni.
2. Se la risposta non è presente nei testi, scrivi solo:
   "Spiacente, l'informazione non è presente nei manuali consultati."
3. Se i testi recuperati parlano di prodotti diversi, distingui chiaramente tra EGO e WENDY.
4. Se la domanda è procedurale, rispondi con passaggi numerati.
5. Se sono presenti codici, menu, opzioni SET o comandi, riportali esattamente.
6. Non aggiungere conoscenze esterne.
7. Alla fine indica da quale fonte hai ricavato la risposta.
8. Non usare informazioni che non compaiono nei testi forniti.

--- TESTI DEL MANUALE ---
{context}

--- DOMANDA UTENTE ---
{query}
"""


# ─── Generazione ─────────────────────────────────────────────────────────────

def ask(query: str) -> Tuple[str, List[dict]]:
    """
    Pipeline RAG completa:
    domanda → retrieve → prompt → Ollama → risposta.
    """
    import requests

    chunks = retrieve(query)

    if not chunks:
        return "Non ho trovato informazioni rilevanti nei manuali.", []

    content_msg = build_prompt(query, chunks)

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": OLLAMA_LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Sei un assistente AI rigoroso per il supporto tecnico "
                        "sui prodotti Buffetti e Olivetti. Rispondi solo usando "
                        "il contesto fornito. Se il contesto non contiene la risposta, "
                        "devi dire che l'informazione non è presente nei manuali consultati."
                    )
                },
                {
                    "role": "user",
                    "content": content_msg
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 250
            },
        },
        timeout=600,
    )

    response.raise_for_status()

    answer = response.json().get("message", {}).get("content", "").strip()

    if not answer:
        answer = "(Nessuna risposta dal modello. Verificare le risorse o riformulare la domanda.)"

    return answer, chunks
