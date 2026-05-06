"""
rag_mcp_server.py
-----------------
Tool MCP per Nanobot: interroga i manuali Buffetti/EGO/WENDY.

Questa versione usa retrieval ibrido:
1. keyword search sui file Markdown;
2. vector search tramite ChromaDB;
3. risposta finale basata solo sugli estratti trovati.
"""

import re
import sys
import contextlib
from pathlib import Path
from fastmcp import FastMCP

from rag_core import retrieve, index_exists, DOCS_DIR, clean_document

mcp = FastMCP("manuali-buffetti-rag")


STOPWORDS = {
    "come", "cosa", "quale", "quali", "dove", "quando", "perche", "perché",
    "devo", "posso", "fare", "faccio", "fa", "si", "mi", "spieghi",
    "su", "sul", "sulla", "nel", "nella", "nei", "nelle", "con", "per",
    "il", "lo", "la", "i", "gli", "le", "un", "una", "uno", "di", "del",
    "della", "dei", "degli", "delle", "a", "ad", "da", "e", "o", "in"
}


def normalize(text: str) -> str:
    """
    Normalizza il testo per confronti semplici.
    """
    text = text.lower()
    text = text.replace("à", "a").replace("è", "e").replace("é", "e")
    text = text.replace("ì", "i").replace("ò", "o").replace("ù", "u")
    return text


def query_terms(query: str) -> list[str]:
    """
    Estrae parole utili dalla domanda, rimuovendo parole troppo comuni.
    Esempio: "Come si fa il backup in WENDY?"
    diventa: ["backup", "wendy"].
    """
    q = normalize(query)
    words = re.findall(r"[a-z0-9]+", q)
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def title_for_file(path: Path) -> str:
    """
    Assegna un titolo leggibile in base al nome del file.
    """
    name = path.stem.upper()

    if "EGO" in name and "ASSISTENZA" in name:
        return "Manuale Assistenza EGO"

    if "WENDY" in name:
        return "Manuale Operativo WENDY"

    if "OUTPUT" in name:
        return "Guida Rapida EGO"

    return path.stem


def split_markdown_sections(text: str) -> list[str]:
    """
    Divide il Markdown in sezioni usando i titoli #, ##, ###.
    Così la keyword search lavora su sezioni intere, non su righe isolate.
    """
    sections = re.split(r"(?=^#{1,3}\s+)", text, flags=re.MULTILINE)
    return [s.strip() for s in sections if len(s.strip()) > 80]


def keyword_section_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Cerca nei file .md sezioni che contengono parole esatte della domanda.

    Questa è la keyword search.
    È utile per termini tecnici come:
    backup, ripristina, rotolo, carta, DGFE, SET, Wi-Fi, fattura, scorte.
    """
    terms = query_terms(query)

    if not terms:
        return []

    results = []

    for md_path in sorted(DOCS_DIR.glob("*.md")):
        try:
            raw = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        cleaned = clean_document(raw)
        sections = split_markdown_sections(cleaned)

        for section in sections:
            ns = normalize(section)

            score = 0

            # Punteggio base: +1 per ogni termine della domanda trovato nella sezione.
            for term in terms:
                if term in ns:
                    score += 1

            # Bonus per casi tecnici importanti.
            if "backup" in terms and ("backup" in ns or "ripristina" in ns or "ripristino" in ns):
                score += 6

            if "wendy" in terms and "wendy" in ns:
                score += 3

            if "rotolo" in terms and "carta" in terms and "rotolo" in ns and "carta" in ns:
                score += 6

            if "ego" in terms and "ego" in ns:
                score += 3

            if "dgfe" in terms and "dgfe" in ns:
                score += 6

            if "fattura" in terms and "fattura" in ns:
                score += 4

            if "wifi" in terms and ("wifi" in ns or "wi-fi" in ns):
                score += 4

            if "scorte" in terms and ("scorte" in ns or "magazzino" in ns):
                score += 4

            # Evita risultati debolissimi.
            if score <= 0:
                continue

            excerpt = " ".join(section.split())

            if len(excerpt) > 1800:
                excerpt = excerpt[:1800] + "..."

            results.append({
                "content": excerpt,
                "source": md_path.name,
                "title": title_for_file(md_path),
                "score": round(score / max(len(terms), 1), 3),
                "kind": "keyword"
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


def vector_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Usa il retrieval vettoriale già presente in rag_core.py.
    """
    if not index_exists():
        return []

    with contextlib.redirect_stdout(sys.stderr):
        chunks = retrieve(query, top_k=max_results)

    for c in chunks:
        c["kind"] = "vector"

    return chunks


def deduplicate(chunks: list[dict]) -> list[dict]:
    """
    Rimuove duplicati semplici tra risultati keyword e vector.
    """
    seen = set()
    output = []

    for chunk in chunks:
        text = chunk.get("content", "")
        key = normalize(text[:300])

        if key in seen:
            continue

        seen.add(key)
        output.append(chunk)

    return output


def build_tool_response(domanda: str, chunks: list[dict]) -> str:
    """
    Costruisce il testo che Nanobot riceve dal tool.
    """
    if not chunks:
        return (
            "NESSUN RISULTATO TROVATO NEI MANUALI.\n"
            "Non rispondere con conoscenza generale."
        )

    parts = []
    parts.append("RISULTATO DEL TOOL MCP MANUALI BUFFETTI")
    parts.append("Rispondi in italiano usando SOLO queste informazioni. Non inventare.\n")
    parts.append(f"Domanda: {domanda}\n")
    parts.append("ESTRATTI TROVATI NEI MANUALI:\n")

    for i, chunk in enumerate(chunks, 1):
        content = " ".join(chunk.get("content", "").split())

        if len(content) > 1600:
            content = content[:1600] + "..."

        parts.append(
            f"[Estratto {i}]\n"
            f"Tipo ricerca: {chunk.get('kind', '')}\n"
            f"Fonte: {chunk.get('title', '')}\n"
            f"File: {chunk.get('source', '')}\n"
            f"Score: {chunk.get('score', '')}\n"
            f"Testo: {content}\n"
        )

    parts.append(
        "\nISTRUZIONE PER L'AGENTE:\n"
        "Rispondi solo usando gli estratti sopra. "
        "Se gli estratti contengono una procedura, trasformala in passaggi numerati. "
        "Se gli estratti non contengono la risposta, scrivi: "
        "\"L'informazione non è presente nei manuali consultati.\""
    )

    return "\n".join(parts)


@mcp.tool()
def ask_manuali_buffetti(domanda: str) -> str:
    """
    Tool MCP chiamato da Nanobot.

    Fa retrieval ibrido:
    1. keyword search nei Markdown;
    2. vector search in ChromaDB;
    3. unione dei risultati.
    """

    keyword_chunks = keyword_section_search(domanda, max_results=5)
    vector_chunks = vector_search(domanda, max_results=5)

    # Prima mettiamo i risultati keyword, poi quelli vettoriali.
    chunks = deduplicate(keyword_chunks + vector_chunks)

    return build_tool_response(domanda, chunks)


if __name__ == "__main__":
    mcp.run(transport="stdio")