"""
chat_terminal.py
----------------
Interfaccia di chat da terminale per il RAG dei manuali Buffetti.

Uso:
    python chat_terminal.py
"""

import sys
import os

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from rag_core import ask, index_exists, OLLAMA_LLM_MODEL, EMBED_MODEL_NAME

# ─── Colori ANSI ─────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
GRAY   = "\033[90m"
RED    = "\033[91m"

def print_banner():
    banner = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════╗
║         ASSISTENTE MANUALI BUFFETTI — RAG Chat           ║
║                                                          ║
║  Modello LLM : {OLLAMA_LLM_MODEL:<25s}          ║
║  Embedding   : {EMBED_MODEL_NAME:<25s}          ║
╚══════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)
    print(f"{GRAY}Comandi speciali:{RESET}")
    print(f"  {YELLOW}esci{RESET} / {YELLOW}exit{RESET} / {YELLOW}quit{RESET}  → chiudi il programma")
    print(f"  {YELLOW}fonti{RESET}              → mostra/nascondi le fonti usate")
    print(f"  {YELLOW}help{RESET}               → esempi di domande\n")

def print_sources(chunks):
    if not chunks:
        return
    print(f"\n{GRAY}{'─'*60}")
    print(f"  📚 Fonti consultate:{RESET}")
    seen = set()
    for i, chunk in enumerate(chunks, 1):
        key = chunk["title"]
        indicator = f"  {i}. [{chunk['score']:.2f}] {chunk['title']}"
        if key not in seen:
            print(f"{GRAY}{indicator}{RESET}")
            seen.add(key)
        # Mostra un'anteprima del testo
        preview = chunk["content"][:120].replace('\n', ' ').strip()
        print(f"{GRAY}     └─ \"{preview}…\"{RESET}")
    print(f"{GRAY}{'─'*60}{RESET}\n")

def print_help():
    examples = [
        "Come si accende il registratore telematico EGO?",
        "Come si inserisce il rotolo di carta?",
        "Come si attiva la lotteria degli scontrini?",
        "Quali sono i codici di errore del registratore?",
        "Come si effettua il backup in WENDY?",
        "Come si crea una fidelity card in WENDY?",
        "Come si programma l'intestazione dello scontrino?",
        "Come si connette il registratore al WiFi?",
        "Cosa fare quando la batteria è scarica?",
        "Come si emette una fattura elettronica?",
    ]
    print(f"\n{YELLOW}💡 Esempi di domande:{RESET}")
    for ex in examples:
        print(f"   • {ex}")
    print()

def main():
    # Verifica prerequisiti
    if not index_exists():
        print(f"{RED}⚠️  Indice vettoriale non trovato!{RESET}")
        print(f"  Prima esegui: {YELLOW}python build_index.py{RESET}")
        sys.exit(1)

    print_banner()

    show_sources = True
    print(f"{GREEN}✅ Indice caricato. Pronto per le domande!{RESET}\n")

    while True:
        try:
            # Prompt di input
            query = input(f"{CYAN}{BOLD}Tu ❯{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{GRAY}Arrivederci!{RESET}")
            break

        if not query:
            continue

        # Comandi speciali
        query_lower = query.lower()
        if query_lower in ("esci", "exit", "quit", "q"):
            print(f"\n{GRAY}Arrivederci!{RESET}")
            break
        elif query_lower == "fonti":
            show_sources = not show_sources
            state = "ATTIVATE" if show_sources else "DISATTIVATE"
            print(f"{GRAY}  → Fonti {state}{RESET}\n")
            continue
        elif query_lower == "help":
            print_help()
            continue

        # Risposta RAG
        print(f"\n{GRAY}⏳ Ricerca in corso...{RESET}", end="\r")
        try:
            answer, chunks = ask(query)
        except Exception as e:
            print(f"\n{RED}❌ Errore: {e}{RESET}\n")
            continue

        # Stampa la risposta
        print(f"\n{GREEN}{BOLD}Assistente ❯{RESET}")
        # Stampa la risposta riga per riga con un piccolo indent
        for line in answer.split('\n'):
            print(f"  {line}")

        # Mostra le fonti
        if show_sources:
            print_sources(chunks)
        else:
            print()


if __name__ == "__main__":
    main()
