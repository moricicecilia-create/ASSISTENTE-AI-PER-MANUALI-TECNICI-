"""
build_index.py
--------------
Eseguire UNA SOLA VOLTA per costruire l'indice vettoriale.
La prima esecuzione scarica automaticamente il modello di embedding (~470 MB).

Uso:
    python build_index.py
    python build_index.py --force   # rigenera l'indice da zero
"""

import sys
import time
import argparse
import chromadb

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from rag_core import (
    load_and_clean_docs,
    split_into_chunks,
    embed_texts,
    get_chroma_collection,
    CHROMA_DIR,
    OLLAMA_LLM_MODEL,
    EMBED_MODEL_NAME,
)


def build_index(force: bool = False):
    print("=" * 62)
    print("  COSTRUZIONE INDICE RAG — Manuali Buffetti")
    print("=" * 62)
    print(f"  LLM       : {OLLAMA_LLM_MODEL}")
    print(f"  Embedding : {EMBED_MODEL_NAME} (sentence-transformers)")
    print(f"  Indice    : {CHROMA_DIR}")
    print()

    collection     = get_chroma_collection()
    existing_count = collection.count()

    if existing_count > 0 and not force:
        print(f"⚠️  Indice già esistente con {existing_count} chunk.")
        print("   Usa --force per rigenerarlo.")
        return

    if existing_count > 0 and force:
        print(f"🗑️  Cancellazione indice esistente ({existing_count} chunk)...")
        chromadb.PersistentClient(path=str(CHROMA_DIR)).delete_collection("manuali_buffetti")
        collection = get_chroma_collection()
        print()

    # 1. Carica e pulisci
    print("📄 Caricamento e pulizia documenti...")
    docs = load_and_clean_docs()
    print()

    # 2. Chunking
    print("✂️  Divisione in chunk semantici...")
    chunks = split_into_chunks(docs)
    print()

    # 3. Embedding (veloce con sentence-transformers)
    print(f"🔢 Calcolo embedding per {len(chunks)} chunk...")
    print("   (Prima esecuzione: scarica il modello ~470 MB, poi è veloce)\n")

    t_start = time.time()
    BATCH_SIZE = 64  # batch ampio perché sentence-transformers è efficiente

    all_ids        = []
    all_docs       = []
    all_embeddings = []
    all_metadatas  = []

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch  = chunks[batch_start: batch_start + BATCH_SIZE]
        texts  = [c["content"] for c in batch]
        embeds = embed_texts(texts)

        all_ids.extend([f"chunk_{batch_start + i}" for i in range(len(batch))])
        all_docs.extend(texts)
        all_embeddings.extend(embeds)
        all_metadatas.extend([c["metadata"] for c in batch])

        done    = min(batch_start + BATCH_SIZE, len(chunks))
        pct     = done / len(chunks) * 100
        elapsed = time.time() - t_start
        bar     = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  [{bar}] {pct:5.1f}%  {done}/{len(chunks)} chunk  {elapsed:.1f}s", end="\r")

    # 4. Inserimento in ChromaDB (un'unica operazione)
    print(f"\n\n💾 Salvataggio in ChromaDB...")
    collection.add(
        ids=all_ids,
        documents=all_docs,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
    )

    elapsed_total = time.time() - t_start
    print(f"✅ Indice costruito in {elapsed_total:.1f}s")
    print(f"   Chunk totali: {collection.count()}")
    print()
    print("Ora puoi usare:")
    print("  python chat_terminal.py   →  chat da terminale")
    print("  python chat_web.py        →  web app (apri http://localhost:5000)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rigenera l'indice anche se esiste già")
    args = parser.parse_args()
    build_index(force=args.force)
