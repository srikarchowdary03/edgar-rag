"""
Re-embed the corpus with the Voyage AI embedding API (replaces local BGE-M3) and
rebuild the Chroma index. Part of the API-retrieval migration that makes the
service deployable on tiny free hosts (no local ML models, no torch).

Free tier: 200M tokens -- re-embedding 982 chunks (~0.8M tokens) is essentially free.

Setup: export VOYAGE_API_KEY=...
       pip install voyageai chromadb
Run:   PYTHONPATH=src python src/embed_index.py
"""

import json
import time
from pathlib import Path

import chromadb
import voyageai

from config import CHROMA_PATH, CHUNKS_PATH, COLLECTION, VOYAGE_EMBED_MODEL

BATCH = 100   # chunks per embed call (stays under Voyage per-call token limits)

vo = voyageai.Client()   # reads VOYAGE_API_KEY

chunks = [json.loads(line) for line in Path(CHUNKS_PATH).open(encoding="utf-8")]
print(f"Loaded {len(chunks)} chunks; embedding with {VOYAGE_EMBED_MODEL}...")

texts = [c["text"] for c in chunks]
ids = [c["chunk_id"] for c in chunks]
metadatas = [{
    "ticker": c["ticker"], "company": c["company"], "fiscal_year": c["fiscal_year"],
    "section": c["section"], "chunk_index": c["chunk_index"], "n_tokens": c["n_tokens"],
} for c in chunks]

# --- embed in batches; input_type="document" optimizes the CORPUS side ---
embeddings, total_tokens = [], 0
for i in range(0, len(texts), BATCH):
    batch = texts[i:i + BATCH]
    r = vo.embed(batch, model=VOYAGE_EMBED_MODEL, input_type="document")
    embeddings.extend(r.embeddings)
    total_tokens += r.total_tokens
    print(f"  embedded {i + len(batch)}/{len(texts)}  (tokens: {total_tokens})")
    time.sleep(0.2)   # gentle on rate limits

# --- rebuild the Chroma collection with the new vectors ---
client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    client.delete_collection(COLLECTION)
except Exception:
    pass
col = client.create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

print(f"\nIndexed {col.count()} chunks with {VOYAGE_EMBED_MODEL} "
      f"(dim={len(embeddings[0])}). Embedding tokens used: {total_tokens}")
print("Note: this overwrote the BGE-M3 index. Compare the new eval numbers against "
      "the saved BGE numbers in FINDINGS (same gold set + metric code = valid comparison).")