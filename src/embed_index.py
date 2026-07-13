"""
Phase 3 embedding + indexing: embed each chunk with a local open-source model
and load them into a persistent Chroma vector store. Ends with test queries so
you can see real retrieval working.

Model: BAAI/bge-m3 (MIT-licensed, strong open retrieval model; its family pairs
cleanly with a BGE reranker in Phase 6). First run downloads ~2GB of weights.
If your machine is constrained or you want the fastest setup, swap MODEL_NAME to
"sentence-transformers/all-MiniLM-L6-v2" (~90MB, lower quality but instant).
Note: BGE-M3 needs NO special query prefix, so we embed queries and documents
the same way. (Some other BGE models require a query instruction; M3 does not.)

Requires: pip install sentence-transformers chromadb
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
CHROMA_PATH = "data/chroma"
COLLECTION = "filings"
BATCH = 32

# --- 1. Load chunks ---
chunks = [json.loads(line) for line in Path("data/chunks.jsonl").open(encoding="utf-8")]
print(f"Loaded {len(chunks)} chunks")

texts = [c["text"] for c in chunks]
ids = [c["chunk_id"] for c in chunks]
# Chroma metadata must be scalar (str/int/float/bool) -- drop 'text', keep the rest
metadatas = [{
    "ticker": c["ticker"],
    "company": c["company"],
    "fiscal_year": c["fiscal_year"],
    "section": c["section"],
    "chunk_index": c["chunk_index"],
    "n_tokens": c["n_tokens"],
} for c in chunks]

# --- 2. Embed (normalized -> cosine similarity) ---
print(f"Loading model {MODEL_NAME} (first run downloads weights)...")
model = SentenceTransformer(MODEL_NAME)
print("Embedding chunks...")
embeddings = model.encode(
    texts, batch_size=BATCH, normalize_embeddings=True, show_progress_bar=True
).tolist()

# --- 3. Load into a fresh Chroma collection (cosine space) ---
client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    client.delete_collection(COLLECTION)   # rebuild cleanly on re-run
except Exception:
    pass
collection = client.create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
print(f"Indexed {collection.count()} chunks -> {CHROMA_PATH} (collection '{COLLECTION}')")


# --- 4. Test queries: prove retrieval works ---
def search(query, k=5, where=None):
    """Embed the query with the SAME model, then retrieve top-k from Chroma."""
    qvec = model.encode([query], normalize_embeddings=True).tolist()
    return collection.query(query_embeddings=qvec, n_results=k, where=where)


def show(query, res):
    print(f"\nQuery: {query}")
    for rank, (doc, meta, dist) in enumerate(zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]), 1):
        snippet = doc[:160].replace("\n", " ")
        print(f"  {rank}. [{meta['ticker']} FY{meta['fiscal_year']} {meta['section']}]"
              f" dist={dist:.3f}")
        print(f"     {snippet}...")


# a) general query across the whole corpus
q1 = "What are the main risk factors the company faces?"
show(q1, search(q1, k=5))

# b) filtered query -- retrieval scoped to one company via metadata
q2 = "impact of the 737 MAX grounding on operations"
show(q2, search(q2, k=5, where={"ticker": "BA"}))