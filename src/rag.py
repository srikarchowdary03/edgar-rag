"""
Phase 4 grounded answering with citations. Connects to the persisted Chroma
index from Phase 3 (no re-embedding), retrieves the top-k chunks for a question,
and asks Claude to answer USING ONLY those chunks -- citing sources by number
and refusing when the context doesn't support an answer.

Setup (run from the repo root; the data/chroma index path resolves from there
and src/ must be importable):
  export ANTHROPIC_API_KEY=sk-ant-...
  pip install anthropic sentence-transformers chromadb
  PYTHONPATH=src python src/rag.py
Model strings update over time; current list:
  https://docs.claude.com/en/docs/about-claude/models
"""

import re

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"          # MUST match the model used in Phase 3
CHROMA_PATH = "data/chroma"
COLLECTION = "filings"
LLM_MODEL = "claude-sonnet-5"       # good quality/cost balance for RAG generation
TOP_K = 6

SECTION_LABELS = {
    "item1_business": "Item 1 Business",
    "item1a_risk_factors": "Item 1A Risk Factors",
    "item7_mdna": "Item 7 MD&A",
}

SYSTEM_PROMPT = (
    "You are a financial-filings analyst. Answer the user's question using ONLY the "
    "numbered context passages provided. Cite the passages you use with their bracket "
    "numbers, e.g. [1], [2]. If the context does not contain enough information to "
    "answer, say exactly: \"I don't have enough information in the retrieved filings "
    "to answer that.\" Do not use outside knowledge. Be concise and precise with figures."
)

# --- connect to the existing index (load model for QUERY embedding only) ---
print("Loading embedding model + index...")
embedder = SentenceTransformer(MODEL_NAME)
collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION)
client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment



def get_text(resp):
    """Concatenate text from all text blocks, ignoring thinking/other blocks."""
    return "".join(b.text for b in resp.content if b.type == "text")

def retrieve(query, k=TOP_K, where=None):
    qvec = embedder.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=qvec, n_results=k, where=where)
    return [
        {"id": cid, "text": doc, "meta": meta, "dist": dist}
        for doc, meta, dist, cid in zip(
            res["documents"][0], res["metadatas"][0],
            res["distances"][0], res["ids"][0])
    ]


def build_context(chunks):
    blocks = []
    for i, c in enumerate(chunks, 1):
        m = c["meta"]
        label = SECTION_LABELS.get(m["section"], m["section"])
        blocks.append(f"[{i}] ({m['ticker']} FY{m['fiscal_year']}, {label})\n{c['text']}")
    return "\n\n".join(blocks)


def answer(query, k=TOP_K, where=None):
    chunks = retrieve(query, k=k, where=where)
    context = build_context(chunks)
    user_msg = (
        f"Question: {query}\n\n"
        f"Context passages:\n\n{context}\n\n"
        "Answer using only the context above, citing sources like [1], [2]."
    )
    resp = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,          # grounding is enforced here, not via temperature
        messages=[{"role": "user", "content": user_msg}],
    )
    text = get_text(resp)

    # map cited [n] back to the source chunks for a citation list
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    sources = []
    for n in cited:
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            m = c["meta"]
            sources.append(
                f"[{n}] {c['id']} ({m['ticker']} FY{m['fiscal_year']}, "
                f"{SECTION_LABELS.get(m['section'], m['section'])})")
    return {"answer": text, "sources": sources, "chunks": chunks}


if __name__ == "__main__":
    demos = [
        ("What are the main risks Boeing highlights about the 737 program?", {"ticker": "BA"}),
        ("How does Microsoft describe its cloud business strategy?", {"ticker": "MSFT"}),
        ("What is the airspeed velocity of an unladen swallow?", None),  # should refuse
    ]
    for q, where in demos:
        print("\n" + "=" * 72)
        print(f"Q: {q}")
        r = answer(q, where=where)
        print(f"\n{r['answer']}\n")
        if r["sources"]:
            print("Sources:")
            for s in r["sources"]:
                print(f"  {s}")