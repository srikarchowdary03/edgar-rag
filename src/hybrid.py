"""
Phase 6 hybrid retrieval + reranking. Combines DENSE (Chroma / BGE-M3) and
SPARSE (BM25) retrieval via reciprocal rank fusion (RRF), then reranks the merged
candidates with a local BGE cross-encoder. Exposes retrieve(query, k, where) with
the SAME signature and return format as rag.retrieve, so the eval harness can swap
it in with a flag.

Pipeline:  query -> [dense top-M  +  BM25 top-M] -> RRF merge -> rerank pool -> top-k

Requires: pip install rank-bm25
(sentence-transformers + chromadb already installed in earlier phases; first run
downloads the reranker weights, ~0.5GB.)
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from rag import embedder, collection   # reuse the already-loaded BGE-M3 + Chroma

CHUNKS_PATH = "data/chunks.jsonl"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
DENSE_M = 40        # candidates pulled from dense search
SPARSE_M = 40       # candidates pulled from BM25
RRF_K = 60          # RRF constant (standard default)
RERANK_POOL = 40    # how many fused candidates to rerank

print("Building BM25 index + loading reranker...")
_chunks = [json.loads(line) for line in Path(CHUNKS_PATH).open(encoding="utf-8")]
_by_id = {c["chunk_id"]: c for c in _chunks}
_corpus_ids = [c["chunk_id"] for c in _chunks]


def _tok(text):
    return re.findall(r"[a-z0-9]+", text.lower())


_bm25 = BM25Okapi([_tok(c["text"]) for c in _chunks])
_reranker = CrossEncoder(RERANKER_NAME, max_length=512)


def _match_where(chunk, where):
    return all(chunk.get(key) == val for key, val in where.items())


def _dense(query, m, where=None):
    qvec = embedder.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=qvec, n_results=m, where=where)
    return res["ids"][0]                       # ranked chunk_ids


def _sparse(query, m, where=None):
    scores = _bm25.get_scores(_tok(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ids = []
    for i in order:
        cid = _corpus_ids[i]
        if where and not _match_where(_by_id[cid], where):
            continue
        ids.append(cid)
        if len(ids) >= m:
            break
    return ids


def _rrf(ranked_lists, k=RRF_K):
    scores = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def retrieve(query, k=6, where=None):
    """Hybrid + reranked retrieval. Same return shape as rag.retrieve."""
    dense_ids = _dense(query, DENSE_M, where=where)
    sparse_ids = _sparse(query, SPARSE_M, where=where)
    fused = _rrf([dense_ids, sparse_ids])[:RERANK_POOL]

    # cross-encoder rerank of the fused candidate pool
    pairs = [(query, _by_id[cid]["text"]) for cid in fused]
    rr_scores = _reranker.predict(pairs)
    reranked = sorted(zip(fused, rr_scores), key=lambda x: x[1], reverse=True)

    out = []
    for cid, score in reranked[:k]:
        c = _by_id[cid]
        out.append({
            "id": cid,
            "text": c["text"],
            "meta": {"ticker": c["ticker"], "company": c["company"],
                     "fiscal_year": c["fiscal_year"], "section": c["section"],
                     "chunk_index": c["chunk_index"], "n_tokens": c["n_tokens"]},
            "dist": float(score),              # here 'dist' holds rerank score (higher=better)
        })
    return out

def retrieve_dense_rerank(query, k=6, where=None):
    """Dense retrieve -> cross-encoder rerank (no BM25). The chosen production path."""
    cand = _dense(query, RERANK_POOL, where=where)          # dense candidates only
    scores = _reranker.predict([(query, _by_id[cid]["text"]) for cid in cand])
    reranked = sorted(zip(cand, scores), key=lambda x: x[1], reverse=True)
    out = []
    for cid, score in reranked[:k]:
        c = _by_id[cid]
        out.append({
            "id": cid, "text": c["text"],
            "meta": {"ticker": c["ticker"], "company": c["company"],
                     "fiscal_year": c["fiscal_year"], "section": c["section"],
                     "chunk_index": c["chunk_index"], "n_tokens": c["n_tokens"]},
            "dist": float(score),
        })
    return out


if __name__ == "__main__":
    for q in ["What technology or competitive risks does Adobe highlight in its risk factors?",
              "impact of the 737 MAX grounding on operations"]:
        print(f"\nQ: {q}")
        for r in retrieve(q, k=5):
            print(f"  {r['dist']:.3f} [{r['meta']['ticker']} {r['meta']['section']}] "
                  f"{r['text'][:90].strip()}...")