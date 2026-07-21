"""
Pool-size sweep (Phase 7 latency tuning).

The reranker is the latency bottleneck (~25s of a 44s query at pool=40), and its
cost scales with the number of candidates it scores. This sweeps DENSE_CANDIDATES
and reports quality (hit@k, MRR) AND mean rerank latency at each size, so the pool
can be chosen on measured tradeoff rather than guesswork.

Retrieval only -- no generation, no judge, NO API calls, no credits.

Run:  PYTHONPATH=src python src/pool_sweep.py
"""

import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (CHROMA_PATH, COLLECTION, EMBED_MODEL, GOLD_PATH,
                    RERANK_MODEL, TOP_K)

POOL_SIZES = [5, 10, 15, 20, 30, 40]     # 40 = current production setting
KS = [1, 3, 5]

print("Loading models + index...")
embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL, max_length=512)
collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION)

gold = [json.loads(line) for line in Path(GOLD_PATH).open(encoding="utf-8")]
print(f"Loaded {len(gold)} gold questions\n")


def is_hit(meta, g):
    if meta["ticker"] != g["expected_ticker"]:
        return False
    if meta["section"] not in g["expected_sections"]:
        return False
    if g.get("expected_year") is not None and meta["fiscal_year"] != g["expected_year"]:
        return False
    return True


# --- embed every query ONCE and cache the max-size candidate set ---
# Larger pools are supersets of smaller ones (same dense ranking), so we fetch
# the biggest pool once and slice it. This isolates rerank cost as the variable.
max_pool = max(POOL_SIZES)
cache = []
t_embed_total = 0.0
for g in gold:
    t0 = time.perf_counter()
    qvec = embedder.encode([g["question"]], normalize_embeddings=True).tolist()
    t_embed_total += time.perf_counter() - t0
    res = collection.query(query_embeddings=qvec, n_results=max_pool)
    cand = [{"text": doc, "meta": meta}
            for doc, meta in zip(res["documents"][0], res["metadatas"][0])]
    cache.append((g, cand))
print(f"Mean query embed time: {t_embed_total / len(gold) * 1000:.0f} ms\n")

rows = []
for pool in POOL_SIZES:
    ranks, rerank_times = [], []
    for g, cand in cache:
        sub = cand[:pool]
        t0 = time.perf_counter()
        scores = reranker.predict([(g["question"], c["text"]) for c in sub])
        rerank_times.append(time.perf_counter() - t0)
        ordered = [c for c, _ in sorted(zip(sub, scores), key=lambda x: x[1], reverse=True)]
        rank = next((i for i, c in enumerate(ordered[:TOP_K], 1) if is_hit(c["meta"], g)), None)
        ranks.append(rank)

    n = len(ranks)
    row = {
        "pool": pool,
        **{f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS},
        "MRR": sum((1.0 / r) if r else 0.0 for r in ranks) / n,
        "rerank_ms": sum(rerank_times) / n * 1000,
    }
    rows.append(row)
    print(f"  pool={pool:<3} hit@1={row['hit@1']:.3f} MRR={row['MRR']:.3f} "
          f"rerank={row['rerank_ms']:.0f}ms")

# --- summary table ---
print("\n" + "=" * 66)
print(f"{'pool':>5}{'hit@1':>9}{'hit@3':>9}{'hit@5':>9}{'MRR':>9}{'rerank_ms':>12}{'speedup':>10}")
print("-" * 66)
base = next(r for r in rows if r["pool"] == max_pool)["rerank_ms"]
for r in rows:
    print(f"{r['pool']:>5}{r['hit@1']:>9.3f}{r['hit@3']:>9.3f}{r['hit@5']:>9.3f}"
          f"{r['MRR']:>9.3f}{r['rerank_ms']:>12.0f}{base / r['rerank_ms']:>9.1f}x")
print("=" * 66)
print("\nPick the smallest pool whose hit@1/MRR match pool=40 -- that's free speed.")

Path("data/eval").mkdir(parents=True, exist_ok=True)
Path("data/eval/pool_sweep.json").write_text(json.dumps(rows, indent=2))
print("Saved -> data/eval/pool_sweep.json")
