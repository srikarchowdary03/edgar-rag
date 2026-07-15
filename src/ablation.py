"""
Retrieval ablation (Phase 6 analysis). Runs FOUR retrieval configs over the gold
set and reports hit@k / MRR for each, so you can see how much each component
(BM25, RRF fusion, reranker) actually contributes.

Retrieval only -- NO answer generation, NO judge, NO API calls, NO credits.
(An ANTHROPIC_API_KEY just needs to exist in the env for the import to succeed;
no request is ever made.)

Run:  PYTHONPATH=src python src/ablation.py
"""

import json
from pathlib import Path

# reuse the exact retrieval internals from Phase 6
from hybrid import (_dense, _sparse, _rrf, _reranker, _by_id,
                    DENSE_M, SPARSE_M, RERANK_POOL)

GOLD_PATH = "data/eval/eval_gold_narrative.jsonl"
TOPK = 10          # evaluation depth (enough for hit@5)
KS = [1, 3, 5]


# --- the four configs: each returns a ranked list of chunk_ids ---
def rank_dense(q):
    return _dense(q, TOPK)


def rank_sparse(q):
    return _sparse(q, TOPK)


def rank_hybrid_norerank(q):
    fused = _rrf([_dense(q, DENSE_M), _sparse(q, SPARSE_M)])
    return fused[:TOPK]


def rank_hybrid_rerank(q):
    fused = _rrf([_dense(q, DENSE_M), _sparse(q, SPARSE_M)])[:RERANK_POOL]
    scores = _reranker.predict([(q, _by_id[cid]["text"]) for cid in fused])
    ranked = [cid for cid, _ in sorted(zip(fused, scores),
                                       key=lambda x: x[1], reverse=True)]
    return ranked[:TOPK]

def rank_dense_rerank(q):
    cand = _dense(q, RERANK_POOL)
    scores = _reranker.predict([(q, _by_id[cid]["text"]) for cid in cand])
    ranked = [cid for cid, _ in sorted(zip(cand, scores), key=lambda x: x[1], reverse=True)]
    return ranked[:TOPK]


CONFIGS = {
    "dense_only":       rank_dense,
    "sparse_only":      rank_sparse,
    "hybrid_norerank":  rank_hybrid_norerank,
    "hybrid_rerank":    rank_hybrid_rerank,
    "dense_rerank": rank_dense_rerank
}


# --- metrics ---
def is_hit(cid, gold):
    c = _by_id[cid]
    if c["ticker"] != gold["expected_ticker"]:
        return False
    if c["section"] not in gold["expected_sections"]:
        return False
    if gold.get("expected_year") is not None and c["fiscal_year"] != gold["expected_year"]:
        return False
    return True


def first_hit_rank(ids, gold):
    for i, cid in enumerate(ids, 1):
        if is_hit(cid, gold):
            return i
    return None


def main():
    gold = [json.loads(line) for line in Path(GOLD_PATH).open(encoding="utf-8")]
    print(f"Ablation over {len(gold)} gold questions "
          f"(DENSE_M={DENSE_M}, SPARSE_M={SPARSE_M}, RERANK_POOL={RERANK_POOL})\n")

    # rank every question under every config once
    ranks = {name: {} for name in CONFIGS}          # name -> {qid -> first_hit_rank}
    for g in gold:
        for name, fn in CONFIGS.items():
            ranks[name][g["id"]] = first_hit_rank(fn(g["question"]), g)

    # --- per-question rank matrix (— = miss) ---
    hdr = f"{'id':<5}" + "".join(f"{n:>18}" for n in CONFIGS)
    print(hdr)
    print("-" * len(hdr))
    for g in gold:
        row = f"{g['id']:<5}"
        for name in CONFIGS:
            r = ranks[name][g["id"]]
            row += f"{(str(r) if r else '—'):>18}"
        print(row)

    # --- aggregate table ---
    n = len(gold)
    def hit_at(name, k):
        return sum(1 for g in gold
                   if ranks[name][g["id"]] and ranks[name][g["id"]] <= k) / n
    def mrr(name):
        return sum((1.0 / ranks[name][g["id"]]) if ranks[name][g["id"]] else 0.0
                   for g in gold) / n

    print("\n" + "=" * 60)
    print(f"{'config':<18}" + "".join(f"{'hit@'+str(k):>9}" for k in KS)
          + f"{'MRR':>9}")
    print("-" * 60)
    for name in CONFIGS:
        print(f"{name:<18}"
              + "".join(f"{hit_at(name, k):>9.3f}" for k in KS)
              + f"{mrr(name):>9.3f}")
    print("=" * 60)

    Path("data/eval").mkdir(parents=True, exist_ok=True)
    summary = {name: {"hit@1": hit_at(name, 1), "hit@3": hit_at(name, 3),
                      "hit@5": hit_at(name, 5), "MRR": mrr(name)}
               for name in CONFIGS}
    Path("data/eval/ablation.json").write_text(json.dumps(summary, indent=2))
    print("\nSummary -> data/eval/ablation.json")


if __name__ == "__main__":
    main()