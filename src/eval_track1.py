"""
Phase 5, Track 1 eval harness: measures RETRIEVAL quality (hit@k, MRR) and
answer FAITHFULNESS on the hand-written narrative gold set. Reuses the retrieval
+ answering pipeline from rag.py.

Key choice: retrieval runs over the FULL corpus (no ticker filter). The test is
whether the correct company+section surfaces on its own -- filtering by ticker
would be grading with the answer key.

Run from the repo root (relative paths like data/chroma and data/eval/ resolve
from there, and src/ must be importable):
  export ANTHROPIC_API_KEY=sk-ant-...
  PYTHONPATH=src python src/eval_track1.py
"""

import json
from pathlib import Path

from rag import answer, client, LLM_MODEL   # reuse the exact pipeline under test

GOLD_PATH = "data/eval/eval_gold_narrative.jsonl"
K = 6                       # retrieve depth (rag.answer uses TOP_K=6)
KS = [1, 3, 5]              # report hit@k at these depths
REFUSAL_MARK = "don't have enough information"

FAITH_SYSTEM = (
    "You are a strict evaluator checking whether an ANSWER is fully supported by "
    "the provided CONTEXT passages. List each distinct factual claim in the answer "
    "and mark supported=true only if the context directly supports it. Return ONLY "
    "JSON: {\"claims\": [{\"claim\": \"...\", \"supported\": true|false}]}. "
    "No prose, no markdown."
)

def get_text(resp):
    """Concatenate text from all text blocks, ignoring thinking/other blocks."""
    return "".join(b.text for b in resp.content if b.type == "text")


def is_hit(meta, gold):
    """A retrieved chunk is a hit if ticker + section (+ year if specified) match."""
    if meta["ticker"] != gold["expected_ticker"]:
        return False
    if meta["section"] not in gold["expected_sections"]:
        return False
    if gold.get("expected_year") is not None and meta["fiscal_year"] != gold["expected_year"]:
        return False
    return True


def first_hit_rank(chunks, gold):
    """1-based rank of the first hit among retrieved chunks, or None."""
    for i, c in enumerate(chunks, 1):
        if is_hit(c["meta"], gold):
            return i
    return None


def faithfulness(ans, chunks):
    """Claim-level faithfulness via an LLM judge: supported_claims / total_claims.
    Returns None for refusals (no claims to check)."""
    if REFUSAL_MARK in ans.lower():
        return None
    context = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1))
    resp = client.messages.create(
        model=LLM_MODEL, max_tokens=4096,   # was 1024 -- claim lists overflow
        system=FAITH_SYSTEM,
        messages=[{"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{ans}"}],
    )
    text = get_text(resp)
    if not text.strip():
        print(f"  [faith] empty judge response")
        return None
    try:
        claims = json.loads(text[text.index("{"):text.rindex("}") + 1])["claims"]
    except Exception as e:
        print(f"  [faith] parse failed: {e} | tail={text[-80:]!r}")
        return None
    if not claims:
        return None
    return sum(1 for c in claims if c.get("supported")) / len(claims)


def main():
    gold = [json.loads(line) for line in Path(GOLD_PATH).open(encoding="utf-8")]
    print(f"Evaluating {len(gold)} gold questions (retrieval over full corpus)...\n")

    rows = []
    for g in gold:
        r = answer(g["question"], k=K, where=None)      # NO ticker filter
        chunks = r["chunks"]
        rank = first_hit_rank(chunks, g)
        refused = REFUSAL_MARK in r["answer"].lower()
        faith = faithfulness(r["answer"], chunks)
        rows.append({"id": g["id"], "ticker": g["expected_ticker"],
                     "first_hit_rank": rank, "refused": refused, "faithfulness": faith})
        top = chunks[0]["meta"]
        flag = f"hit@{rank}" if rank else "MISS"
        print(f"  {g['id']} [{g['expected_ticker']}] {flag:<7} "
              f"top=[{top['ticker']} {top['section']}]"
              f"{'  REFUSED' if refused else ''}")

    # --- aggregate ---
    n = len(rows)
    def hit_at(k):
        return sum(1 for r in rows if r["first_hit_rank"] and r["first_hit_rank"] <= k) / n
    mrr = sum((1.0 / r["first_hit_rank"]) if r["first_hit_rank"] else 0.0 for r in rows) / n
    faiths = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    mean_faith = (sum(faiths) / len(faiths)) if faiths else float("nan")
    refusal_rate = sum(1 for r in rows if r["refused"]) / n

    print("\n" + "=" * 48)
    print(f"{'Metric':<24}{'Value':>10}")
    print("-" * 48)
    for k in KS:
        print(f"{'hit@' + str(k):<24}{hit_at(k):>10.3f}")
    print(f"{'MRR':<24}{mrr:>10.3f}")
    print(f"{'faithfulness (mean)':<24}{mean_faith:>10.3f}   (n={len(faiths)})")
    print(f"{'refusal rate':<24}{refusal_rate:>10.3f}")
    print("=" * 48)

    Path("data/eval").mkdir(parents=True, exist_ok=True)
    out = "data/eval/track1_results_baseline.json"
    Path(out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nPer-question results -> {out}")


if __name__ == "__main__":
    main()