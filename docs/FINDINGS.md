# Engineering Findings & Evaluation Results

A running log of the decisions, measurements, and tradeoffs behind this SEC-filings
RAG system. Every design choice here was driven by the evaluation harness, not by
assumption — the recurring principle is *change one thing, re-run the table, keep it
only if the numbers move.*

---

## System overview

A retrieval system over SEC 10-K filings that answers narrative questions with
grounded, cited answers, plus a two-track evaluation that measures retrieval quality
and answer faithfulness.

- **Corpus:** 6 companies chosen for sector/vocabulary diversity — Microsoft, Adobe
  (software), JPMorgan (bank), American Water Works (utility), Boeing (aerospace),
  Pfizer (pharma). ~12 10-Ks, pulled from SEC EDGAR via `edgartools`.
- **Sections indexed:** Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A).
  **982 chunks** (~800 tokens, section-aware).
- **Embeddings:** BGE-M3 (local, MIT-licensed), stored in Chroma (cosine).
- **Generation:** Claude (via a provider-swappable wrapper; OpenAI supported as fallback).
- **Retrieval (final):** dense + cross-encoder reranker (BGE-reranker-v2-m3). See ablation.

---

## Key design decisions (and why)

**Section-aware chunking, not blob chunking.** Filings were parsed into labeled
sections and chunked *within* each section, so no chunk spans two sections. This gives
clean, single-topic chunks and free section metadata for citations and filtered retrieval.

**Item 8 (financial statements) excluded from the retrieval index.** Financial-statement
tables mangle under text splitting (rows lose their column headers), which produced
ambiguous, noisy chunks — Item 8 alone was ~47% of the raw corpus. Decision: narrative
questions are served by retrieval; numeric questions are to be served by structured XBRL
exact-match (Track 2, planned). This separated concerns and roughly halved the index.

**Two-track evaluation.** FinanceBench, for this corpus, turned out to be overwhelmingly
a *numeric* benchmark (its questions need the financial tables). So:
- **Track 1 (built):** hand-written narrative gold set (22 Qs) over Item 1/1A/7,
  measuring retrieval (hit@k, MRR) + faithfulness (claim-level LLM judge).
- **Track 2 (planned):** FinanceBench numeric questions answered from structured XBRL,
  scored by exact match.

**Faithfulness needs no gold answers.** It's judged as answer-vs-retrieved-context
(claim decomposition → supported fraction), which avoids fabricated answer keys.

**Provider-agnostic LLM layer.** Generation and judging route through one `complete()`
function; switching Anthropic/OpenAI is a one-env-var change. Real production pattern
(vendor fallback / cost routing).

---

## Evaluation results

### Track 1 — retrieval + faithfulness (baseline vs final)

Measured on 22 narrative gold questions; retrieval over the full corpus (no ticker
filter). Baseline = dense-only. Final = dense + reranker (BM25 dropped, see ablation).
Both faithfulness/refusal figures below are from runs on the final gold set (n07 reworded
to a corpus-answerable question).

| Config              | hit@1 | hit@3 | hit@5 |  MRR  | faithfulness | refusal |
|---------------------|:-----:|:-----:|:-----:|:-----:|:------------:|:-------:|
| dense (baseline)    | 0.636 | 0.909 | 0.955 | 0.782 | —            | —       |
| dense + reranker    | 0.727 | 0.909 | 0.955 | 0.830 | 0.998 (n=22) | 0.000   |

The reranker lifted top-1 precision (hit@1 +0.09, MRR +0.05) while recall (hit@3/hit@5)
held at ceiling. On the final config every question was answered (refusal 0/22) and
faithfulness reached 0.998 — the reranker reliably surfaces chunks that support grounded,
cited answers.

### Retrieval ablation — what each component contributes

Retrieval-only (no generation), 22 gold questions, candidate pool = 40.

| Config             | hit@1 | hit@3 | hit@5 |  MRR  |
|--------------------|:-----:|:-----:|:-----:|:-----:|
| sparse_only (BM25) | 0.455 | 0.682 | 0.864 | 0.599 |
| dense_only         | 0.636 | 0.909 | 0.955 | 0.782 |
| hybrid_norerank    | 0.682 | 0.818 | 0.909 | 0.777 |
| hybrid_rerank      | 0.727 | 0.909 | 0.955 | 0.828 |
| dense_rerank       | 0.727 | 0.909 | 0.955 | 0.830 |

**Findings:**
1. **The reranker is the largest single contributor** to ranking precision (MRR 0.782 → 0.830).
2. **BM25 alone is weak** on this gold set (hit@1 0.455) and misses paraphrased questions
   entirely — but it does out-rank dense on a few exact-term questions (e.g. n02, n03).
3. **Fusion alone slightly hurt recall:** adding BM25 without a reranker pushed some good
   dense hits down (hit@3 0.909 → 0.818). The reranker afterward recovered it to ceiling.
4. **Decision — drop BM25.** `dense_rerank` equals or beats `hybrid_rerank` on every metric
   (MRR 0.830 vs 0.828) while being simpler and faster (no BM25 index). Once a reranker is
   present, BM25 doesn't earn its place *on this corpus.*

**Caveat:** this gold set skews toward conceptual/paraphrased questions (dense's strength).
On a more exact-match / keyword-heavy question set, BM25 could pull its weight; the honest
claim is "dense+rerank wins *here*," not universally.

---

## Known limitations / open items

- **n10 / n12** (AWK questions) — retriever prefers the business section over risk
  factors; correct chunk present but ranked 4th–6th, not top-1. Not blocking (still hits).
- **Numeric questions (Track 2)** not yet built — structured XBRL exact-match is planned.
- **Table handling** — deferred by design (see Item 8 decision).
- **Faithfulness judge** uses the same model family that generates answers; a known
  self-preference bias exists. Reported number is directional.
- Reranking is CPU-bound and is the current latency bottleneck (production consideration).

---

## Config (final)

- Chunk size 800 tokens, overlap 120, section-aware.
- Embedder: `BAAI/bge-m3` (normalized, cosine).
- Retriever: dense top-40 → `BAAI/bge-reranker-v2-m3` → top-k (BM25 dropped after ablation).
- Generator: `claude-sonnet-5` (grounded, cited, refuses on insufficient context).
- Reproducibility: any config runs via `--retriever {dense,hybrid,dense_rerank}`.