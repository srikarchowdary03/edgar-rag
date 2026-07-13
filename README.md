# edgar-rag

A retrieval-augmented QA system over SEC 10-K filings, with a measured evaluation
of both retrieval quality and answer faithfulness.

The corpus is the narrative sections of 10-Ks (Item 1 Business, Item 1A Risk
Factors, Item 7 MD&A) for six companies chosen for sector spread and retrieval
difficulty. Questions are answered strictly from retrieved passages, with
citations and refusal when the context doesn't support an answer.

## Corpus

| Ticker | Company | Years | Why it's in the set |
|--------|---------|-------|---------------------|
| MSFT | Microsoft | 2016, 2023 | Spans the cloud pivot; good cross-year comparisons |
| JPM | JPMorgan Chase | 2022 | Bank 10-K is structurally alien (regulatory capital, credit-loss provisions) — a retrieval stressor |
| ADBE | Adobe | 2015-2017, 2022 | Subscription/Creative Cloud business model |
| PFE | Pfizer | 2021 | Pharma vocabulary (pipeline, patents, COVID products) that overlaps with nothing else |
| AWK | American Water Works | 2020-2022 | Regulated utility; best multi-year coverage for trend questions |
| BA | Boeing | 2018, 2022 | 737 MAX crisis makes the risk-factor sections diverge sharply |

Each filing is stored on disk at `data/raw/{TICKER}/{YEAR}/{section}.txt`, where
`section` is one of `item1_business`, `item1a_risk_factors`, `item7_mdna`.

## Pipeline

The pipeline runs in phases, each a script in `src/`:

1. **`fetch_filings.py` / `ingest.py`** — pull target 10-Ks from SEC EDGAR and
   extract clean narrative sections.
2. **`chunk.py`** — split each section into token-sized chunks with overlap.
3. **`embed_index.py`** — embed chunks with a local `BAAI/bge-m3` model and
   persist them to a Chroma index at `data/chroma` (collection `filings`).
4. **`rag.py`** — retrieve the top-k chunks for a question and ask Claude to
   answer using only those chunks, citing sources by number and refusing when
   unsupported.
5. **`eval_track1.py`** — evaluate retrieval (hit@k, MRR) and answer
   faithfulness against the hand-written gold set in
   `data/eval/eval_gold_narrative.jsonl`.

## Evaluation

The gold set (`data/eval/eval_gold_narrative.jsonl`) is a set of narrative
questions, each labeled with `expected_ticker`, `expected_year`,
`expected_sections`, and `source`. Item 8 (financial statements) is deliberately
excluded — this track evaluates narrative retrieval only.

- **Retrieval** runs over the full corpus with no ticker filter: the test is
  whether the correct company + section surfaces on its own. A retrieved chunk
  is a hit when ticker + section (+ year, if the label specifies one) match.
- **Faithfulness** uses an LLM judge that lists each factual claim in the answer
  and marks whether the context supports it; the score is
  supported / total claims. Refusals score no claims and are excluded from the mean.

## Setup

Requires Python >= 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...     # LLM answering + judge
export EDGAR_CONTACT_EMAIL=you@example.com   # SEC EDGAR identity for fetching
```

Scripts resolve paths (`data/chroma`, `data/eval/`) from the repo root and
import sibling modules from `src/`, so run them from the repo root with `src`
on the path:

```bash
PYTHONPATH=src python src/rag.py
PYTHONPATH=src python src/eval_track1.py
```

The `data/` directory (raw filings, Chroma index, gold set) is gitignored and
not distributed with the repo.
