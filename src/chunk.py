"""
Phase 2 chunking: split each narrative section file into token-sized chunks with
metadata, ready for embedding. Reads data/manifest.json, writes data/chunks.jsonl.

v1 retrieves over NARRATIVE sections only (Item 1, 1A, 7). Item 8 is excluded
because its tables mangle under text splitting; numeric answers will be served
from structured XBRL, and clean structured financial chunks can be added later
as a MEASURED enhancement.

Requires: pip install langchain-text-splitters tiktoken
"""

import json

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    ROOT, CHUNKS_PATH, MANIFEST_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
    TIKTOKEN_ENCODING, SKIP_SECTIONS,
)

enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)


def n_tokens(text: str) -> int:
    return len(enc.encode(text))


splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=TIKTOKEN_ENCODING,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

chunks = []
for rec in manifest:
    ticker = rec["ticker"]
    year = rec["fiscal_year"]
    company = rec.get("company", ticker)
    folder = ROOT / rec["path"]                # manifest path is repo-root-relative

    for section in rec["sections"]:            # iterate the sections we saved...
        if section in SKIP_SECTIONS:           # ...but skip excluded ones
            continue

        fpath = folder / f"{section}.txt"
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8").strip()
        if not text:
            continue

        pieces = splitter.split_text(text)     # one section at a time
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append({
                "chunk_id": f"{ticker}_{year}_{section}_{i:04d}",
                "ticker": ticker,
                "company": company,
                "fiscal_year": year,
                "section": section,            # <- comes from the loop, never hardcoded
                "chunk_index": i,
                "text": piece,
                "n_tokens": n_tokens(piece),
                "source_path": str(fpath),
            })

if not chunks:
    print("No chunks produced -- check data/manifest.json and your section files.")
    raise SystemExit

out = CHUNKS_PATH
with out.open("w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

# --- Summary ---
tok_counts = [c["n_tokens"] for c in chunks]
by_ticker, by_section = {}, {}
for c in chunks:
    by_ticker[c["ticker"]] = by_ticker.get(c["ticker"], 0) + 1
    by_section[c["section"]] = by_section.get(c["section"], 0) + 1

print(f"Total chunks: {len(chunks)} -> {out}")
print(f"Tokens/chunk: min={min(tok_counts)}, "
      f"avg={sum(tok_counts) // len(tok_counts)}, max={max(tok_counts)}")
print("Chunks per company:")
for t, n in sorted(by_ticker.items()):
    print(f"  {t}: {n}")
print("Chunks per section type:")
for s, n in sorted(by_section.items()):
    print(f"  {s}: {n}")

# sanity guard: catch the "everything became one section" class of bug
if len(by_section) < 2:
    print("\n[!] Only one section type present -- that's unexpected. "
          "Check that the loop reads `section` from rec['sections'].")