"""
Phase 7 (#3): single source of configuration truth.

Every script imports its settings from here, so the production pipeline and the
evaluation harness can never silently drift apart (e.g. pipeline reranking 40
candidates while the eval measures 20).

Paths are anchored to the REPO ROOT via __file__, not the current working
directory -- so scripts run correctly from anywhere, not just repo root with
PYTHONPATH=src. This fixes the path-fragility issue that created a stray
Chroma DB under src/.

Any value can be overridden by an environment variable (useful for deploys).
"""

import os
from pathlib import Path

# --- paths (anchored to repo root: src/config.py -> parent.parent) ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EVAL_DIR = DATA_DIR / "eval"

MANIFEST_PATH = DATA_DIR / "manifest.json"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
CHROMA_PATH = str(DATA_DIR / "chroma")          # chromadb wants a str
GOLD_PATH = EVAL_DIR / "eval_gold_narrative.jsonl"

# --- models ---
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5")
COLLECTION = os.environ.get("CHROMA_COLLECTION", "filings")

# --- chunking (must match what's in the index; changing these needs a re-embed) ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TIKTOKEN_ENCODING = "cl100k_base"
SKIP_SECTIONS = {"item8_financials"}            # tables excluded by design

# --- retrieval (final config: dense -> rerank; BM25 dropped after ablation) ---
DENSE_CANDIDATES = int(os.environ.get("DENSE_CANDIDATES", 10))   # was 40; see pool_sweep
TOP_K = int(os.environ.get("TOP_K", 6))                          # chunks given to the LLM

# --- generation ---
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 1024))
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", 4096))

# --- display / prompts ---
SECTION_LABELS = {
    "item1_business": "Item 1 Business",
    "item1a_risk_factors": "Item 1A Risk Factors",
    "item7_mdna": "Item 7 MD&A",
}

REFUSAL = "I don't have enough information in the retrieved filings to answer that."

SYSTEM_PROMPT = (
    "You are a financial-filings analyst. Answer the user's question using ONLY the "
    "numbered context passages provided. Cite the passages you use with their bracket "
    "numbers, e.g. [1], [2]. If the context does not contain enough information to "
    f"answer, say exactly: \"{REFUSAL}\" "
    "Do not use outside knowledge. Be concise and precise with figures."
)