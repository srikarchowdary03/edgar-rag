"""
Phase 8: FastAPI service wrapping the RAG pipeline (simple, non-streaming).

- Loads RAGPipeline ONCE at startup (models load once, not per request).
- POST /query  -> {answer, sources, retrieved, metrics, error}
- GET  /health -> readiness + index size
- GET  /       -> the single-page UI

Run:  PYTHONPATH=src uvicorn api:app --port 8000
Then open http://localhost:8000
Requires: pip install fastapi "uvicorn[standard]"
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from config import ROOT, SECTION_LABELS
from pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["pipeline"] = RAGPipeline()          # construct once; models load here
    yield
    STATE.clear()


app = FastAPI(title="SEC Filings RAG", lifespan=lifespan)

_INDEX_HTML = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

# error type -> HTTP status (anything not listed, but with an error, is 503)
STATUS_FOR_ERROR = {"empty_query": 400, "llm_timeout": 504}


class QueryRequest(BaseModel):
    question: str
    ticker: Optional[str] = None
    year: Optional[int] = None


def _where(ticker, year):
    conds = []
    if ticker:
        conds.append({"ticker": ticker.upper()})
    if year:
        conds.append({"fiscal_year": year})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


@app.get("/", response_class=HTMLResponse)
def index():
    return _INDEX_HTML


@app.get("/health")
def health():
    rag = STATE.get("pipeline")
    if not rag:
        return JSONResponse({"status": "starting"}, status_code=503)
    return {"status": "ok", "chunks": rag.collection.count()}


@app.post("/query")
def query(req: QueryRequest):
    rag = STATE["pipeline"]
    r = rag.answer(req.question, where=_where(req.ticker, req.year))

    retrieved = [{
        "chunk_id": c["id"],
        "ticker": c["meta"]["ticker"],
        "fiscal_year": c["meta"]["fiscal_year"],
        "section": SECTION_LABELS.get(c["meta"]["section"], c["meta"]["section"]),
        "snippet": c["text"][:240].strip(),
        "rerank_score": round(c.get("rerank_score", 0.0), 3),
    } for c in r["chunks"]]

    body = {
        "answer": r["answer"],
        "sources": r["sources"],
        "retrieved": retrieved,
        "metrics": r["metrics"],
        "error": r["error"],
    }
    if r["error"]:
        return JSONResponse(body, status_code=STATUS_FOR_ERROR.get(r["error"]["type"], 503))
    return body