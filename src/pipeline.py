"""
Phase 7/8: the production RAG core, migrated to Voyage AI API retrieval.

Retrieval now uses the Voyage API for BOTH embedding (query side) and reranking,
so the server holds NO local ML models (no torch / sentence-transformers). This
makes it deployable on tiny free hosts and cuts query latency from ~30s to ~6s.

Import-safe: nothing loads at import time; construct RAGPipeline() explicitly.
answer() never raises -- failures return a typed {error: ...} result.
"""

import logging
import re
import time

import anthropic
import chromadb
import voyageai

from config import (COLLECTION, CHROMA_PATH, DENSE_CANDIDATES, LLM_MODEL,
                    MAX_TOKENS, REFUSAL, SECTION_LABELS, SYSTEM_PROMPT, TOP_K,
                    VOYAGE_EMBED_MODEL, VOYAGE_RERANK_MODEL)

logger = logging.getLogger(__name__)

LLM_MAX_RETRIES = 3
LLM_TIMEOUT_S = 60.0

PRICE_PER_MTOK_IN = 3.00
PRICE_PER_MTOK_OUT = 15.00


class RAGPipeline:
    """Dense retrieve (Voyage) -> Voyage rerank -> grounded, cited answer (Claude)."""

    def __init__(self):
        t0 = time.perf_counter()
        self.vo = voyageai.Client()          # reads VOYAGE_API_KEY
        self.collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION)
        self.llm = anthropic.Anthropic(max_retries=LLM_MAX_RETRIES, timeout=LLM_TIMEOUT_S)
        logger.info("pipeline loaded in %.1fs (%d chunks indexed)",
                    time.perf_counter() - t0, self.collection.count())

    # --- retrieval: Voyage query embedding -> Chroma -> Voyage rerank ---
    def retrieve(self, query, k=TOP_K, where=None, timings=None):
        t0 = time.perf_counter()
        # input_type="query" -- the asymmetric counterpart to "document" at index time
        qvec = self.vo.embed([query], model=VOYAGE_EMBED_MODEL,
                             input_type="query").embeddings[0]
        t_embed = time.perf_counter()

        res = self.collection.query(query_embeddings=[qvec], n_results=DENSE_CANDIDATES, where=where)
        cand = [
            {"id": cid, "text": doc, "meta": meta}
            for cid, doc, meta in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
        ]
        t_search = time.perf_counter()

        out = []
        if cand:
            rr = self.vo.rerank(query, [c["text"] for c in cand],
                                model=VOYAGE_RERANK_MODEL, top_k=k)
            for item in rr.results:
                c = cand[item.index]
                c["rerank_score"] = float(item.relevance_score)
                out.append(c)
        t_rerank = time.perf_counter()

        if timings is not None:
            timings.update({
                "embed_ms": round((t_embed - t0) * 1000, 1),
                "search_ms": round((t_search - t_embed) * 1000, 1),
                "rerank_ms": round((t_rerank - t_search) * 1000, 1),
                "candidates": len(cand),
            })
        return out

    def _build_context(self, chunks):
        blocks = []
        for i, c in enumerate(chunks, 1):
            m = c["meta"]
            label = SECTION_LABELS.get(m["section"], m["section"])
            blocks.append(f"[{i}] ({m['ticker']} FY{m['fiscal_year']}, {label})\n{c['text']}")
        return "\n\n".join(blocks)

    def _generate(self, query, chunks):
        context = self._build_context(chunks)
        user = (f"Question: {query}\n\nContext passages:\n\n{context}\n\n"
                "Answer using only the context above, citing sources like [1], [2].")
        resp = self.llm.messages.create(
            model=LLM_MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = {"input_tokens": resp.usage.input_tokens,
                 "output_tokens": resp.usage.output_tokens}
        return text, usage

    def _result(self, answer, sources, chunks, metrics, error=None):
        return {"answer": answer, "sources": sources, "chunks": chunks,
                "metrics": metrics, "error": error}

    def _fail(self, message, err_type, t0, timings):
        metrics = {**(timings or {}),
                   "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                   "error": err_type}
        return self._result(message, [], [], metrics, error={"type": err_type, "message": message})

    def answer(self, query, k=TOP_K, where=None):
        """Never raises. Returns {answer, sources, chunks, metrics, error}."""
        t0 = time.perf_counter()
        if not query or not query.strip():
            return self._fail("Please provide a question.", "empty_query", t0, {})

        timings = {}
        try:
            chunks = self.retrieve(query, k=k, where=where, timings=timings)
        except Exception:
            logger.exception("retrieval failed")
            return self._fail(
                "Sorry, I couldn't search the filings right now. Please try again.",
                "retrieval_error", t0, timings)

        if not chunks:
            timings["generate_ms"] = 0.0
            metrics = self._metrics(timings, {"input_tokens": 0, "output_tokens": 0}, t0, refused=True)
            return self._result(REFUSAL, [], [], metrics, error=None)

        t_gen0 = time.perf_counter()
        try:
            text, usage = self._generate(query, chunks)
        except anthropic.APIStatusError as e:
            code = getattr(e, "status_code", "?")
            logger.error("LLM status error %s", code)
            return self._fail(
                "The answer service is temporarily unavailable. Please try again in a moment.",
                f"llm_status_{code}", t0, timings)
        except (anthropic.APIConnectionError, anthropic.APITimeoutError):
            logger.error("LLM connection/timeout after retries")
            return self._fail("The answer service timed out. Please try again.",
                              "llm_timeout", t0, timings)
        except anthropic.APIError:
            logger.exception("LLM error")
            return self._fail("The answer service failed. Please try again.",
                              "llm_error", t0, timings)
        timings["generate_ms"] = round((time.perf_counter() - t_gen0) * 1000, 1)

        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
        sources = []
        for n in cited:
            if 1 <= n <= len(chunks):
                m = chunks[n - 1]["meta"]
                sources.append({
                    "n": n, "chunk_id": chunks[n - 1]["id"], "ticker": m["ticker"],
                    "fiscal_year": m["fiscal_year"],
                    "section": SECTION_LABELS.get(m["section"], m["section"]),
                })

        metrics = self._metrics(timings, usage, t0, refused=(text.strip() == REFUSAL))
        logger.info("query=%r total=%.0fms cost=$%.5f",
                    query[:60], metrics["total_ms"], metrics["est_cost_usd"])
        return self._result(text, sources, chunks, metrics, error=None)

    def _metrics(self, timings, usage, t0, refused):
        cost = (usage["input_tokens"] / 1e6 * PRICE_PER_MTOK_IN
                + usage["output_tokens"] / 1e6 * PRICE_PER_MTOK_OUT)
        return {
            **timings,
            "retrieval_ms": round(timings.get("embed_ms", 0) + timings.get("search_ms", 0)
                                  + timings.get("rerank_ms", 0), 1),
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "est_cost_usd": round(cost, 6),
            "refused": refused,
            "error": None,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rag = RAGPipeline()
    for q in ["What risks does Boeing highlight about the 737 program?",
              "How does JPMorgan describe the components of its net revenue?"]:
        print("\n" + "=" * 72)
        print(f"Q: {q}")
        r = rag.answer(q)
        print(f"\n{r['answer'][:300]}\n")
        m = r["metrics"]
        print(f"  embed {m.get('embed_ms')}ms | search {m.get('search_ms')}ms | "
              f"rerank {m.get('rerank_ms')}ms | generate {m.get('generate_ms')}ms")
        print(f"  TOTAL {m['total_ms']}ms | ${m['est_cost_usd']}")