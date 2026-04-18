"""
app.py — Flask API server

Single responsibility: HTTP routing, request validation, and response formatting.
All business logic lives in scraper.py, embeddings.py, and rag_pipeline.py.

Endpoints:
  POST /ingest   — scrape a URL, chunk it, and build the FAISS index
  POST /ask      — retrieve relevant chunks and generate a Gemini answer
  GET  /status   — report current index state
"""

from flask import Flask, request, jsonify, Response, send_from_directory

from embeddings import EmbeddingStore
from scraper import fetch_and_clean
from rag_pipeline import chunk_text, generate_answer
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# App + global state
# ---------------------------------------------------------------------------

app = Flask(__name__)

# One shared EmbeddingStore for the lifetime of the server process.
# Loading the sentence-transformers model is expensive (~300ms);
# this ensures it happens once at startup, not per request.
_store = EmbeddingStore()

# Lightweight metadata kept on the app object to power GET /status
_state: dict = {
    "index_loaded": False,
    "source_url": None,
    "chunks_indexed": 0,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(data: dict, status: int = 200) -> Response:
    """Returns a success envelope response."""
    return jsonify({"success": True, "data": data}), status


def _err(code: str, message: str, status: int = 400) -> Response:
    """Returns an error envelope response."""
    return jsonify({"success": False, "error": {"code": code, "message": message}}), status


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def index():
    return send_from_directory(".", "index.html")


@app.post("/ingest")
def ingest() -> Response:
    """
    Scrapes a documentation URL, chunks the text, and builds the FAISS index.

    Request body (JSON):
      { "url": "https://example.com/docs/page" }

    Responses:
      201 — index built successfully
      400 — missing or blank URL field
      422 — URL unreachable, non-2xx, no meaningful content, or scrape too short
      500 — unexpected internal error
    """
    body = request.get_json(silent=True) or {}
    url: str = (body.get("url") or "").strip()

    if not url:
        return _err(
            "MISSING_URL",
            "Request body must include a non-empty 'url' field.",
            status=400,
        )

    # --- Scrape ---
    try:
        raw_text = fetch_and_clean(url)
    except ValueError as e:
        return _err("INGEST_FAILED", str(e), status=422)
    except Exception as e:
        return _err("INGEST_ERROR", f"Unexpected error during scraping: {e}", status=500)

    # --- Chunk ---
    try:
        chunks = chunk_text(raw_text)
    except ValueError as e:
        return _err("CHUNK_FAILED", str(e), status=422)
    except Exception as e:
        return _err("CHUNK_ERROR", f"Unexpected error during chunking: {e}", status=500)

    if not chunks:
        return _err(
            "NO_CHUNKS",
            "The page was scraped successfully but produced no indexable content. "
            "The page may be too short or structured in an unsupported way.",
            status=422,
        )

    # --- Index ---
    try:
        _store.clear()
        count = _store.build_index(chunks, source_url=url)
    except (ValueError, RuntimeError) as e:
        return _err("INDEX_FAILED", str(e), status=422)
    except Exception as e:
        return _err("INDEX_ERROR", f"Unexpected error building index: {e}", status=500)

    # Update server state
    _state["index_loaded"] = True
    _state["source_url"] = url
    _state["chunks_indexed"] = count
    print(f"[INGEST] URL: {url}")
    print(f"[INFO] Chunks created: {len(chunks)}")
    return _ok(
        {
            "url": url,
            "chunks_indexed": count,
            "message": "Documentation ingested successfully.",
        },
        status=201,
    )


@app.post("/ask")
def ask() -> Response:
    """
    Retrieves the most relevant chunks for a question and returns a
    Gemini-generated answer grounded strictly in that context.

    Request body (JSON):
      { "question": "What does zip() do?" }

    Optional fields:
      { "top_k": 3 }   — number of chunks to retrieve (default 3, max 10)

    Responses:
      200 — answer generated successfully
      400 — no index loaded yet (call /ingest first)
      422 — missing/blank question or invalid top_k
      503 — Gemini API key missing or API call failed
      500 — unexpected internal error
    """
    if not _store.is_ready:
        return _err(
            "NO_INDEX",
            "No documentation has been ingested yet. Call POST /ingest first.",
            status=400,
        )

    body = request.get_json(silent=True) or {}
    question: str = (body.get("question") or "").strip()

    if not question:
        return _err(
            "INVALID_QUESTION",
            "Request body must include a non-empty 'question' field.",
            status=422,
        )

    # Validate optional top_k
    raw_top_k = body.get("top_k", 3)
    try:
        top_k = int(raw_top_k)
        if not (1 <= top_k <= 10):
            raise ValueError()
    except (ValueError, TypeError):
        return _err(
            "INVALID_TOP_K",
            "'top_k' must be an integer between 1 and 10.",
            status=422,
        )

    # --- Retrieve ---
    try:
        search_results = _store.search(question, top_k=top_k)
    except (ValueError, RuntimeError) as e:
        return _err("RETRIEVAL_FAILED", str(e), status=500)
    except Exception as e:
        return _err("RETRIEVAL_ERROR", f"Unexpected retrieval error: {e}", status=500)

    if not search_results:
        return _ok(
            {
                "answer": "Not found in the documentation.",
                "sources": [],
            }
        )

    # --- Generate ---
    try:
        answer = generate_answer(question, search_results)
    except EnvironmentError as e:
        # Missing API key — actionable config error, return 503
        return _err("LLM_CONFIG_ERROR", str(e), status=503)
    except RuntimeError as e:
        # Gemini API call failed
        return _err("Gemini API call failed. Check API key and network.", str(e), status=503)
    except Exception as e:
        return _err("GENERATION_ERROR", f"Unexpected error during generation: {e}", status=500)
    print(f"[ASK] Question: {question}")
    print(f"[INFO] Retrieved {len(search_results)} chunks")
    return _ok(
        {
            "answer": answer,
            "sources": [
                {"text": r["text"], "source": r["source"], "score": round(r["score"], 4)}
                for r in search_results
            ],
        }
    )


@app.get("/status")
def status() -> Response:
    """
    Returns the current index state.

    Response 200:
      {
        "success": true,
        "data": {
          "index_loaded": true,
          "source_url": "https://...",
          "chunks_indexed": 42
        }
      }
    """
    return _ok(_state)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
