# Documentation RAG Q&A System — Design Document

> Validated design. All decisions confirmed before this document was written.

---

## Understanding Summary

- **What:** Flask-based RAG documentation Q&A — user provides a URL, system scrapes & indexes it, user asks questions, system answers using retrieved context
- **Why:** Natural language Q&A over any documentation page without manual reading
- **Who:** Developer/personal use (single user, local or simple hosted deployment)
- **Constraints:** Minimal, modular implementation; each layer independently replaceable
- **Non-goals:** Multi-page crawling, user auth, persistence, production scaling (Phase 2+)

---

## Tech Stack

| Concern          | Choice                                  |
|------------------|-----------------------------------------|
| Framework        | Flask                                   |
| Answer generation| Google Gemini API (`gemini-1.5-flash`)  |
| Embeddings       | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector store     | FAISS (in-memory, `IndexFlatL2`)        |
| Scraping         | `requests` + `BeautifulSoup4`           |
| Chunking         | Paragraph (`\n\n`) + fallback fixed split|

---

## Module Breakdown

```
doc-chatbot/
├── app.py            # Flask app — routes, request/response only
├── scraper.py        # URL → clean text
├── embeddings.py     # Chunks → FAISS index (EmbeddingStore class)
├── rag_pipeline.py   # chunk_text() + generate_answer() orchestration
├── requirements.txt  # Dependencies
└── .env              # GEMINI_API_KEY (not committed)
```

---

## Data Flow

```
POST /ingest { url }
  scraper.fetch_and_clean(url)         → raw_text
  rag_pipeline.chunk_text(raw_text)    → [chunk1, chunk2, ...]
  index.build_index(chunks)            → FAISS index (in-memory)

POST /ask { question }
  index.search(question, top_k=3)      → [chunk1, chunk2, chunk3]
  rag_pipeline.generate_answer(q, ctx) → answer string (Gemini)
```

---

## API Endpoints

All responses use the envelope pattern: `{ success, data?, error? }`

### POST /ingest

**Request**
```json
{ "url": "https://docs.python.org/3/library/functions.html" }
```

**201 Created**
```json
{
  "success": true,
  "data": {
    "url": "https://docs.python.org/3/library/functions.html",
    "chunks_indexed": 42,
    "message": "Documentation ingested successfully."
  }
}
```

**422 — invalid/unreachable URL**
```json
{
  "success": false,
  "error": { "code": "INGEST_FAILED", "message": "Could not fetch or parse the provided URL." }
}
```

---

### POST /ask

**Request**
```json
{ "question": "What does the zip() function do?" }
```

**200 OK**
```json
{
  "success": true,
  "data": {
    "answer": "The zip() function returns an iterator of tuples...",
    "sources": [
      "zip() makes an iterator that aggregates elements from each of the iterables...",
      "If the iterables are of uneven length, missing values are filled-in with fillvalue..."
    ]
  }
}
```

**400 — no index loaded**
```json
{
  "success": false,
  "error": { "code": "NO_INDEX", "message": "No documentation has been ingested. Call POST /ingest first." }
}
```

**422 — empty question**
```json
{
  "success": false,
  "error": { "code": "INVALID_QUESTION", "message": "Question must be a non-empty string." }
}
```

---

### GET /status

**200 OK**
```json
{
  "success": true,
  "data": {
    "index_loaded": true,
    "source_url": "https://docs.python.org/3/library/functions.html",
    "chunks_indexed": 42
  }
}
```

---

## Module Contracts

### scraper.py
```python
def fetch_and_clean(url: str) -> str:
    # requests.get → BeautifulSoup
    # Extract <main> or <article>, fallback to <body>
    # Remove <nav>, <header>, <footer>, <script>, <style>
    # Return stripped plain text
```

### embeddings.py
```python
class EmbeddingStore:
    def __init__(self): ...           # Load SentenceTransformer
    def build_index(self, chunks: list[str]): ...   # Build FAISS index
    def search(self, query: str, top_k=3) -> list[str]: ...  # Return top chunks
    # SWAP POINT: replace internals here to change embedding backend
```

### rag_pipeline.py
```python
def chunk_text(text: str, max_len=500, min_len=50) -> list[str]:
    # Split on \n\n → filter short → split oversized chunks

def generate_answer(question: str, context_chunks: list[str]) -> str:
    # Build prompt with context → call Gemini → return answer
    # SWAP POINT: replace this function to change LLM backend
```

### app.py
```python
# State: index (EmbeddingStore | None), source_url, chunk_count
# Routes: POST /ingest, POST /ask, GET /status
# Validates inputs, calls modules, formats envelope responses
```

---

## Decision Log

| # | Decision | Alternatives | Reason |
|---|----------|-------------|--------|
| 1 | Gemini for generation | OpenAI, Ollama | Existing stack; single key; modular swap point |
| 2 | sentence-transformers for embeddings | Gemini Embeddings, OpenAI Ada | Free, local, no per-chunk cost |
| 3 | FAISS in-memory | Chroma, Pinecone, disk FAISS | Zero infra; `faiss.write_index()` is trivial upgrade |
| 4 | Single-page scrape + clean content | Full crawl, raw HTML | Avoids crawl complexity; cleaner embeddings |
| 5 | Paragraph chunking + fallback split | Fixed char, sentence-level | Semantic alignment with doc structure |
| 6 | Envelope response pattern | Direct return, JSON:API | Consistent; explicit error codes; client-friendly |
| 7 | GET /status endpoint | None | Guards against NO_INDEX confusion |

---

## Implementation Phases

- [ ] **Phase 0** — `requirements.txt`, `.env` setup
- [ ] **Phase 1** — `scraper.py`: `fetch_and_clean()`
- [ ] **Phase 2** — `embeddings.py`: `EmbeddingStore` class
- [ ] **Phase 3** — `rag_pipeline.py`: `chunk_text()` + `generate_answer()`
- [ ] **Phase 4** — `app.py`: routes wired to modules
- [ ] **Phase 5** — Verification: ingest real URL, ask grounded question, test error cases

---

## Future Upgrade Points (Phase 2+)

- Persistence: `faiss.write_index()` / `faiss.read_index()`
- Swap LLM: edit only `generate_answer()` in `rag_pipeline.py`
- Swap embeddings: edit only `EmbeddingStore.__init__` and `search()`
- Multi-page crawl: extend `scraper.py` with BFS over same-domain links
- Simple HTML frontend: add `GET /` serving a chat UI
