"""
rag_pipeline.py — Chunking + Answer Generation

Two responsibilities (deliberately co-located as they are both
pipeline concerns, not storage or I/O concerns):

  1. chunk_text()      — split raw documentation text into indexable chunks
  2. generate_answer() — assemble retrieved context + call Gemini

Swap points:
  - chunk_text():      replace with any strategy (semantic, sliding-window, etc.)
  - generate_answer(): replace the inner Gemini call with any LLM backend.
                       The function signature and return type stay the same.
"""

import os
import re
import textwrap

from google import genai
from dotenv import load_dotenv

from embeddings import SearchResult

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_CHUNK_LEN: int = 1500   # characters — split further if a chunk exceeds this
_MIN_CHUNK_LEN: int = 50     # characters — discard chunks shorter than this
_GEMINI_MODEL: str = "gemini-3.1-flash-lite-preview"

# ---------------------------------------------------------------------------
# 1. Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    max_len: int = _MAX_CHUNK_LEN,
    min_len: int = _MIN_CHUNK_LEN,
) -> list[str]:
    """
    Splits raw documentation text into clean, indexable chunks.

    Strategy (in order):
      1. Split on paragraph boundaries (double newlines).
      2. Discard chunks shorter than min_len (noise, headings, lone symbols).
      3. Keep paragraphs that fit within max_len as-is.
      4. For oversized paragraphs: split on sentence boundaries, accumulating
         sentences into windows ≤ max_len.
      5. If a single sentence is itself too long: hard-cut via textwrap.wrap().

    Args:
        text:    Raw plain text from scraper.fetch_and_clean().
        max_len: Maximum chunk length in characters (default 1000).
        min_len: Minimum chunk length in characters (default 50).

    Returns:
        Ordered list of non-empty text chunks, ready for embedding.

    Raises:
        ValueError: if text is empty or blank.
    """
    if not text or not text.strip():
        raise ValueError("chunk_text received empty or blank text.")

    raw_paragraphs = text.split("\n\n")
    chunks: list[str] = []

    for para in raw_paragraphs:
        para = para.strip()

        # Skip noise: empty lines, lone headings, short fragments
        if len(para) < min_len:
            continue

        if len(para) <= max_len:
            chunks.append(para)
        else:
            # Paragraph too long — split on sentence boundaries
            sub_chunks = _split_on_sentences(para, max_len, min_len)
            chunks.extend(sub_chunks)

    return chunks


def _split_on_sentences(text: str, max_len: int, min_len: int) -> list[str]:
    """
    Splits a long paragraph into windows of sentences, each ≤ max_len chars.

    Accumulates sentences greedily until adding the next would exceed max_len,
    then flushes the buffer. Falls back to textwrap.wrap() for sentences that
    are individually longer than max_len.

    Args:
        text:    A single long paragraph string.
        max_len: Maximum window character length.
        min_len: Minimum output chunk length (short results are discarded).

    Returns:
        List of sub-chunk strings.
    """
    # Split on terminal punctuation followed by whitespace
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_pattern.split(text)

    chunks: list[str] = []
    buffer: str = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate = (buffer + " " + sentence).strip() if buffer else sentence

        if len(candidate) <= max_len:
            buffer = candidate
        else:
            # Flush current buffer before starting new window
            if len(buffer) >= min_len:
                chunks.append(buffer)

            if len(sentence) > max_len:
                # Single sentence is itself too long — hard cut
                for segment in textwrap.wrap(sentence, width=max_len):
                    if len(segment) >= min_len:
                        chunks.append(segment)
                buffer = ""
            else:
                buffer = sentence

    # Flush any remaining content
    if len(buffer) >= min_len:
        chunks.append(buffer)

    return chunks


# ---------------------------------------------------------------------------
# 2. Answer Generation
# ---------------------------------------------------------------------------


def generate_answer(question: str, search_results: list[SearchResult]) -> str:
    """
    Generates a grounded answer using Gemini, restricted strictly to context.

    Prompt engineering principles applied:
      - Role assignment: "documentation assistant" (scopes the task)
      - Hard constraints: no general knowledge, no inference beyond context
      - Explicit fallback phrase: "Not found in the documentation."
        (consistent wording the caller can detect programmatically)
      - Numbered, source-labelled context blocks (helps Gemini cite correctly)

    Args:
        question:       The user's question string.
        search_results: Top-k SearchResult dicts from EmbeddingStore.search().
                        Each has keys: text, source, score.

    Returns:
        A grounded answer string, or exactly "Not found in the documentation."

    Raises:
        ValueError:       if question is empty or search_results is empty.
        EnvironmentError: if GEMINI_API_KEY is not set in the environment.
        RuntimeError:     if the Gemini API call fails for any reason.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")
    if not search_results:
        raise ValueError(
            "generate_answer() requires at least one search result. "
            "Ensure EmbeddingStore.search() returned results before calling this."
        )

    context_block = _assemble_context(search_results)

    prompt = f"""You are a documentation assistant. Your only job is to answer \
questions using the documentation excerpts provided below.

STRICT RULES — follow these exactly:
1. Answer ONLY using information explicitly present in the CONTEXT section.
2. Do NOT use your general knowledge, prior training data, or outside information.
3. Do NOT speculate, infer, or extend beyond what the CONTEXT states.
4. If the answer is not clearly present in the CONTEXT, respond:
   Not found in the documentation.
5. If partial information is available, provide the best possible answer using only the CONTEXT.
6. Keep answers concise and precise. Quote from the context when it is helpful.

---
CONTEXT:
{context_block}
---

QUESTION: {question}

ANSWER:"""

    try:
        client = _init_gemini()
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip()
    except (ValueError, EnvironmentError):
        raise
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _init_gemini() -> genai.Client:
    """
    Returns a configured google.genai Client.

    Reads GEMINI_API_KEY from the environment (populated by python-dotenv
    from the project .env file).

    Swap point: replace this function to use a different LLM SDK.
    The rest of generate_answer() remains unchanged.

    Raises:
        EnvironmentError: if GEMINI_API_KEY is missing or blank.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Create a .env file in doc-chatbot/ with: GEMINI_API_KEY=your_key_here"
        )
    return genai.Client(api_key=api_key)


def _assemble_context(search_results: list[SearchResult]) -> str:
    """
    Formats retrieved chunks into a numbered, source-labelled context block.

    Output format (each result):
        [1] (source: https://example.com/docs)
        The actual chunk text goes here...

        [2] (source: https://example.com/docs)
        Another chunk text...

    Numbered entries make it easier for Gemini to attribute answers
    and for callers to correlate sources with the response.
    """
    parts = []
    for i, result in enumerate(search_results, start=1):
        parts.append(
            f"[{i}] (source: {result['source']})\n{result['text']}"
        )
    return "\n\n".join(parts)
