"""
embeddings.py — Text chunks → FAISS vector index

Single responsibility: embed chunks and provide semantic search.

Swap point: to change the embedding backend (e.g. Gemini Embeddings,
OpenAI Ada), subclass EmbeddingStore or replace _embed() only.
The rest of the system (app.py, rag_pipeline.py) never touches this directly.

Similarity metric: cosine similarity via FAISS IndexFlatIP
  → vectors are L2-normalised before indexing/querying so that
    inner-product == cosine similarity (range [-1, 1], higher = more similar).
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import TypedDict, Union

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ChunkMetadata(TypedDict):
    text: str
    source: str  # source URL the chunk came from


class SearchResult(TypedDict):
    text: str
    source: str
    score: float  # cosine similarity score (-1.0 to 1.0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality for docs


# ---------------------------------------------------------------------------
# EmbeddingStore
# ---------------------------------------------------------------------------


class EmbeddingStore:
    """
    Manages embedding generation and FAISS-based semantic search.

    Lifecycle:
      store = EmbeddingStore()          # loads the model once
      store.build_index(chunks, url)    # indexes a list of text chunks
      results = store.search(query)     # returns top-k ChunkMetadata + scores
      store.clear()                     # resets for a new document
    """

    def __init__(self) -> None:
        """
        Loads the SentenceTransformer model.

        The model is downloaded once and cached by sentence-transformers
        in ~/.cache/huggingface. Subsequent instantiations are instant.

        Raises:
            RuntimeError: if the model cannot be loaded.
        """
        try:
            self._model = SentenceTransformer(_MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{_MODEL_NAME}': {e}"
            )

        self._dimension: int = self._model.get_sentence_embedding_dimension()
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[ChunkMetadata] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self, chunks: Union[list[str], list[dict]], source_url: str = "") -> int:
        """
        Embeds a list of text chunks and builds a fresh FAISS index.

        Calling this again replaces the previous index entirely
        (in-memory design: one active document at a time).

        Args:
            chunks:     Non-empty list of text strings or dicts to index.
            source_url: Default URL if chunks are strings.

        Returns:
            Number of chunks successfully indexed.

        Raises:
            ValueError: if chunks list is empty.
            RuntimeError: if embedding fails.
        """
        if not chunks:
            raise ValueError("Cannot build index from an empty chunk list.")

        if isinstance(chunks[0], dict):
            text_values = [c["text"] for c in chunks]
            self._chunks = [
                ChunkMetadata(text=c["text"].strip(), source=c.get("source_url", source_url))
                for c in chunks
            ]
        else:
            text_values = chunks
            self._chunks = [
                ChunkMetadata(text=c.strip(), source=source_url)
                for c in chunks
            ]

        # Embed all chunks in one batched call (efficient)
        try:
            embeddings = self._embed(text_values)
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")

        # Build a fresh IndexFlatIP (inner product = cosine after normalisation)
        index = faiss.IndexFlatIP(self._dimension)
        index.add(embeddings)

        # Persist index and metadata
        self._index = index
        return len(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Finds the top-k chunks most semantically similar to the query.

        Args:
            query:  The user's question string.
            top_k:  Number of chunks to return (default 3).

        Returns:
            List of SearchResult dicts, ordered by descending similarity.
            Each result contains: text, source, score.

        Raises:
            RuntimeError: if no index has been built yet.
            ValueError:   if query is empty.
        """

        k = min(top_k, len(self._chunks))
        if k == 0:
            return []
        
        if self._index is None:
            raise RuntimeError(
                "No index loaded. Call build_index() before search()."
            )

        query = query.strip()
        if not query:
            raise ValueError("Search query must not be empty.")

        # Embed and normalise the query vector
        query_embedding = self._embed([query])  # shape (1, dim)

        # Clamp top_k to available chunks
        k = min(top_k, len(self._chunks))

        scores, indices = self._index.search(query_embedding, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 for padding when fewer results exist
                continue
            chunk = self._chunks[idx]
            results.append(
                SearchResult(
                    text=chunk["text"],
                    source=chunk["source"],
                    score=float(score),
                )
            )

        return results

    def clear(self) -> None:
        """Resets the store. The loaded model is retained (expensive to reload)."""
        self._index = None
        self._chunks = []

    @property
    def is_ready(self) -> bool:
        """True if an index has been built and is ready to search."""
        return self._index is not None and len(self._chunks) > 0

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently indexed."""
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        """
        Encodes texts and returns L2-normalised float32 vectors.

        L2 normalisation converts IndexFlatIP inner-product search into
        cosine similarity search (dot product of unit vectors = cosine).

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32.
        """
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalisation built in
            show_progress_bar=False,
        )
        faiss.normalize_L2(vectors)
        # sentence-transformers returns float32 by default; FAISS requires it
        return vectors.astype(np.float32)
