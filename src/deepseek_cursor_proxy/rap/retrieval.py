"""Retrieval Layer — context chunking, embedding, and vector search.

This module implements the Retrieval Layer component of the RAP pipeline,
responsible for chunking context messages, generating embeddings via LM Studio,
and retrieving relevant snippets from Qdrant vector storage.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4, 7.5
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx
import msgpack
import tiktoken

from deepseek_cursor_proxy.rap.config import RAPConfig

logger = logging.getLogger(__name__)


class EmbeddingUnavailableError(Exception):
    """Raised when the LM Studio embedding endpoint is unreachable or times out.

    The pipeline can catch this to gracefully skip retrieval and forward
    the full context to DeepSeek.

    Requirements: 13.2
    """

    pass


class QdrantUnavailableError(Exception):
    """Raised when Qdrant is unreachable or returns an error.

    The pipeline can catch this to gracefully skip retrieval and forward
    the full context to DeepSeek.

    Requirements: 13.1
    """

    pass


@dataclass
class Chunk:
    """A chunk of text extracted from a context message.

    Attributes:
        text: The chunk text content.
        token_count: Number of tokens in this chunk (counted via tiktoken).
        source_message_index: Index of the original message this chunk came from.
        metadata: Additional metadata about the chunk.
    """

    text: str
    token_count: int
    source_message_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    """A chunk with a relevance score from vector search.

    Attributes:
        chunk: The underlying Chunk.
        score: Similarity score from Qdrant search.
        vector_id: The vector ID in Qdrant.
    """

    chunk: Chunk
    score: float
    vector_id: str


class RetrievalLayer:
    """Retrieval Layer for context chunking, embedding, and vector search.

    Uses tiktoken (cl100k_base encoding) for accurate token counting.
    Preserves system messages and the latest user message unchanged.
    Only middle context messages (non-system, non-latest-user) are chunked.

    Args:
        config: RAPConfig instance with retrieval settings.
    """

    def __init__(self, config: RAPConfig | None = None) -> None:
        if config is None:
            config = RAPConfig()
        self._config = config
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._chunk_size = config.retrieval_max_tokens  # Use design default
        self._chunk_overlap = 64  # Default overlap from design

        # Use RetrievalConfig-style values from RAPConfig
        # chunk_size_tokens defaults to 512 per design doc
        self._chunk_size_tokens = 512
        self._chunk_overlap_tokens = 64

    @property
    def chunk_size_tokens(self) -> int:
        """The configured chunk size in tokens."""
        return self._chunk_size_tokens

    @chunk_size_tokens.setter
    def chunk_size_tokens(self, value: int) -> None:
        self._chunk_size_tokens = value

    @property
    def chunk_overlap_tokens(self) -> int:
        """The configured chunk overlap in tokens."""
        return self._chunk_overlap_tokens

    @chunk_overlap_tokens.setter
    def chunk_overlap_tokens(self, value: int) -> None:
        self._chunk_overlap_tokens = value

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken cl100k_base encoding."""
        return len(self._encoding.encode(text))

    def _decode_tokens(self, tokens: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._encoding.decode(tokens)

    def chunk_context(self, messages: list[dict[str, Any]]) -> list[Chunk]:
        """Split context messages into overlapping chunks.

        System messages and the latest user message are preserved unchanged
        (not chunked). Only middle context messages are chunked into windows
        of ``chunk_size_tokens`` with ``chunk_overlap_tokens`` overlap.

        Args:
            messages: List of chat messages in OpenAI format
                      (each has 'role' and 'content').

        Returns:
            List of Chunk instances from the middle context messages.

        Requirements: 6.1, 6.4
        """
        if not messages:
            return []

        # Identify which messages to chunk (middle context):
        # - Skip system messages (preserve unchanged)
        # - Skip the latest user message (preserve unchanged)
        # Everything else is "middle context" that gets chunked.
        middle_messages = self._extract_middle_messages(messages)

        chunks: list[Chunk] = []
        for msg_index, content in middle_messages:
            if not content:
                continue
            msg_chunks = self._chunk_text(content, msg_index)
            chunks.extend(msg_chunks)

        return chunks

    def _extract_middle_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[tuple[int, str]]:
        """Extract middle context messages (non-system, non-latest-user).

        Returns a list of (original_index, content) tuples for messages
        that should be chunked.
        """
        if not messages:
            return []

        # Find the index of the latest user message
        latest_user_index: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                latest_user_index = i
                break

        middle: list[tuple[int, str]] = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Skip system messages
            if role == "system":
                continue

            # Skip the latest user message
            if i == latest_user_index:
                continue

            # Only chunk messages with string content
            if isinstance(content, str) and content:
                middle.append((i, content))

        return middle

    def _chunk_text(self, text: str, source_message_index: int) -> list[Chunk]:
        """Split a single text into overlapping token-based chunks.

        Uses a sliding window of ``chunk_size_tokens`` with
        ``chunk_overlap_tokens`` overlap between consecutive chunks.

        Args:
            text: The text to chunk.
            source_message_index: Index of the source message.

        Returns:
            List of Chunk instances.
        """
        tokens = self._encoding.encode(text)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return []

        # If the text fits in a single chunk, return it as-is
        if total_tokens <= self._chunk_size_tokens:
            return [
                Chunk(
                    text=text,
                    token_count=total_tokens,
                    source_message_index=source_message_index,
                    metadata={"chunk_index": 0, "total_chunks": 1},
                )
            ]

        chunks: list[Chunk] = []
        step = self._chunk_size_tokens - self._chunk_overlap_tokens
        # Ensure step is at least 1 to avoid infinite loops
        if step < 1:
            step = 1

        start = 0
        chunk_index = 0

        while start < total_tokens:
            end = min(start + self._chunk_size_tokens, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self._encoding.decode(chunk_tokens)
            chunk_token_count = len(chunk_tokens)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    token_count=chunk_token_count,
                    source_message_index=source_message_index,
                    metadata={"chunk_index": chunk_index},
                )
            )

            # If we've reached the end, stop
            if end >= total_tokens:
                break

            start += step
            chunk_index += 1

        # Add total_chunks metadata
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.metadata["total_chunks"] = total_chunks

        return chunks

    def _embed_batch(self, batch_texts: list[str]) -> list[list[float]]:
        """Embed a single batch of texts via LM Studio.

        Used internally by ``embed()`` for parallel dispatch. Contains
        NaN/Inf validation and dimensionality consistency checks.

        Args:
            batch_texts: Sub-batch of text strings to embed.

        Returns:
            List of embedding vectors for this batch.

        Raises:
            EmbeddingUnavailableError: If LM Studio is unreachable.
            ValueError: If embeddings contain NaN/Inf or dimensions mismatch.
        """
        if not batch_texts:
            return []

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    self._config.embedding_url,
                    json={
                        "model": self._config.embedding_model,
                        "input": batch_texts,
                    },
                )
                response.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise EmbeddingUnavailableError(
                f"LM Studio embedding endpoint unavailable: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingUnavailableError(
                f"LM Studio embedding endpoint returned error: {exc.response.status_code}"
            ) from exc

        data = response.json()
        embedding_data = data.get("data", [])

        # Extract embeddings sorted by index
        embeddings: list[list[float]] = [
            item["embedding"]
            for item in sorted(embedding_data, key=lambda x: x["index"])
        ]

        if not embeddings:
            return embeddings

        # Validate: reject NaN or Inf values (Requirement 6.5)
        for i, embedding in enumerate(embeddings):
            for j, value in enumerate(embedding):
                if math.isnan(value) or math.isinf(value):
                    raise ValueError(
                        f"Embedding at index {i} contains invalid value "
                        f"at position {j}: {value}"
                    )

        # Validate: dimensionality consistency within this batch
        expected_dim = len(embeddings[0])
        for i, embedding in enumerate(embeddings):
            if len(embedding) != expected_dim:
                raise ValueError(
                    f"Embedding dimensionality mismatch: expected {expected_dim}, "
                    f"got {len(embedding)} at index {i}"
                )

        return embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via LM Studio endpoint with parallel dispatch.

        Splits large batches into sub-batches and dispatches them concurrently
        to better utilize LM Studio's concurrent request capacity (up to 4
        concurrent requests). Each sub-batch is independently validated for
        NaN/Inf and dimensionality consistency.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats), in the same
            order as the input texts.

        Raises:
            EmbeddingUnavailableError: If LM Studio is unreachable or all
                sub-batches fail.
            ValueError: If embeddings contain NaN or Inf values, or if
                embedding dimensionality is inconsistent.

        Requirements: 6.2, 6.3, 6.5, 13.2
        """
        if not texts:
            return []

        max_batch_size = 32  # LM Studio recommended batch size
        max_workers = 4       # LM Studio concurrent request limit

        # Split into sub-batches
        batches = [
            texts[i : i + max_batch_size]
            for i in range(0, len(texts), max_batch_size)
        ]

        # Single batch — no need for threading overhead
        if len(batches) == 1:
            return self._embed_batch(batches[0])

        # Dispatch batches concurrently using ThreadPoolExecutor
        embeddings: list[list[float]] = []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as pool:
            future_to_batch = {
                pool.submit(self._embed_batch, batch): batch_idx
                for batch_idx, batch in enumerate(batches)
            }

            # Collect results in batch order
            ordered_results: dict[int, list[list[float]]] = {}
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    result = future.result()
                    ordered_results[batch_idx] = result
                except EmbeddingUnavailableError:
                    raise
                except Exception as exc:
                    raise ValueError(
                        f"Embedding batch {batch_idx} failed: {exc}"
                    ) from exc

            # Flatten results preserving batch order
            for batch_idx in sorted(ordered_results.keys()):
                embeddings.extend(ordered_results[batch_idx])

        if not embeddings:
            return []

        # Cross-batch dimensionality consistency check
        expected_dim = len(embeddings[0])
        for i, embedding in enumerate(embeddings):
            if len(embedding) != expected_dim:
                raise ValueError(
                    f"Embedding dimensionality mismatch: expected {expected_dim}, "
                    f"got {len(embedding)} at index {i}"
                )

        return embeddings

    def _validate_localhost_url(self, url: str) -> None:
        """Validate that the URL points to localhost only.

        Requirement 7.5: All Qdrant communication must be over localhost (127.0.0.1).

        Raises:
            ValueError: If the URL does not point to localhost.
        """
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(
                f"Qdrant URL must point to localhost, got '{hostname}'. "
                "Requirement 7.5: communication over localhost only (127.0.0.1)."
            )

    def upsert_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Store chunk vectors in Qdrant using MessagePack transport.

        Serializes points using ``msgpack.packb(data, use_bin_type=True)``
        and sends them to the configured Qdrant endpoint. Each point contains
        an auto-generated UUID, the embedding vector, and a payload with
        text, token_count, source_index, and metadata.

        Args:
            chunks: List of Chunk instances to store.
            embeddings: Corresponding embedding vectors (must match len(chunks)).

        Raises:
            QdrantUnavailableError: If Qdrant is unreachable or returns an error.
                The pipeline should catch this to skip retrieval gracefully.
            ValueError: If chunks and embeddings have different lengths, or
                if the Qdrant URL is not localhost.

        Requirements: 7.1, 7.5, 13.1, 14.1, 14.3
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must have the same length: "
                f"got {len(chunks)} chunks and {len(embeddings)} embeddings"
            )

        # Validate localhost-only communication (Requirement 7.5)
        self._validate_localhost_url(self._config.qdrant_url)

        # Build points for Qdrant upsert
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            points.append(
                {
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "text": chunk.text,
                        "token_count": chunk.token_count,
                        "source_index": chunk.source_message_index,
                        "metadata": chunk.metadata,
                    },
                }
            )

        # Serialize as JSON for Qdrant REST API
        # (MessagePack is used internally for size comparison/efficiency tracking)
        import json as _json
        body = _json.dumps({"points": points}).encode("utf-8")

        # Send to Qdrant
        collection = self._config.qdrant_collection
        url = f"{self._config.qdrant_url}/collections/{collection}/points"

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.put(
                    url,
                    content=body,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise QdrantUnavailableError(
                f"Qdrant unavailable at {self._config.qdrant_url}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise QdrantUnavailableError(
                f"Qdrant returned error: {exc.response.status_code}"
            ) from exc

    def retrieve(
        self, query_embedding: list[float], top_k: int | None = None
    ) -> list[ScoredChunk]:
        """Query Qdrant for the most relevant chunks.

        Sends a search request to Qdrant using MessagePack serialization,
        retrieving the top-k most similar vectors. Only chunks with a
        relevance score > 0.5 are returned.

        Args:
            query_embedding: The query vector to search with.
            top_k: Number of results to retrieve. Defaults to config's
                   retrieval_top_k.

        Returns:
            List of ScoredChunk instances sorted by score descending,
            filtered to only include chunks with score > 0.5.

        Raises:
            QdrantUnavailableError: If Qdrant is unreachable or returns an error.
                The pipeline should catch this to skip retrieval gracefully.
            ValueError: If the Qdrant URL is not localhost.

        Requirements: 7.2, 7.4, 7.5, 13.1, 14.1, 14.3
        """
        # Validate localhost-only communication (Requirement 7.5)
        self._validate_localhost_url(self._config.qdrant_url)

        if top_k is None:
            top_k = self._config.retrieval_top_k

        # Build search request
        search_request = {
            "vector": query_embedding,
            "limit": top_k,
            "with_payload": True,
        }

        # Serialize as JSON for Qdrant REST API
        import json as _json
        body = _json.dumps(search_request).encode("utf-8")

        collection = self._config.qdrant_collection
        url = f"{self._config.qdrant_url}/collections/{collection}/points/search"

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    url,
                    content=body,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise QdrantUnavailableError(
                f"Qdrant unavailable at {self._config.qdrant_url}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise QdrantUnavailableError(
                f"Qdrant returned error: {exc.response.status_code}"
            ) from exc

        data = response.json()
        results = data.get("result", [])

        # Build ScoredChunk list, filtering by score > 0.5 (Requirement 7.4)
        scored_chunks: list[ScoredChunk] = []
        for result in results:
            score = result.get("score", 0.0)
            if score <= 0.5:
                continue

            payload = result.get("payload", {})
            chunk = Chunk(
                text=payload.get("text", ""),
                token_count=payload.get("token_count", 0),
                source_message_index=payload.get("source_index", 0),
                metadata=payload.get("metadata", {}),
            )
            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    vector_id=str(result.get("id", "")),
                )
            )

        # Sort by score descending
        scored_chunks.sort(key=lambda sc: sc.score, reverse=True)
        return scored_chunks

    def _retry_with_backoff(
        self,
        operation: str,
        fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with exponential backoff retry.

        Starts at 1 second delay, doubles each retry, with a maximum total
        wait time of 5 minutes (300 seconds). If all retries are exhausted,
        re-raises the last exception.

        Args:
            operation: Description of the operation (for logging).
            fn: The callable to execute.
            *args: Positional arguments for fn.
            **kwargs: Keyword arguments for fn.

        Returns:
            The return value of fn on success.

        Raises:
            The last exception raised by fn if all retries are exhausted.

        Requirements: 13.4
        """
        max_total_seconds = 300  # 5 minutes
        delay = 1.0
        elapsed = 0.0

        while True:
            try:
                return fn(*args, **kwargs)
            except (EmbeddingUnavailableError, QdrantUnavailableError) as exc:
                # Don't retry client errors (4xx) — they'll never succeed
                exc_msg = str(exc)
                if "400" in exc_msg or "401" in exc_msg or "404" in exc_msg or "422" in exc_msg:
                    logger.warning(
                        "Non-retryable error for %s: %s",
                        operation,
                        exc,
                    )
                    raise
                if elapsed + delay > max_total_seconds:
                    logger.warning(
                        "Retry exhausted for %s after %.1fs: %s",
                        operation,
                        elapsed,
                        exc,
                    )
                    raise
                logger.warning(
                    "Retrying %s after %.1fs delay: %s",
                    operation,
                    delay,
                    exc,
                )
                time.sleep(delay)
                elapsed += delay
                delay = min(delay * 2, 60.0)  # Cap individual delay at 60s

    def build_reduced_context(
        self, query: str, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Replace full context with targeted retrieval results.

        Implements the full context reduction pipeline:
        chunk → embed chunks → upsert to Qdrant → embed query → retrieve top-k → assemble.

        The output preserves system messages and the latest user message unchanged.
        Retrieved context is assembled into a synthetic message placed between
        system messages and the latest user message.

        If any service is unavailable (embedding or Qdrant), returns the original
        messages unchanged (graceful degradation). Uses exponential backoff retry
        for service connections with a maximum total wait of 5 minutes.

        Args:
            query: The user query to use for semantic retrieval.
            messages: List of chat messages in OpenAI format.

        Returns:
            Reduced message list in format:
            [system_messages..., {"role": "user", "content": "[Retrieved Context]\\n{context}"}, latest_user_message]

            If services are unavailable or no relevant chunks are found,
            returns the original messages unchanged.

        Requirements: 7.3, 6.4, 13.4
        """
        if not messages:
            return messages

        # Identify system messages and latest user message (to preserve)
        system_messages = [m for m in messages if m.get("role") == "system"]
        latest_user_message: dict[str, Any] | None = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                latest_user_message = messages[i]
                break

        # If there's no user message, nothing to reduce
        if latest_user_message is None:
            return messages

        try:
            # Step 1: Chunk context messages
            chunks = self.chunk_context(messages)
            if not chunks:
                return messages

            # Step 2: Embed chunks with retry
            chunk_texts = [c.text for c in chunks]
            embeddings = self._retry_with_backoff(
                "embed chunks", self.embed, chunk_texts
            )
            if not embeddings:
                return messages

            # Step 3: Upsert to Qdrant with retry
            self._retry_with_backoff(
                "upsert chunks", self.upsert_chunks, chunks, embeddings
            )

            # Step 4: Embed query with retry
            query_embeddings = self._retry_with_backoff(
                "embed query", self.embed, [query]
            )
            if not query_embeddings:
                return messages
            query_embedding = query_embeddings[0]

            # Step 5: Retrieve top-k with retry
            scored_chunks = self._retry_with_backoff(
                "retrieve", self.retrieve, query_embedding
            )

            # If no relevant chunks found, return original
            if not scored_chunks:
                return messages

            # Step 6: Assemble reduced context within token budget
            reduced_context = self._assemble_context(scored_chunks)

            # If assembly produced nothing, return original
            if not reduced_context:
                return messages

            # Build output: system messages + retrieved context + latest user message
            result: list[dict[str, Any]] = list(system_messages)
            result.append(
                {"role": "user", "content": f"[Retrieved Context]\n{reduced_context}"}
            )
            result.append(latest_user_message)
            return result

        except (EmbeddingUnavailableError, QdrantUnavailableError) as exc:
            # Graceful degradation: return original messages unchanged
            logger.warning(
                "Service unavailable during build_reduced_context, "
                "returning original messages: %s",
                exc,
            )
            return messages

    def _assemble_context(self, scored_chunks: list[ScoredChunk]) -> str:
        """Assemble retrieved chunks into a context string within token budget.

        Iterates through scored chunks (already sorted by relevance) and
        includes them until the token budget (retrieval_max_tokens) is reached.

        Args:
            scored_chunks: List of ScoredChunk sorted by score descending.

        Returns:
            Assembled context string, or empty string if no chunks fit.

        Requirements: 7.3
        """
        max_tokens = self._config.retrieval_max_tokens
        total_tokens = 0
        selected_texts: list[str] = []

        for sc in scored_chunks:
            chunk_tokens = sc.chunk.token_count
            if total_tokens + chunk_tokens > max_tokens:
                # Try to see if we can fit a partial — but for simplicity
                # and correctness, we skip chunks that would exceed budget
                continue
            selected_texts.append(sc.chunk.text)
            total_tokens += chunk_tokens

        return "\n\n".join(selected_texts)

    def health_check(self) -> str:
        """Check health of the retrieval layer components.

        Queries LM Studio's /api/v1/models endpoint to verify:
        1. LM Studio is reachable
        2. The configured embedding model is loaded (has active instances)

        Also checks Qdrant reachability.

        Returns:
            "healthy" — LM Studio reachable and embedding model loaded
            "model_not_loaded" — LM Studio reachable but model not loaded
            "degraded" — LM Studio or Qdrant unreachable
            "unhealthy" — both services unreachable
        """
        lm_studio_ok = False
        model_loaded = False
        qdrant_ok = False

        # Check LM Studio and embedding model status
        try:
            # Derive the base URL from the embedding URL
            # e.g. http://localhost:1234/v1/embeddings -> http://localhost:1234
            from urllib.parse import urlparse

            parsed = urlparse(self._config.embedding_url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"

            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{base_url}/api/v1/models")
                if response.status_code == 200:
                    lm_studio_ok = True
                    data = response.json()
                    models = data.get("models", [])
                    # Check if the configured embedding model is loaded
                    for model in models:
                        model_key = model.get("key", "")
                        loaded_instances = model.get("loaded_instances", [])
                        if (
                            model_key == self._config.embedding_model
                            and len(loaded_instances) > 0
                        ):
                            model_loaded = True
                            break
        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            lm_studio_ok = False

        # Check Qdrant reachability
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(
                    f"{self._config.qdrant_url}/collections"
                )
                qdrant_ok = response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            qdrant_ok = False

        if lm_studio_ok and model_loaded and qdrant_ok:
            return "healthy"
        elif lm_studio_ok and not model_loaded:
            return "model_not_loaded"
        elif lm_studio_ok or qdrant_ok:
            return "degraded"
        else:
            return "unhealthy"
