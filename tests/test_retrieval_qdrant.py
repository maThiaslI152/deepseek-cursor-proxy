"""Unit tests for RetrievalLayer Qdrant vector storage — upsert_chunks() and retrieve().

Tests the upsert_chunks() and retrieve() methods which communicate with Qdrant
using MessagePack serialization, validate localhost-only communication, and
handle Qdrant unavailability gracefully.

Requirements: 7.1, 7.2, 7.4, 7.5, 13.1, 14.1, 14.3
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import json
import pytest

from deepseek_cursor_proxy.rap.config import RAPConfig
from deepseek_cursor_proxy.rap.retrieval import (
    Chunk,
    QdrantUnavailableError,
    RetrievalLayer,
    ScoredChunk,
)


class TestUpsertChunks:
    """Tests for RetrievalLayer.upsert_chunks()."""

    def setup_method(self) -> None:
        self.config = RAPConfig()
        self.layer = RetrievalLayer(self.config)

    def test_empty_chunks_does_nothing(self) -> None:
        """Empty chunk list returns without calling Qdrant."""
        # Should not raise or make any HTTP calls
        self.layer.upsert_chunks([], [])

    def test_mismatched_lengths_raises_value_error(self) -> None:
        """Chunks and embeddings must have the same length."""
        chunks = [Chunk(text="hello", token_count=1, source_message_index=0)]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # 2 embeddings for 1 chunk

        with pytest.raises(ValueError, match="same length"):
            self.layer.upsert_chunks(chunks, embeddings)

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_upsert_sends_json_body(self, mock_client_cls: MagicMock) -> None:
        """Requirement 14.1: Payloads are serialized as JSON for Qdrant REST API."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.put.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        chunks = [
            Chunk(text="hello world", token_count=2, source_message_index=1, metadata={"chunk_index": 0}),
        ]
        embeddings = [[0.1, 0.2, 0.3]]

        self.layer.upsert_chunks(chunks, embeddings)

        # Verify the call was made
        mock_client.put.assert_called_once()
        call_kwargs = mock_client.put.call_args

        # Verify URL contains collection name
        url = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("url", "")
        assert "rap_context" in url
        assert "/points" in url

        # Verify content-type header
        headers = call_kwargs[1].get("headers", {})
        assert headers.get("Content-Type") == "application/json"

        # Verify body is valid JSON
        body = call_kwargs[1].get("content", b"")
        decoded = json.loads(body)
        assert "points" in decoded
        assert len(decoded["points"]) == 1

        point = decoded["points"][0]
        assert "id" in point
        assert point["vector"] == [0.1, 0.2, 0.3]
        assert point["payload"]["text"] == "hello world"
        assert point["payload"]["token_count"] == 2
        assert point["payload"]["source_index"] == 1
        assert point["payload"]["metadata"] == {"chunk_index": 0}

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_upsert_multiple_chunks(self, mock_client_cls: MagicMock) -> None:
        """Multiple chunks are all included in the upsert payload."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.put.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        chunks = [
            Chunk(text="chunk one", token_count=2, source_message_index=0),
            Chunk(text="chunk two", token_count=3, source_message_index=1),
            Chunk(text="chunk three", token_count=2, source_message_index=2),
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        self.layer.upsert_chunks(chunks, embeddings)

        call_kwargs = mock_client.put.call_args
        body = call_kwargs[1].get("content", b"")
        decoded = json.loads(body)
        assert len(decoded["points"]) == 3

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_upsert_generates_unique_ids(self, mock_client_cls: MagicMock) -> None:
        """Each point gets a unique UUID."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.put.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        chunks = [
            Chunk(text="a", token_count=1, source_message_index=0),
            Chunk(text="b", token_count=1, source_message_index=0),
        ]
        embeddings = [[0.1], [0.2]]

        self.layer.upsert_chunks(chunks, embeddings)

        call_kwargs = mock_client.put.call_args
        body = call_kwargs[1].get("content", b"")
        decoded = json.loads(body)
        ids = [p["id"] for p in decoded["points"]]
        assert len(set(ids)) == 2  # All unique

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_upsert_connection_error_raises_qdrant_unavailable(
        self, mock_client_cls: MagicMock
    ) -> None:
        """Requirement 13.1: Connection error raises QdrantUnavailableError."""
        mock_client = MagicMock()
        mock_client.put.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        chunks = [Chunk(text="test", token_count=1, source_message_index=0)]
        embeddings = [[0.1, 0.2]]

        with pytest.raises(QdrantUnavailableError, match="unavailable"):
            self.layer.upsert_chunks(chunks, embeddings)

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_upsert_timeout_raises_qdrant_unavailable(
        self, mock_client_cls: MagicMock
    ) -> None:
        """Requirement 13.1: Timeout raises QdrantUnavailableError."""
        mock_client = MagicMock()
        mock_client.put.side_effect = httpx.TimeoutException("Request timed out")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        chunks = [Chunk(text="test", token_count=1, source_message_index=0)]
        embeddings = [[0.1, 0.2]]

        with pytest.raises(QdrantUnavailableError, match="unavailable"):
            self.layer.upsert_chunks(chunks, embeddings)

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_upsert_http_error_raises_qdrant_unavailable(
        self, mock_client_cls: MagicMock
    ) -> None:
        """Requirement 13.1: HTTP error raises QdrantUnavailableError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.put.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        chunks = [Chunk(text="test", token_count=1, source_message_index=0)]
        embeddings = [[0.1, 0.2]]

        with pytest.raises(QdrantUnavailableError, match="error"):
            self.layer.upsert_chunks(chunks, embeddings)

    def test_upsert_non_localhost_url_raises_value_error(self) -> None:
        """Requirement 7.5: Non-localhost URL raises ValueError."""
        config = RAPConfig(qdrant_url="http://remote-server.com:6333")
        layer = RetrievalLayer(config)

        chunks = [Chunk(text="test", token_count=1, source_message_index=0)]
        embeddings = [[0.1, 0.2]]

        with pytest.raises(ValueError, match="localhost"):
            layer.upsert_chunks(chunks, embeddings)

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_upsert_localhost_127_0_0_1_accepted(self, mock_client_cls: MagicMock) -> None:
        """Requirement 7.5: 127.0.0.1 is accepted as localhost."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.put.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        config = RAPConfig(qdrant_url="http://127.0.0.1:6333")
        layer = RetrievalLayer(config)

        chunks = [Chunk(text="test", token_count=1, source_message_index=0)]
        embeddings = [[0.1, 0.2]]

        # Should not raise
        layer.upsert_chunks(chunks, embeddings)


class TestRetrieve:
    """Tests for RetrievalLayer.retrieve()."""

    def setup_method(self) -> None:
        self.config = RAPConfig()
        self.layer = RetrievalLayer(self.config)

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_retrieve_sends_json_request(self, mock_client_cls: MagicMock) -> None:
        """Requirement 14.1: Search request is serialized as JSON for Qdrant REST API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        self.layer.retrieve([0.1, 0.2, 0.3])

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args

        # Verify URL
        url = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("url", "")
        assert "rap_context" in url
        assert "/points/search" in url

        # Verify content-type
        headers = call_kwargs[1].get("headers", {})
        assert headers.get("Content-Type") == "application/json"

        # Verify body is valid JSON
        body = call_kwargs[1].get("content", b"")
        decoded = json.loads(body)
        assert decoded["vector"] == [0.1, 0.2, 0.3]
        assert decoded["limit"] == 5  # default top_k
        assert decoded["with_payload"] is True

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_retrieve_uses_custom_top_k(self, mock_client_cls: MagicMock) -> None:
        """Custom top_k overrides config default."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        self.layer.retrieve([0.1, 0.2], top_k=10)

        call_kwargs = mock_client.post.call_args
        body = call_kwargs[1].get("content", b"")
        decoded = json.loads(body)
        assert decoded["limit"] == 10

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_retrieve_filters_by_score_threshold(self, mock_client_cls: MagicMock) -> None:
        """Requirement 7.4: Only chunks with score > 0.5 are returned."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": [
                {
                    "id": "id-1",
                    "score": 0.9,
                    "payload": {"text": "high score", "token_count": 2, "source_index": 0, "metadata": {}},
                },
                {
                    "id": "id-2",
                    "score": 0.5,  # Exactly 0.5 — should be excluded (> 0.5 required)
                    "payload": {"text": "boundary", "token_count": 1, "source_index": 1, "metadata": {}},
                },
                {
                    "id": "id-3",
                    "score": 0.3,
                    "payload": {"text": "low score", "token_count": 1, "source_index": 2, "metadata": {}},
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        results = self.layer.retrieve([0.1, 0.2])

        # Only the chunk with score 0.9 should be returned
        assert len(results) == 1
        assert results[0].score == 0.9
        assert results[0].chunk.text == "high score"

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_retrieve_returns_scored_chunks_sorted_by_score(
        self, mock_client_cls: MagicMock
    ) -> None:
        """Results are sorted by score descending."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": [
                {
                    "id": "id-1",
                    "score": 0.7,
                    "payload": {"text": "medium", "token_count": 1, "source_index": 0, "metadata": {}},
                },
                {
                    "id": "id-2",
                    "score": 0.95,
                    "payload": {"text": "highest", "token_count": 1, "source_index": 1, "metadata": {}},
                },
                {
                    "id": "id-3",
                    "score": 0.8,
                    "payload": {"text": "high", "token_count": 1, "source_index": 2, "metadata": {}},
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        results = self.layer.retrieve([0.1, 0.2])

        assert len(results) == 3
        assert results[0].score == 0.95
        assert results[1].score == 0.8
        assert results[2].score == 0.7

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_retrieve_builds_correct_scored_chunks(self, mock_client_cls: MagicMock) -> None:
        """ScoredChunk objects are correctly constructed from Qdrant response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": [
                {
                    "id": "abc-123",
                    "score": 0.85,
                    "payload": {
                        "text": "relevant code snippet",
                        "token_count": 5,
                        "source_index": 3,
                        "metadata": {"chunk_index": 2, "total_chunks": 4},
                    },
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        results = self.layer.retrieve([0.1, 0.2])

        assert len(results) == 1
        sc = results[0]
        assert sc.score == 0.85
        assert sc.vector_id == "abc-123"
        assert sc.chunk.text == "relevant code snippet"
        assert sc.chunk.token_count == 5
        assert sc.chunk.source_message_index == 3
        assert sc.chunk.metadata == {"chunk_index": 2, "total_chunks": 4}

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_retrieve_empty_results(self, mock_client_cls: MagicMock) -> None:
        """Empty result from Qdrant returns empty list."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        results = self.layer.retrieve([0.1, 0.2])
        assert results == []

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_retrieve_connection_error_raises_qdrant_unavailable(
        self, mock_client_cls: MagicMock
    ) -> None:
        """Requirement 13.1: Connection error raises QdrantUnavailableError."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(QdrantUnavailableError, match="unavailable"):
            self.layer.retrieve([0.1, 0.2])

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_retrieve_timeout_raises_qdrant_unavailable(
        self, mock_client_cls: MagicMock
    ) -> None:
        """Requirement 13.1: Timeout raises QdrantUnavailableError."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("Timed out")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(QdrantUnavailableError, match="unavailable"):
            self.layer.retrieve([0.1, 0.2])

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_retrieve_http_error_raises_qdrant_unavailable(
        self, mock_client_cls: MagicMock
    ) -> None:
        """Requirement 13.1: HTTP error raises QdrantUnavailableError."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service Unavailable", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(QdrantUnavailableError, match="error"):
            self.layer.retrieve([0.1, 0.2])

    def test_retrieve_non_localhost_url_raises_value_error(self) -> None:
        """Requirement 7.5: Non-localhost URL raises ValueError."""
        config = RAPConfig(qdrant_url="http://external-qdrant.io:6333")
        layer = RetrievalLayer(config)

        with pytest.raises(ValueError, match="localhost"):
            layer.retrieve([0.1, 0.2])

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_retrieve_graceful_degradation_pattern(self, mock_client_cls: MagicMock) -> None:
        """Requirement 13.1: Pipeline can catch QdrantUnavailableError to skip retrieval."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        # Simulate pipeline graceful degradation
        try:
            self.layer.retrieve([0.1, 0.2])
            retrieval_available = True
        except QdrantUnavailableError:
            retrieval_available = False

        assert retrieval_available is False

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_retrieve_all_below_threshold_returns_empty(
        self, mock_client_cls: MagicMock
    ) -> None:
        """When all results have score <= 0.5, returns empty list."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": [
                {
                    "id": "id-1",
                    "score": 0.4,
                    "payload": {"text": "low", "token_count": 1, "source_index": 0, "metadata": {}},
                },
                {
                    "id": "id-2",
                    "score": 0.2,
                    "payload": {"text": "very low", "token_count": 1, "source_index": 1, "metadata": {}},
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        results = self.layer.retrieve([0.1, 0.2])
        assert results == []
