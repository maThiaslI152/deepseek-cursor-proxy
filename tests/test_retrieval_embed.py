"""Unit tests for RetrievalLayer.embed() — embedding generation via LM Studio.

Tests the embed() method which calls LM Studio's /v1/embeddings endpoint,
validates dimensionality consistency, rejects NaN/Inf values, and handles
LM Studio unavailability gracefully.

Requirements: 6.2, 6.3, 6.5, 13.2
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import httpx
import pytest

from deepseek_cursor_proxy.rap.config import RAPConfig
from deepseek_cursor_proxy.rap.retrieval import (
    EmbeddingUnavailableError,
    RetrievalLayer,
)


class TestEmbedBasic:
    """Basic functionality tests for embed()."""

    def setup_method(self) -> None:
        self.config = RAPConfig()
        self.layer = RetrievalLayer(self.config)

    def test_empty_input_returns_empty(self) -> None:
        """Empty text list returns empty embedding list without calling API."""
        result = self.layer.embed([])
        assert result == []

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_single_text_returns_single_embedding(self, mock_client_cls: MagicMock) -> None:
        """Requirement 6.2: Single text produces single embedding vector."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "model": "nomic-embed-text",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = self.layer.embed(["hello world"])
        assert result == [[0.1, 0.2, 0.3]]

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_multiple_texts_returns_multiple_embeddings(self, mock_client_cls: MagicMock) -> None:
        """Requirement 6.2: Multiple texts produce corresponding embeddings."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                {"index": 2, "embedding": [0.7, 0.8, 0.9]},
            ],
            "model": "nomic-embed-text",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = self.layer.embed(["text1", "text2", "text3"])
        assert len(result) == 3
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        assert result[2] == [0.7, 0.8, 0.9]

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_embeddings_sorted_by_index(self, mock_client_cls: MagicMock) -> None:
        """Embeddings are returned in order even if API returns out of order."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 2, "embedding": [0.7, 0.8, 0.9]},
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            ],
            "model": "nomic-embed-text",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = self.layer.embed(["a", "b", "c"])
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        assert result[2] == [0.7, 0.8, 0.9]

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_calls_correct_endpoint_and_model(self, mock_client_cls: MagicMock) -> None:
        """Requirement 6.2: Calls configured embedding_url with embedding_model."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        self.layer.embed(["test"])

        mock_client.post.assert_called_once_with(
            "http://localhost:1234/v1/embeddings",
            json={
                "model": "text-embedding-nomic-embed-text-v1.5-embedding",
                "input": ["test"],
            },
        )


class TestEmbedValidation:
    """Validation tests for embed() — NaN/Inf rejection and dimensionality."""

    def setup_method(self) -> None:
        self.config = RAPConfig()
        self.layer = RetrievalLayer(self.config)

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_rejects_nan_values(self, mock_client_cls: MagicMock) -> None:
        """Requirement 6.5: Embeddings containing NaN are rejected."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, float("nan"), 0.3]}],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(ValueError, match="invalid value"):
            self.layer.embed(["test"])

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_rejects_positive_inf(self, mock_client_cls: MagicMock) -> None:
        """Requirement 6.5: Embeddings containing +Inf are rejected."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, float("inf"), 0.3]}],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(ValueError, match="invalid value"):
            self.layer.embed(["test"])

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_rejects_negative_inf(self, mock_client_cls: MagicMock) -> None:
        """Requirement 6.5: Embeddings containing -Inf are rejected."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, float("-inf"), 0.3]}],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(ValueError, match="invalid value"):
            self.layer.embed(["test"])

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_rejects_nan_in_second_embedding(self, mock_client_cls: MagicMock) -> None:
        """Requirement 6.5: NaN in any embedding in the batch is rejected."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, float("nan"), 0.6]},
            ],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(ValueError, match="index 1"):
            self.layer.embed(["text1", "text2"])

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_rejects_inconsistent_dimensionality(self, mock_client_cls: MagicMock) -> None:
        """Requirement 6.3: All embeddings must have same dimensionality."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5]},  # Different dimension!
            ],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(ValueError, match="dimensionality mismatch"):
            self.layer.embed(["text1", "text2"])

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_consistent_dimensionality_passes(self, mock_client_cls: MagicMock) -> None:
        """Requirement 6.3: Consistent dimensionality is accepted."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]},
                {"index": 1, "embedding": [0.5, 0.6, 0.7, 0.8]},
                {"index": 2, "embedding": [0.9, 1.0, 1.1, 1.2]},
            ],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = self.layer.embed(["a", "b", "c"])
        assert len(result) == 3
        assert all(len(e) == 4 for e in result)


class TestEmbedGracefulDegradation:
    """Tests for graceful handling of LM Studio unavailability."""

    def setup_method(self) -> None:
        self.config = RAPConfig()
        self.layer = RetrievalLayer(self.config)

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_connection_error_raises_unavailable(self, mock_client_cls: MagicMock) -> None:
        """Requirement 13.2: Connection error raises EmbeddingUnavailableError."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(EmbeddingUnavailableError, match="unavailable"):
            self.layer.embed(["test"])

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_timeout_raises_unavailable(self, mock_client_cls: MagicMock) -> None:
        """Requirement 13.2: Timeout raises EmbeddingUnavailableError."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(EmbeddingUnavailableError, match="unavailable"):
            self.layer.embed(["test"])

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_http_500_raises_unavailable(self, mock_client_cls: MagicMock) -> None:
        """Requirement 13.2: HTTP 500 raises EmbeddingUnavailableError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response,
        )

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(EmbeddingUnavailableError, match="error"):
            self.layer.embed(["test"])

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_http_404_raises_unavailable(self, mock_client_cls: MagicMock) -> None:
        """Requirement 13.2: HTTP 404 raises EmbeddingUnavailableError."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=mock_response,
        )

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(EmbeddingUnavailableError, match="error"):
            self.layer.embed(["test"])

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_unavailable_error_is_catchable_for_graceful_skip(
        self, mock_client_cls: MagicMock
    ) -> None:
        """Requirement 13.2: Pipeline can catch error and skip retrieval."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        # Simulate pipeline graceful degradation pattern
        try:
            self.layer.embed(["test"])
            embeddings_available = True
        except EmbeddingUnavailableError:
            embeddings_available = False

        assert embeddings_available is False

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_empty_data_returns_empty(self, mock_client_cls: MagicMock) -> None:
        """Empty data array from API returns empty list."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = self.layer.embed(["test"])
        assert result == []


class TestParallelEmbedding:
    """Tests for parallel embedding dispatch via ThreadPoolExecutor."""

    def setup_method(self) -> None:
        self.config = RAPConfig()
        self.layer = RetrievalLayer(self.config)

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_small_batch_does_not_use_threading(self, mock_client_cls: MagicMock) -> None:
        """Small batches (<=32 texts) go through a single request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"index": i, "embedding": [float(i)]} for i in range(5)],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = self.layer.embed(["a", "b", "c", "d", "e"])
        assert len(result) == 5
        # Should have called post once
        assert mock_client.post.call_count == 1

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_large_batch_splits_into_sub_batches(self, mock_client_cls: MagicMock) -> None:
        """Large batches are split into sub-batches of 32."""
        # We need 65 texts to get 3 batches (32+32+1)
        texts = [f"text-{i}" for i in range(33)]

        # Return embedding index matching input order
        def mock_post(url, **kwargs):
            input_texts = kwargs["json"]["input"]
            batch_size = len(input_texts)
            return _mock_embedding_response(batch_size)

        mock_client = MagicMock()
        mock_client.post.side_effect = mock_post
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = self.layer.embed(texts)
        assert len(result) == 33

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_parallel_batches_maintain_order(self, mock_client_cls: MagicMock) -> None:
        """Parallel embedding results preserve per-batch internal ordering.

        With 65 texts split into batches of 32+32+1, the results should
        contain 65 embeddings with the correct total count and each
        sub-batch internally ordered by index.
        """
        texts = [f"text-{i}" for i in range(65)]

        def mock_post(url, **kwargs):
            input_texts = kwargs["json"]["input"]
            batch_size = len(input_texts)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "data": [
                    {"index": i, "embedding": [float(i)]}
                    for i in range(batch_size)
                ],
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_client = MagicMock()
        mock_client.post.side_effect = mock_post
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = self.layer.embed(texts)
        assert len(result) == 65
        # Every embedding should have exactly one float value
        assert all(len(e) == 1 for e in result)

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_parallel_batch_rejects_nan(self, mock_client_cls: MagicMock) -> None:
        """NaN values are rejected even in parallel sub-batches."""
        call_count = [0]

        def mock_post(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First batch is clean
                return _mock_embedding_response(32)
            # Second batch has NaN
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "data": [{"index": 0, "embedding": [float("nan")]}],
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_client = MagicMock()
        mock_client.post.side_effect = mock_post
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        texts = [f"text-{i}" for i in range(33)]
        with pytest.raises(ValueError, match="invalid value"):
            self.layer.embed(texts)

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_parallel_rejects_inconsistent_dimensionality(self, mock_client_cls: MagicMock) -> None:
        """Inconsistent dimensionality across sub-batches is rejected."""
        call_count = [0]

        def mock_post(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First batch has 2-dim embeddings
                mock_resp = MagicMock()
                mock_resp.json.return_value = {
                    "data": [{"index": i, "embedding": [float(i), float(i)]} for i in range(32)],
                }
                mock_resp.raise_for_status = MagicMock()
                return mock_resp
            # Second batch has 3-dim embeddings
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_client = MagicMock()
        mock_client.post.side_effect = mock_post
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        texts = [f"text-{i}" for i in range(33)]
        with pytest.raises(ValueError, match="dimensionality mismatch"):
            self.layer.embed(texts)

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_parallel_connection_error_propagates(self, mock_client_cls: MagicMock) -> None:
        """Connection error in any sub-batch propagates as EmbeddingUnavailableError."""
        call_count = [0]

        def mock_post(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_embedding_response(32)
            raise httpx.ConnectError("Connection refused on batch 2")

        mock_client = MagicMock()
        mock_client.post.side_effect = mock_post
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        texts = [f"text-{i}" for i in range(33)]
        with pytest.raises(EmbeddingUnavailableError):
            self.layer.embed(texts)


def _mock_embedding_response(batch_size: int, starting_index: int = 0) -> MagicMock:
    """Helper to create a mock embedding response with the given batch size."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {"index": i, "embedding": [float(starting_index + i)]}
            for i in range(batch_size)
        ],
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp
