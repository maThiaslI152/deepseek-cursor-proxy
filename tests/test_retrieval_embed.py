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
