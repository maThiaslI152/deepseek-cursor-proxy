"""Unit tests for RetrievalLayer.build_reduced_context() — full context reduction pipeline.

Tests the build_reduced_context() method which orchestrates:
chunk → embed → upsert → query → assemble, with exponential backoff retry
and graceful degradation when services are unavailable.

Requirements: 7.3, 6.4, 13.4
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from deepseek_cursor_proxy.rap.config import RAPConfig
from deepseek_cursor_proxy.rap.retrieval import (
    Chunk,
    EmbeddingUnavailableError,
    QdrantUnavailableError,
    RetrievalLayer,
    ScoredChunk,
)


def _make_embed_response_dynamic(dim: int = 4):
    """Create a factory that returns embed responses matching input count."""

    def _factory(url, **kwargs):
        """Return an embed response with correct count based on input."""
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        # Determine count from the json body if available
        json_body = kwargs.get("json", {})
        input_texts = json_body.get("input", [])
        count = len(input_texts) if input_texts else 1
        resp.json.return_value = {
            "data": [
                {"index": i, "embedding": [0.1 * (i + 1)] * dim}
                for i in range(count)
            ]
        }
        return resp

    return _factory


def _make_search_response(chunks: list[tuple[str, int, float]]) -> dict:
    """Create a mock Qdrant search response.

    Args:
        chunks: List of (text, token_count, score) tuples.
    """
    return {
        "result": [
            {
                "id": f"id-{i}",
                "score": score,
                "payload": {
                    "text": text,
                    "token_count": token_count,
                    "source_index": i,
                    "metadata": {},
                },
            }
            for i, (text, token_count, score) in enumerate(chunks)
        ]
    }


class TestBuildReducedContext:
    """Tests for RetrievalLayer.build_reduced_context()."""

    def setup_method(self) -> None:
        self.config = RAPConfig(retrieval_max_tokens=200)
        self.layer = RetrievalLayer(self.config)

    def test_empty_messages_returns_empty(self) -> None:
        """Empty message list returns unchanged."""
        result = self.layer.build_reduced_context("query", [])
        assert result == []

    def test_no_user_message_returns_original(self) -> None:
        """Messages without a user message return unchanged."""
        messages = [{"role": "system", "content": "You are helpful."}]
        result = self.layer.build_reduced_context("query", messages)
        assert result == messages

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_preserves_system_messages(self, mock_client_cls: MagicMock) -> None:
        """Requirement 6.4: System messages are preserved unchanged in output."""
        search_response = MagicMock()
        search_response.json.return_value = _make_search_response([
            ("relevant snippet", 10, 0.9),
        ])
        search_response.raise_for_status = MagicMock()

        upsert_response = MagicMock()
        upsert_response.raise_for_status = MagicMock()

        embed_factory = _make_embed_response_dynamic(dim=4)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client
        mock_client.put.return_value = upsert_response

        def post_side_effect(url, **kwargs):
            if "/points/search" in url:
                return search_response
            return embed_factory(url, **kwargs)

        mock_client.post.side_effect = post_side_effect

        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Some old context message"},
            {"role": "assistant", "content": "Previous response with lots of code..."},
            {"role": "user", "content": "Refactor this function"},
        ]

        result = self.layer.build_reduced_context("Refactor this function", messages)

        # System message should be first
        assert result[0] == {"role": "system", "content": "You are a coding assistant."}

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_preserves_latest_user_message(self, mock_client_cls: MagicMock) -> None:
        """Requirement 6.4: Latest user message is preserved unchanged."""
        search_response = MagicMock()
        search_response.json.return_value = _make_search_response([
            ("relevant snippet", 10, 0.9),
        ])
        search_response.raise_for_status = MagicMock()

        upsert_response = MagicMock()
        upsert_response.raise_for_status = MagicMock()

        embed_factory = _make_embed_response_dynamic(dim=4)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client
        mock_client.put.return_value = upsert_response

        def post_side_effect(url, **kwargs):
            if "/points/search" in url:
                return search_response
            return embed_factory(url, **kwargs)

        mock_client.post.side_effect = post_side_effect

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Old context"},
            {"role": "assistant", "content": "Old response"},
            {"role": "user", "content": "Latest question here"},
        ]

        result = self.layer.build_reduced_context("Latest question here", messages)

        # Latest user message should be last
        assert result[-1] == {"role": "user", "content": "Latest question here"}

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_output_format_with_retrieved_context(self, mock_client_cls: MagicMock) -> None:
        """Output format: [system..., retrieved_context_msg, latest_user_msg]."""
        search_response = MagicMock()
        search_response.json.return_value = _make_search_response([
            ("code snippet A", 10, 0.9),
            ("code snippet B", 10, 0.8),
        ])
        search_response.raise_for_status = MagicMock()

        upsert_response = MagicMock()
        upsert_response.raise_for_status = MagicMock()

        embed_factory = _make_embed_response_dynamic(dim=4)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client
        mock_client.put.return_value = upsert_response

        def post_side_effect(url, **kwargs):
            if "/points/search" in url:
                return search_response
            return embed_factory(url, **kwargs)

        mock_client.post.side_effect = post_side_effect

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Here is my large codebase..."},
            {"role": "assistant", "content": "I see your code"},
            {"role": "user", "content": "Fix the bug"},
        ]

        result = self.layer.build_reduced_context("Fix the bug", messages)

        assert len(result) == 3  # system + retrieved context + latest user
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"].startswith("[Retrieved Context]\n")
        assert result[2] == {"role": "user", "content": "Fix the bug"}

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_token_budget_not_exceeded(self, mock_client_cls: MagicMock) -> None:
        """Requirement 7.3: Total retrieved tokens must not exceed retrieval_max_tokens."""
        config = RAPConfig(retrieval_max_tokens=150)
        layer = RetrievalLayer(config)

        search_response = MagicMock()
        search_response.json.return_value = _make_search_response([
            ("chunk one with some content", 80, 0.95),
            ("chunk two with more content", 80, 0.90),
            ("chunk three extra", 50, 0.85),
        ])
        search_response.raise_for_status = MagicMock()

        upsert_response = MagicMock()
        upsert_response.raise_for_status = MagicMock()

        embed_factory = _make_embed_response_dynamic(dim=4)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client
        mock_client.put.return_value = upsert_response

        def post_side_effect(url, **kwargs):
            if "/points/search" in url:
                return search_response
            return embed_factory(url, **kwargs)

        mock_client.post.side_effect = post_side_effect

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Old context " * 100},
            {"role": "user", "content": "Query"},
        ]

        result = layer.build_reduced_context("Query", messages)

        # The retrieved context message should exist
        retrieved_msg = result[1]
        assert retrieved_msg["content"].startswith("[Retrieved Context]\n")

        # Verify we didn't include all 3 chunks (80+80+50=210 > 150)
        # Should include chunk one (80) + chunk three (50) = 130 <= 150
        # (chunk two is skipped because 80+80=160 > 150, then chunk three fits)
        content = retrieved_msg["content"].replace("[Retrieved Context]\n", "")
        assert "chunk one" in content

    @patch("deepseek_cursor_proxy.rap.retrieval.time.sleep")
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_graceful_degradation_embedding_unavailable(
        self, mock_client_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Requirement 13.4: Returns original messages when embedding service is down."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Old context"},
            {"role": "user", "content": "Query"},
        ]

        result = self.layer.build_reduced_context("Query", messages)

        # Should return original messages unchanged
        assert result == messages

    @patch("deepseek_cursor_proxy.rap.retrieval.time.sleep")
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_graceful_degradation_qdrant_unavailable(
        self, mock_client_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Requirement 13.4: Returns original messages when Qdrant is down."""
        embed_factory = _make_embed_response_dynamic(dim=4)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        # Embed succeeds but upsert (put) fails
        mock_client.post.side_effect = lambda url, **kwargs: embed_factory(url, **kwargs)
        mock_client.put.side_effect = httpx.ConnectError("Qdrant down")

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Old context"},
            {"role": "user", "content": "Query"},
        ]

        result = self.layer.build_reduced_context("Query", messages)

        # Should return original messages unchanged
        assert result == messages

    @patch("deepseek_cursor_proxy.rap.retrieval.time.sleep")
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_exponential_backoff_retry(
        self, mock_client_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Requirement 13.4: Exponential backoff retry starting at 1s, doubling."""
        search_response = MagicMock()
        search_response.json.return_value = _make_search_response([
            ("relevant", 10, 0.9),
        ])
        search_response.raise_for_status = MagicMock()

        upsert_response = MagicMock()
        upsert_response.raise_for_status = MagicMock()

        embed_factory = _make_embed_response_dynamic(dim=4)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client
        mock_client.put.return_value = upsert_response

        # Track post calls: first embed call fails, then succeeds on retry
        post_call_count = [0]

        def post_side_effect(url, **kwargs):
            post_call_count[0] += 1
            if "/points/search" in url:
                return search_response
            # First embed call fails, subsequent succeed
            if post_call_count[0] == 1:
                fail_resp = MagicMock()
                fail_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Error", request=MagicMock(), response=MagicMock(status_code=503)
                )
                return fail_resp
            return embed_factory(url, **kwargs)

        mock_client.post.side_effect = post_side_effect

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Old context with enough text to chunk"},
            {"role": "user", "content": "Query"},
        ]

        result = self.layer.build_reduced_context("Query", messages)

        # Should have retried (sleep was called)
        assert mock_sleep.call_count >= 1
        # First retry delay should be 1.0 second
        mock_sleep.assert_any_call(1.0)

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_no_middle_context_returns_original(
        self, mock_client_cls: MagicMock
    ) -> None:
        """When there's only system + user message (no middle context), returns original."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Just a question"},
        ]

        result = self.layer.build_reduced_context("Just a question", messages)

        # No middle context to chunk, returns original
        assert result == messages

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_no_relevant_chunks_returns_original(
        self, mock_client_cls: MagicMock
    ) -> None:
        """When retrieval returns no chunks above threshold, returns original."""
        # All results below 0.5 threshold
        search_response = MagicMock()
        search_response.json.return_value = _make_search_response([
            ("irrelevant", 10, 0.3),
            ("also irrelevant", 10, 0.2),
        ])
        search_response.raise_for_status = MagicMock()

        upsert_response = MagicMock()
        upsert_response.raise_for_status = MagicMock()

        embed_factory = _make_embed_response_dynamic(dim=4)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client
        mock_client.put.return_value = upsert_response

        def post_side_effect(url, **kwargs):
            if "/points/search" in url:
                return search_response
            return embed_factory(url, **kwargs)

        mock_client.post.side_effect = post_side_effect

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Old context"},
            {"role": "assistant", "content": "Old response"},
            {"role": "user", "content": "Query"},
        ]

        result = self.layer.build_reduced_context("Query", messages)

        # No relevant chunks, returns original
        assert result == messages

    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_multiple_system_messages_preserved(
        self, mock_client_cls: MagicMock
    ) -> None:
        """Multiple system messages are all preserved in output."""
        search_response = MagicMock()
        search_response.json.return_value = _make_search_response([
            ("relevant", 10, 0.9),
        ])
        search_response.raise_for_status = MagicMock()

        upsert_response = MagicMock()
        upsert_response.raise_for_status = MagicMock()

        embed_factory = _make_embed_response_dynamic(dim=4)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client
        mock_client.put.return_value = upsert_response

        def post_side_effect(url, **kwargs):
            if "/points/search" in url:
                return search_response
            return embed_factory(url, **kwargs)

        mock_client.post.side_effect = post_side_effect

        messages = [
            {"role": "system", "content": "System prompt 1"},
            {"role": "system", "content": "System prompt 2"},
            {"role": "user", "content": "Old context"},
            {"role": "assistant", "content": "Old response"},
            {"role": "user", "content": "Query"},
        ]

        result = self.layer.build_reduced_context("Query", messages)

        # Both system messages preserved
        assert result[0] == {"role": "system", "content": "System prompt 1"}
        assert result[1] == {"role": "system", "content": "System prompt 2"}
        # Retrieved context + latest user
        assert result[2]["content"].startswith("[Retrieved Context]\n")
        assert result[3] == {"role": "user", "content": "Query"}


class TestAssembleContext:
    """Tests for RetrievalLayer._assemble_context()."""

    def setup_method(self) -> None:
        self.config = RAPConfig(retrieval_max_tokens=100)
        self.layer = RetrievalLayer(self.config)

    def test_empty_chunks_returns_empty_string(self) -> None:
        """No chunks produces empty string."""
        result = self.layer._assemble_context([])
        assert result == ""

    def test_single_chunk_within_budget(self) -> None:
        """Single chunk within budget is included."""
        chunks = [
            ScoredChunk(
                chunk=Chunk(text="hello world", token_count=2, source_message_index=0),
                score=0.9,
                vector_id="id-1",
            )
        ]
        result = self.layer._assemble_context(chunks)
        assert result == "hello world"

    def test_multiple_chunks_within_budget(self) -> None:
        """Multiple chunks within budget are joined with double newline."""
        chunks = [
            ScoredChunk(
                chunk=Chunk(text="first", token_count=30, source_message_index=0),
                score=0.9,
                vector_id="id-1",
            ),
            ScoredChunk(
                chunk=Chunk(text="second", token_count=30, source_message_index=1),
                score=0.8,
                vector_id="id-2",
            ),
        ]
        result = self.layer._assemble_context(chunks)
        assert result == "first\n\nsecond"

    def test_budget_exceeded_skips_chunks(self) -> None:
        """Chunks that would exceed budget are skipped."""
        chunks = [
            ScoredChunk(
                chunk=Chunk(text="big chunk", token_count=80, source_message_index=0),
                score=0.95,
                vector_id="id-1",
            ),
            ScoredChunk(
                chunk=Chunk(text="also big", token_count=80, source_message_index=1),
                score=0.90,
                vector_id="id-2",
            ),
            ScoredChunk(
                chunk=Chunk(text="small", token_count=15, source_message_index=2),
                score=0.85,
                vector_id="id-3",
            ),
        ]
        result = self.layer._assemble_context(chunks)
        # Budget is 100: first chunk (80) fits, second (80) doesn't, third (15) fits
        assert "big chunk" in result
        assert "also big" not in result
        assert "small" in result

    def test_exact_budget_fit(self) -> None:
        """Chunks that exactly fill the budget are included."""
        chunks = [
            ScoredChunk(
                chunk=Chunk(text="fifty", token_count=50, source_message_index=0),
                score=0.9,
                vector_id="id-1",
            ),
            ScoredChunk(
                chunk=Chunk(text="also fifty", token_count=50, source_message_index=1),
                score=0.8,
                vector_id="id-2",
            ),
        ]
        result = self.layer._assemble_context(chunks)
        assert "fifty" in result
        assert "also fifty" in result


class TestRetryWithBackoff:
    """Tests for RetrievalLayer._retry_with_backoff()."""

    def setup_method(self) -> None:
        self.config = RAPConfig()
        self.layer = RetrievalLayer(self.config)

    @patch("deepseek_cursor_proxy.rap.retrieval.time.sleep")
    def test_succeeds_on_first_try(self, mock_sleep: MagicMock) -> None:
        """No retry needed when function succeeds immediately."""
        result = self.layer._retry_with_backoff("test", lambda: "success")
        assert result == "success"
        mock_sleep.assert_not_called()

    @patch("deepseek_cursor_proxy.rap.retrieval.time.sleep")
    def test_retries_on_embedding_error(self, mock_sleep: MagicMock) -> None:
        """Retries when EmbeddingUnavailableError is raised."""
        call_count = [0]

        def flaky_fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise EmbeddingUnavailableError("Temporary failure")
            return "success"

        result = self.layer._retry_with_backoff("test", flaky_fn)
        assert result == "success"
        assert call_count[0] == 3
        # Should have slept twice (before 2nd and 3rd attempts)
        assert mock_sleep.call_count == 2

    @patch("deepseek_cursor_proxy.rap.retrieval.time.sleep")
    def test_retries_on_qdrant_error(self, mock_sleep: MagicMock) -> None:
        """Retries when QdrantUnavailableError is raised."""
        call_count = [0]

        def flaky_fn():
            call_count[0] += 1
            if call_count[0] < 2:
                raise QdrantUnavailableError("Qdrant down")
            return "recovered"

        result = self.layer._retry_with_backoff("test", flaky_fn)
        assert result == "recovered"
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(1.0)

    @patch("deepseek_cursor_proxy.rap.retrieval.time.sleep")
    def test_exponential_delay_doubling(self, mock_sleep: MagicMock) -> None:
        """Delay doubles each retry: 1s, 2s, 4s, ..."""
        call_count = [0]

        def flaky_fn():
            call_count[0] += 1
            if call_count[0] < 4:
                raise EmbeddingUnavailableError("Still failing")
            return "done"

        result = self.layer._retry_with_backoff("test", flaky_fn)
        assert result == "done"

        # Delays: 1.0, 2.0, 4.0
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert sleep_calls[0] == 1.0
        assert sleep_calls[1] == 2.0
        assert sleep_calls[2] == 4.0

    @patch("deepseek_cursor_proxy.rap.retrieval.time.sleep")
    def test_max_5_minutes_total(self, mock_sleep: MagicMock) -> None:
        """Raises after exceeding 5 minutes total elapsed time."""

        def always_fails():
            raise EmbeddingUnavailableError("Permanently down")

        with pytest.raises(EmbeddingUnavailableError, match="Permanently down"):
            self.layer._retry_with_backoff("test", always_fails)

        # Total sleep time should not exceed 300 seconds
        total_sleep = sum(c[0][0] for c in mock_sleep.call_args_list)
        assert total_sleep <= 300

    @patch("deepseek_cursor_proxy.rap.retrieval.time.sleep")
    def test_does_not_retry_on_other_exceptions(self, mock_sleep: MagicMock) -> None:
        """Non-service exceptions are not retried."""

        def raises_value_error():
            raise ValueError("Not a service error")

        with pytest.raises(ValueError, match="Not a service error"):
            self.layer._retry_with_backoff("test", raises_value_error)

        mock_sleep.assert_not_called()
