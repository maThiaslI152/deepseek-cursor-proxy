"""Unit tests for the Retrieval Layer — context chunking.

Tests chunk_context() splitting messages into chunks of chunk_size_tokens
with chunk_overlap_tokens overlap, tiktoken-based token counting,
and preservation of system messages and latest user message.

Requirements: 6.1, 6.4
"""

import tiktoken

from deepseek_cursor_proxy.rap.config import RAPConfig
from deepseek_cursor_proxy.rap.retrieval import RetrievalLayer


class TestChunkContext:
    """Tests for RetrievalLayer.chunk_context()."""

    def setup_method(self) -> None:
        """Set up a RetrievalLayer with default config."""
        self.config = RAPConfig()
        self.layer = RetrievalLayer(self.config)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def test_empty_messages_returns_empty(self) -> None:
        """Edge case: empty message list returns no chunks."""
        chunks = self.layer.chunk_context([])
        assert chunks == []

    def test_only_system_message_returns_empty(self) -> None:
        """Requirement 6.4: System messages are preserved (not chunked)."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        chunks = self.layer.chunk_context(messages)
        assert chunks == []

    def test_only_user_message_returns_empty(self) -> None:
        """Requirement 6.4: Latest user message is preserved (not chunked)."""
        messages = [
            {"role": "user", "content": "What is the meaning of life?"},
        ]
        chunks = self.layer.chunk_context(messages)
        assert chunks == []

    def test_system_and_user_only_returns_empty(self) -> None:
        """Requirement 6.4: System + latest user message both preserved."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        chunks = self.layer.chunk_context(messages)
        assert chunks == []

    def test_middle_messages_are_chunked(self) -> None:
        """Requirement 6.1: Middle context messages are chunked."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Here is some context about the project."},
            {"role": "assistant", "content": "I understand. What would you like to do?"},
            {"role": "user", "content": "Fix the bug."},
        ]
        chunks = self.layer.chunk_context(messages)
        # The first user message (index 1) and assistant message (index 2)
        # are middle context. The latest user message (index 3) is preserved.
        assert len(chunks) > 0
        # Check that chunks come from the middle messages
        source_indices = {c.source_message_index for c in chunks}
        assert 1 in source_indices or 2 in source_indices
        # Latest user message (index 3) should NOT be chunked
        assert 3 not in source_indices
        # System message (index 0) should NOT be chunked
        assert 0 not in source_indices

    def test_chunk_token_count_accuracy(self) -> None:
        """Requirement 6.1: Token counts are accurate (tiktoken-based)."""
        text = "Hello world, this is a test message for chunking."
        messages = [
            {"role": "user", "content": text},
            {"role": "user", "content": "Final question?"},
        ]
        chunks = self.layer.chunk_context(messages)
        # First user message is middle context (not the latest user)
        assert len(chunks) >= 1
        for chunk in chunks:
            expected_tokens = len(self.encoding.encode(chunk.text))
            assert chunk.token_count == expected_tokens

    def test_chunk_size_respects_limit(self) -> None:
        """Requirement 6.1: Each chunk has at most chunk_size_tokens tokens."""
        # Create a long message that will need multiple chunks
        long_text = "This is a sentence with several words. " * 200
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": long_text},
            {"role": "user", "content": "What do you think?"},
        ]
        # Use a small chunk size for testing
        self.layer.chunk_size_tokens = 50
        chunks = self.layer.chunk_context(messages)

        assert len(chunks) > 1  # Should produce multiple chunks
        for chunk in chunks:
            assert chunk.token_count <= 50

    def test_chunk_overlap(self) -> None:
        """Requirement 6.1: Chunks have chunk_overlap_tokens overlap."""
        # Create a message that will produce multiple chunks
        long_text = "word " * 500  # ~500 tokens
        messages = [
            {"role": "user", "content": long_text},
            {"role": "user", "content": "Final question?"},
        ]
        self.layer.chunk_size_tokens = 100
        self.layer.chunk_overlap_tokens = 20

        chunks = self.layer.chunk_context(messages)
        assert len(chunks) > 1

        # Verify overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            # There should be some token overlap
            # (exact overlap depends on token boundaries)
            current_text_tokens = self.encoding.encode(chunks[i].text)
            next_text_tokens = self.encoding.encode(chunks[i + 1].text)
            # The end of current chunk should overlap with start of next
            overlap_size = self.layer.chunk_overlap_tokens
            if len(current_text_tokens) >= overlap_size and len(next_text_tokens) >= overlap_size:
                tail = current_text_tokens[-overlap_size:]
                head = next_text_tokens[:overlap_size]
                assert tail == head

    def test_preserves_system_messages(self) -> None:
        """Requirement 6.4: System messages are never chunked."""
        messages = [
            {"role": "system", "content": "A very long system prompt. " * 200},
            {"role": "user", "content": "Some context here."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "Do the thing."},
        ]
        chunks = self.layer.chunk_context(messages)
        # No chunk should come from the system message (index 0)
        for chunk in chunks:
            assert chunk.source_message_index != 0

    def test_preserves_latest_user_message(self) -> None:
        """Requirement 6.4: Latest user message is never chunked."""
        messages = [
            {"role": "user", "content": "First user message with context."},
            {"role": "assistant", "content": "I see."},
            {"role": "user", "content": "A very long final question. " * 200},
        ]
        chunks = self.layer.chunk_context(messages)
        # The latest user message is at index 2 — should not be chunked
        for chunk in chunks:
            assert chunk.source_message_index != 2

    def test_single_short_middle_message_one_chunk(self) -> None:
        """A short middle message produces exactly one chunk."""
        messages = [
            {"role": "system", "content": "System."},
            {"role": "assistant", "content": "Short reply."},
            {"role": "user", "content": "Question?"},
        ]
        chunks = self.layer.chunk_context(messages)
        assert len(chunks) == 1
        assert chunks[0].text == "Short reply."
        assert chunks[0].source_message_index == 1

    def test_chunk_metadata_includes_indices(self) -> None:
        """Chunks include chunk_index and total_chunks in metadata."""
        long_text = "word " * 500
        messages = [
            {"role": "user", "content": long_text},
            {"role": "user", "content": "Final?"},
        ]
        self.layer.chunk_size_tokens = 100
        chunks = self.layer.chunk_context(messages)

        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_non_string_content_skipped(self) -> None:
        """Messages with non-string content (None, tool_calls) are skipped."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "user", "content": "Hello"},
        ]
        chunks = self.layer.chunk_context(messages)
        # assistant message has None content, latest user is preserved
        assert chunks == []

    def test_multiple_middle_messages_all_chunked(self) -> None:
        """All middle messages (not system, not latest user) are chunked."""
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "First context."},
            {"role": "assistant", "content": "First reply."},
            {"role": "user", "content": "Second context."},
            {"role": "assistant", "content": "Second reply."},
            {"role": "user", "content": "Final question."},
        ]
        chunks = self.layer.chunk_context(messages)
        source_indices = {c.source_message_index for c in chunks}
        # Messages at indices 1, 2, 3, 4 are middle context
        # Index 0 is system, index 5 is latest user
        assert 0 not in source_indices
        assert 5 not in source_indices
        assert source_indices.issubset({1, 2, 3, 4})

    def test_assistant_messages_are_chunked(self) -> None:
        """Assistant messages in the middle are chunked."""
        messages = [
            {"role": "system", "content": "System."},
            {"role": "assistant", "content": "I provided some context earlier."},
            {"role": "user", "content": "Thanks, now help me."},
        ]
        chunks = self.layer.chunk_context(messages)
        assert len(chunks) == 1
        assert chunks[0].source_message_index == 1
        assert chunks[0].text == "I provided some context earlier."

    def test_uses_tiktoken_cl100k_base(self) -> None:
        """Verifies tiktoken cl100k_base encoding is used for token counting."""
        # "Hello" is 1 token in cl100k_base
        # Use a known multi-token word
        text = "antidisestablishmentarianism"
        expected_tokens = len(self.encoding.encode(text))

        messages = [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Question?"},
        ]
        chunks = self.layer.chunk_context(messages)
        assert len(chunks) == 1
        assert chunks[0].token_count == expected_tokens

    def test_chunk_text_reconstruction(self) -> None:
        """Chunk texts together cover the original message content."""
        # Use non-overlapping for simpler verification
        self.layer.chunk_size_tokens = 50
        self.layer.chunk_overlap_tokens = 0

        text = "word " * 200  # ~200 tokens
        messages = [
            {"role": "user", "content": text},
            {"role": "user", "content": "End."},
        ]
        chunks = self.layer.chunk_context(messages)

        # Concatenating all chunk texts should reconstruct the original
        reconstructed = "".join(c.text for c in chunks)
        assert reconstructed == text

    def test_default_chunk_size_is_512(self) -> None:
        """Default chunk_size_tokens is 512 per design doc."""
        layer = RetrievalLayer()
        assert layer.chunk_size_tokens == 512

    def test_default_chunk_overlap_is_64(self) -> None:
        """Default chunk_overlap_tokens is 64 per design doc."""
        layer = RetrievalLayer()
        assert layer.chunk_overlap_tokens == 64
