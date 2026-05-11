"""Property-based tests for the Retrieval Layer — context chunking.

**Validates: Requirement 6.1**

Property 10: Chunking Produces Correct Size and Overlap
For any list of messages with middle context:
- Each chunk has at most chunk_size_tokens tokens
- Consecutive chunks from the same message have exactly chunk_overlap_tokens overlap
- All middle context text is covered by the chunks (no data loss)
"""

from __future__ import annotations

import tiktoken
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from deepseek_cursor_proxy.rap.config import RAPConfig
from deepseek_cursor_proxy.rap.retrieval import RetrievalLayer


# --- Strategies ---


def _word() -> st.SearchStrategy[str]:
    """Generate a single word-like token."""
    return st.from_regex(r"[a-z]{1,8}", fullmatch=True)


def _middle_content() -> st.SearchStrategy[str]:
    """Generate content for a middle context message.

    Produces space-separated words to ensure predictable tokenization.
    """
    return st.lists(
        _word(),
        min_size=10,
        max_size=200,
    ).map(lambda words: " ".join(words))


def _middle_message(role: st.SearchStrategy[str]) -> st.SearchStrategy[dict[str, str]]:
    """Generate a middle context message with a given role."""
    return st.fixed_dictionaries({
        "role": role,
        "content": _middle_content(),
    })


def _messages_with_middle_context() -> st.SearchStrategy[list[dict[str, str]]]:
    """Generate a message list that has middle context messages.

    Structure: optional system + 1..3 middle messages + final user message.
    Middle messages are user/assistant messages that are NOT the last user message.
    """
    system_msg = st.just({"role": "system", "content": "You are a helpful assistant."})
    middle_msgs = st.lists(
        _middle_message(st.sampled_from(["user", "assistant"])),
        min_size=1,
        max_size=3,
    )
    final_user = st.just({"role": "user", "content": "What should I do next?"})

    return st.tuples(system_msg, middle_msgs, final_user).map(
        lambda t: [t[0]] + t[1] + [t[2]]
    )


def _chunk_params() -> st.SearchStrategy[tuple[int, int]]:
    """Generate valid (chunk_size_tokens, chunk_overlap_tokens) pairs.

    Ensures chunk_size > chunk_overlap and both are reasonable values.
    """
    return st.integers(min_value=20, max_value=128).flatmap(
        lambda size: st.tuples(
            st.just(size),
            st.integers(min_value=1, max_value=max(1, size // 2)),
        )
    )


# --- Property Tests ---


class TestChunkingProperty:
    """Property 10: Chunking Produces Correct Size and Overlap."""

    @given(
        messages=_messages_with_middle_context(),
        params=_chunk_params(),
    )
    @settings(max_examples=20, deadline=5000)
    def test_each_chunk_respects_size_limit(
        self, messages: list[dict[str, str]], params: tuple[int, int]
    ) -> None:
        """Each chunk has at most chunk_size_tokens tokens.

        **Validates: Requirements 6.1**
        """
        chunk_size, chunk_overlap = params
        config = RAPConfig()
        layer = RetrievalLayer(config)
        layer.chunk_size_tokens = chunk_size
        layer.chunk_overlap_tokens = chunk_overlap

        chunks = layer.chunk_context(messages)

        encoding = tiktoken.get_encoding("cl100k_base")
        for chunk in chunks:
            actual_tokens = len(encoding.encode(chunk.text))
            assert actual_tokens <= chunk_size, (
                f"Chunk has {actual_tokens} tokens, exceeds limit of {chunk_size}"
            )
            # Also verify the stored token_count matches actual
            assert chunk.token_count == actual_tokens

    @given(
        messages=_messages_with_middle_context(),
        params=_chunk_params(),
    )
    @settings(max_examples=20, deadline=5000)
    def test_consecutive_chunks_have_exact_overlap(
        self, messages: list[dict[str, str]], params: tuple[int, int]
    ) -> None:
        """Consecutive chunks from the same message have exactly chunk_overlap_tokens overlap.

        **Validates: Requirements 6.1**
        """
        chunk_size, chunk_overlap = params
        config = RAPConfig()
        layer = RetrievalLayer(config)
        layer.chunk_size_tokens = chunk_size
        layer.chunk_overlap_tokens = chunk_overlap

        chunks = layer.chunk_context(messages)

        encoding = tiktoken.get_encoding("cl100k_base")

        # Group chunks by source message
        from collections import defaultdict
        chunks_by_source: dict[int, list] = defaultdict(list)
        for chunk in chunks:
            chunks_by_source[chunk.source_message_index].append(chunk)

        for source_idx, source_chunks in chunks_by_source.items():
            if len(source_chunks) <= 1:
                continue  # No overlap to check for single-chunk messages

            for i in range(len(source_chunks) - 1):
                current_tokens = encoding.encode(source_chunks[i].text)
                next_tokens = encoding.encode(source_chunks[i + 1].text)

                # The last chunk_overlap tokens of current should equal
                # the first chunk_overlap tokens of next
                # (unless the current chunk is shorter than chunk_overlap)
                effective_overlap = min(chunk_overlap, len(current_tokens), len(next_tokens))
                tail = current_tokens[-effective_overlap:]
                head = next_tokens[:effective_overlap]
                assert tail == head, (
                    f"Chunks {i} and {i+1} from message {source_idx} "
                    f"do not have expected overlap of {effective_overlap} tokens. "
                    f"Tail: {tail[:5]}..., Head: {head[:5]}..."
                )

    @given(
        messages=_messages_with_middle_context(),
        params=_chunk_params(),
    )
    @settings(max_examples=20, deadline=5000)
    def test_all_middle_context_covered(
        self, messages: list[dict[str, str]], params: tuple[int, int]
    ) -> None:
        """All middle context text is covered by the chunks (no data loss).

        **Validates: Requirements 6.1**
        """
        chunk_size, chunk_overlap = params
        config = RAPConfig()
        layer = RetrievalLayer(config)
        layer.chunk_size_tokens = chunk_size
        layer.chunk_overlap_tokens = chunk_overlap

        chunks = layer.chunk_context(messages)

        encoding = tiktoken.get_encoding("cl100k_base")

        # Identify middle messages (same logic as RetrievalLayer)
        # System messages and latest user message are excluded
        latest_user_index = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                latest_user_index = i
                break

        middle_messages: list[tuple[int, str]] = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                continue
            if i == latest_user_index:
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                middle_messages.append((i, content))

        # For each middle message, verify all its tokens are covered by chunks
        from collections import defaultdict
        chunks_by_source: dict[int, list] = defaultdict(list)
        for chunk in chunks:
            chunks_by_source[chunk.source_message_index].append(chunk)

        for msg_idx, content in middle_messages:
            original_tokens = encoding.encode(content)
            if not original_tokens:
                continue

            msg_chunks = chunks_by_source[msg_idx]
            assert len(msg_chunks) > 0, (
                f"Message at index {msg_idx} has content but no chunks"
            )

            # Verify coverage: every token in the original message appears
            # in at least one chunk. We do this by collecting the set of
            # token positions covered by all chunks.
            step = chunk_size - chunk_overlap
            if step < 1:
                step = 1

            covered_positions: set[int] = set()
            expected_start = 0
            for chunk in msg_chunks:
                chunk_tokens = encoding.encode(chunk.text)
                # Verify this chunk's tokens match the expected window
                expected_end = min(expected_start + chunk_size, len(original_tokens))
                expected_window = original_tokens[expected_start:expected_end]
                assert chunk_tokens == expected_window, (
                    f"Chunk at expected start {expected_start} does not match "
                    f"expected token window"
                )
                # Mark positions as covered
                for pos in range(expected_start, expected_end):
                    covered_positions.add(pos)

                if expected_end >= len(original_tokens):
                    break
                expected_start += step

            # All positions must be covered
            all_positions = set(range(len(original_tokens)))
            uncovered = all_positions - covered_positions
            assert not uncovered, (
                f"Token positions {sorted(uncovered)} in message {msg_idx} "
                f"are not covered by any chunk"
            )
