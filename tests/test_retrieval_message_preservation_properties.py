"""Property-based tests for message preservation in build_reduced_context().

**Validates: Requirement 6.4**

Property 12: System and User Message Preservation
For any message list processed by build_reduced_context(), all system messages
and the latest user message SHALL appear in the output unchanged.

This property verifies:
1. System messages are always preserved unchanged in the output
2. The latest user message is always preserved unchanged in the output
3. These messages are never modified, reordered, or dropped regardless of
   what retrieval returns
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from deepseek_cursor_proxy.rap.config import RAPConfig
from deepseek_cursor_proxy.rap.retrieval import RetrievalLayer


# --- Strategies ---


def _message_content() -> st.SearchStrategy[str]:
    """Generate arbitrary message content (non-empty strings)."""
    return st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "Z"),
            whitelist_characters=" \n\t",
        ),
        min_size=1,
        max_size=200,
    ).filter(lambda s: s.strip())


def _system_message() -> st.SearchStrategy[dict[str, str]]:
    """Generate a system message with arbitrary content."""
    return st.fixed_dictionaries({
        "role": st.just("system"),
        "content": _message_content(),
    })


def _user_message() -> st.SearchStrategy[dict[str, str]]:
    """Generate a user message with arbitrary content."""
    return st.fixed_dictionaries({
        "role": st.just("user"),
        "content": _message_content(),
    })


def _assistant_message() -> st.SearchStrategy[dict[str, str]]:
    """Generate an assistant message with arbitrary content."""
    return st.fixed_dictionaries({
        "role": st.just("assistant"),
        "content": _message_content(),
    })


def _middle_messages() -> st.SearchStrategy[list[dict[str, str]]]:
    """Generate middle context messages (user/assistant pairs)."""
    return st.lists(
        st.one_of(_user_message(), _assistant_message()),
        min_size=1,
        max_size=5,
    )


def _messages_with_system_and_user() -> st.SearchStrategy[list[dict[str, str]]]:
    """Generate a message list with system messages, middle context, and a final user message.

    Structure: system_messages + middle_messages + final_user_message
    Ensures there is at least one system message and a final user message,
    with middle context messages in between to trigger chunking.
    """
    return st.tuples(
        st.lists(_system_message(), min_size=1, max_size=3),
        _middle_messages(),
        _user_message(),
    ).map(lambda t: t[0] + t[1] + [t[2]])


def _make_mock_client_with_results(dim: int = 4):
    """Create a mock httpx.Client that simulates successful retrieval.

    Returns embeddings for any embed call and search results with
    chunks above the 0.5 score threshold.
    """
    def setup_mock(mock_client_cls):
        search_response = MagicMock()
        search_response.json.return_value = {
            "result": [
                {
                    "id": "id-0",
                    "score": 0.9,
                    "payload": {
                        "text": "retrieved context snippet",
                        "token_count": 10,
                        "source_index": 0,
                        "metadata": {},
                    },
                }
            ]
        }
        search_response.raise_for_status = MagicMock()

        upsert_response = MagicMock()
        upsert_response.raise_for_status = MagicMock()

        def embed_factory(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
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

    return setup_mock


# --- Property Tests ---


class TestMessagePreservationProperty:
    """Property 12: System and User Message Preservation.

    **Validates: Requirements 6.4**
    """

    @given(messages=_messages_with_system_and_user())
    @settings(max_examples=20, deadline=10000)
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_system_messages_preserved_on_success(
        self, mock_client_cls: MagicMock, messages: list[dict[str, str]]
    ) -> None:
        """System messages are always preserved unchanged in the output when retrieval succeeds.

        **Validates: Requirements 6.4**
        """
        _make_mock_client_with_results()(mock_client_cls)

        config = RAPConfig(retrieval_max_tokens=200)
        layer = RetrievalLayer(config)

        # Extract the expected system messages from input
        input_system_messages = [m for m in messages if m.get("role") == "system"]

        # Find the latest user message content to use as query
        latest_user_content = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                latest_user_content = m["content"]
                break

        result = layer.build_reduced_context(latest_user_content, messages)

        # Extract system messages from output
        output_system_messages = [m for m in result if m.get("role") == "system"]

        # All input system messages must appear in output, unchanged and in order
        assert len(output_system_messages) == len(input_system_messages), (
            f"Expected {len(input_system_messages)} system messages in output, "
            f"got {len(output_system_messages)}"
        )
        for i, (expected, actual) in enumerate(
            zip(input_system_messages, output_system_messages)
        ):
            assert expected == actual, (
                f"System message {i} was modified. "
                f"Expected: {expected}, Got: {actual}"
            )

    @given(messages=_messages_with_system_and_user())
    @settings(max_examples=20, deadline=10000)
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_latest_user_message_preserved_on_success(
        self, mock_client_cls: MagicMock, messages: list[dict[str, str]]
    ) -> None:
        """The latest user message is always preserved unchanged in the output when retrieval succeeds.

        **Validates: Requirements 6.4**
        """
        _make_mock_client_with_results()(mock_client_cls)

        config = RAPConfig(retrieval_max_tokens=200)
        layer = RetrievalLayer(config)

        # Find the latest user message from input
        latest_user_message = None
        for m in reversed(messages):
            if m.get("role") == "user":
                latest_user_message = m
                break

        assert latest_user_message is not None  # guaranteed by strategy

        result = layer.build_reduced_context(latest_user_message["content"], messages)

        # The latest user message must be the last message in the output
        assert result[-1] == latest_user_message, (
            f"Latest user message was not preserved as last message. "
            f"Expected: {latest_user_message}, Got: {result[-1]}"
        )

    @given(messages=_messages_with_system_and_user())
    @settings(max_examples=20, deadline=10000)
    @patch("deepseek_cursor_proxy.rap.retrieval.time.sleep")
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_messages_preserved_on_service_failure(
        self, mock_client_cls: MagicMock, mock_sleep: MagicMock, messages: list[dict[str, str]]
    ) -> None:
        """System messages and latest user message are preserved when services fail (graceful degradation).

        When embedding or Qdrant services are unavailable, build_reduced_context()
        returns the original messages unchanged — which trivially preserves all messages.

        **Validates: Requirements 6.4**
        """
        # Simulate service failure
        import httpx as httpx_mod

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client
        mock_client.post.side_effect = httpx_mod.ConnectError("Connection refused")

        config = RAPConfig(retrieval_max_tokens=200)
        layer = RetrievalLayer(config)

        # Find the latest user message content
        latest_user_content = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                latest_user_content = m["content"]
                break

        result = layer.build_reduced_context(latest_user_content, messages)

        # On failure, original messages are returned unchanged
        assert result == messages, (
            "On service failure, build_reduced_context should return original messages unchanged"
        )

    @given(messages=_messages_with_system_and_user())
    @settings(max_examples=20, deadline=10000)
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_system_messages_never_reordered(
        self, mock_client_cls: MagicMock, messages: list[dict[str, str]]
    ) -> None:
        """System messages maintain their relative order in the output.

        **Validates: Requirements 6.4**
        """
        _make_mock_client_with_results()(mock_client_cls)

        config = RAPConfig(retrieval_max_tokens=200)
        layer = RetrievalLayer(config)

        # Find the latest user message content
        latest_user_content = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                latest_user_content = m["content"]
                break

        result = layer.build_reduced_context(latest_user_content, messages)

        # System messages should appear at the beginning of the output
        # and before the retrieved context message
        output_system_messages = [m for m in result if m.get("role") == "system"]
        input_system_messages = [m for m in messages if m.get("role") == "system"]

        # Verify order: system messages come first in the result
        system_indices = [i for i, m in enumerate(result) if m.get("role") == "system"]
        if system_indices:
            # All system messages should be at the start (indices 0, 1, 2, ...)
            for idx, expected_idx in enumerate(system_indices):
                assert expected_idx == idx, (
                    f"System message at position {expected_idx} should be at position {idx}. "
                    "System messages were reordered."
                )

        # Content order preserved
        for i in range(len(output_system_messages)):
            assert output_system_messages[i]["content"] == input_system_messages[i]["content"], (
                f"System message {i} content was modified or reordered"
            )
