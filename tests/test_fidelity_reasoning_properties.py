"""Property-based tests for reasoning token extraction integrity.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 3: Reasoning Token Extraction Integrity
- For any SSE chunk containing reasoning_content, the extracted value equals
  the original (no modification).
- For any SSE chunk without reasoning_content, the method returns None
  without error.
- For any arbitrary string as reasoning_content, it is returned unchanged
  (integrity).
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from deepseek_cursor_proxy.rap.fidelity import FidelityConfig, FidelityModule


# --- Strategies ---

# Arbitrary reasoning content: any text including unicode, empty, whitespace
arbitrary_reasoning_content = st.text(min_size=0, max_size=500)

# Strategy for a valid SSE chunk WITH reasoning_content
def sse_chunk_with_reasoning(reasoning: str) -> dict:
    """Build a well-formed SSE chunk containing reasoning_content."""
    return {
        "choices": [
            {
                "delta": {
                    "reasoning_content": reasoning,
                }
            }
        ]
    }


# Strategy for SSE chunks WITHOUT reasoning_content
sse_chunks_without_reasoning = st.one_of(
    # Empty dict
    st.just({}),
    # No choices key
    st.dictionaries(
        keys=st.text(min_size=1, max_size=20).filter(lambda k: k != "choices"),
        values=st.text(min_size=0, max_size=50),
        min_size=0,
        max_size=5,
    ),
    # Choices is empty list
    st.just({"choices": []}),
    # Choices is not a list
    st.just({"choices": "not_a_list"}),
    # Delta without reasoning_content
    st.builds(
        lambda content: {"choices": [{"delta": {"content": content}}]},
        content=st.text(min_size=0, max_size=100),
    ),
    # Delta is not a dict
    st.just({"choices": [{"delta": "not_a_dict"}]}),
    # First choice is not a dict
    st.just({"choices": ["not_a_dict"]}),
    # Delta missing entirely
    st.just({"choices": [{"index": 0}]}),
    # reasoning_content is explicitly None
    st.just({"choices": [{"delta": {"reasoning_content": None}}]}),
)


# --- Fixtures ---

def _make_module(enabled: bool = True) -> FidelityModule:
    """Create a FidelityModule with reasoning stream enabled/disabled."""
    return FidelityModule(FidelityConfig(reasoning_stream_enabled=enabled))


# --- Property 3: Reasoning Token Extraction Integrity ---


class TestProperty3ReasoningTokenExtractionIntegrity:
    """For any SSE chunk, reasoning extraction preserves content integrity
    and handles missing fields gracefully.

    **Validates: Requirements 2.1, 2.2, 2.3**
    """

    @given(reasoning=arbitrary_reasoning_content)
    def test_extracted_value_equals_original(self, reasoning: str) -> None:
        """For any SSE chunk containing reasoning_content, the extracted
        value equals the original without modification.

        **Validates: Requirements 2.1, 2.2**
        """
        module = _make_module(enabled=True)
        chunk = sse_chunk_with_reasoning(reasoning)
        result = module.extract_reasoning_stream(chunk)
        assert result == reasoning

    @given(chunk=sse_chunks_without_reasoning)
    def test_missing_reasoning_returns_none(self, chunk: dict) -> None:
        """For any SSE chunk without reasoning_content, the method returns
        None without raising an error.

        **Validates: Requirements 2.3**
        """
        module = _make_module(enabled=True)
        result = module.extract_reasoning_stream(chunk)
        assert result is None

    @given(reasoning=arbitrary_reasoning_content)
    def test_arbitrary_string_returned_unchanged(self, reasoning: str) -> None:
        """For any arbitrary string as reasoning_content, it is returned
        unchanged (integrity check with identity comparison).

        **Validates: Requirements 2.2**
        """
        module = _make_module(enabled=True)
        chunk = sse_chunk_with_reasoning(reasoning)
        result = module.extract_reasoning_stream(chunk)
        # Verify byte-for-byte identity — no stripping, encoding, or mutation
        assert result is not None
        assert len(result) == len(reasoning)
        assert result == reasoning

    @given(reasoning=arbitrary_reasoning_content)
    def test_disabled_reasoning_returns_none(self, reasoning: str) -> None:
        """When reasoning_stream_enabled is False, extraction always returns
        None regardless of chunk content.

        **Validates: Requirements 2.1**
        """
        module = _make_module(enabled=False)
        chunk = sse_chunk_with_reasoning(reasoning)
        result = module.extract_reasoning_stream(chunk)
        assert result is None
