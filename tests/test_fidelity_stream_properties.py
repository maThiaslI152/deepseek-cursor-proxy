"""Property-based tests for stream integrity under heartbeat injection.

**Property 4: Stream Integrity Under Heartbeat Injection**
**Validates: Requirement 3.2**

For any sequence of upstream SSE chunks with heartbeats injected by
heartbeat_wrapper(), the subsequence of non-heartbeat chunks SHALL be
identical in content and order to the original upstream stream.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from deepseek_cursor_proxy.rap.fidelity import FidelityConfig, FidelityModule

# --- Constants ---

HEARTBEAT_BYTES = b": heartbeat\n\n"

# --- Strategies ---

# Generate arbitrary non-empty byte chunks (simulating SSE data)
# Exclude the exact heartbeat bytes to avoid ambiguity in filtering
arbitrary_chunk = st.binary(min_size=1, max_size=512).filter(
    lambda b: b != HEARTBEAT_BYTES
)

# Generate lists of byte chunks representing an upstream SSE stream
arbitrary_chunk_list = st.lists(arbitrary_chunk, min_size=0, max_size=20)


# --- Helpers ---


async def async_iter_from_list(
    items: list[bytes], delay: float = 0.0
) -> AsyncIterator[bytes]:
    """Create an async iterator from a list with optional delay between items."""
    for item in items:
        if delay > 0:
            await asyncio.sleep(delay)
        yield item


async def collect_stream(stream: AsyncIterator[bytes]) -> list[bytes]:
    """Collect all items from an async iterator into a list."""
    result = []
    async for chunk in stream:
        result.append(chunk)
    return result


def filter_heartbeats(chunks: list[bytes]) -> list[bytes]:
    """Remove heartbeat comments from a list of chunks."""
    return [c for c in chunks if c != HEARTBEAT_BYTES]


def _make_module(interval: float = 0.05) -> FidelityModule:
    """Create a FidelityModule with a short heartbeat interval for testing."""
    return FidelityModule(FidelityConfig(heartbeat_interval_seconds=interval))


# --- Property 4: Stream Integrity Under Heartbeat Injection ---


@pytest.mark.asyncio
class TestProperty4StreamIntegrityUnderHeartbeatInjection:
    """For any list of byte chunks passed through heartbeat_wrapper, all
    original chunks appear in the output in their original order (filtering
    out heartbeat comments), the output never modifies the content of
    original chunks, and heartbeat comments are always exactly
    b": heartbeat\\n\\n".

    **Validates: Requirement 3.2**
    """

    @given(chunks=arbitrary_chunk_list)
    @settings(max_examples=20, deadline=10000)
    @pytest.mark.asyncio
    async def test_original_chunks_preserved_in_order(
        self, chunks: list[bytes]
    ) -> None:
        """All original chunks appear in the output in their original order.

        **Validates: Requirements 3.2**
        """
        module = _make_module()
        stream = async_iter_from_list(chunks)

        result = await collect_stream(module.heartbeat_wrapper(stream))
        non_heartbeat = filter_heartbeats(result)

        assert non_heartbeat == chunks

    @given(chunks=arbitrary_chunk_list)
    @settings(max_examples=20, deadline=10000)
    @pytest.mark.asyncio
    async def test_chunk_content_never_modified(
        self, chunks: list[bytes]
    ) -> None:
        """The output never modifies the content of original chunks.

        Each non-heartbeat chunk in the output must be byte-for-byte
        identical to the corresponding input chunk.

        **Validates: Requirements 3.2**
        """
        module = _make_module()
        stream = async_iter_from_list(chunks)

        result = await collect_stream(module.heartbeat_wrapper(stream))
        non_heartbeat = filter_heartbeats(result)

        for original, output in zip(chunks, non_heartbeat):
            assert original == output
            assert id(original) != id(output) or original is output
            # Byte-for-byte equality
            assert len(original) == len(output)

    @given(chunks=arbitrary_chunk_list)
    @settings(max_examples=20, deadline=10000)
    @pytest.mark.asyncio
    async def test_heartbeats_are_exact_format(
        self, chunks: list[bytes]
    ) -> None:
        """Heartbeat comments are always exactly b": heartbeat\\n\\n".

        Any chunk in the output that is not from the original stream
        must be exactly the heartbeat comment format.

        **Validates: Requirements 3.2**
        """
        module = _make_module()
        stream = async_iter_from_list(chunks)

        result = await collect_stream(module.heartbeat_wrapper(stream))

        # Partition output into original chunks and injected chunks
        original_set = set()
        remaining_originals = list(chunks)

        for chunk in result:
            if remaining_originals and chunk == remaining_originals[0]:
                remaining_originals.pop(0)
                original_set.add(id(chunk))
            else:
                # Any non-original chunk must be exactly the heartbeat
                assert chunk == HEARTBEAT_BYTES

    @given(
        chunks=st.lists(
            st.binary(min_size=1, max_size=256).filter(
                lambda b: b != HEARTBEAT_BYTES
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=10, deadline=15000)
    @pytest.mark.asyncio
    async def test_stream_integrity_with_delays(
        self, chunks: list[bytes]
    ) -> None:
        """Stream integrity holds even when delays trigger heartbeats.

        With a very short heartbeat interval and small delays between
        chunks, heartbeats may be injected, but original chunk order
        and content must still be preserved.

        **Validates: Requirements 3.2**
        """
        # Use a very short interval so heartbeats are likely injected
        module = _make_module(interval=0.01)
        # Small delay to potentially trigger heartbeats
        stream = async_iter_from_list(chunks, delay=0.02)

        result = await collect_stream(module.heartbeat_wrapper(stream))
        non_heartbeat = filter_heartbeats(result)

        assert non_heartbeat == chunks
