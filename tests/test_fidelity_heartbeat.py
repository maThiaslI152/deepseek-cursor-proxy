"""Unit tests for FidelityModule.heartbeat_wrapper().

Tests stream health monitoring with heartbeat injection.
Requirements: 3.1, 3.2, 3.3, 3.4
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from deepseek_cursor_proxy.rap.fidelity import FidelityConfig, FidelityModule


# --- Helpers ---


async def async_iter_from_list(
    items: list[bytes], delay: float = 0.0
) -> AsyncIterator[bytes]:
    """Create an async iterator from a list with optional delay between items."""
    for item in items:
        if delay > 0:
            await asyncio.sleep(delay)
        yield item


async def slow_stream(
    items: list[tuple[bytes, float]],
) -> AsyncIterator[bytes]:
    """Create an async iterator where each item has a specific delay before it."""
    for item, delay in items:
        await asyncio.sleep(delay)
        yield item


async def stalling_stream(
    stall_seconds: float,
) -> AsyncIterator[bytes]:
    """A stream that stalls for a given duration then ends."""
    await asyncio.sleep(stall_seconds)
    # Never yields — simulates a completely stalled upstream
    return
    yield  # noqa: unreachable — makes this a generator


async def collect_stream(stream: AsyncIterator[bytes]) -> list[bytes]:
    """Collect all items from an async iterator into a list."""
    result = []
    async for chunk in stream:
        result.append(chunk)
    return result


# --- Tests ---


@pytest.mark.asyncio
class TestHeartbeatWrapper:
    """Tests for FidelityModule.heartbeat_wrapper()."""

    def setup_method(self) -> None:
        """Set up a FidelityModule with a short heartbeat interval for testing."""
        self.config = FidelityConfig(heartbeat_interval_seconds=0.1)
        self.module = FidelityModule(self.config)

    async def test_passes_through_chunks_unchanged(self) -> None:
        """Requirement 3.2: All original chunks are preserved without modification."""
        chunks = [b"data: chunk1\n\n", b"data: chunk2\n\n", b"data: chunk3\n\n"]
        stream = async_iter_from_list(chunks)

        result = await collect_stream(self.module.heartbeat_wrapper(stream))

        # All original chunks must be present in order
        original_chunks = [c for c in result if c != b": heartbeat\n\n"]
        assert original_chunks == chunks

    async def test_preserves_chunk_order(self) -> None:
        """Requirement 3.2: Original chunks maintain their relative order."""
        chunks = [b"first", b"second", b"third"]
        stream = async_iter_from_list(chunks)

        result = await collect_stream(self.module.heartbeat_wrapper(stream))

        # Filter out heartbeats and verify order
        original_chunks = [c for c in result if c != b": heartbeat\n\n"]
        assert original_chunks == chunks

    async def test_injects_heartbeat_on_timeout(self) -> None:
        """Requirement 3.1: Heartbeat injected when no data within interval."""
        # Stream that delays 0.25s between chunks (interval is 0.1s)
        items = [
            (b"data: first\n\n", 0.0),
            (b"data: second\n\n", 0.25),
        ]
        stream = slow_stream(items)

        result = await collect_stream(self.module.heartbeat_wrapper(stream))

        # Should have heartbeats between the two data chunks
        assert b": heartbeat\n\n" in result
        # Both original chunks should be present
        original_chunks = [c for c in result if c != b": heartbeat\n\n"]
        assert original_chunks == [b"data: first\n\n", b"data: second\n\n"]

    async def test_no_heartbeat_when_data_arrives_quickly(self) -> None:
        """No heartbeat injected when data arrives within interval."""
        chunks = [b"data: a\n\n", b"data: b\n\n", b"data: c\n\n"]
        # No delay — data arrives immediately
        stream = async_iter_from_list(chunks, delay=0.0)

        result = await collect_stream(self.module.heartbeat_wrapper(stream))

        # No heartbeats should be present
        assert b": heartbeat\n\n" not in result
        assert result == chunks

    async def test_closes_after_60s_no_data(self) -> None:
        """Requirement 3.3: Stream closes after max_no_data_seconds of no real data."""
        # Use a very short interval and short max_no_data for testing
        config = FidelityConfig(heartbeat_interval_seconds=0.05, max_no_data_seconds=0.5)
        module = FidelityModule(config)

        # Stream that never yields data
        stream = stalling_stream(stall_seconds=120.0)

        start = time.monotonic()
        result = await collect_stream(module.heartbeat_wrapper(stream))
        elapsed = time.monotonic() - start

        # Should have closed within ~0.5s (with some tolerance)
        assert elapsed < 1.5
        # All items should be heartbeats (no real data)
        for chunk in result:
            assert chunk == b": heartbeat\n\n"

    async def test_heartbeat_format(self) -> None:
        """Requirement 3.1: Heartbeat is a valid SSE comment."""
        items = [(b"data: hello\n\n", 0.0), (b"data: world\n\n", 0.25)]
        stream = slow_stream(items)

        result = await collect_stream(self.module.heartbeat_wrapper(stream))

        heartbeats = [c for c in result if c == b": heartbeat\n\n"]
        # At least one heartbeat should have been injected
        assert len(heartbeats) >= 1
        # Verify the format: SSE comment starts with `:` and ends with `\n\n`
        for hb in heartbeats:
            assert hb.startswith(b":")
            assert hb.endswith(b"\n\n")

    async def test_empty_stream(self) -> None:
        """Edge case: empty upstream stream ends immediately."""
        stream = async_iter_from_list([])

        result = await collect_stream(self.module.heartbeat_wrapper(stream))

        # Empty stream should produce no output (or only heartbeats if there's a delay)
        # Since the stream ends immediately via StopAsyncIteration, no heartbeats
        assert result == []

    async def test_time_between_emissions_bounded(self) -> None:
        """Requirement 3.4: Time since last emitted byte <= 2× heartbeat interval."""
        config = FidelityConfig(heartbeat_interval_seconds=0.1)
        module = FidelityModule(config)

        # Stream with a 0.5s gap (should trigger multiple heartbeats)
        items = [(b"data: start\n\n", 0.0), (b"data: end\n\n", 0.5)]
        stream = slow_stream(items)

        max_gap = 0.0
        last_emit_time = time.monotonic()

        async for chunk in module.heartbeat_wrapper(stream):
            now = time.monotonic()
            gap = now - last_emit_time
            if gap > max_gap:
                max_gap = gap
            last_emit_time = now

        # Gap should never exceed 2× heartbeat interval (0.2s)
        # Allow some tolerance for scheduling jitter
        assert max_gap < 2 * config.heartbeat_interval_seconds + 0.05

    async def test_multiple_heartbeats_during_long_gap(self) -> None:
        """Multiple heartbeats injected during a long gap between data."""
        config = FidelityConfig(heartbeat_interval_seconds=0.1)
        module = FidelityModule(config)

        # 0.35s gap should produce ~3 heartbeats
        items = [(b"data: first\n\n", 0.0), (b"data: second\n\n", 0.35)]
        stream = slow_stream(items)

        result = await collect_stream(module.heartbeat_wrapper(stream))

        heartbeats = [c for c in result if c == b": heartbeat\n\n"]
        # Should have at least 2 heartbeats (0.35 / 0.1 = 3.5, minus scheduling)
        assert len(heartbeats) >= 2

    async def test_stream_terminates_normally(self) -> None:
        """Stream ends cleanly when upstream finishes."""
        chunks = [b"data: only\n\n"]
        stream = async_iter_from_list(chunks)

        result = await collect_stream(self.module.heartbeat_wrapper(stream))

        assert b"data: only\n\n" in result

    async def test_large_chunks_passed_through(self) -> None:
        """Large binary chunks are passed through without modification."""
        large_chunk = b"x" * 65536
        stream = async_iter_from_list([large_chunk])

        result = await collect_stream(self.module.heartbeat_wrapper(stream))

        original_chunks = [c for c in result if c != b": heartbeat\n\n"]
        assert original_chunks == [large_chunk]
