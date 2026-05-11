"""Fidelity Module for the RAP middleware stack.

Ensures seamless integration between Cursor and DeepSeek V4 by managing
header spoofing, reasoning token pass-through, and stream health monitoring.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FidelityConfig:
    """Configuration for the Fidelity Module.

    Attributes:
        spoof_headers: Headers to inject into outbound requests.
        heartbeat_interval_seconds: Interval for SSE heartbeat injection.
        reasoning_stream_enabled: Whether to extract reasoning_content as a
            distinct stream.
        byok_endpoint: The BYOK endpoint URL to route inference requests to.
    """

    spoof_headers: dict[str, str] = field(default_factory=lambda: {
        "X-Cursor-Plan": "pro",
        "X-Cursor-Tier": "unlimited",
    })
    heartbeat_interval_seconds: float = 15.0
    reasoning_stream_enabled: bool = True
    byok_endpoint: str = "https://api.deepseek.com"
    max_no_data_seconds: float = 60.0


class FidelityModule:
    """Manages header spoofing, reasoning pass-through, and stream health.

    The module intercepts outbound requests to inject Pro/Unlimited headers,
    routes to the configured BYOK endpoint, extracts reasoning tokens from
    SSE chunks, and injects heartbeat comments during long reasoning cycles.
    """

    def __init__(self, config: FidelityConfig) -> None:
        self._config = config

    @property
    def config(self) -> FidelityConfig:
        """Return the current fidelity configuration."""
        return self._config

    def intercept_request(
        self, headers: dict[str, str], body: dict[str, Any]
    ) -> dict[str, str]:
        """Inject spoofed headers mimicking Pro/Unlimited state.

        This method returns a new dict containing all original headers
        plus the configured spoof headers. The operation is idempotent:
        applying it multiple times produces the same result as applying
        it once.

        Requirements:
            1.1 — Injects X-Cursor-Plan: pro
            1.2 — Injects X-Cursor-Tier: unlimited
            1.3 — Preserves all original headers
            1.4 — Idempotent (applying twice == applying once)
            1.5 — Routes to configured BYOK endpoint

        Args:
            headers: The original request headers.
            body: The request body (unused in header injection but
                available for future routing logic).

        Returns:
            A new dict with all original headers plus spoofed headers.
            Spoof headers overwrite any existing values for the same keys,
            ensuring idempotency.
        """
        # Start with a copy of the original headers to preserve them
        result = dict(headers)

        # Inject (or overwrite) spoof headers — idempotent because
        # re-applying the same key-value pairs produces identical output
        for key, value in self._config.spoof_headers.items():
            result[key] = value

        return result

    def extract_reasoning_stream(self, chunk: dict[str, Any]) -> str | None:
        """Extract reasoning_content from an SSE chunk as a distinct stream.

        When reasoning_passthrough is enabled, this method extracts the
        `reasoning_content` field from the SSE chunk's delta and returns
        it without modification. If the field is absent or reasoning
        pass-through is disabled, returns None without raising an error.

        Requirements:
            2.1 — Extracts reasoning_content and emits as distinct stream
            2.2 — Forwards all reasoning tokens without modification
            2.3 — Handles missing reasoning_content gracefully (returns None)

        Args:
            chunk: A parsed SSE chunk dict, typically containing a
                'choices' list with delta objects that may include
                a 'reasoning_content' field.

        Returns:
            The reasoning content string if present and pass-through is
            enabled, None otherwise.
        """
        if not self._config.reasoning_stream_enabled:
            return None

        # Navigate the standard SSE chunk structure:
        # {"choices": [{"delta": {"reasoning_content": "..."}}]}
        choices = chunk.get("choices")
        if not choices or not isinstance(choices, list):
            return None

        # Extract from the first choice's delta
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return None

        delta = first_choice.get("delta")
        if not isinstance(delta, dict):
            return None

        reasoning_content = delta.get("reasoning_content")
        if reasoning_content is None:
            return None

        # Return the reasoning content without modification (Req 2.2)
        return reasoning_content

    def get_endpoint(self) -> str:
        """Return the configured BYOK endpoint for routing.

        Requirement 1.5: Route inference requests to the configured
        BYOK endpoint.

        Returns:
            The BYOK endpoint URL string.
        """
        return self._config.byok_endpoint

    async def heartbeat_wrapper(
        self, stream: AsyncIterator[bytes]
    ) -> AsyncIterator[bytes]:
        """Wrap an upstream SSE stream with keep-alive heartbeat injection.

        This async generator monitors the upstream stream and injects SSE
        comment heartbeats (`: heartbeat\\n\\n`) whenever no data is received
        within the configured heartbeat interval. All original chunks are
        yielded unchanged and in order.

        The stream is closed gracefully after 60 seconds of receiving no
        real data from upstream, even if heartbeats are being injected.

        Requirements:
            3.1 — Injects heartbeat when no data within interval
            3.2 — Preserves all original chunks in order without modification
            3.3 — Closes stream after 60s of no real data
            3.4 — Time since last emitted byte never exceeds 2× interval

        Args:
            stream: An async iterator yielding SSE-formatted bytes from
                the upstream DeepSeek API.

        Yields:
            Original upstream bytes chunks interspersed with heartbeat
            comments as needed to keep the connection alive.
        """
        heartbeat_interval = self._config.heartbeat_interval_seconds
        max_no_data_seconds = self._config.max_no_data_seconds
        heartbeat_bytes = b": heartbeat\n\n"

        last_real_data_time = time.monotonic()
        stream_iter = stream.__aiter__()
        pending_next: asyncio.Task[bytes] | None = None

        try:
            while True:
                now = time.monotonic()
                elapsed_since_real_data = now - last_real_data_time

                # Req 3.3: Close stream after 60s of no real data
                if elapsed_since_real_data >= max_no_data_seconds:
                    return

                # Create a task for the next chunk if we don't have one pending
                if pending_next is None:
                    pending_next = asyncio.ensure_future(
                        stream_iter.__anext__()
                    )

                try:
                    chunk = await asyncio.wait_for(
                        asyncio.shield(pending_next),
                        timeout=heartbeat_interval,
                    )
                    # Task completed successfully — clear it
                    pending_next = None
                    # Req 3.2: Yield original chunk unchanged
                    yield chunk
                    last_real_data_time = time.monotonic()
                except asyncio.TimeoutError:
                    # Req 3.1: No data within interval, inject heartbeat
                    # Req 3.4: Ensures emitted byte gap <= 2× interval
                    yield heartbeat_bytes
                except StopAsyncIteration:
                    # Upstream stream ended normally
                    pending_next = None
                    return
        finally:
            # Clean up any pending task
            if pending_next is not None and not pending_next.done():
                pending_next.cancel()
                try:
                    await pending_next
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
