"""Property-based tests for RAP Pipeline Orchestrator.

**Validates: Requirements 11.3, 11.4, 11.5**

Property 19: Pipeline Phase Skip on Disable
For any combination of phase enable/disable flags, disabled phases are never executed.

Property 20: Pipeline Graceful Degradation on Phase Failure
For any phase that raises an exception, the pipeline continues with remaining phases
and returns valid data.

Property 21: Message Format Preservation Through Pipeline
For any valid OpenAI chat message list, the pipeline output maintains valid format
(messages have role + content or tool_calls).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from deepseek_cursor_proxy.rap.config import RAPConfig
from deepseek_cursor_proxy.rap.pipeline import PipelineOrchestrator


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

phase_flags = st.fixed_dictionaries({
    "phase_bridge": st.booleans(),
    "phase_compression": st.booleans(),
    "phase_retrieval": st.booleans(),
    "phase_security": st.booleans(),
})

# Strategy for valid OpenAI chat message content
message_content = st.text(min_size=0, max_size=200)

# Strategy for a message with role + content
content_message = st.fixed_dictionaries({
    "role": st.sampled_from(["system", "user", "assistant"]),
    "content": message_content,
})

# Strategy for a message with role + tool_calls (assistant only)
tool_call_message = st.fixed_dictionaries({
    "role": st.just("assistant"),
    "tool_calls": st.lists(
        st.fixed_dictionaries({
            "id": st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_"),
            "function": st.fixed_dictionaries({
                "name": st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz_"),
                "arguments": st.just("{}"),
            }),
        }),
        min_size=1,
        max_size=3,
    ),
})

# Strategy for any valid message (either content or tool_calls)
valid_message = st.one_of(content_message, tool_call_message)

# Strategy for a valid message list (at least one message, ending with user message)
valid_message_list = st.lists(valid_message, min_size=1, max_size=10)

# Strategy for a valid request
valid_request = st.builds(
    lambda messages: {"model": "deepseek-v4-flash", "messages": messages},
    messages=valid_message_list,
)

# Strategy for a valid response
valid_response = st.builds(
    lambda messages: {
        "id": "chatcmpl-test",
        "choices": [
            {"index": i, "message": msg, "finish_reason": "stop"}
            for i, msg in enumerate(messages)
        ],
    },
    messages=st.lists(content_message, min_size=1, max_size=3),
)

# Strategy for which phases should fail (subset of phase names)
failing_phases = st.lists(
    st.sampled_from(["fidelity", "security", "toon", "retrieval"]),
    min_size=1,
    max_size=4,
    unique=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides: Any) -> RAPConfig:
    """Create a RAPConfig with optional overrides."""
    return RAPConfig(**overrides)


def _wire_tracking_modules(
    pipeline: PipelineOrchestrator,
) -> dict[str, MagicMock]:
    """Wire mock modules that track whether they were called.

    Returns a dict of phase_name -> mock for inspection.
    """
    mocks: dict[str, MagicMock] = {}

    # Fidelity
    mock_fidelity = MagicMock()
    mock_fidelity.intercept_request.side_effect = lambda r: r
    mock_fidelity.health_check.return_value = "healthy"
    pipeline._fidelity = mock_fidelity
    mocks["fidelity"] = mock_fidelity

    # Security
    mock_security = MagicMock()
    mock_security.scan_outbound.side_effect = lambda r: (r, [])
    mock_security.scan_inbound.side_effect = lambda r: (r, [])
    mock_security.health_check.return_value = "healthy"
    pipeline._security = mock_security
    mocks["security"] = mock_security

    # TOON
    mock_toon = MagicMock()
    mock_toon.compress.side_effect = lambda msgs: msgs
    mock_toon.rehydrate.side_effect = lambda c: c
    mock_toon.health_check.return_value = "healthy"
    pipeline._toon = mock_toon
    mocks["toon"] = mock_toon

    # Retrieval
    mock_retrieval = MagicMock()
    mock_retrieval.build_reduced_context.side_effect = lambda q, msgs: msgs
    mock_retrieval.health_check.return_value = "healthy"
    pipeline._retrieval = mock_retrieval
    mocks["retrieval"] = mock_retrieval

    return mocks


def _wire_failing_modules(
    pipeline: PipelineOrchestrator,
    phases_to_fail: list[str],
) -> dict[str, MagicMock]:
    """Wire mock modules where specified phases raise exceptions.

    Non-failing phases pass data through normally.
    Returns a dict of phase_name -> mock for inspection.
    """
    mocks = _wire_tracking_modules(pipeline)

    if "fidelity" in phases_to_fail:
        mocks["fidelity"].intercept_request.side_effect = RuntimeError("fidelity failure")

    if "security" in phases_to_fail:
        mocks["security"].scan_outbound.side_effect = RuntimeError("security failure")
        mocks["security"].scan_inbound.side_effect = RuntimeError("security failure")

    if "toon" in phases_to_fail:
        mocks["toon"].compress.side_effect = RuntimeError("toon failure")
        mocks["toon"].rehydrate.side_effect = RuntimeError("toon failure")

    if "retrieval" in phases_to_fail:
        mocks["retrieval"].build_reduced_context.side_effect = RuntimeError("retrieval failure")

    return mocks


# ---------------------------------------------------------------------------
# Property 19: Pipeline Phase Skip on Disable
# ---------------------------------------------------------------------------

class TestProperty19PhaseSkipOnDisable:
    """For any combination of phase enable/disable flags, disabled phases
    are never executed.

    **Validates: Requirements 11.3**
    """

    @given(flags=phase_flags, request=valid_request)
    @settings(max_examples=20)
    def test_disabled_phases_not_called_on_request(
        self, flags: dict[str, bool], request: dict[str, Any]
    ) -> None:
        """Disabled phases are never invoked during process_request."""
        config = _make_config(**flags)
        pipeline = PipelineOrchestrator(config)
        mocks = _wire_tracking_modules(pipeline)

        pipeline.process_request(request)

        # Check that disabled phases were not called
        if not flags["phase_bridge"]:
            mocks["fidelity"].intercept_request.assert_not_called()
        if not flags["phase_security"]:
            mocks["security"].scan_outbound.assert_not_called()
        if not flags["phase_compression"]:
            mocks["toon"].compress.assert_not_called()
        if not flags["phase_retrieval"]:
            mocks["retrieval"].build_reduced_context.assert_not_called()

    @given(flags=phase_flags, response=valid_response)
    @settings(max_examples=20)
    def test_disabled_phases_not_called_on_response(
        self, flags: dict[str, bool], response: dict[str, Any]
    ) -> None:
        """Disabled phases are never invoked during process_response."""
        config = _make_config(**flags)
        pipeline = PipelineOrchestrator(config)
        mocks = _wire_tracking_modules(pipeline)

        pipeline.process_response(response)

        # Stream health is gated by phase_bridge
        # TOON rehydrate is gated by phase_compression
        # Security inbound is gated by phase_security
        if not flags["phase_compression"]:
            mocks["toon"].rehydrate.assert_not_called()
        if not flags["phase_security"]:
            mocks["security"].scan_inbound.assert_not_called()

    @given(flags=phase_flags, request=valid_request)
    @settings(max_examples=20)
    def test_all_disabled_returns_input_unchanged(
        self, flags: dict[str, bool], request: dict[str, Any]
    ) -> None:
        """When all phases are disabled, the output equals the input."""
        # Force all phases disabled
        all_disabled = {
            "phase_bridge": False,
            "phase_compression": False,
            "phase_retrieval": False,
            "phase_security": False,
        }
        config = _make_config(**all_disabled)
        pipeline = PipelineOrchestrator(config)

        result = pipeline.process_request(request)
        assert result == request


# ---------------------------------------------------------------------------
# Property 20: Pipeline Graceful Degradation on Phase Failure
# ---------------------------------------------------------------------------

class TestProperty20GracefulDegradation:
    """For any phase that raises an exception, the pipeline continues with
    remaining phases and returns valid data.

    **Validates: Requirements 11.4**
    """

    @given(
        phases_to_fail=failing_phases,
        request=valid_request,
    )
    @settings(max_examples=20)
    def test_failing_phases_do_not_crash_request_pipeline(
        self, phases_to_fail: list[str], request: dict[str, Any]
    ) -> None:
        """Pipeline does not raise even when phases fail."""
        config = _make_config(
            phase_bridge=True,
            phase_compression=True,
            phase_retrieval=True,
            phase_security=True,
        )
        pipeline = PipelineOrchestrator(config)
        _wire_failing_modules(pipeline, phases_to_fail)

        # Should not raise
        result = pipeline.process_request(request)
        assert result is not None

    @given(
        phases_to_fail=failing_phases,
        response=valid_response,
    )
    @settings(max_examples=20)
    def test_failing_phases_do_not_crash_response_pipeline(
        self, phases_to_fail: list[str], response: dict[str, Any]
    ) -> None:
        """Pipeline does not raise even when phases fail on response."""
        config = _make_config(
            phase_bridge=True,
            phase_compression=True,
            phase_security=True,
        )
        pipeline = PipelineOrchestrator(config)
        _wire_failing_modules(pipeline, phases_to_fail)

        # Should not raise
        result = pipeline.process_response(response)
        assert result is not None

    @given(
        phases_to_fail=failing_phases,
        request=valid_request,
    )
    @settings(max_examples=20)
    def test_non_failing_phases_still_execute(
        self, phases_to_fail: list[str], request: dict[str, Any]
    ) -> None:
        """Phases that don't fail are still called after a failing phase."""
        config = _make_config(
            phase_bridge=True,
            phase_compression=True,
            phase_retrieval=True,
            phase_security=True,
        )
        pipeline = PipelineOrchestrator(config)
        mocks = _wire_failing_modules(pipeline, phases_to_fail)

        pipeline.process_request(request)

        # Verify non-failing phases were still called
        if "security" not in phases_to_fail:
            mocks["security"].scan_outbound.assert_called()
        if "toon" not in phases_to_fail:
            mocks["toon"].compress.assert_called()
        # Retrieval may not be called if messages list is empty or no user message
        # so we only check it if there's a user message at the end
        messages = request.get("messages", [])
        if (
            "retrieval" not in phases_to_fail
            and messages
            and messages[-1].get("role") == "user"
            and messages[-1].get("content", "")
        ):
            mocks["retrieval"].build_reduced_context.assert_called()

    @given(request=valid_request)
    @settings(max_examples=10)
    def test_result_is_valid_dict_even_on_all_failures(
        self, request: dict[str, Any]
    ) -> None:
        """Even when all phases fail, the result is a valid dict."""
        config = _make_config(
            phase_bridge=True,
            phase_compression=True,
            phase_retrieval=True,
            phase_security=True,
        )
        pipeline = PipelineOrchestrator(config)
        _wire_failing_modules(
            pipeline, ["fidelity", "security", "toon", "retrieval"]
        )

        result = pipeline.process_request(request)
        assert isinstance(result, dict)
        # The original request should be returned when all phases fail
        assert "messages" in result or "model" in result


# ---------------------------------------------------------------------------
# Property 21: Message Format Preservation Through Pipeline
# ---------------------------------------------------------------------------

class TestProperty21MessageFormatPreservation:
    """For any valid OpenAI chat message list, the pipeline output maintains
    valid format (messages have role + content or tool_calls).

    **Validates: Requirements 11.5**
    """

    @given(flags=phase_flags, request=valid_request)
    @settings(max_examples=20)
    def test_request_messages_preserve_format(
        self, flags: dict[str, bool], request: dict[str, Any]
    ) -> None:
        """Output messages always have role + (content or tool_calls)."""
        config = _make_config(**flags)
        pipeline = PipelineOrchestrator(config)
        _wire_tracking_modules(pipeline)

        result = pipeline.process_request(request)

        # Verify all messages in result maintain valid format
        messages = result.get("messages", [])
        for msg in messages:
            assert "role" in msg, f"Message missing 'role': {msg}"
            assert (
                "content" in msg or "tool_calls" in msg
            ), f"Message must have 'content' or 'tool_calls': {msg}"

    @given(flags=phase_flags, response=valid_response)
    @settings(max_examples=20)
    def test_response_messages_preserve_format(
        self, flags: dict[str, bool], response: dict[str, Any]
    ) -> None:
        """Output response messages always have role + (content or tool_calls)."""
        config = _make_config(**flags)
        pipeline = PipelineOrchestrator(config)
        _wire_tracking_modules(pipeline)

        result = pipeline.process_response(response)

        # Verify all choice messages maintain valid format
        for choice in result.get("choices", []):
            msg = choice.get("message", {})
            assert "role" in msg, f"Response message missing 'role': {msg}"
            assert (
                "content" in msg or "tool_calls" in msg
            ), f"Response message must have 'content' or 'tool_calls': {msg}"

    @given(request=valid_request)
    @settings(max_examples=10)
    def test_format_preserved_even_on_phase_failures(
        self, request: dict[str, Any]
    ) -> None:
        """Message format is preserved even when phases fail."""
        config = _make_config(
            phase_bridge=True,
            phase_compression=True,
            phase_retrieval=True,
            phase_security=True,
        )
        pipeline = PipelineOrchestrator(config)
        _wire_failing_modules(
            pipeline, ["fidelity", "security", "toon", "retrieval"]
        )

        result = pipeline.process_request(request)

        messages = result.get("messages", [])
        for msg in messages:
            assert "role" in msg, f"Message missing 'role': {msg}"
            assert (
                "content" in msg or "tool_calls" in msg
            ), f"Message must have 'content' or 'tool_calls': {msg}"

    @given(messages=valid_message_list)
    @settings(max_examples=20)
    def test_message_roles_are_valid_strings(
        self, messages: list[dict[str, Any]]
    ) -> None:
        """All message roles remain valid strings through the pipeline."""
        request = {"model": "deepseek-v4-flash", "messages": messages}
        config = _make_config(
            phase_bridge=False,
            phase_compression=False,
            phase_retrieval=False,
            phase_security=False,
        )
        pipeline = PipelineOrchestrator(config)

        result = pipeline.process_request(request)

        for msg in result.get("messages", []):
            assert isinstance(msg["role"], str)
            assert len(msg["role"]) > 0
