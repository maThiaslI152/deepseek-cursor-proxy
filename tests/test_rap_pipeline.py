"""Unit tests for the RAP Pipeline Orchestrator.

Tests cover:
- Phase ordering (Requirements 11.1, 11.2)
- Phase enable/disable (Requirement 11.3)
- Graceful degradation on phase failure (Requirement 11.4)
- Message format preservation (Requirement 11.5)
- Health check reporting (Requirement 13.5)
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

from deepseek_cursor_proxy.rap.config import RAPConfig
from deepseek_cursor_proxy.rap.pipeline import PipelineOrchestrator


def _make_config(**overrides: Any) -> RAPConfig:
    """Create a RAPConfig with optional overrides."""
    return RAPConfig(**overrides)


def _valid_request() -> dict[str, Any]:
    """Return a minimal valid OpenAI chat completion request."""
    return {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"},
        ],
    }


def _valid_response() -> dict[str, Any]:
    """Return a minimal valid OpenAI chat completion response."""
    return {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there!"},
                "finish_reason": "stop",
            }
        ],
    }


class TestPipelineConstruction(unittest.TestCase):
    """Test that PipelineOrchestrator can be constructed."""

    def test_construction_with_default_config(self) -> None:
        config = _make_config()
        pipeline = PipelineOrchestrator(config)
        self.assertIsNotNone(pipeline)

    def test_construction_with_all_phases_enabled(self) -> None:
        config = _make_config(
            phase_bridge=True,
            phase_compression=True,
            phase_retrieval=True,
            phase_security=True,
        )
        pipeline = PipelineOrchestrator(config)
        self.assertIsNotNone(pipeline)


class TestProcessRequestPhaseSkip(unittest.TestCase):
    """Requirement 11.3: Disabled phases are skipped."""

    def test_all_phases_disabled_returns_request_unchanged(self) -> None:
        config = _make_config(
            phase_bridge=False,
            phase_compression=False,
            phase_retrieval=False,
            phase_security=False,
        )
        pipeline = PipelineOrchestrator(config)
        request = _valid_request()
        result = pipeline.process_request(request)
        self.assertEqual(result, request)

    def test_disabled_phase_not_called(self) -> None:
        config = _make_config(phase_bridge=False)
        pipeline = PipelineOrchestrator(config)
        # Mock fidelity module to verify it's not called
        mock_fidelity = MagicMock()
        pipeline._fidelity = mock_fidelity

        pipeline.process_request(_valid_request())
        mock_fidelity.intercept_request.assert_not_called()

    def test_enabled_phase_is_called(self) -> None:
        config = _make_config(phase_security=True)
        pipeline = PipelineOrchestrator(config)
        mock_security = MagicMock()
        mock_security.scan_outbound.return_value = (_valid_request(), [])
        pipeline._security = mock_security

        pipeline.process_request(_valid_request())
        mock_security.scan_outbound.assert_called_once()


class TestProcessRequestPhaseOrder(unittest.TestCase):
    """Requirement 11.1: Outbound order is Fidelity → Security → TOON → Retrieval."""

    def test_outbound_phase_order(self) -> None:
        config = _make_config(
            phase_bridge=True,
            phase_security=True,
            phase_compression=True,
            phase_retrieval=True,
        )
        pipeline = PipelineOrchestrator(config)

        call_order: list[str] = []

        # Create mocks that record call order
        mock_fidelity = MagicMock()
        mock_fidelity.intercept_request.side_effect = lambda h, b: (
            call_order.append("fidelity") or h
        )

        mock_security = MagicMock()
        mock_security.scan_outbound.side_effect = lambda r: (
            call_order.append("security") or (r, [])
        )

        mock_toon = MagicMock()
        mock_toon.compress.side_effect = lambda msgs: (
            call_order.append("toon") or msgs
        )

        mock_retrieval = MagicMock()
        mock_retrieval.build_reduced_context.side_effect = lambda q, msgs: (
            call_order.append("retrieval") or msgs
        )

        pipeline._fidelity = mock_fidelity
        pipeline._security = mock_security
        pipeline._toon = mock_toon
        pipeline._retrieval = mock_retrieval

        pipeline.process_request(_valid_request())

        self.assertEqual(call_order, ["fidelity", "security", "toon", "retrieval"])


class TestProcessResponsePhaseOrder(unittest.TestCase):
    """Requirement 11.2: Inbound order is Stream Health → TOON → Security."""

    def test_inbound_phase_order(self) -> None:
        config = _make_config(
            phase_bridge=True,
            phase_compression=True,
            phase_security=True,
        )
        pipeline = PipelineOrchestrator(config)

        call_order: list[str] = []

        mock_fidelity = MagicMock()
        pipeline._fidelity = mock_fidelity
        # Stream health phase just passes through when fidelity is set
        # We track it via the phase being entered

        mock_toon = MagicMock()
        mock_toon.rehydrate.side_effect = lambda c: (
            call_order.append("toon") or c
        )

        mock_security = MagicMock()
        mock_security.scan_inbound.side_effect = lambda r: (
            call_order.append("security") or (r, [])
        )

        pipeline._toon = mock_toon
        pipeline._security = mock_security

        pipeline.process_response(_valid_response())

        self.assertEqual(call_order, ["toon", "security"])


class TestProcessResponsePhaseSkip(unittest.TestCase):
    """Requirement 11.3: Disabled phases are skipped in response pipeline."""

    def test_all_phases_disabled_returns_response_unchanged(self) -> None:
        config = _make_config(
            phase_bridge=False,
            phase_compression=False,
            phase_security=False,
        )
        pipeline = PipelineOrchestrator(config)
        response = _valid_response()
        result = pipeline.process_response(response)
        self.assertEqual(result, response)


class TestGracefulDegradation(unittest.TestCase):
    """Requirement 11.4: Failing phases are skipped gracefully."""

    def test_failing_phase_does_not_crash_pipeline(self) -> None:
        config = _make_config(phase_security=True, phase_bridge=False)
        pipeline = PipelineOrchestrator(config)

        mock_security = MagicMock()
        mock_security.scan_outbound.side_effect = RuntimeError("Service unavailable")
        pipeline._security = mock_security

        request = _valid_request()
        # Should not raise — graceful degradation
        result = pipeline.process_request(request)
        self.assertEqual(result, request)

    def test_failing_phase_allows_subsequent_phases(self) -> None:
        config = _make_config(
            phase_bridge=True,
            phase_security=True,
            phase_compression=True,
        )
        pipeline = PipelineOrchestrator(config)

        # Fidelity fails
        mock_fidelity = MagicMock()
        mock_fidelity.intercept_request.side_effect = RuntimeError("Fidelity error")
        pipeline._fidelity = mock_fidelity

        # Security should still be called
        mock_security = MagicMock()
        mock_security.scan_outbound.return_value = (_valid_request(), [])
        pipeline._security = mock_security

        # TOON should still be called
        mock_toon = MagicMock()
        mock_toon.compress.return_value = _valid_request()["messages"]
        pipeline._toon = mock_toon

        pipeline.process_request(_valid_request())

        mock_security.scan_outbound.assert_called_once()
        mock_toon.compress.assert_called_once()

    def test_inbound_failing_phase_graceful(self) -> None:
        config = _make_config(phase_compression=True, phase_security=True)
        pipeline = PipelineOrchestrator(config)

        mock_toon = MagicMock()
        mock_toon.rehydrate.side_effect = RuntimeError("Rehydration failed")
        pipeline._toon = mock_toon

        mock_security = MagicMock()
        mock_security.scan_inbound.return_value = (_valid_response(), [])
        pipeline._security = mock_security

        response = _valid_response()
        result = pipeline.process_response(response)
        # Security should still be called despite TOON failure
        mock_security.scan_inbound.assert_called_once()
        self.assertIsNotNone(result)


class TestMessageFormatPreservation(unittest.TestCase):
    """Requirement 11.5: Messages maintain valid OpenAI chat format."""

    def test_request_preserves_message_structure(self) -> None:
        config = _make_config(
            phase_bridge=False,
            phase_compression=False,
            phase_retrieval=False,
            phase_security=False,
        )
        pipeline = PipelineOrchestrator(config)
        request = _valid_request()
        result = pipeline.process_request(request)

        # Verify messages still have role and content
        for msg in result["messages"]:
            self.assertIn("role", msg)
            self.assertTrue(
                "content" in msg or "tool_calls" in msg,
                "Message must have 'content' or 'tool_calls'",
            )

    def test_response_preserves_choice_structure(self) -> None:
        config = _make_config(
            phase_bridge=False,
            phase_compression=False,
            phase_security=False,
        )
        pipeline = PipelineOrchestrator(config)
        response = _valid_response()
        result = pipeline.process_response(response)

        for choice in result["choices"]:
            msg = choice["message"]
            self.assertIn("role", msg)
            self.assertTrue(
                "content" in msg or "tool_calls" in msg,
                "Message must have 'content' or 'tool_calls'",
            )

    def test_tool_calls_message_preserved(self) -> None:
        """Messages with tool_calls instead of content are valid."""
        config = _make_config(
            phase_bridge=False,
            phase_compression=False,
            phase_retrieval=False,
            phase_security=False,
        )
        pipeline = PipelineOrchestrator(config)
        request = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "test"}}]},
                {"role": "user", "content": "Continue"},
            ],
        }
        result = pipeline.process_request(request)
        self.assertEqual(result["messages"][0]["tool_calls"][0]["id"], "call_1")


class TestHealthCheck(unittest.TestCase):
    """Requirement 13.5: Health check reports component status."""

    def test_health_check_returns_dict(self) -> None:
        config = _make_config()
        pipeline = PipelineOrchestrator(config)
        health = pipeline.health_check()
        self.assertIsInstance(health, dict)

    def test_health_check_contains_pipeline_status(self) -> None:
        config = _make_config()
        pipeline = PipelineOrchestrator(config)
        health = pipeline.health_check()
        self.assertIn("pipeline", health)

    def test_health_check_contains_phase_statuses(self) -> None:
        config = _make_config()
        pipeline = PipelineOrchestrator(config)
        health = pipeline.health_check()
        self.assertIn("phases", health)
        self.assertIn("fidelity", health["phases"])
        self.assertIn("security", health["phases"])
        self.assertIn("toon", health["phases"])
        self.assertIn("retrieval", health["phases"])

    def test_health_check_contains_config_flags(self) -> None:
        config = _make_config()
        pipeline = PipelineOrchestrator(config)
        health = pipeline.health_check()
        self.assertIn("config", health)
        self.assertIn("phase_bridge", health["config"])

    def test_disabled_phase_reports_disabled(self) -> None:
        config = _make_config(phase_compression=False)
        pipeline = PipelineOrchestrator(config)
        health = pipeline.health_check()
        self.assertEqual(health["phases"]["toon"], "disabled")

    def test_enabled_phase_without_module_reports_unavailable(self) -> None:
        config = _make_config(phase_bridge=True)
        pipeline = PipelineOrchestrator(config)
        # Force fidelity to None to simulate unavailable state
        pipeline._fidelity = None
        health = pipeline.health_check()
        self.assertEqual(health["phases"]["fidelity"], "unavailable")

    def test_enabled_phase_with_module_reports_healthy(self) -> None:
        config = _make_config(phase_security=True)
        pipeline = PipelineOrchestrator(config)
        mock_security = MagicMock()
        mock_security.health_check.return_value = "healthy"
        pipeline._security = mock_security
        health = pipeline.health_check()
        self.assertEqual(health["phases"]["security"], "healthy")

    def test_pipeline_degraded_when_enabled_component_unavailable(self) -> None:
        config = _make_config(phase_bridge=True)
        pipeline = PipelineOrchestrator(config)
        # Force fidelity to None to simulate unavailable state
        pipeline._fidelity = None
        health = pipeline.health_check()
        self.assertEqual(health["pipeline"], "degraded")

    def test_pipeline_healthy_when_all_enabled_components_available(self) -> None:
        config = _make_config(
            phase_bridge=True,
            phase_compression=False,
            phase_retrieval=False,
            phase_security=False,
        )
        pipeline = PipelineOrchestrator(config)
        mock_fidelity = MagicMock()
        mock_fidelity.health_check.return_value = "healthy"
        pipeline._fidelity = mock_fidelity
        health = pipeline.health_check()
        self.assertEqual(health["pipeline"], "healthy")

    def test_health_check_with_module_health_check_method(self) -> None:
        config = _make_config(phase_retrieval=True)
        pipeline = PipelineOrchestrator(config)
        mock_retrieval = MagicMock()
        mock_retrieval.health_check.return_value = "healthy"
        pipeline._retrieval = mock_retrieval
        health = pipeline.health_check()
        self.assertEqual(health["phases"]["retrieval"], "healthy")

    def test_health_check_with_module_health_check_failure(self) -> None:
        config = _make_config(phase_retrieval=True)
        pipeline = PipelineOrchestrator(config)
        mock_retrieval = MagicMock()
        mock_retrieval.health_check.side_effect = RuntimeError("Connection refused")
        pipeline._retrieval = mock_retrieval
        health = pipeline.health_check()
        self.assertEqual(health["phases"]["retrieval"], "unhealthy")


class TestNullModulePassthrough(unittest.TestCase):
    """When modules are None (not yet wired or forced to None), data passes through unchanged."""

    def test_request_passthrough_with_null_modules(self) -> None:
        config = _make_config(
            phase_bridge=True,
            phase_security=True,
            phase_compression=True,
            phase_retrieval=True,
        )
        pipeline = PipelineOrchestrator(config)
        # Force all modules to None to test passthrough behavior
        pipeline._fidelity = None
        pipeline._security = None
        pipeline._toon = None
        pipeline._retrieval = None
        request = _valid_request()
        result = pipeline.process_request(request)
        self.assertEqual(result, request)

    def test_response_passthrough_with_null_modules(self) -> None:
        config = _make_config(
            phase_bridge=True,
            phase_compression=True,
            phase_security=True,
        )
        pipeline = PipelineOrchestrator(config)
        # Force all modules to None to test passthrough behavior
        pipeline._fidelity = None
        pipeline._security = None
        pipeline._toon = None
        pipeline._retrieval = None
        response = _valid_response()
        result = pipeline.process_response(response)
        self.assertEqual(result, response)


if __name__ == "__main__":
    unittest.main()
