"""Pipeline Orchestrator for the RAP middleware stack.

Coordinates the execution order of all middleware modules with
graceful degradation and phase enable/disable control.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 13.5
"""

from __future__ import annotations

import logging
from typing import Any

from deepseek_cursor_proxy.rap.config import RAPConfig
from deepseek_cursor_proxy.rap.fidelity import FidelityConfig, FidelityModule
from deepseek_cursor_proxy.rap.retrieval import RetrievalLayer
from deepseek_cursor_proxy.rap.security import SecurityConfig, SecurityGateway
from deepseek_cursor_proxy.rap.toon import TOONEngine

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Coordinates the RAP middleware pipeline.

    Manages the execution order of pipeline phases for both outbound
    requests and inbound responses. Each phase can be independently
    enabled/disabled via configuration, and failures in any phase
    trigger graceful degradation (skip and continue).

    Outbound order (Requirement 11.1):
        Fidelity → Security (outbound) → TOON (compress) → Retrieval

    Inbound order (Requirement 11.2):
        Stream Health → TOON (re-hydrate) → Security (inbound scan)
    """

    def __init__(self, config: RAPConfig) -> None:
        self._config = config

        # Wire modules based on configuration (Requirement 11.1)
        self._fidelity: Any = None
        self._security: Any = None
        self._toon: Any = None
        self._retrieval: Any = None

        # Instantiate modules for enabled phases
        if config.phase_bridge:
            self._fidelity = FidelityModule(FidelityConfig(
                spoof_headers={
                    "X-Cursor-Plan": "pro",
                    "X-Cursor-Tier": "unlimited",
                },
                heartbeat_interval_seconds=config.heartbeat_interval,
                reasoning_stream_enabled=config.reasoning_passthrough,
                byok_endpoint=config.upstream_base_url,
            ))

        if config.phase_security:
            self._security = SecurityGateway(SecurityConfig(
                redaction_enabled=config.redaction_enabled,
                cve_scanning_enabled=config.cve_scanning_enabled,
                audit_logging_enabled=True,
                audit_db_path=str(config.audit_db_path),
                local_security_model_url=config.security_model_url,
                security_model_name=config.security_model_name,
                entropy_threshold=config.entropy_threshold,
            ))

        if config.phase_compression:
            self._toon = TOONEngine(config)

        if config.phase_retrieval:
            self._retrieval = RetrievalLayer(config)

    # ------------------------------------------------------------------
    # Outbound pipeline
    # ------------------------------------------------------------------

    @property
    def last_outbound_actions(self) -> list[str]:
        """Return the list of actions taken during the last process_request() call."""
        return getattr(self, "_last_outbound_actions", [])

    def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run the full outbound pipeline on a request.

        Phase order (Requirement 11.1):
            1. Fidelity — header injection
            2. Security — outbound secret redaction
            3. TOON — structured block compression
            4. Retrieval — context reduction via vector search

        Each phase is wrapped in try/except for graceful degradation
        (Requirement 11.4). Disabled phases are skipped (Requirement 11.3).
        The message list maintains valid OpenAI chat format at every
        stage (Requirement 11.5).
        """
        result = request
        self._last_outbound_actions: list[str] = []

        # Phase 1: Fidelity (header injection)
        if self._config.phase_bridge:
            result = self._run_phase("fidelity", self._phase_fidelity_outbound, result)
            if result.get("_headers"):
                self._last_outbound_actions.append("headers")

        # Phase 2: Security (outbound redaction)
        if self._config.phase_security:
            before = result.get("messages", [])
            result = self._run_phase("security_outbound", self._phase_security_outbound, result)
            # Check if any redaction happened
            after_content = "".join(
                m.get("content", "") for m in result.get("messages", []) if isinstance(m.get("content"), str)
            )
            if "[REDACTED]" in after_content:
                self._last_outbound_actions.append("redacted")

        # Phase 3: TOON (compression)
        if self._config.phase_compression:
            before_size = sum(
                len(m.get("content", "")) for m in result.get("messages", []) if isinstance(m.get("content"), str)
            )
            result = self._run_phase("toon_compress", self._phase_toon_compress, result)
            after_size = sum(
                len(m.get("content", "")) for m in result.get("messages", []) if isinstance(m.get("content"), str)
            )
            if after_size < before_size:
                ratio = round((1 - after_size / before_size) * 100) if before_size > 0 else 0
                self._last_outbound_actions.append(f"compressed({ratio}%)")

        # Phase 4: Retrieval (context reduction)
        if self._config.phase_retrieval:
            before_msgs = len(result.get("messages", []))
            result = self._run_phase("retrieval", self._phase_retrieval_outbound, result)
            after_msgs = len(result.get("messages", []))
            if after_msgs < before_msgs:
                self._last_outbound_actions.append(f"retrieved({before_msgs}→{after_msgs}msgs)")

        return result

    # ------------------------------------------------------------------
    # Inbound pipeline
    # ------------------------------------------------------------------

    def process_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Run the full inbound pipeline on a response.

        Phase order (Requirement 11.2):
            1. Stream Health — heartbeat monitoring
            2. TOON — re-hydrate compressed content
            3. Security — inbound CVE scanning

        Each phase is wrapped in try/except for graceful degradation
        (Requirement 11.4). Disabled phases are skipped (Requirement 11.3).
        """
        result = response

        # Phase 0: HITL interception (convert AskQuestion tool_calls to content)
        if self._config.phase_bridge:
            result = self._run_phase("hitl_intercept", self._phase_hitl_intercept, result)

        # Phase 1: Stream Health (heartbeat / reasoning extraction)
        if self._config.phase_bridge:
            result = self._run_phase("stream_health", self._phase_stream_health, result)

        # Phase 2: TOON (re-hydration)
        if self._config.phase_compression:
            result = self._run_phase("toon_rehydrate", self._phase_toon_rehydrate, result)

        # Phase 3: Security (inbound CVE scan)
        if self._config.phase_security:
            result = self._run_phase("security_inbound", self._phase_security_inbound, result)

        return result

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Return the health status of each pipeline component.

        Requirement 13.5: Expose a health check reporting the status
        of each component (Qdrant, LM Studio, audit database).
        """
        status: dict[str, Any] = {
            "pipeline": "healthy",
            "phases": {
                "fidelity": self._check_component_health("fidelity"),
                "security": self._check_component_health("security"),
                "toon": self._check_component_health("toon"),
                "retrieval": self._check_component_health("retrieval"),
            },
            "config": {
                "phase_bridge": self._config.phase_bridge,
                "phase_compression": self._config.phase_compression,
                "phase_retrieval": self._config.phase_retrieval,
                "phase_security": self._config.phase_security,
            },
        }
        # Mark pipeline as degraded if any enabled component is unhealthy
        phase_map = {
            "fidelity": self._config.phase_bridge,
            "security": self._config.phase_security,
            "toon": self._config.phase_compression,
            "retrieval": self._config.phase_retrieval,
        }
        for name, enabled in phase_map.items():
            if enabled and status["phases"][name] != "healthy":
                status["pipeline"] = "degraded"
                break

        return status

    # ------------------------------------------------------------------
    # Phase implementations (stubs — wired in task 11.1)
    # ------------------------------------------------------------------

    def _phase_fidelity_outbound(self, request: dict[str, Any]) -> dict[str, Any]:
        """Fidelity phase: inject spoofed headers.

        Adapts the request dict to the FidelityModule.intercept_request(headers, body)
        interface. Headers are stored/updated in request["_headers"].
        """
        if self._fidelity is None:
            return request
        headers = request.get("_headers", {})
        body = {k: v for k, v in request.items() if k != "_headers"}
        new_headers = self._fidelity.intercept_request(headers, body)
        return {**request, "_headers": new_headers}

    def _phase_security_outbound(self, request: dict[str, Any]) -> dict[str, Any]:
        """Security phase: scan and redact secrets from outbound payload."""
        if self._security is None:
            return request
        sanitized, _redactions = self._security.scan_outbound(request)
        return sanitized

    def _phase_toon_compress(self, request: dict[str, Any]) -> dict[str, Any]:
        """TOON phase: compress structured blocks in messages."""
        if self._toon is None:
            return request
        messages = request.get("messages", [])
        compressed = self._toon.compress(messages)
        return {**request, "messages": compressed}

    def _phase_retrieval_outbound(self, request: dict[str, Any]) -> dict[str, Any]:
        """Retrieval phase: reduce context via vector search."""
        if self._retrieval is None:
            return request
        messages = request.get("messages", [])
        if not messages:
            return request
        query = messages[-1].get("content", "") if messages[-1].get("role") == "user" else ""
        if not query:
            return request
        reduced = self._retrieval.build_reduced_context(query, messages)
        return {**request, "messages": reduced}

    def _phase_hitl_intercept(self, response: dict[str, Any]) -> dict[str, Any]:
        """Intercept HITL tool_calls (AskQuestion) and convert to content.

        Cursor's Plan mode uses an AskQuestion tool_call to pause and ask
        the user for approval. On free/limited tiers, this gets blocked
        behind an "upgrade" button. This phase converts AskQuestion tool_calls
        into regular assistant content messages so the response flows through
        without requiring HITL approval.
        """
        choices = response.get("choices", [])
        if not choices:
            return response

        modified = False
        for choice in choices:
            finish_reason = choice.get("finish_reason", "")
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if finish_reason != "tool_calls" or not tool_calls:
                continue

            # Check if any tool_call is AskQuestion
            ask_questions = []
            other_tool_calls = []
            for tc in tool_calls:
                func = tc.get("function", {})
                func_name = func.get("name", "")
                if func_name.lower() in ("askquestion", "ask_question", "ask_user"):
                    ask_questions.append(tc)
                else:
                    other_tool_calls.append(tc)

            if not ask_questions:
                continue

            # Extract the question text from AskQuestion arguments
            question_texts = []
            for aq in ask_questions:
                func = aq.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    import json as _json
                    args = _json.loads(args_str) if isinstance(args_str, str) else args_str
                    question = args.get("question", args.get("message", args.get("content", str(args))))
                    question_texts.append(question)
                except Exception:
                    question_texts.append(args_str)

            # Convert to regular content message
            combined_question = "\n".join(question_texts)
            existing_content = message.get("content", "") or ""
            new_content = existing_content
            if combined_question:
                if new_content:
                    new_content += "\n\n"
                new_content += combined_question

            # Update the message
            message["content"] = new_content
            if other_tool_calls:
                message["tool_calls"] = other_tool_calls
            else:
                # Remove tool_calls entirely and change finish_reason
                message.pop("tool_calls", None)
                choice["finish_reason"] = "stop"

            modified = True
            logger.info("HITL intercepted: converted AskQuestion to content")

        return response

    def _phase_stream_health(self, response: dict[str, Any]) -> dict[str, Any]:
        """Stream health phase: extract reasoning tokens."""
        if self._fidelity is None:
            return response
        return response

    def _phase_toon_rehydrate(self, response: dict[str, Any]) -> dict[str, Any]:
        """TOON phase: re-hydrate compressed content in response."""
        if self._toon is None:
            return response
        choices = response.get("choices", [])
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", "")
            if isinstance(content, str) and content:
                message["content"] = self._toon.rehydrate(content)
        return response

    def _phase_security_inbound(self, response: dict[str, Any]) -> dict[str, Any]:
        """Security phase: scan inbound code blocks for CVEs."""
        if self._security is None:
            return response
        scanned, _findings = self._security.scan_inbound(response)
        return scanned

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        phase_name: str,
        phase_fn: Any,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single pipeline phase with graceful degradation.

        Requirement 11.4: If a phase fails, skip it, log the error,
        and continue with remaining phases.
        """
        try:
            return phase_fn(data)
        except Exception:
            logger.exception(
                "Pipeline phase '%s' failed; skipping (graceful degradation)",
                phase_name,
            )
            return data

    def _check_component_health(self, component_name: str) -> str:
        """Check health of a single component.

        Returns 'healthy', 'unavailable', or 'disabled'.
        """
        module_map: dict[str, Any] = {
            "fidelity": (self._fidelity, self._config.phase_bridge),
            "security": (self._security, self._config.phase_security),
            "toon": (self._toon, self._config.phase_compression),
            "retrieval": (self._retrieval, self._config.phase_retrieval),
        }
        module, enabled = module_map.get(component_name, (None, False))
        if not enabled:
            return "disabled"
        if module is None:
            return "unavailable"
        # If the module has a health_check method, call it
        if hasattr(module, "health_check"):
            try:
                return module.health_check()
            except Exception:
                return "unhealthy"
        return "healthy"
