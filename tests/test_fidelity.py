"""Unit tests for the Fidelity Module.

Tests header spoofing, idempotency, header preservation, endpoint routing,
and reasoning token pass-through.
Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3
"""

from deepseek_cursor_proxy.rap.fidelity import FidelityConfig, FidelityModule


class TestFidelityModule:
    """Tests for FidelityModule.intercept_request()."""

    def setup_method(self) -> None:
        """Set up a default FidelityModule for tests."""
        self.config = FidelityConfig()
        self.module = FidelityModule(self.config)

    def test_injects_cursor_plan_header(self) -> None:
        """Requirement 1.1: Injects X-Cursor-Plan: pro."""
        headers: dict[str, str] = {"Authorization": "Bearer sk-test"}
        result = self.module.intercept_request(headers, {})
        assert result["X-Cursor-Plan"] == "pro"

    def test_injects_cursor_tier_header(self) -> None:
        """Requirement 1.2: Injects X-Cursor-Tier: unlimited."""
        headers: dict[str, str] = {"Authorization": "Bearer sk-test"}
        result = self.module.intercept_request(headers, {})
        assert result["X-Cursor-Tier"] == "unlimited"

    def test_preserves_original_headers(self) -> None:
        """Requirement 1.3: All original headers are preserved."""
        headers = {
            "Authorization": "Bearer sk-test",
            "Content-Type": "application/json",
            "X-Custom": "value",
        }
        result = self.module.intercept_request(headers, {})
        for key, value in headers.items():
            assert result[key] == value

    def test_does_not_mutate_original_headers(self) -> None:
        """Requirement 1.3: Original dict is not mutated."""
        headers = {"Authorization": "Bearer sk-test"}
        original_copy = dict(headers)
        self.module.intercept_request(headers, {})
        assert headers == original_copy

    def test_idempotency(self) -> None:
        """Requirement 1.4: Applying twice produces same result as once."""
        headers = {"Authorization": "Bearer sk-test", "X-Foo": "bar"}
        first_result = self.module.intercept_request(headers, {})
        second_result = self.module.intercept_request(first_result, {})
        assert first_result == second_result

    def test_idempotency_with_existing_spoof_headers(self) -> None:
        """Requirement 1.4: Idempotent even if spoof headers already present."""
        headers = {
            "Authorization": "Bearer sk-test",
            "X-Cursor-Plan": "pro",
            "X-Cursor-Tier": "unlimited",
        }
        result = self.module.intercept_request(headers, {})
        assert result == headers

    def test_routes_to_byok_endpoint(self) -> None:
        """Requirement 1.5: Routes to configured BYOK endpoint."""
        assert self.module.get_endpoint() == "https://api.deepseek.com"

    def test_custom_byok_endpoint(self) -> None:
        """Requirement 1.5: Custom BYOK endpoint is respected."""
        config = FidelityConfig(byok_endpoint="https://custom.api.example.com")
        module = FidelityModule(config)
        assert module.get_endpoint() == "https://custom.api.example.com"

    def test_empty_headers_input(self) -> None:
        """Edge case: empty headers dict still gets spoof headers."""
        result = self.module.intercept_request({}, {})
        assert result["X-Cursor-Plan"] == "pro"
        assert result["X-Cursor-Tier"] == "unlimited"
        assert len(result) == 2

    def test_custom_spoof_headers(self) -> None:
        """Custom spoof headers are injected correctly."""
        config = FidelityConfig(spoof_headers={
            "X-Cursor-Plan": "pro",
            "X-Cursor-Tier": "unlimited",
            "X-Custom-Spoof": "enabled",
        })
        module = FidelityModule(config)
        result = module.intercept_request({"Host": "example.com"}, {})
        assert result["X-Custom-Spoof"] == "enabled"
        assert result["Host"] == "example.com"

    def test_body_does_not_affect_header_injection(self) -> None:
        """Body content does not influence header injection."""
        headers = {"Authorization": "Bearer sk-test"}
        body = {"model": "deepseek-v4", "messages": [{"role": "user", "content": "hi"}]}
        result = self.module.intercept_request(headers, body)
        assert result["X-Cursor-Plan"] == "pro"
        assert result["X-Cursor-Tier"] == "unlimited"
        assert result["Authorization"] == "Bearer sk-test"


class TestReasoningExtraction:
    """Tests for FidelityModule.extract_reasoning_stream().

    Requirements: 2.1, 2.2, 2.3
    """

    def setup_method(self) -> None:
        """Set up a default FidelityModule with reasoning enabled."""
        self.config = FidelityConfig(reasoning_stream_enabled=True)
        self.module = FidelityModule(self.config)

    def test_extracts_reasoning_content(self) -> None:
        """Requirement 2.1: Extracts reasoning_content from SSE chunk."""
        chunk = {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "Let me think about this step by step..."
                    }
                }
            ]
        }
        result = self.module.extract_reasoning_stream(chunk)
        assert result == "Let me think about this step by step..."

    def test_forwards_reasoning_without_modification(self) -> None:
        """Requirement 2.2: Reasoning tokens forwarded without modification."""
        reasoning_text = "Step 1: Analyze the input.\nStep 2: Consider edge cases.\n"
        chunk = {"choices": [{"delta": {"reasoning_content": reasoning_text}}]}
        result = self.module.extract_reasoning_stream(chunk)
        assert result == reasoning_text

    def test_returns_none_when_reasoning_content_absent(self) -> None:
        """Requirement 2.3: Returns None when reasoning_content is missing."""
        chunk = {"choices": [{"delta": {"content": "Hello, world!"}}]}
        result = self.module.extract_reasoning_stream(chunk)
        assert result is None

    def test_returns_none_when_delta_has_no_reasoning(self) -> None:
        """Requirement 2.3: Returns None for delta without reasoning_content."""
        chunk = {"choices": [{"delta": {"role": "assistant"}}]}
        result = self.module.extract_reasoning_stream(chunk)
        assert result is None

    def test_returns_none_when_choices_empty(self) -> None:
        """Requirement 2.3: Returns None when choices list is empty."""
        chunk = {"choices": []}
        result = self.module.extract_reasoning_stream(chunk)
        assert result is None

    def test_returns_none_when_choices_missing(self) -> None:
        """Requirement 2.3: Returns None when choices key is absent."""
        chunk = {"id": "chatcmpl-123", "object": "chat.completion.chunk"}
        result = self.module.extract_reasoning_stream(chunk)
        assert result is None

    def test_returns_none_when_delta_missing(self) -> None:
        """Requirement 2.3: Returns None when delta key is absent."""
        chunk = {"choices": [{"index": 0, "finish_reason": None}]}
        result = self.module.extract_reasoning_stream(chunk)
        assert result is None

    def test_returns_none_when_disabled(self) -> None:
        """Requirement 2.2: Returns None when reasoning_passthrough disabled."""
        config = FidelityConfig(reasoning_stream_enabled=False)
        module = FidelityModule(config)
        chunk = {
            "choices": [{"delta": {"reasoning_content": "some reasoning"}}]
        }
        result = module.extract_reasoning_stream(chunk)
        assert result is None

    def test_handles_empty_reasoning_content(self) -> None:
        """Edge case: empty string reasoning_content is returned as-is."""
        chunk = {"choices": [{"delta": {"reasoning_content": ""}}]}
        result = self.module.extract_reasoning_stream(chunk)
        assert result == ""

    def test_handles_empty_chunk(self) -> None:
        """Edge case: completely empty chunk returns None."""
        result = self.module.extract_reasoning_stream({})
        assert result is None

    def test_handles_non_dict_choices_entry(self) -> None:
        """Edge case: non-dict entry in choices returns None."""
        chunk = {"choices": ["not a dict"]}
        result = self.module.extract_reasoning_stream(chunk)
        assert result is None

    def test_handles_non_dict_delta(self) -> None:
        """Edge case: non-dict delta returns None."""
        chunk = {"choices": [{"delta": "not a dict"}]}
        result = self.module.extract_reasoning_stream(chunk)
        assert result is None

    def test_handles_none_choices(self) -> None:
        """Edge case: None choices value returns None."""
        chunk = {"choices": None}
        result = self.module.extract_reasoning_stream(chunk)
        assert result is None

    def test_preserves_special_characters_in_reasoning(self) -> None:
        """Requirement 2.2: Special characters preserved without modification."""
        reasoning = "∀x ∈ ℝ: x² ≥ 0\n```python\ndef f(x): return x**2\n```"
        chunk = {"choices": [{"delta": {"reasoning_content": reasoning}}]}
        result = self.module.extract_reasoning_stream(chunk)
        assert result == reasoning

    def test_preserves_multiline_reasoning(self) -> None:
        """Requirement 2.2: Multi-line reasoning preserved exactly."""
        reasoning = "Line 1\nLine 2\nLine 3\n"
        chunk = {"choices": [{"delta": {"reasoning_content": reasoning}}]}
        result = self.module.extract_reasoning_stream(chunk)
        assert result == reasoning
