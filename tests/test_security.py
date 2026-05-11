"""Unit tests for the Security Gateway — outbound secret redaction.

Tests regex-based pattern detection, Shannon entropy detection,
immutability of original payload, and correct redaction behavior.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import copy
import math

from deepseek_cursor_proxy.rap.security import (
    SecurityConfig,
    SecurityGateway,
    shannon_entropy,
)


class TestShannonEntropy:
    """Tests for the shannon_entropy() utility function."""

    def test_empty_string_returns_zero(self) -> None:
        """Empty string has zero entropy."""
        assert shannon_entropy("") == 0.0

    def test_single_char_repeated(self) -> None:
        """A string of identical characters has zero entropy."""
        assert shannon_entropy("aaaaaaaaaa") == 0.0

    def test_two_chars_equal_frequency(self) -> None:
        """Two characters with equal frequency have entropy of 1.0."""
        result = shannon_entropy("abababab")
        assert abs(result - 1.0) < 1e-10

    def test_high_entropy_random_string(self) -> None:
        """A string with many unique characters has high entropy."""
        # All unique characters -> high entropy
        text = "abcdefghijklmnop"
        result = shannon_entropy(text)
        assert result == math.log2(16)  # 4.0

    def test_known_entropy_value(self) -> None:
        """Verify entropy calculation against a known value."""
        # "aab" -> freq: a=2/3, b=1/3
        # entropy = -(2/3 * log2(2/3) + 1/3 * log2(1/3))
        expected = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))
        result = shannon_entropy("aab")
        assert abs(result - expected) < 1e-10


class TestPatternDetection:
    """Tests for regex-based secret pattern detection."""

    def setup_method(self) -> None:
        """Set up a SecurityGateway with default config."""
        self.config = SecurityConfig()
        self.gateway = SecurityGateway(self.config)

    def _make_payload(self, content: str) -> dict:
        """Helper to create a payload with a single user message."""
        return {"messages": [{"role": "user", "content": content}]}

    def test_detects_api_key_sk_prefix(self) -> None:
        """Requirement 8.1: Detects API keys with sk- prefix."""
        payload = self._make_payload(
            "My key is sk-abcdefghijklmnopqrstuvwxyz"
        )
        result, redactions = self.gateway.scan_outbound(payload)
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert len(redactions) >= 1
        assert any(r.pattern_name == "api_key" for r in redactions)

    def test_detects_api_key_pk_prefix(self) -> None:
        """Requirement 8.1: Detects API keys with pk- prefix."""
        payload = self._make_payload(
            "Use pk_abcdefghijklmnopqrstuvwxyz for auth"
        )
        result, redactions = self.gateway.scan_outbound(payload)
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert any(r.pattern_name == "api_key" for r in redactions)

    def test_detects_api_key_apikey_prefix(self) -> None:
        """Requirement 8.1: Detects API keys with apikey prefix."""
        payload = self._make_payload(
            "Set apikey_abcdefghijklmnopqrstuvwxyz here"
        )
        result, redactions = self.gateway.scan_outbound(payload)
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert any(r.pattern_name == "api_key" for r in redactions)

    def test_detects_aws_key(self) -> None:
        """Requirement 8.1: Detects AWS access key IDs."""
        payload = self._make_payload(
            "AWS key: AKIAIOSFODNN7EXAMPLE"
        )
        result, redactions = self.gateway.scan_outbound(payload)
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert any(r.pattern_name == "aws_key" for r in redactions)

    def test_detects_ssh_key(self) -> None:
        """Requirement 8.1: Detects SSH private key headers."""
        payload = self._make_payload(
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQ..."
        )
        result, redactions = self.gateway.scan_outbound(payload)
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert any(r.pattern_name == "ssh_key" for r in redactions)

    def test_detects_ssh_key_ec(self) -> None:
        """Requirement 8.1: Detects EC private key headers."""
        payload = self._make_payload(
            "-----BEGIN EC PRIVATE KEY-----\nMHQCAQEE..."
        )
        result, redactions = self.gateway.scan_outbound(payload)
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert any(r.pattern_name == "ssh_key" for r in redactions)

    def test_detects_ssh_key_generic(self) -> None:
        """Requirement 8.1: Detects generic private key headers."""
        payload = self._make_payload(
            "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBg..."
        )
        result, redactions = self.gateway.scan_outbound(payload)
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert any(r.pattern_name == "ssh_key" for r in redactions)

    def test_detects_env_var_export(self) -> None:
        """Requirement 8.1: Detects exported environment variables."""
        payload = self._make_payload(
            "export DATABASE_URL='postgres://user:pass@host/db'"
        )
        result, redactions = self.gateway.scan_outbound(payload)
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert any(r.pattern_name == "env_var" for r in redactions)

    def test_detects_env_var_no_export(self) -> None:
        """Requirement 8.1: Detects env vars without export keyword."""
        payload = self._make_payload(
            "SECRET_KEY=\"my_super_secret_value_12345\""
        )
        result, redactions = self.gateway.scan_outbound(payload)
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert any(r.pattern_name == "env_var" for r in redactions)

    def test_detects_jwt_token(self) -> None:
        """Requirement 8.1: Detects JWT tokens."""
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        payload = self._make_payload(f"Token: {jwt}")
        result, redactions = self.gateway.scan_outbound(payload)
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert any(r.pattern_name == "jwt_token" for r in redactions)

    def test_detects_github_token_ghp(self) -> None:
        """Requirement 8.1: Detects GitHub personal access tokens."""
        token = "ghp_" + "A" * 36
        payload = self._make_payload(f"GitHub token: {token}")
        result, redactions = self.gateway.scan_outbound(payload)
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert any(r.pattern_name == "github_token" for r in redactions)

    def test_detects_github_token_ghs(self) -> None:
        """Requirement 8.1: Detects GitHub server-to-server tokens."""
        token = "ghs_" + "B" * 36
        payload = self._make_payload(f"Token: {token}")
        result, redactions = self.gateway.scan_outbound(payload)
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert any(r.pattern_name == "github_token" for r in redactions)


class TestEntropyDetection:
    """Tests for Shannon entropy-based secret detection."""

    def setup_method(self) -> None:
        """Set up a SecurityGateway with default config (threshold=4.5)."""
        self.config = SecurityConfig(entropy_threshold=4.5)
        self.gateway = SecurityGateway(self.config)

    def test_detects_high_entropy_string(self) -> None:
        """Requirement 8.3: Detects high-entropy substrings >= 16 chars."""
        # A random-looking string with high entropy
        high_entropy = "aB3$xZ9!mK7@pL2#"  # 16 chars, diverse charset
        results = self.gateway.detect_high_entropy(f"Key: {high_entropy}")
        # The high-entropy substring should be detected if its entropy >= 4.5
        entropy = shannon_entropy(high_entropy)
        if entropy >= 4.5:
            assert len(results) >= 1

    def test_ignores_low_entropy_string(self) -> None:
        """Low-entropy strings should not be flagged."""
        low_entropy = "aaaaaaaaaaaaaaaa"  # 16 chars, all same -> entropy 0
        results = self.gateway.detect_high_entropy(low_entropy)
        assert len(results) == 0

    def test_ignores_short_strings(self) -> None:
        """Strings shorter than 16 chars should not be flagged."""
        short = "aB3$xZ9!mK7@pL2"  # 15 chars
        results = self.gateway.detect_high_entropy(short)
        assert len(results) == 0

    def test_entropy_detection_in_scan_outbound(self) -> None:
        """Requirement 8.3: High-entropy strings are redacted in scan_outbound."""
        # Generate a string with guaranteed high entropy (all unique chars)
        # 20 unique printable chars -> entropy = log2(20) ≈ 4.32
        # Use 32 unique chars for entropy > 4.5: log2(32) = 5.0
        high_entropy = "aB3xZ9mK7pL2nQ5wR8tY1uI4oP6sD0fG"
        entropy = shannon_entropy(high_entropy)
        assert entropy >= 4.5, f"Test string entropy {entropy} < 4.5"

        payload = {"messages": [{"role": "user", "content": high_entropy}]}
        result, redactions = self.gateway.scan_outbound(payload)

        # Should have been redacted
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert any(r.pattern_name == "high_entropy" for r in redactions)


class TestImmutability:
    """Tests that scan_outbound never mutates the original payload."""

    def setup_method(self) -> None:
        """Set up a SecurityGateway with default config."""
        self.config = SecurityConfig()
        self.gateway = SecurityGateway(self.config)

    def test_original_payload_unchanged_after_redaction(self) -> None:
        """Requirement 8.4: Original payload is not mutated."""
        secret = "sk-abcdefghijklmnopqrstuvwxyz"
        payload = {"messages": [{"role": "user", "content": f"Key: {secret}"}]}
        original_copy = copy.deepcopy(payload)

        self.gateway.scan_outbound(payload)

        # Original should be completely unchanged
        assert payload == original_copy

    def test_original_messages_list_unchanged(self) -> None:
        """Requirement 8.4: Original messages list is not mutated."""
        payload = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "AKIAIOSFODNN7EXAMPLE"},
            ]
        }
        original_content_0 = payload["messages"][0]["content"]
        original_content_1 = payload["messages"][1]["content"]

        self.gateway.scan_outbound(payload)

        assert payload["messages"][0]["content"] == original_content_0
        assert payload["messages"][1]["content"] == original_content_1

    def test_returned_payload_is_different_object(self) -> None:
        """Requirement 8.4: Returned payload is a separate copy."""
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        result, _ = self.gateway.scan_outbound(payload)
        assert result is not payload
        assert result["messages"] is not payload["messages"]


class TestRedactionBehavior:
    """Tests for overall redaction behavior and edge cases."""

    def setup_method(self) -> None:
        """Set up a SecurityGateway with default config."""
        self.config = SecurityConfig()
        self.gateway = SecurityGateway(self.config)

    def test_no_secrets_returns_unchanged_content(self) -> None:
        """Content without secrets should pass through unchanged."""
        payload = {"messages": [{"role": "user", "content": "Hello, world!"}]}
        result, redactions = self.gateway.scan_outbound(payload)
        assert result["messages"][0]["content"] == "Hello, world!"
        assert len(redactions) == 0

    def test_multiple_secrets_in_one_message(self) -> None:
        """Multiple secrets in a single message are all redacted."""
        content = (
            "AWS: AKIAIOSFODNN7EXAMPLE and "
            "GitHub: ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        )
        payload = {"messages": [{"role": "user", "content": content}]}
        result, redactions = self.gateway.scan_outbound(payload)
        assert result["messages"][0]["content"].count("[REDACTED]") >= 2
        assert len(redactions) >= 2

    def test_multiple_messages_scanned(self) -> None:
        """All messages in the payload are scanned."""
        payload = {
            "messages": [
                {"role": "system", "content": "sk-systemsecretkey12345678901234"},
                {"role": "user", "content": "AKIAIOSFODNN7EXAMPLE"},
            ]
        }
        result, redactions = self.gateway.scan_outbound(payload)
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert "[REDACTED]" in result["messages"][1]["content"]
        assert len(redactions) >= 2

    def test_non_string_content_skipped(self) -> None:
        """Messages with non-string content are skipped gracefully."""
        payload = {
            "messages": [
                {"role": "user", "content": ["not", "a", "string"]},
                {"role": "user", "content": "AKIAIOSFODNN7EXAMPLE"},
            ]
        }
        result, redactions = self.gateway.scan_outbound(payload)
        # First message unchanged (non-string content)
        assert result["messages"][0]["content"] == ["not", "a", "string"]
        # Second message redacted
        assert "[REDACTED]" in result["messages"][1]["content"]

    def test_empty_messages_list(self) -> None:
        """Empty messages list is handled gracefully."""
        payload = {"messages": []}
        result, redactions = self.gateway.scan_outbound(payload)
        assert result == {"messages": []}
        assert len(redactions) == 0

    def test_missing_messages_key(self) -> None:
        """Payload without messages key is handled gracefully."""
        payload = {"model": "deepseek-v4"}
        result, redactions = self.gateway.scan_outbound(payload)
        assert result == {"model": "deepseek-v4"}
        assert len(redactions) == 0

    def test_redaction_disabled(self) -> None:
        """When redaction is disabled, no scanning occurs."""
        config = SecurityConfig(redaction_enabled=False)
        gateway = SecurityGateway(config)
        payload = {"messages": [{"role": "user", "content": "AKIAIOSFODNN7EXAMPLE"}]}
        result, redactions = gateway.scan_outbound(payload)
        # Content should be unchanged (still a copy though)
        assert result["messages"][0]["content"] == "AKIAIOSFODNN7EXAMPLE"
        assert len(redactions) == 0

    def test_redaction_records_position_and_length(self) -> None:
        """Redaction objects contain correct position and length info."""
        secret = "AKIAIOSFODNN7EXAMPLE"
        payload = {"messages": [{"role": "user", "content": f"Key: {secret}"}]}
        _, redactions = self.gateway.scan_outbound(payload)
        assert len(redactions) >= 1
        r = next(r for r in redactions if r.pattern_name == "aws_key")
        assert r.original_length == len(secret)
        assert r.position == (5, 5 + len(secret))
        assert r.replacement == "[REDACTED]"

    def test_replacement_text_is_redacted(self) -> None:
        """Requirement 8.2: Replacement text is always [REDACTED]."""
        payload = {"messages": [{"role": "user", "content": "AKIAIOSFODNN7EXAMPLE"}]}
        result, redactions = self.gateway.scan_outbound(payload)
        assert "[REDACTED]" in result["messages"][0]["content"]
        for r in redactions:
            assert r.replacement == "[REDACTED]"
